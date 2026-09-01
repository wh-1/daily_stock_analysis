#!/usr/bin/env python3
"""无头运行每日选股，结果写入 reports/。

设计目标：
- 不依赖 server / DB（db_manager=None 时不写历史，不影响产出）。
- 用于 GitHub Actions workflow 的「执行股票分析」step（已注入 LLM / 数据源 env）。
- 失败（如 LLM 限流、数据源不可用）以非零退出，由调用方决定是否软跳过。
"""
import argparse
import datetime
import json
import os
import sys

# 允许从仓库根目录运行：python scripts/run_screening.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import get_config
from src.services.screening_service import ScreeningService


def _first(d: dict, *keys, default=""):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


# 报告展示用中文名（代码保留在括号里，便于与 dispatch 输入对照）。
# 中文名对齐上游 strategies/*.yaml 的 description 语义，不是随便翻译。
_STRATEGY_CN = {
    "balanced_alpha": "均衡优选",
    "blue_chip_income": "蓝筹红利",
    "capital_heat": "资金热度",
    "dual_low": "双低稳健",
    "low_volatility_quality": "低波质量",
    "momentum_quality": "趋势质量",
    "oversold_reversal": "超跌反转",
    "quality_value": "质量价值",
    "shrink_pullback": "缩量回踩",
    "volume_breakout": "放量突破",
}
_MARKET_CN = {"cn": "A股", "hk": "港股", "us": "美股"}
_RANKING_CN = {"llm": "AI 排序", "screen_score": "本地评分"}


def _strategy_label(code: str) -> str:
    cn = _STRATEGY_CN.get(code)
    return f"{cn}（{code}）" if cn else str(code)


def _market_label(code: str) -> str:
    return _MARKET_CN.get(code, str(code))


def _ranking_label(mode: str) -> str:
    return _RANKING_CN.get(mode, str(mode))


# auto 模式：市场状态 → 策略映射。与三态攻守体系的 CSI300 闸门口径对齐，
# 但这里是轻量代理实现，权威状态仍以本地三态引擎为准。
# 每态取 [首选, 替补] 两个策略交叉比对：双策略共振的票置信度更高。
_AUTO_STRATEGIES_BY_STATE = {
    "STRONG": ["volume_breakout", "capital_heat"],      # 进攻态：放量突破 × 资金热度
    "MIX": ["momentum_quality", "balanced_alpha"],      # 震荡态：趋势质量 × 均衡优选
    "BEAR": ["blue_chip_income", "low_volatility_quality"],  # 防御态：蓝筹红利 × 低波质量
}


def _fetch_csi300_closes(lmt: int = 120) -> list[float]:
    """拉沪深300日K收盘价（东财 push2his，与快照主源同族，海外 runner 实测可达）。"""
    import urllib.request

    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        "?secid=1.000300&fields1=f1,f2,f3,f4,f5,f6"
        f"&fields2=f51,f53&klt=101&fqt=1&end=20500101&lmt={lmt}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    klines = (payload.get("data") or {}).get("klines") or []
    closes = []
    for row in klines:
        parts = str(row).split(",")
        if len(parts) >= 2:
            try:
                closes.append(float(parts[1]))
            except ValueError:
                continue
    return closes


def _detect_market_state() -> tuple[str, dict]:
    """三态代理判定：STRONG=收盘>MA20>MA60；BEAR=收盘<MA20 且 MA20<MA60；其余 MIX。"""
    closes = _fetch_csi300_closes()
    if len(closes) < 60:
        raise ValueError(f"CSI300 日K不足（{len(closes)} 根），无法判定市场状态")
    close = closes[-1]
    ma20 = sum(closes[-20:]) / 20
    ma60 = sum(closes[-60:]) / 60
    if close > ma20 > ma60:
        state = "STRONG"
    elif close < ma20 and ma20 < ma60:
        state = "BEAR"
    else:
        state = "MIX"
    return state, {"close": close, "ma20": round(ma20, 2), "ma60": round(ma60, 2)}


def _resolve_auto_strategies() -> tuple[str | None, list[str]]:
    """auto 模式：按市场状态返回 [首选, 替补] 两个策略；判定失败回退单策略 momentum_quality。"""
    try:
        state, detail = _detect_market_state()
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 市场状态判定失败（{exc}），回退默认策略 momentum_quality")
        return None, ["momentum_quality"]
    strategies = _AUTO_STRATEGIES_BY_STATE[state]
    print(
        f"🧭 市场状态 {state}（close={detail['close']} MA20={detail['ma20']} MA60={detail['ma60']}）"
        f"→ 双策略交叉 {strategies[0]} × {strategies[1]}"
    )
    return state, strategies


def _merge_strategy_results(results: list[dict], max_results: int) -> dict:
    """合并多策略结果：去重、标记共振、共振优先排序、截断到 max_results。

    排序键 = (命中策略数, 最高分)：共振票排前，同命中数内按分排。
    分数口径：优先 llm_score（同模型下可比），缺失则退 screen_score。
    """
    by_code: dict[str, dict] = {}
    for res in results:
        strat = str(res.get("strategy") or "")
        for c in res.get("candidates") or []:
            if not isinstance(c, dict):
                continue
            code = str(_first(c, "code", "symbol", "stock_code", "ts_code") or "")
            if not code:
                continue
            score = c.get("llm_score")
            if score is None:
                score = c.get("screen_score")
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            if code not in by_code:
                merged = dict(c)
                merged["source_strategies"] = [strat]
                merged["merge_score"] = score
                # 报告显示分与排序口径统一
                merged["score"] = score
                by_code[code] = merged
            else:
                m = by_code[code]
                if strat and strat not in m["source_strategies"]:
                    m["source_strategies"].append(strat)
                m["merge_score"] = max(m["merge_score"], score)
                m["score"] = m["merge_score"]

    ordered = sorted(
        by_code.values(),
        key=lambda x: (len(x["source_strategies"]), x["merge_score"]),
        reverse=True,
    )[:max_results]

    base = dict(results[0])
    base["candidates"] = ordered
    base["strategies_used"] = [str(r.get("strategy") or "") for r in results]
    base["resonance_count"] = sum(
        1 for c in ordered if len(c.get("source_strategies") or []) >= 2
    )
    return base


def _candidate_row(c: dict, max_reason: int = 30) -> str:
    """精简候选行：编号外的代码/名称/评分/短理由，理由超长截断。

    说明：飞书 text 消息不渲染 markdown，这里刻意不用 `**` 等符号。
    """
    code = _first(c, "code", "symbol", "stock_code", "ts_code")
    name = _first(c, "name", "stock_name")
    score = _first(c, "score", "rating", "rank_score", "total_score")
    reason = _first(c, "reason", "selection_logic", "comment", "logic")
    # 双策略共振标记（★=首选+替补同时命中）
    prefix = "★ " if len(c.get("source_strategies") or []) >= 2 else ""
    line = f"- {prefix}{code}" + (f" {name}" if name else "")
    if score != "":
        line += f"（{score}分）"
    if reason:
        reason = " ".join(str(reason).split()).strip()
        if len(reason) > max_reason:
            reason = reason[:max_reason] + "…"
        line += f"：{reason}"
    return line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy",
        default=os.environ.get("SCREENING_STRATEGY", "auto"),
        help=(
            "选股策略；auto=按市场状态自动选双策略交叉"
            "（STRONG→volume_breakout×capital_heat / MIX→momentum_quality×balanced_alpha / BEAR→blue_chip_income×low_volatility_quality）"
        ),
    )
    ap.add_argument("--market", default=os.environ.get("SCREENING_MARKET", "cn"))
    ap.add_argument("--max-results", type=int, default=int(os.environ.get("SCREENING_MAX_RESULTS", "20")))
    ap.add_argument("--seed", default=os.environ.get("SCREENING_SEED", ""))
    args = ap.parse_args()

    try:
        cfg = get_config()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 加载配置失败: {exc}")
        return 1

    svc = ScreeningService(cfg)
    status = svc.status()
    if not status.get("enabled"):
        print("⚠️ 选股功能未启用（SCREENING_ENABLED != true），跳过。")
        return 0

    market_state: str | None = None
    if args.strategy == "auto":
        market_state, strategies = _resolve_auto_strategies()
    else:
        strategies = [args.strategy]

    print(
        f"🎯 选股参数: strategy={'+'.join(strategies)} "
        f"market={args.market} max={args.max_results} seed={args.seed!r}"
    )

    def _screen_err_msg(exc: Exception) -> str:
        detail = getattr(exc, "detail", None)
        if isinstance(detail, dict):
            return json.dumps(detail, ensure_ascii=False)
        if isinstance(detail, str) and detail:
            return detail
        return str(exc)

    # 逐策略运行（共享同一份上游快照，边际成本低）；auto 下单边失败软降级
    results: list[dict] = []
    for i, strat in enumerate(strategies):
        try:
            res = svc.screen(
                strategy=strat,
                market=args.market,
                max_results=args.max_results,
                selection_seed=args.seed,
            )
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            if len(strategies) == 1:
                print(f"❌ 选股运行失败: {_screen_err_msg(exc)}")
                return 1
            role = "首选" if i == 0 else "替补"
            print(f"⚠️ {role}策略 {strat} 失败（{_screen_err_msg(exc)}），继续尝试剩余策略")

    if not results:
        print("❌ 选股运行失败：所有策略均无结果")
        return 1
    if len(results) == 1:
        result = results[0]
        if market_state:
            result["market_state"] = market_state
    else:
        result = _merge_strategy_results(results, args.max_results)
        if market_state:
            result["market_state"] = market_state
        print(
            f"🔀 双策略合并：{result['strategies_used']} → {len(result['candidates'])} 只"
            f"（共振 {result['resonance_count']} 只）"
        )

    # 报告日期统一用北京时间（UTC+8）：GitHub runner 是 UTC，直接用 today()
    # 会在北京凌晨跑出前一天的文件名，导致下游（如飞书推送按北京日期找文件）错位。
    from datetime import timezone, timedelta
    bjt_now = datetime.datetime.now(timezone(timedelta(hours=8)))
    date = bjt_now.strftime("%Y%m%d")
    os.makedirs("reports", exist_ok=True)
    json_path = f"reports/screening_{date}.json"
    md_path = f"reports/screening_{date}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    candidates = result.get("candidates") or []
    if candidates and isinstance(candidates[0], dict):
        print("🔎 candidate keys:", list(candidates[0].keys()))

    lines = []
    lines.append(f"# 每日选股 · {date}")
    lines.append("")
    strategies_used = [s for s in (result.get("strategies_used") or []) if s]
    if strategies_used:
        strategy_label = " + ".join(_STRATEGY_CN.get(s, s) for s in strategies_used)
        strategy_detail = " + ".join(strategies_used)
        resonance_note = f"（共振 {result.get('resonance_count', 0)} 只）"
    else:
        strategy_label = _strategy_label(result.get("strategy", args.strategy))
        strategy_detail = ""
        resonance_note = ""
    lines.append(
        f"- 策略 {strategy_label}"
        f" ｜ 市场 {_market_label(result.get('market', args.market))}"
        f" ｜ 入选 {len(candidates)} 只{resonance_note}"
    )
    lines.append(
        f"- 快照 {result.get('snapshot_source', 'n/a')}"
        f"（{result.get('snapshot_count', 'n/a')} 只）"
        f" ｜ 排名 {_ranking_label(result.get('ranking_mode', 'n/a'))}"
    )
    if strategies_used:
        lines.append(f"- ★ = 双策略共振（{strategy_detail}）")
    lines.append("")
    lines.append("## 入选标的")
    lines.append("")
    if candidates:
        for c in candidates:
            if isinstance(c, dict):
                lines.append(_candidate_row(c))
            else:
                lines.append(f"- {c}")
    else:
        lines.append("本轮未选出符合条件的标的。")
    # 风险提示用 LLM 的组合风险（中文）；warnings 是数据源诊断噪音（英文，
    # 如 sina 超时/efinance 解析失败），对报告读者无价值，只保留在 json 里。
    risk = str(result.get("llm_portfolio_risk") or "").strip()
    if risk:
        lines.append("")
        lines.append("## 风险提示")
        lines.append("")
        lines.append(risk)
    md = "\n".join(lines) + "\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    top = ", ".join(
        str(_first(c, "code", "symbol", "name") or c)
        for c in candidates[:10]
    )
    print(f"✅ 选股完成：{len(candidates)} 只 → {md_path} / {json_path}")
    print(f"📌 入选：{top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
