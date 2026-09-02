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
import re
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
# 市场态中文语义（与下方 _AUTO_STRATEGIES_BY_STATE 的进攻/震荡/防御注释对齐）
_STATE_CN = {"STRONG": "进攻", "MIX": "震荡", "BEAR": "防御"}


def _strategy_label(code: str) -> str:
    cn = _STRATEGY_CN.get(code)
    return f"{cn}（{code}）" if cn else str(code)


def _market_label(code: str) -> str:
    return _MARKET_CN.get(code, str(code))


def _ranking_label(mode: str) -> str:
    return _RANKING_CN.get(mode, str(mode))


# 数据源中文名。选股链路有两套互不相干的源：
#   快照链 src/services/screening/snapshot.py —— 全市场一次拉全，SNAPSHOT_SOURCE_PRIORITY 控制顺序
#   日K链 src/services/screening/daily.py    —— 每票各拉一次，DAILY_SOURCE 控制顺序
# 两者各自维护健康度（连挂 3 次冷却 300s），所以健康度不能只看快照。
_SOURCE_CN = {
    "sina": "新浪",
    "tencent": "腾讯",
    "efinance": "东财(efinance)",
    "akshare_em": "东财(akshare)",
    "em_datacenter": "东财(datacenter)",
    "akshare": "AKShare",
    "baostock": "Baostock",
    "tushare": "Tushare",
    "yfinance": "Yfinance",
    "last_good_cache": "快照缓存",
}


def _source_label(code: str) -> str:
    return _SOURCE_CN.get(str(code), str(code))


# 引擎已经把日 K 采集情况写进了 degradation 文本（pipeline.py:251-272），
# 只是没结构化进 json。这里做纯文本解析，避免改动 src/（engine-zero-modification）。
_DAILY_LINE_PATTERNS = {
    "attempt": re.compile(
        r"Daily K-line enrichment attempted (\d+) candidates, succeeded (\d+)"
    ),
    "sources": re.compile(r"Daily K-line sources: (.+)"),
    "quality": re.compile(r"Daily K-line quality flags: (.+)"),
    "ordering": re.compile(r"Daily K-line source ordering: (.+)"),
    "health": re.compile(r"Daily K-line source health: (.+)"),
    "errors": re.compile(r"Daily K-line enrichment row errors: (.+)"),
    "skipped": re.compile(r"Daily K-line enrichment skipped: (.+)"),
}


def _parse_counts(text: str) -> dict[str, int]:
    """解析 'tencent=3, sina=1' 这类计数串，忽略非整数项。"""
    counts: dict[str, int] = {}
    for item in str(text).split(","):
        name, _, raw = item.partition("=")
        name = name.strip()
        if not name:
            continue
        try:
            counts[name] = counts.get(name, 0) + int(raw.strip())
        except ValueError:
            continue
    return counts


def _daily_kline_summary(result: dict) -> dict:
    """从 result 还原本轮日 K 采集情况（源、命中次数、降级、健康度、错误）。"""
    info: dict = {
        "requested": False,
        "attempted": 0,
        "succeeded": int(result.get("daily_enrich_count") or 0),
        "sources": {},
        "primary": "",
        "quality_flags": {},
        "source_order_notes": [],
        "health_notes": [],
        "error_samples": [],
        "skipped_reason": "",
        "status": "not_requested",
    }
    for line in (str(x) for x in (result.get("degradation") or [])):
        m = _DAILY_LINE_PATTERNS["attempt"].search(line)
        if m:
            info["attempted"] = int(m.group(1))
            info["succeeded"] = int(m.group(2))
            info["requested"] = True
            continue
        m = _DAILY_LINE_PATTERNS["sources"].search(line)
        if m:
            info["sources"] = _parse_counts(m.group(1))
            info["requested"] = True
            continue
        m = _DAILY_LINE_PATTERNS["quality"].search(line)
        if m:
            info["quality_flags"] = _parse_counts(m.group(1))
            continue
        m = _DAILY_LINE_PATTERNS["health"].search(line)
        if m:
            info["health_notes"] = [x.strip() for x in m.group(1).split(";") if x.strip()]
            continue
        m = _DAILY_LINE_PATTERNS["ordering"].search(line)
        if m:
            info["source_order_notes"] = [x.strip() for x in m.group(1).split("|") if x.strip()]
            continue
        m = _DAILY_LINE_PATTERNS["errors"].search(line)
        if m:
            info["error_samples"] = [x.strip() for x in m.group(1).split("|") if x.strip()]
            continue
        m = _DAILY_LINE_PATTERNS["skipped"].search(line)
        if m:
            info["skipped_reason"] = m.group(1).strip()
            info["requested"] = True

    if info["sources"]:
        info["primary"] = max(info["sources"].items(), key=lambda kv: kv[1])[0]

    if not info["requested"]:
        info["status"] = "not_requested"
    elif info["skipped_reason"] and info["attempted"] == 0:
        info["status"] = "skipped"
    elif info["attempted"] > 0 and info["succeeded"] <= 0:
        info["status"] = "failed"
    elif info["succeeded"] < info["attempted"]:
        info["status"] = "partial"
    else:
        info["status"] = "ok"
    return info


def _merge_daily_summaries(entries: list[tuple[str, dict]]) -> dict:
    """多策略各跑一次日 K，合并成一份：计数相加、备注去重、保留分策略明细。"""
    if not entries:
        return {}
    if len(entries) == 1:
        return entries[0][1]

    merged = dict(entries[0][1])
    sources: dict[str, int] = {}
    flags: dict[str, int] = {}
    attempted = succeeded = 0
    for _strat, info in entries:
        attempted += int(info.get("attempted") or 0)
        succeeded += int(info.get("succeeded") or 0)
        for name, count in (info.get("sources") or {}).items():
            sources[name] = sources.get(name, 0) + int(count)
        for name, count in (info.get("quality_flags") or {}).items():
            flags[name] = flags.get(name, 0) + int(count)
    merged["attempted"] = attempted
    merged["succeeded"] = succeeded
    merged["sources"] = sources
    merged["quality_flags"] = flags
    merged["requested"] = any(info.get("requested") for _, info in entries)
    if sources:
        merged["primary"] = max(sources.items(), key=lambda kv: kv[1])[0]
    for key in ("source_order_notes", "health_notes", "error_samples"):
        seen: list[str] = []
        for _strat, info in entries:
            for item in info.get(key) or []:
                if item not in seen:
                    seen.append(item)
        merged[key] = seen
    skipped = [info.get("skipped_reason") for _, info in entries if info.get("skipped_reason")]
    merged["skipped_reason"] = " | ".join(skipped)
    statuses = [info.get("status") for _, info in entries]
    for candidate in ("failed", "partial", "ok", "skipped", "not_requested"):
        if candidate in statuses:
            merged["status"] = candidate
            break
    merged["per_strategy"] = {strat: info for strat, info in entries}
    return merged


def _daily_kline_brief(info: dict) -> str:
    """给报告/日志用的一行中文摘要。"""
    if not info:
        return "日K：未知"
    sources = info.get("sources") or {}
    attempted = int(info.get("attempted") or 0)
    succeeded = int(info.get("succeeded") or 0)
    if sources:
        detail = " ".join(f"{_kline_source_label(k)}×{v}" for k, v in sorted(sources.items(), key=lambda kv: -kv[1]))
        return f"日K：{detail}（{succeeded}/{attempted} 成功）"
    if not info.get("requested"):
        return "日K：未拉取（策略无需）"
    reason = info.get("skipped_reason") or (info.get("error_samples") or ["取数失败"])[0]
    return f"日K：拉取失败（0/{attempted}，{reason}）"


def _kline_source_label(key: str) -> str:
    """日K fetcher 名 → 中文（引擎内部名形如 'dsa:TencentFetcher'，剥壳后查中文表）。"""
    short = str(key).replace("dsa:", "").removesuffix("Fetcher").strip().lower()
    return _SOURCE_CN.get(short, short)


# auto 模式：市场状态 → 策略映射。与三态攻守体系的 CSI300 闸门口径对齐，
# 但这里是轻量代理实现，权威状态仍以本地三态引擎为准。
# 每态取 [首选, 替补] 两个策略交叉比对：双策略共振的票置信度更高。
_AUTO_STRATEGIES_BY_STATE = {
    "STRONG": ["volume_breakout", "capital_heat"],      # 进攻态：放量突破 × 资金热度
    "MIX": ["momentum_quality", "balanced_alpha"],      # 震荡态：趋势质量 × 均衡优选
    "BEAR": ["blue_chip_income", "low_volatility_quality"],  # 防御态：蓝筹红利 × 低波质量
}


def _http_json(url: str, timeout: float = 10.0):
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _csi300_from_eastmoney(lmt: int) -> list[float]:
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        "?secid=1.000300&fields1=f1,f2,f3,f4,f5,f6"
        f"&fields2=f51,f53&klt=101&fqt=1&end=20500101&lmt={lmt}"
    )
    klines = (_http_json(url).get("data") or {}).get("klines") or []
    closes = []
    for row in klines:
        parts = str(row).split(",")
        if len(parts) >= 2:
            try:
                closes.append(float(parts[1]))
            except ValueError:
                continue
    return closes


def _csi300_from_tencent(lmt: int) -> list[float]:
    # 返回 data.sh000300.qfqday（或 day）：每行 [date, open, close, high, low, volume, ...]
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000300,day,,,{lmt},qfq"
    data = _http_json(url).get("data") or {}
    node = data.get("sh000300") or {}
    rows = node.get("qfqday") or node.get("day") or []
    closes = []
    for row in rows:
        if len(row) >= 3:
            try:
                closes.append(float(row[2]))
            except (TypeError, ValueError):
                continue
    return closes


def _csi300_from_sina(lmt: int) -> list[float]:
    # 返回 [{"day": "...", "close": "..."}, ...] 按时间升序
    url = (
        "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
        f"?symbol=sh000300&scale=240&ma=no&datalen={lmt}"
    )
    rows = _http_json(url)
    if not isinstance(rows, list):
        return []
    closes = []
    for row in rows:
        try:
            closes.append(float(row["close"]))
        except (KeyError, TypeError, ValueError):
            continue
    return closes


_CSI300_SOURCES = (
    ("eastmoney", _csi300_from_eastmoney),
    ("tencent", _csi300_from_tencent),
    ("sina", _csi300_from_sina),
)


def _fetch_csi300_closes(lmt: int = 120) -> list[float]:
    """拉沪深300日K收盘价，多源依次尝试：东财 push2his → 腾讯 → 新浪。

    海外 runner 上 push2his 频繁断连（Run #6/#7 实测），腾讯/新浪接口独立可用。
    """
    errors = []
    for name, fetch in _CSI300_SOURCES:
        try:
            closes = fetch(lmt)
            if len(closes) >= 60:
                return closes
            errors.append(f"{name}: 仅 {len(closes)} 根")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    raise RuntimeError("沪深300日K全部数据源失败: " + "; ".join(errors))


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


def _candidate_row(c: dict, max_reason: int = 56) -> str:
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
    daily_entries: list[tuple[str, dict]] = []
    for i, strat in enumerate(strategies):
        try:
            res = svc.screen(
                strategy=strat,
                market=args.market,
                max_results=args.max_results,
                selection_seed=args.seed,
            )
            results.append(res)
            # 日 K 采集情况要趁 res 还在时抽出来：合并后只保留第一个策略的 degradation
            daily_entries.append((strat, _daily_kline_summary(res)))
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

    # 数据源健康度：快照链与日 K 链各走一套，健康度互不相通，分开记。
    daily_info = _merge_daily_summaries(daily_entries)
    result["data_sources"] = {
        "snapshot": {
            "source": result.get("snapshot_source") or "",
            "count": result.get("snapshot_count") or 0,
            "errors": [str(x) for x in (result.get("source_errors") or [])],
        },
        "daily_kline": daily_info,
    }
    print(f"🛰 数据源：快照 {_source_label(result.get('snapshot_source') or '未知')}"
          f"（{result.get('snapshot_count') or 0} 只）｜{_daily_kline_brief(daily_info)}")

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
        # auto 双策略交叉：中文名用 × 强调两策略并联关系（与 --strategy auto 帮助文案口径一致）
        strategy_label = " × ".join(_STRATEGY_CN.get(s, s) for s in strategies_used)
        resonance_count = int(result.get("resonance_count") or 0)
        resonance_note = f"（共振 ★{resonance_count}）" if resonance_count else ""
    else:
        strategy_label = _strategy_label(result.get("strategy", args.strategy))
        resonance_note = ""

    # 概要：行情环境(组1) + 运行环境(组2)，每行一个字段、行内不串接。
    # 飞书 text 消息不渲染 md，一行一字段避免手机窄屏(~16字/行)随机折行；md 桌面阅读同样整齐。
    state = str(result.get("market_state") or "").strip().upper()
    if state:
        state_cn = _STATE_CN.get(state)
        lines.append(f"- 市场态：{state} · {state_cn}" if state_cn else f"- 市场态：{state}")
    market = str(result.get("market") or args.market).lower()
    if market and market != "cn":
        # A股是默认市场，不写以免噪音；港/美股才标注
        lines.append(f"- 市场：{_market_label(market)}")
    lines.append(f"- 策略：{strategy_label}")
    lines.append(f"- 入选：{len(candidates)} 只{resonance_note}")
    lines.append("")
    model = str(result.get("llm_model_used") or "").strip()
    if model:
        # provider 前缀只是 litellm 通道标识（如 gemini/xxx），对读者是噪音，只留模型名
        lines.append(f"- 模型：{model.split('/', 1)[-1]}")
    lines.append(
        f"- 快照：{_source_label(result.get('snapshot_source', 'n/a'))}"
        f" {result.get('snapshot_count', 'n/a')} 只"
    )
    lines.append(f"- {_daily_kline_brief(daily_info)}")
    lines.append(f"- 排名：{_ranking_label(result.get('ranking_mode', 'n/a'))}")

    if strategies_used and resonance_note:
        lines.append("")
        lines.append(f"★ = 双策略共振（{strategy_label}）")
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
