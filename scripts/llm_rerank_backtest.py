#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3 实验：LLM 排序层增量 α。

背景（L1 已证结论）：规则层筛选（缩量回调+均线支撑）在 6 年样本上无正向 α 且方向反向。
L3 回答最后一个 α 假设：在同等信息下，LLM 重排/选股能否跑赢规则排序与随机？

盲测协议（防前视）：
- LLM 只能看到 ≤t 日的指标快照（价格/均线/量比/动量/波动率），绝无任何 t 日之后的数据；
- 前向收益 = close(t+hold)/close(t) - 1，收盘进收盘出，无成本口径（与 L1 一致，便于对照）；
- 股票池展示顺序随机打乱，防位置偏差。

双臂设计：
- Arm A（生产问题）：池 = 规则分 Top-20 → LLM 重排 Top-10，对比 规则 Top-10 / 池内随机 10；
  回答"LLM 能否在规则层给定的池子里排序出增量"。
- Arm B（选股能力）：池 = 当日全宇宙随机 20 → LLM 选 Top-10，对比 池内随机 10；
  回答"LLM 自己有没有选股 α"。

用法：
  python scripts/llm_rerank_backtest.py --pilot 3          # 小样冒烟（6 次 LLM 调用）
  python scripts/llm_rerank_backtest.py --max-days 30      # 正式实验（~60 次调用）
  python scripts/llm_rerank_backtest.py --max-days 30 --raw-out reports/l3_raw.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from offline_rule_backtest import (  # noqa: E402
    _mean,
    _stdev,
    build_indicators,
    load_bars,
    load_index,
    paired_t_pvalue,
    score_stock,
)

DB_PATH = ROOT / "data" / "stock_history.db"

POOL_SIZE = 20
PICK_K = 10
CALL_INTERVAL = 20.0   # 智谱免费档限流：两次调用间隔（秒）
LLM_TIMEOUT = 120
RETRY_BACKOFF = (20, 40, 60, 90)  # 重试退避（秒）


# ---------------------------------------------------------------- LLM

def _llm_config() -> Tuple[str, str, str]:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    model = os.environ.get("LITELLM_MODEL", "openai/glm-4.5-flash")
    base = os.environ.get("LLM_ZHIPU_BASE_URL", "")
    key = os.environ.get("LLM_ZHIPU_API_KEY") or (
        os.environ.get("LLM_ZHIPU_API_KEYS", "").split(",")[0]
    )
    if not (base and key):
        raise SystemExit("缺少 LLM_ZHIPU_BASE_URL / LLM_ZHIPU_API_KEY 配置")
    return model, base, key


def llm_pick(pool_codes: List[str], snapshot: Dict[str, dict], rng: random.Random,
             model: str, base: str, key: str, retries: int = 4) -> Optional[List[str]]:
    """让 LLM 从池中选未来 hold 天最可能涨的 K 只。只喂 ≤t 日数据。"""
    import litellm

    shown = list(pool_codes)
    rng.shuffle(shown)  # 防位置偏差
    lines = []
    for code in shown:
        ind = snapshot[code]
        align = "多" if ind["ma5"] > ind["ma10"] > ind["ma20"] else (
            "半" if ind["ma5"] > ind["ma10"] else "空")
        lines.append(
            f"{code}: 收盘{ind['close']:.2f} 均线{align} "
            f"20日动量{ind.get('mom20') or 0:+.1%} 量比{ind.get('vol_ratio') or 1:.2f} "
            f"5日乖离{ind.get('bias5') or 0:+.1%} 20日波动{ind.get('vol20') or 0:.3f}"
        )
    prompt = (
        f"以下是 {len(shown)} 只 A 股在某一交易日收盘后的技术面快照"
        f"（均线多头=MA5>MA10>MA20）：\n" + "\n".join(lines) +
        f"\n\n假设次日开盘等权买入，请从中选出未来 {PICK_K} 个交易日最可能上涨的 "
        f"{PICK_K} 只。只输出 JSON 数组（如 [\"600519\",...]），不要解释。"
    )
    for attempt in range(retries + 1):
        try:
            time.sleep(CALL_INTERVAL)
            r = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=key,
                api_base=base,
                max_tokens=800,
                temperature=0.2,
                timeout=LLM_TIMEOUT,
                # 排序任务无需长推理；GLM-4.5 关闭 thinking 可避开超时且不被限流拖死
                extra_body={"thinking": {"type": "disabled"}},
            )
            text = (r.choices[0].message.content or "").strip()
            m = re.search(r"\[.*?\]", text, re.S)
            if m:
                codes = re.findall(r"\d{6}", m.group(0))
                valid = [c for c in codes if c in pool_codes][:PICK_K]
                if len(valid) >= PICK_K // 2:
                    return valid
            # 解析失败兜底：按出现顺序取代码
            codes = re.findall(r"\d{6}", text)
            valid = [c for c in codes if c in pool_codes][:PICK_K]
            if len(valid) >= PICK_K // 2:
                return valid
        except Exception as e:  # noqa: BLE001
            print(f"    LLM 调用失败({attempt + 1}/{retries + 1}): {type(e).__name__}: {str(e)[:120]}", flush=True)
            if attempt < retries:
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
    return None


# ---------------------------------------------------------------- 主流程

def fwd_return(ind_map: Dict[str, dict], series: List[dict],
               t_date: str, t_end: str) -> Optional[float]:
    """t→t_end 收盘收益。两日都有数据才算。"""
    a = ind_map.get(t_date)
    b = ind_map.get(t_end)
    if a is None or b is None or not a["close"]:
        return None
    return b["close"] / a["close"] - 1.0


def portfolio_ret(picks: List[str], rets: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [rets[c] for c in picks if rets.get(c) is not None]
    if len(vals) < max(3, len(picks) // 2):
        return None
    return sum(vals) / len(vals)


def main() -> int:
    ap = argparse.ArgumentParser(description="L3：LLM 排序层增量 α 实验")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--hold", type=int, default=5, help="前向持有交易日数")
    ap.add_argument("--step", type=int, default=10, help="采样步长（交易日）")
    ap.add_argument("--max-days", type=int, default=30, help="最多采样天数")
    ap.add_argument("--mc", type=int, default=200, help="随机基准模拟次数/日")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="", help="覆盖 LITELLM_MODEL（如 openai/glm-4.5-air）；仍走智谱通道")
    ap.add_argument("--pilot", type=int, default=0, help="冒烟模式：只跑前 N 天")
    ap.add_argument("--out", default=str(ROOT / "reports" / "l3_llm_rerank.md"))
    ap.add_argument("--raw-out", default="", help="逐日原始结果 JSON 路径")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model, base, key = _llm_config()
    if args.model:
        model = args.model

    conn_path = args.db
    import sqlite3

    conn = sqlite3.connect(conn_path)
    bars = load_bars(conn)
    index_series = load_index(conn)
    indicators = build_indicators(bars)
    conn.close()
    print(f"[load] 个股 {len(bars)} 只 | 指数 {len(index_series)} 根", flush=True)

    trading_days = [r["date"] for r in index_series]
    idx_closes = [r["close"] for r in index_series]

    def state_at(i: int) -> str:
        if i < 59:
            return "unknown"
        ma20 = _mean(idx_closes[i - 19: i + 1])
        ma60 = _mean(idx_closes[i - 59: i + 1])
        c = idx_closes[i]
        return "STRONG" if (ma20 and ma60 and c > ma20 and ma20 > ma60) else "BEAR"

    # 采样日（非重叠：步长 ≥ hold）
    # 指数起点可能早于个股数据起点（本次：指数 2020-02 vs 个股 2020-07），
    # 必须把采样日约束在个股数据覆盖区间内，否则宇宙为空。
    from collections import Counter

    date_counts: Counter = Counter()
    for ind_map in indicators.values():
        date_counts.update(ind_map.keys())
    usable = {d for d, n in date_counts.items() if n >= 100}
    sample_idx = [
        s for s in range(59, len(trading_days) - args.hold, args.step)
        if trading_days[s] in usable
    ]
    if args.pilot:
        sample_idx = sample_idx[: args.pilot]
    elif len(sample_idx) > args.max_days:
        pick_at = [round(i * (len(sample_idx) - 1) / (args.max_days - 1)) for i in range(args.max_days)]
        sample_idx = [sample_idx[i] for i in sorted(set(pick_at))]
    print(f"[plan] 采样 {len(sample_idx)} 天 | hold={args.hold} | 模型 {model}", flush=True)

    day_rows: List[dict] = []
    for n, s in enumerate(sample_idx, 1):
        t_date, t_end = trading_days[s], trading_days[s + args.hold]
        state = state_at(s)

        # 当日可用宇宙（有 t 指标且 t_end 有数据）
        universe: Dict[str, dict] = {}
        for code, ind_map in indicators.items():
            ind = ind_map.get(t_date)
            if ind is not None and ind_map.get(t_end) is not None:
                universe[code] = ind
        if len(universe) < POOL_SIZE * 2:
            print(f"({n}/{len(sample_idx)}) {t_date} 宇宙不足({len(universe)})，跳过", flush=True)
            continue

        vols20 = [i["vol20"] for i in universe.values() if i.get("vol20") is not None]
        med_vol = _mean(sorted(vols20)[len(vols20) // 2: len(vols20) // 2 + 1]) if vols20 else None

        rets = {c: fwd_return(indicators[c], bars[c], t_date, t_end) for c in universe}
        rets = {c: r for c, r in rets.items() if r is not None}

        # 双臂池
        scored = sorted(
            ((score_stock(ind, med_vol), c) for c, ind in universe.items()),
            key=lambda x: (-x[0], x[1]),
        )
        pool_a = [c for _, c in scored[:POOL_SIZE] if c in rets]
        pool_b = rng.sample([c for c in universe if c in rets], POOL_SIZE)

        # LLM 两臂
        t0 = time.time()
        picks_a = llm_pick(pool_a, universe, rng, model, base, key)
        picks_b = llm_pick(pool_b, universe, rng, model, base, key)
        dt = time.time() - t0

        if not picks_a or not picks_b:
            print(f"({n}/{len(sample_idx)}) {t_date} LLM 失败，跳过", flush=True)
            continue

        # 基准：各池内随机 K 只（MC 均值）
        def mc_mean(pool: List[str]) -> float:
            ms = []
            for _ in range(args.mc):
                picks = rng.sample(pool, PICK_K)
                ms.append(_mean([rets[c] for c in picks]))
            return _mean(ms)

        row = {
            "date": t_date,
            "state": state,
            "armA_llm": portfolio_ret(picks_a, rets),
            "armA_rule": portfolio_ret(pool_a[:PICK_K], rets),
            "armA_rand": mc_mean(pool_a),
            "armA_pool": _mean([rets[c] for c in pool_a]),
            "armB_llm": portfolio_ret(picks_b, rets),
            "armB_rand": mc_mean(pool_b),
            "armB_pool": _mean([rets[c] for c in pool_b]),
            "picks_a": picks_a,
            "picks_b": picks_b,
            "llm_sec": round(dt, 1),
        }
        if None in (row["armA_llm"], row["armA_rule"], row["armB_llm"]):
            print(f"({n}/{len(sample_idx)}) {t_date} 收益数据缺失，跳过", flush=True)
            continue
        day_rows.append(row)
        print(
            f"({n}/{len(sample_idx)}) {t_date} {state} | A: LLM{row['armA_llm']:+.2%} "
            f"vs 规则{row['armA_rule']:+.2%} 随机{row['armA_rand']:+.2%} | "
            f"B: LLM{row['armB_llm']:+.2%} vs 随机{row['armB_rand']:+.2%} | {dt:.0f}s",
            flush=True,
        )
        time.sleep(0.5)

    if len(day_rows) < 5:
        print(f"有效样本仅 {len(day_rows)} 天，不足以出结论（需 ≥5）")
        return 1

    report = render(day_rows, args, model)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n[done] 报告 → {args.out}（N={len(day_rows)}）")

    if args.raw_out:
        Path(args.raw_out).write_text(
            json.dumps(day_rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"[done] 原始数据 → {args.raw_out}")
    return 0


def _t_block(llm: List[float], bench: List[float], label: str) -> str:
    diffs = [a - b for a, b in zip(llm, bench)]
    n = len(diffs)
    mean_d = _mean(diffs)
    sd = _stdev(diffs) or 0.0
    t, p = paired_t_pvalue(diffs)
    ir = mean_d / sd if sd else None
    wins = sum(1 for d in diffs if d > 0)
    sig = "✅ 显著" if (p is not None and p < 0.05) else "❌ 不显著"
    return (
        f"| {label} | {n} | {_mean(llm):+.3%} | {_mean(bench):+.3%} | {mean_d:+.3%} "
        f"| {wins}/{n} | {t:.2f} | {p:.3f} | {ir:.2f} | {sig} |"
    )


def render(rows: List[dict], args, model: str) -> str:
    dates = [r["date"] for r in rows]
    states = {}
    for r in rows:
        states[r["state"]] = states.get(r["state"], 0) + 1

    def col(key: str) -> List[float]:
        return [r[key] for r in rows]

    lines = [
        "# L3 实验报告：LLM 排序层增量 α",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}",
        f"- 模型：`{model}`（temperature=0.2，盲测：LLM 仅见 ≤t 日技术面快照）",
        f"- 采样：{len(rows)} 个交易日（步长 {args.step}，非重叠）| 区间 {dates[0]} → {dates[-1]}",
        f"- 市场态分布：{'、'.join(f'{k} {v}天' for k, v in states.items())}",
        f"- 前向口径：hold={args.hold} 交易日收盘→收盘，等权，未计成本（与 L1 同口径）",
        f"- 池：A=规则分Top-{POOL_SIZE}；B=当日随机 {POOL_SIZE} 只；各选/排 Top-{PICK_K}",
        f"- 随机基准：每池 {args.mc} 次蒙特卡洛取均值；t 检验为逐日配对",
        "",
        "## 结果",
        "",
        "| 对比 | N | LLM 组合 | 基准 | 增量 α | LLM 胜/负 | t | p | IR | 判定 |",
        "|---|---|---|---|---|---|---|---|---|---|",
        _t_block(col("armA_llm"), col("armA_rule"), "A: LLM重排 vs 规则Top10"),
        _t_block(col("armA_llm"), col("armA_rand"), "A: LLM重排 vs 池内随机10"),
        _t_block(col("armB_llm"), col("armB_rand"), "B: LLM选股 vs 池内随机10"),
        "",
        "## 参照（组合日均收益水平）",
        "",
        "| 组合 | 日均收益 |",
        "|---|---|",
    ]
    for key, label in [
        ("armA_rule", "A 规则Top10"),
        ("armA_pool", "A 规则池全池均值"),
        ("armB_pool", "B 随机池全池均值"),
        ("armA_rand", "A 池内随机10"),
        ("armB_rand", "B 池内随机10"),
        ("armA_llm", "A LLM重排Top10"),
        ("armB_llm", "B LLM选股Top10"),
    ]:
        lines.append(f"| {label} | {_mean(col(key)):+.3%} |")

    # 结论
    a_llm, a_rule = _mean(col("armA_llm")), _mean(col("armA_rule"))
    b_llm, b_rand = _mean(col("armB_llm")), _mean(col("armB_rand"))
    _, p_a = paired_t_pvalue([a - b for a, b in zip(col("armA_llm"), col("armA_rand"))])
    _, p_b = paired_t_pvalue([a - b for a, b in zip(col("armB_llm"), col("armB_rand"))])
    lines += [
        "",
        "## 结论",
        "",
    ]
    if b_llm > b_rand and (p_b is not None and p_b < 0.05):
        lines.append("- **Arm B 显著为正：LLM 选股层本身有 α** → 值得把 LLM 排序接入生产验证（L2 前瞻跟踪并行）。")
    elif b_llm > b_rand:
        lines.append(f"- Arm B 方向为正但不显著（p={p_b:.3f}）→ 方向上有希望，加样本再验。")
    else:
        lines.append("- **Arm B 为负或零：LLM 在纯技术面信息下没有选股 α**（与 L1 规则层证伪合并 → 当前筛选链路的 α 假设全部落空，需换信息源或换策略层）。")
    if a_llm > a_rule and (p_a is not None and p_a < 0.05):
        lines.append("- Arm A 显著为正：LLM 能在规则池内重排出增量（但若池本身反向，重排的天花板有限）。")
    elif a_llm > a_rule:
        lines.append(f"- Arm A 方向为正但不显著（p={p_a:.3f}）。")
    else:
        lines.append("- Arm A 无增量：LLM 在规则池内重排跑不赢规则排序。")
    lines += [
        "",
        "## 局限",
        "",
        "1. LLM 只喂技术面快照（无新闻/热点/资金流）——比生产环境信息少，此结论是 LLM 增量的**下界**；",
        "2. glm-4.5-flash 是免费档模型，生产用模型更强则上限更高；",
        "3. 未计交易成本；hold=5 收盘对收盘，与 L1 可直接对照；",
        "4. 样本内抽样而非全量逐日，置信区间见上表 p 值。",
        "",
        "## 附：逐日明细",
        "",
        "| 日期 | 市场态 | A:LLM | A:规则 | A:随机 | B:LLM | B:随机 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['date']} | {r['state']} | {r['armA_llm']:+.2%} | {r['armA_rule']:+.2%} "
            f"| {r['armA_rand']:+.2%} | {r['armB_llm']:+.2%} | {r['armB_rand']:+.2%} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
