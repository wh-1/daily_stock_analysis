#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 离线规则层历史回放：验证「技术面筛选」是否存在 alpha。

## 与线上真实策略的关系（务必理解，避免误读结论）

线上 run_screening = 技术面筛选 + 新闻热点 + LLM 排序。
本脚本**只能离线复刻"技术面筛选层"**（新闻/LLM 无法在 250 个历史交易日重跑）。

因此本脚本回答且仅回答一个问题：
    「如果只用规则层选股，相对同宇宙随机选股有没有超额收益？」

结论的两种走向：
- 规则层**无 alpha** → LLM 层无从拯救，策略需要重构；
- 规则层**有 alpha** → 再进一步验证 LLM 层是增益还是拖累（L3）。

## 方法

- 数据：data/stock_history.db（stock_daily 成分股 + index_daily 沪深300）
- 每个交易日 t：用 t 日及之前的数据打分（无前视泄漏），取 Top-K，
  计量 t 收盘 → t+N 收盘的等权组合收益
- 基准 1（核心）：同日同宇宙随机抽 K 只，蒙特卡洛 M 次 → 配对日超额
- 基准 2：沪深300 指数同期收益
- 统计：胜率、期望值、alpha、配对 t 检验 p 值、IR、最大回撤、Calmar
- 时间分段：--split-date 之前样本内、之后样本外

用法:
    python scripts/offline_rule_backtest.py
    python scripts/offline_rule_backtest.py --hold 10 --top-k 8 --mc 2000
    python scripts/offline_rule_backtest.py --non-overlap --split-date 2026-04-01
"""
from __future__ import annotations

import argparse
import math
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "stock_history.db"
REPORT_DIR = REPO_ROOT / "reports"

# 打分权重（参考 strategies/shrink_pullback.yaml + bull_trend.yaml 的可离线部分）
W_UPTREND_FULL = 2.0      # MA5 > MA10 > MA20
W_UPTREND_PART = 1.0      # MA5 > MA10
W_BIAS_TIGHT = 2.0        # |乖离率| < 2%
W_BIAS_LOOSE = 1.0        # |乖离率| < 5%
W_SHRINK_VOL = 1.0        # 当日量 < 5 日均量 70%
W_PULLBACK = 1.0          # 回踩 MA5(±1%) 或 MA10(±2%)
W_MOMENTUM = 1.0          # 近 20 日涨幅落在 [0, 20%]
W_LOW_VOL = 1.0           # 波动率低于当日全池中位数


# ---------------------------------------------------------------- 数据加载

def load_bars(conn: sqlite3.Connection) -> Dict[str, List[dict]]:
    """读取全部个股日K，按 code 分组并按日期升序。"""
    rows = conn.execute(
        "SELECT code, date, open, high, low, close, volume FROM stock_daily ORDER BY code, date"
    ).fetchall()
    bars: Dict[str, List[dict]] = defaultdict(list)
    for code, date, o, h, l, c, v in rows:
        if c is None or c <= 0:
            continue
        bars[code].append(
            {"date": date, "open": o, "high": h, "low": l, "close": c, "volume": v or 0.0}
        )
    for code in bars:
        bars[code].sort(key=lambda r: r["date"])
    return dict(bars)


def load_index(conn: sqlite3.Connection, code: str = "000300") -> List[dict]:
    rows = conn.execute(
        "SELECT date, close FROM index_daily WHERE code = ? ORDER BY date", (code,)
    ).fetchall()
    return [{"date": d, "close": c} for d, c in rows if c and c > 0]


# ---------------------------------------------------------------- 指标

def _mean(seq: Sequence[float]) -> Optional[float]:
    return sum(seq) / len(seq) if seq else None


def _stdev(seq: Sequence[float]) -> Optional[float]:
    if len(seq) < 2:
        return None
    m = _mean(seq)
    if m is None:
        return None
    return math.sqrt(sum((x - m) ** 2 for x in seq) / (len(seq) - 1))


def build_indicators(bars: Dict[str, List[dict]], min_history: int = 60) -> Dict[str, Dict[str, dict]]:
    """为每只股票的每个交易日预计算技术指标。返回 {code: {date: ind}}。"""
    out: Dict[str, Dict[str, dict]] = {}
    for code, series in bars.items():
        n = len(series)
        ind_map: Dict[str, dict] = {}
        closes = [b["close"] for b in series]
        vols = [b["volume"] for b in series]
        rets = [None] + [
            (closes[i] / closes[i - 1] - 1.0) if closes[i - 1] else 0.0 for i in range(1, n)
        ]
        for i in range(min_history - 1, n):
            c = closes[i]
            ma5 = _mean(closes[i - 4 : i + 1])
            ma10 = _mean(closes[i - 9 : i + 1])
            ma20 = _mean(closes[i - 19 : i + 1])
            ma60 = _mean(closes[i - 59 : i + 1])
            if None in (ma5, ma10, ma20, ma60):
                continue
            vol_ma5 = _mean(vols[i - 4 : i + 1]) or 0.0
            window_rets = [r for r in rets[i - 19 : i + 1] if r is not None]
            ind_map[series[i]["date"]] = {
                "idx": i,
                "close": c,
                "ma5": ma5,
                "ma10": ma10,
                "ma20": ma20,
                "ma60": ma60,
                "vol_ratio": (vols[i] / vol_ma5) if vol_ma5 > 0 else None,
                "bias5": (c - ma5) / ma5 if ma5 else None,
                "mom20": (c / closes[i - 20] - 1.0) if i >= 20 and closes[i - 20] else None,
                "vol20": _stdev(window_rets),
            }
        if ind_map:
            out[code] = ind_map
    return out


# ---------------------------------------------------------------- 打分

def score_stock(ind: dict, med_vol20: Optional[float]) -> float:
    """规则层打分（全部由历史 OHLCV 离线可得）。"""
    s = 0.0
    if ind["ma5"] > ind["ma10"] > ind["ma20"]:
        s += W_UPTREND_FULL
    elif ind["ma5"] > ind["ma10"]:
        s += W_UPTREND_PART

    bias = ind.get("bias5")
    if bias is not None:
        ab = abs(bias)
        if ab < 0.02:
            s += W_BIAS_TIGHT
        elif ab < 0.05:
            s += W_BIAS_LOOSE

    vr = ind.get("vol_ratio")
    if vr is not None and vr < 0.7:
        s += W_SHRINK_VOL

    # 回踩检测：贴近 MA5(±1%) 或 MA10(±2%)
    c = ind["close"]
    if abs(c / ind["ma5"] - 1.0) < 0.01 or abs(c / ind["ma10"] - 1.0) < 0.02:
        s += W_PULLBACK

    mom = ind.get("mom20")
    if mom is not None and 0.0 <= mom <= 0.20:
        s += W_MOMENTUM

    v20 = ind.get("vol20")
    if v20 is not None and med_vol20 is not None and v20 < med_vol20:
        s += W_LOW_VOL
    return s


# ---------------------------------------------------------------- 统计

def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_t_pvalue(diffs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """对配对差值序列做 t 检验（H0: 均值=0），大样本用正态近似。返回 (t, p)。"""
    n = len(diffs)
    if n < 2:
        return None, None
    m = sum(diffs) / n
    sd = _stdev(diffs)
    if not sd:
        return None, None
    t = m / (sd / math.sqrt(n))
    p = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return t, p


def max_drawdown(nav: Sequence[float]) -> float:
    """最大回撤（%）。"""
    peak = nav[0]
    mdd = 0.0
    for v in nav:
        if v > peak:
            peak = v
        dd = (v / peak - 1.0) * 100.0 if peak else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def summarize(
    strat: Sequence[float],
    bench: Sequence[float],
    label: str,
    allow_nav: bool = True,
) -> dict:
    """strat/bench 为同长度的日收益序列（%），bench 为随机基准配对值。

    allow_nav=False 时跳过累计收益/最大回撤/Calmar —— 重叠窗口下把各期收益
    连乘会重复计算重叠区间（如持有 20 日则放大约 20 倍），净值指标无意义。
    """
    n = len(strat)
    if n == 0:
        return {"label": label, "n": 0}
    diffs = [s - b for s, b in zip(strat, bench)]
    wins = sum(1 for d in diffs if d > 0)
    m_strat = sum(strat) / n
    m_bench = sum(bench) / n
    m_diff = sum(diffs) / n
    sd_diff = _stdev(diffs) or 0.0
    t, p = paired_t_pvalue(diffs)

    total_ret: Optional[float] = None
    mdd: Optional[float] = None
    calmar: Optional[float] = None
    if allow_nav:
        nav = [1.0]
        for r in strat:
            nav.append(nav[-1] * (1.0 + r / 100.0))
        mdd = max_drawdown(nav)
        total_ret = (nav[-1] - 1.0) * 100.0
        calmar = (total_ret / abs(mdd)) if mdd else None

    wins_abs = sum(1 for s in strat if s > 0)
    gains = [s for s in strat if s > 0]
    losses = [s for s in strat if s < 0]
    avg_gain = sum(gains) / len(gains) if gains else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    win_rate = wins_abs / n * 100.0
    expectancy = (win_rate / 100.0) * avg_gain + (1 - win_rate / 100.0) * avg_loss

    return {
        "label": label,
        "n": n,
        "win_rate_pct": win_rate,
        "avg_gain_pct": avg_gain,
        "avg_loss_pct": avg_loss,
        "expectancy_pct": expectancy,
        "strat_mean_pct": m_strat,
        "bench_mean_pct": m_bench,
        "alpha_pct": m_diff,
        "sd_diff_pct": sd_diff,
        "ir": (m_diff / sd_diff) if sd_diff else None,
        "t_stat": t,
        "p_value": p,
        "beat_bench_rate_pct": wins / n * 100.0,
        "total_return_pct": total_ret,
        "max_drawdown_pct": mdd,
        "calmar": calmar,
    }


# ---------------------------------------------------------------- 主流程

def run_backtest(
    bars: Dict[str, List[dict]],
    indicators: Dict[str, Dict[str, dict]],
    index_series: List[dict],
    *,
    hold: int,
    top_k: int,
    mc: int,
    seed: int,
    non_overlap: bool,
    require_uptrend: bool,
    pick_mode: str = "top",
) -> dict:
    rng = random.Random(seed)

    # 全局交易日（以指数为准，保证指数与个股日期对齐）
    trading_days = [r["date"] for r in index_series]
    idx_close = {r["date"]: r["close"] for r in index_series}

    # 沪深300 market_state（close>MA20 且 MA20>MA60）
    idx_closes = [r["close"] for r in index_series]
    state_by_date: Dict[str, str] = {}
    for i in range(len(idx_closes)):
        if i < 59:
            state_by_date[trading_days[i]] = "unknown"
            continue
        ma20 = _mean(idx_closes[i - 19 : i + 1])
        ma60 = _mean(idx_closes[i - 59 : i + 1])
        c = idx_closes[i]
        if ma20 and ma60 and c > ma20 and ma20 > ma60:
            state_by_date[trading_days[i]] = "STRONG"
        else:
            state_by_date[trading_days[i]] = "BEAR"

    total_days = len(trading_days)
    step = hold if non_overlap else 1
    strat_rets: List[float] = []
    bench_rets: List[float] = []
    mkt_rets: List[float] = []
    dates_used: List[str] = []
    states_used: List[str] = []
    picks_log: List[dict] = []

    for s in range(0, total_days - hold, step):
        t_date = trading_days[s]
        t_end = trading_days[s + hold]
        state = state_by_date.get(t_date, "unknown")
        if state == "unknown":
            continue

        # 当日候选：既有 t 日指标、也能取到 t_end 收盘
        cands: List[Tuple[float, float, str]] = []  # (score, mom, code)
        pool: List[str] = []
        vols20: List[float] = []
        snapshot: Dict[str, dict] = {}
        for code, ind_map in indicators.items():
            ind = ind_map.get(t_date)
            if ind is None:
                continue
            series = bars[code]
            end_ind = ind_map.get(t_end)
            if end_ind is None:
                # 停牌等导致终点无数据，跳过
                continue
            if require_uptrend and not (ind["ma5"] > ind["ma10"] > ind["ma20"]):
                continue
            snapshot[code] = ind
            pool.append(code)
            if ind.get("vol20") is not None:
                vols20.append(ind["vol20"])

        if len(pool) < top_k * 2:
            continue

        med_vol20 = sorted(vols20)[len(vols20) // 2] if vols20 else None
        for code in pool:
            ind = snapshot[code]
            sc = score_stock(ind, med_vol20)
            cands.append((sc, ind.get("mom20") or 0.0, code))

        cands.sort(key=lambda x: (-x[0], -x[1]))
        if pick_mode == "bottom":
            picked = [c for _, _, c in cands[-top_k:]]  # 反向诊断：得分最低
        else:
            picked = [c for _, _, c in cands[:top_k]]

        # 策略收益：t 收盘 → t_end 收盘 等权
        rets = []
        for code in picked:
            c0 = snapshot[code]["close"]
            c1 = indicators[code][t_end]["close"]
            rets.append((c1 / c0 - 1.0) * 100.0)
        strat_ret = sum(rets) / len(rets)

        # 随机基准：同日同池随机 top_k 只，蒙特卡洛
        mc_sum = 0.0
        for _ in range(mc):
            sample = rng.sample(pool, top_k)
            r = sum(
                (indicators[c][t_end]["close"] / snapshot[c]["close"] - 1.0) * 100.0
                for c in sample
            ) / top_k
            mc_sum += r
        bench_ret = mc_sum / mc

        # 市场基准：沪深300 同期
        c0i = idx_close.get(t_date)
        c1i = idx_close.get(t_end)
        mkt_ret = ((c1i / c0i - 1.0) * 100.0) if (c0i and c1i) else 0.0

        strat_rets.append(strat_ret)
        bench_rets.append(bench_ret)
        mkt_rets.append(mkt_ret)
        dates_used.append(t_date)
        states_used.append(state)
        picks_log.append({"date": t_date, "state": state, "picks": picked, "ret": strat_ret})

    return {
        "strat": strat_rets,
        "bench": bench_rets,
        "mkt": mkt_rets,
        "dates": dates_used,
        "states": states_used,
        "picks_log": picks_log,
        "hold": hold,
        "top_k": top_k,
        "mc": mc,
        "non_overlap": non_overlap,
        "require_uptrend": require_uptrend,
        "pick_mode": pick_mode,
    }


def split_by_date(res: dict, split_date: str) -> Tuple[dict, dict]:
    """按日期切分样本内/样本外。"""
    idxs_in = [i for i, d in enumerate(res["dates"]) if d < split_date]
    idxs_out = [i for i, d in enumerate(res["dates"]) if d >= split_date]

    def subset(idxs: List[int]) -> dict:
        return {
            "strat": [res["strat"][i] for i in idxs],
            "bench": [res["bench"][i] for i in idxs],
            "mkt": [res["mkt"][i] for i in idxs],
            "dates": [res["dates"][i] for i in idxs],
        }

    return subset(idxs_in), subset(idxs_out)


# ---------------------------------------------------------------- 报告

def _fmt(v: Optional[float], nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def render_report(res: dict, sub_in: dict, sub_out: dict, args) -> str:
    hold, top_k, mc = res["hold"], res["top_k"], res["mc"]
    overlap_note = "非重叠窗口" if res["non_overlap"] else "重叠窗口（日频滚动，存在自相关）"
    pick_note = "得分最高（正常）" if res["pick_mode"] == "top" else "得分最低（**反向诊断**）"
    up_note = "强制多头排列" if res["require_uptrend"] else "不强制多头排列（作为加分项）"

    allow_nav = res["non_overlap"]
    full = summarize(res["strat"], res["bench"], "全样本", allow_nav)
    in_s = summarize(sub_in["strat"], sub_in["bench"], "样本内", allow_nav)
    out_s = summarize(sub_out["strat"], sub_out["bench"], "样本外", allow_nav)
    mkt_s = summarize(res["strat"], res["mkt"], "vs 沪深300", allow_nav)

    def block(d: dict) -> str:
        if not d.get("n"):
            return f"**{d['label']}**：样本不足\n"
        verdict = "—"
        if d.get("p_value") is not None:
            verdict = "显著 ✅" if d["p_value"] < 0.05 else "不显著 ❌"
        return (
            f"**{d['label']}**（N={d['n']}）\n\n"
            f"| 指标 | 数值 |\n|---|---|\n"
            f"| 组合胜率（绝对正收益） | {_fmt(d['win_rate_pct'],1)}% |\n"
            f"| 平均盈利 / 平均亏损 | {_fmt(d['avg_gain_pct'],2)}% / {_fmt(d['avg_loss_pct'],2)}% |\n"
            f"| **期望值（每期）** | **{_fmt(d['expectancy_pct'],3)}%** |\n"
            f"| 策略均值 / 随机基准均值 | {_fmt(d['strat_mean_pct'],3)}% / {_fmt(d['bench_mean_pct'],3)}% |\n"
            f"| **超额 α（vs 随机）** | **{_fmt(d['alpha_pct'],3)}%** |\n"
            f"| 跑赢随机基准的天数占比 | {_fmt(d['beat_bench_rate_pct'],1)}% |\n"
            f"| t 统计量 / p 值 | {_fmt(d['t_stat'],3)} / {_fmt(d['p_value'],4)} → {verdict} |\n"
            f"| 信息比率 IR | {_fmt(d['ir'],3)} |\n"
            f"| 累计收益 / 最大回撤 | {_fmt(d['total_return_pct'],2)}% / {_fmt(d['max_drawdown_pct'],2)}% |\n"
            f"| Calmar | {_fmt(d['calmar'],3)} |\n"
        )

    is_bottom = res["pick_mode"] == "bottom"
    p = full.get("p_value")
    a = full.get("alpha_pct")
    if p is None or a is None:
        verdict = "样本不足，无法判定"
    elif a > 0 and p < 0.05:
        if is_bottom:
            verdict = (
                "⚠️ **因子方向是反的**：得分**最低**的组合存在统计显著的正 alpha，"
                "而得分最高的组合 alpha≈0 甚至为负。\n>\n> "
                "含义：当前规则层偏好的技术形态（多头排列 / 低乖离 / 缩量回踩 / 低波动）"
                "在样本期内**并未带来超额收益**，反而是被它打低分的股票（超跌 / 放量 / 高波动）"
                "跑得更好——这与 A 股的**短期反转效应**一致。\n>\n> "
                "**但不要据此直接反向交易**，先做完下面两项排查："
                "① 多重检验（本次跑了多组 hold×方向）；② 交易成本（见第八节）。"
            )
        else:
            verdict = "✅ 规则层存在统计显著的正 alpha，可进入 L3（验证 LLM 层增量）"
    elif a > 0:
        if is_bottom:
            verdict = "⚠️ 反向组合 alpha 为正但不显著，需更多样本确认（见样本外结果）"
        else:
            verdict = "⚠️ alpha 为正但不显著，样本不足或信号太弱，需继续累积真实信号"
    else:
        if is_bottom:
            verdict = "❌ 反向同样无 alpha，说明该套因子对收益**没有区分度**，筛选逻辑需重构"
        else:
            verdict = (
                "❌ 规则层无 alpha（甚至为负）。建议立刻跑 `--pick bottom` 做反向诊断："
                "若反向显著为正，说明**因子方向搞反了**（常见原因是 A 股短期反转效应），"
                "不是因子无效。"
            )
    # 样本外是否复现（防过拟合的关键一步）
    if out_s.get("n"):
        op = out_s.get("p_value")
        oa = out_s.get("alpha_pct")
        if oa is not None and oa > 0:
            if op is not None and op < 0.05:
                verdict += "\n>\n> ✅ **样本外复现**：方向一致且显著，过拟合风险较低。"
            else:
                verdict += (
                    f"\n>\n> ⚠️ **样本外方向一致但功效不足**：样本外 α={oa:.3f}%（甚至更大），"
                    f"但 N={out_s['n']} 太小导致 p={op:.3f} 未达显著。"
                    "属于「效应量在、样本不够」，需继续累积，不能算已证实。"
                )
        elif oa is not None:
            verdict += "\n>\n> ❌ **样本外未复现**（α 为负），高度怀疑为样本内过拟合/数据挖掘。"

    return f"""# L1 离线规则层历史回放报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、这次验证的是什么

**只验证「技术面筛选层」有没有 alpha**，不是线上完整策略。

线上 `run_screening` = 技术面筛选 + 新闻热点 + LLM 排序。
新闻与 LLM 无法在 {len(res['dates']) + hold} 个历史交易日重跑，因此本回放只复刻可离线计算的部分
（MA 排列、乖离率、缩量、回踩、动量、低波）。

> **读法**：若规则层都没有 alpha，LLM 层无从拯救；若规则层有 alpha，
> 才值得进一步验证 LLM 排序是增益还是拖累（L3）。

## 二、参数

| 参数 | 值 |
|---|---|
| 持有期 | {hold} 个交易日 |
| 每期选股数 Top-K | {top_k} |
| 蒙特卡洛次数 | {mc} |
| 采样方式 | {overlap_note} |
| 选股方向 | {pick_note} |
| 趋势门控 | {up_note} |
| 股票池 | 沪深300 成分（data/stock_history.db） |
| 基准 | ①同日同池随机 {top_k} 只（蒙特卡洛 {mc} 次）②沪深300 指数 |
| 起止 | {res['dates'][0] if res['dates'] else 'n/a'} → {res['dates'][-1] if res['dates'] else 'n/a'} |

评分权重：多头排列 +{W_UPTREND_FULL}/+{W_UPTREND_PART}，乖离率 <2%/+<5% +{W_BIAS_TIGHT}/+{W_BIAS_LOOSE}，
缩量(<0.7×5日均量) +{W_SHRINK_VOL}，回踩 MA5/MA10 +{W_PULLBACK}，
温和动量(0~20%) +{W_MOMENTUM}，低波动 +{W_LOW_VOL}。

## 三、核心结论

> {verdict}

## 四、全样本（vs 随机基准）

{block(full)}

## 五、vs 沪深300 指数（判断是否为 beta）

{block(mkt_s)}

## 六、样本内 / 样本外

切分日期：{args.split_date}

{block(in_s)}

{block(out_s)}

## 七、怎么读这些数字

- **期望值**必须为正，否则策略长期必亏（胜率高但盈亏比差也会亏）。
- **α（vs 随机）**是关键：它扣除了"同期同池随便买"的收益，才是选股本身的贡献。
- **p 值 < 0.05** 才说明 α 不太可能是运气；样本外 p 值同样重要。
- **IR = α / 波动**，>0.5 才算稳定的选股能力。
- **重叠窗口警告**：默认日频滚动会产生自相关，真实独立样本量小于 N；
  用 `--non-overlap` 可得近似独立样本的结果。

## 八、交易成本敏感度（决定能否落地）

未计成本的 α 不能直接当收益。按期换仓、双边成本按 0.1%~0.3% 估算：

| 项 | 数值 |
|---|---|
| 每年期数（250 交易日 / {hold}） | {250 / hold:.0f} 期 |
| 年化毛 α | {_fmt((full.get('alpha_pct') or 0) * 250 / hold, 1)}% |
| 年化交易成本（@0.1% / 期） | {_fmt(0.1 * 250 / hold, 1)}% |
| 年化交易成本（@0.3% / 期） | {_fmt(0.3 * 250 / hold, 1)}% |
| **年化净 α 区间** | **{_fmt((full.get('alpha_pct') or 0) * 250 / hold - 0.3 * 250 / hold, 1)}% ~ {_fmt((full.get('alpha_pct') or 0) * 250 / hold - 0.1 * 250 / hold, 1)}%** |

> 持有期越短、换手越高，成本吞噬越致命。若净 α 区间下沿为负，则该频率不可交易，
> 需拉长持有期或降低换手（例如只在新信号出现时调仓，而非每期全换）。

## 九、局限（必须知道）

1. **多重检验风险**：本次扫描了 hold×方向 多组组合。多组同时检验会抬高"偶然显著"的概率，
   因此**样本外复现比样本内 p 值更重要**，切勿只看全样本 p 值就下结论。
2. 只复刻**技术面规则层**，未包含新闻热点与 LLM 排序——这不等于线上完整策略的表现。
3. 股票池是**沪深300 成分股**，结论不推广到全市场小票。
4. 回测为**等权、满仓、无止损**的理想化模拟，未考虑资金曲线与仓位管理。
5. 历史区间约 6 年，覆盖了一轮牛熊但不含极端行情（如 2015、2018 全年）。
6. 重叠窗口会产生自相关、虚增显著性；本报告的净值类指标仅在非重叠模式下给出。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="L1 离线规则层历史回放")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--hold", type=int, default=20, help="持有交易日数，默认 20")
    parser.add_argument("--top-k", type=int, default=10, help="每期选股数，默认 10")
    parser.add_argument("--mc", type=int, default=1000, help="蒙特卡洛次数，默认 1000")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--non-overlap", action="store_true", help="非重叠窗口（近似独立样本）")
    parser.add_argument("--require-uptrend", action="store_true", help="强制 MA5>MA10>MA20 才入选")
    parser.add_argument(
        "--pick",
        choices=("top", "bottom"),
        default="top",
        help="top=得分最高 K 只（默认）；bottom=得分最低 K 只（反向诊断，检验因子方向是否搞反）",
    )
    parser.add_argument("--split-date", default="2026-04-01", help="样本内/外切分日期")
    parser.add_argument("--out", default="", help="报告输出路径（默认 reports/l1_offline_rule_backtest_YYYYMMDD.md）")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[err] 历史库不存在：{db_path}，先跑 scripts/backfill_stock_daily_history.py")
        return 1

    conn = sqlite3.connect(db_path)
    try:
        bars = load_bars(conn)
        index_series = load_index(conn)
    finally:
        conn.close()

    if not bars:
        print("[err] stock_daily 无数据")
        return 1
    if not index_series:
        print("[err] index_daily 无数据，先跑 scripts/fetch_index_history.py")
        return 1

    print(f"[load] 个股 {len(bars)} 只，指数 {len(index_series)} 根")
    indicators = build_indicators(bars)
    print(f"[load] 指标构建完成，可用标的 {len(indicators)} 只")

    res = run_backtest(
        bars,
        indicators,
        index_series,
        hold=args.hold,
        top_k=args.top_k,
        mc=args.mc,
        seed=args.seed,
        non_overlap=args.non_overlap,
        require_uptrend=args.require_uptrend,
        pick_mode=args.pick,
    )
    if not res["dates"]:
        print("[err] 无可用交易日，检查数据区间或降低 --hold")
        return 1
    print(f"[run] 完成 {len(res['dates'])} 期，{res['dates'][0]} → {res['dates'][-1]}")

    sub_in, sub_out = split_by_date(res, args.split_date)
    report = render_report(res, sub_in, sub_out, args)

    out_path = Path(args.out) if args.out else (
        REPORT_DIR / f"l1_offline_rule_backtest_{datetime.now().strftime('%Y%m%d')}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    full = summarize(res["strat"], res["bench"], "全样本", res["non_overlap"])
    print()
    print("=" * 60)
    print(f"全样本 N={full['n']}  ({'非重叠' if res['non_overlap'] else '重叠'}窗口)")
    print(f"  策略均值      {_fmt(full['strat_mean_pct'],3)}%")
    print(f"  随机基准均值  {_fmt(full['bench_mean_pct'],3)}%")
    print(f"  期望值        {_fmt(full['expectancy_pct'],3)}%")
    print(f"  超额 α(vs随机) {_fmt(full['alpha_pct'],3)}%")
    print(f"  p 值          {_fmt(full['p_value'],4)}")
    print(f"  IR            {_fmt(full['ir'],3)}")
    if res["non_overlap"]:
        print(f"  累计收益      {_fmt(full['total_return_pct'],2)}%")
        print(f"  最大回撤      {_fmt(full['max_drawdown_pct'],2)}%")
    else:
        print("  累计/回撤     n/a（重叠窗口净值无意义，加 --non-overlap 可得）")
    print("=" * 60)
    print(f"[report] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
