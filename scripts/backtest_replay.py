# -*- coding: utf-8 -*-
"""P1 历史回放器：按项目真实持仓纪律模拟买入→持有→退出，统计胜率/收益。

与仓库自带 BacktestEngine 的区别（该引擎只做固定窗口方向收益，不含持仓规则）：
本脚本实现完整规则：
  - 入场：信号日 T+1 开盘价 + 0.05% 成本
  - +8% 卖半仓（锁定部分利润）
  - -5% 止损（清仓）
  - 峰值 -5% 移动止盈（剩余仓）
  - 最长持仓 20 个交易日（第 20 日收盘清仓）
混合收益 = 0.5*止盈价 + 0.5*最终退出价，再扣入场成本。

用法：
  python scripts/backtest_replay.py            # 跑历史窗口自测 + 真实 09-02 选股复盘
  python scripts/backtest_replay.py --help

注意：本脚本是"外围新文件"，不触碰 src/services/screening、src/core/pipeline 等引擎。
"""
import sys, json, argparse, sqlite3, urllib.request
from datetime import datetime, date
from collections import defaultdict

REPO = r"D:/w-dev/stock/daily_stock_analysis"
HIST_DB = REPO + "/data/stock_history.db"
sys.path.insert(0, REPO)

from src.storage import DatabaseManager, StockDaily

# ---- 持仓规则常量（与项目纪律一致）----
COST_PCT = 0.05      # 买入成本
TP_PCT = 8.0         # 止盈（卖半）
SL_PCT = 5.0         # 止损
TRAIL_PCT = 5.0      # 移动止盈（峰值回撤）
MAX_DAYS = 20        # 最长持仓交易日


def fetch_sina_hist(code: str, days: int = 140):
    """从新浪拉取日K历史（gbk）。返回 [{date,open,high,low,close,volume}] 升序。"""
    sym = "sh" + code if code[0] == "6" else "sz" + code
    url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen={days}")
    raw = json.loads(urllib.request.urlopen(url, timeout=20).read().decode("gbk"))
    out = []
    for b in raw:
        out.append({
            "date": datetime.strptime(b["day"], "%Y-%m-%d").date(),
            "open": float(b["open"]),
            "high": float(b["high"]),
            "low": float(b["low"]),
            "close": float(b["close"]),
            "volume": float(b.get("volume", 0) or 0),
        })
    return out


def simulate_hold(entry_open: float, forward_bars: list) -> dict:
    """对一段前向 K 线模拟持仓，返回退出与收益明细。

    forward_bars: 从 T+1 起的日K（升序），每个含 open/high/low/close/date。
    """
    if not forward_bars or entry_open <= 0:
        return {"status": "insufficient", "return_pct": None, "outcome": None}
    entry = entry_open * (1 + COST_PCT / 100.0)
    tp_price = entry_open * (1 + TP_PCT / 100.0)
    sl_price = entry_open * (1 - SL_PCT / 100.0)

    half_taken = False
    half_exit_price = None
    peak = entry_open
    position_open = True
    exit_price = None
    exit_reason = None
    exit_day = None
    hit_stop = False
    hit_tp = False

    for i, bar in enumerate(forward_bars):
        day = i + 1
        if not position_open:
            break
        lo, hi, cl = bar["low"], bar["high"], bar["close"]
        # 1) 止损优先（同日若止盈止损同触，按止损论）
        if lo <= sl_price:
            exit_price = min(lo, sl_price)
            exit_reason = "stop_loss"
            hit_stop = True
            exit_day = day
            position_open = False
            break
        # 2) 止盈卖半（仅一次）
        if (not half_taken) and hi >= tp_price:
            half_taken = True
            half_exit_price = tp_price
            hit_tp = True
        # 3) 更新峰值
        if cl > peak:
            peak = cl
        # 4) 移动止盈（峰值回撤）
        trail_trigger = peak * (1 - TRAIL_PCT / 100.0)
        if lo <= trail_trigger:
            exit_price = min(lo, trail_trigger)
            exit_reason = "trailing_stop"
            exit_day = day
            position_open = False
            break
        # 5) 最长持仓
        if day >= MAX_DAYS:
            exit_price = cl
            exit_reason = "max_days"
            exit_day = day
            position_open = False
            break

    if position_open:  # 数据在退出前耗尽
        last = forward_bars[-1]
        exit_price = last["close"]
        exit_reason = "truncated"
        exit_day = len(forward_bars)

    if exit_price is None:
        return {"status": "insufficient", "return_pct": None, "outcome": None}

    blended = (0.5 * half_exit_price + 0.5 * exit_price) if half_taken else exit_price
    ret = (blended / entry - 1) * 100.0
    outcome = "win" if ret > 0 else "loss"
    return {
        "status": "completed",
        "entry_open": entry_open,
        "entry_cost_adjusted": entry,
        "half_taken": half_taken,
        "half_exit_price": half_exit_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "exit_day": exit_day,
        "return_pct": ret,
        "outcome": outcome,
        "hit_stop": hit_stop,
        "hit_tp": hit_tp,
    }


def run_self_test(codes: list, hist_days: int = 140, forward: int = 21):
    """用真实历史窗口自测回放器：每只股票取一段含 >=forward 根前向K线的入场点。

    说明：这是「回放器正确性 + 规则在真实价格上的表现」自测，使用 17 只股票作为代理宇宙，
    并非策略在历史上某日的真实选股结果（历史选股需回放 run_screening，当前无该数据）。
    """
    print(f"\n=== [自测] 历史窗口回放（{len(codes)} 只，前向 {forward} 日，规则 +8%/-5%/峰值-5%/≤20日）===")
    results = []
    for code in codes:
        try:
            bars = fetch_sina_hist(code, hist_days)
        except Exception as e:
            print(f"  ! {code} 历史拉取失败: {str(e)[:40]}"); continue
        if len(bars) < forward + 1:
            print(f"  ! {code} 历史不足({len(bars)})"); continue
        # 入场点：取倒数第 (forward+1) 根之后的第一根开盘 = T+1 开盘
        idx = len(bars) - (forward + 1)
        entry_bar = bars[idx]          # 信号日
        t1_bar = bars[idx + 1]         # T+1
        fwd = bars[idx + 1: idx + 1 + forward]
        sim = simulate_hold(t1_bar["open"], fwd)
        sim["code"] = code
        sim["signal_date"] = entry_bar["date"].isoformat()
        sim["t1_date"] = t1_bar["date"].isoformat()
        results.append(sim)
    _print_aggregate(results, label="自测历史窗口")
    return results


def run_real_picks():
    """真实 09-02 / 09-03 选股复盘（用已入库的 sina_p2 K 线）。

    注意：当前沙箱时钟仅到 09-03，09-02 选股的前向数据只到 09-03（1 根），
    无法跑满 20 日规则——这里如实标注「前向窗口不足」。
    """
    print("\n=== [真实] 09-02 选股复盘（前向数据仅到 09-03）===")
    db = DatabaseManager()
    session = db.get_session()
    import glob, os
    files = sorted(glob.glob(f"{REPO}/reports/run*_screening_*.json"))
    picks = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for c in d.get("candidates", []):
            code = str(c.get("code", "")).strip()
            if code:
                picks.append((code, c.get("name"), d.get("market_state"), os.path.basename(f)))
    print(f"  汇总候选 {len(picks)} 只（来自 {len(files)} 份报告）")
    done = 0
    signal_date = date(2026, 9, 2)
    for code, name, mstate, src in picks:
        rows = session.query(StockDaily).filter_by(code=code, data_source="sina_p2")\
            .order_by(StockDaily.date).all()
        # 找信号日 09-02 所在位置，T+1 = 其后第一根（09-03）
        idx = next((i for i, r in enumerate(rows) if r.date == signal_date), None)
        if idx is None or idx + 1 >= len(rows):
            print(f"  {code} {name} 无 T+1 前向K线，跳过"); continue
        t1 = rows[idx + 1]
        fwd = [{"date": r.date, "open": r.open, "high": r.high, "low": r.low, "close": r.close}
               for r in rows[idx + 1:]]
        sim = simulate_hold(t1.open, fwd)
        tag = "✓" if sim["status"] == "completed" else "✗"
        print(f"  {tag} {code} {name:<8} T+1开={t1.open:.2f} 前向{len(fwd)}日 "
              f"收益={sim.get('return_pct'):.3f}% 退出={sim.get('exit_reason')} ({src})")
        done += 1
    print(f"  可复盘 {done} 只（多数因前向仅 1 日，仅能看单日收益，非完整规则回测）")


def run_histdb_backtest(codes: list, forward: int = 20):
    """基于已补数的真实历史库 data/stock_history.db，对 17 只候选做完整 20 日规则回测。

    取每只股票『距最新约 forward+1 根K线前』的入场点（T+1 开盘），用真实历史价格
    跑满 +8%/-5%/峰值-5%/≤20日 规则。这证明 ETL→回放器链路在真实数据上打通，
    且一旦未来真实选股累积，回放器可直接复用此历史库出统计。
    """
    import os
    if not os.path.exists(HIST_DB):
        print(f"  [历史库回放] 跳过：{HIST_DB} 不存在（先跑 backfill_stock_daily_history.py）")
        return []
    print(f"\n=== [历史库回放] 17 候选 / 真实历史 / 完整 {forward} 日规则（源 data/stock_history.db）===")
    conn = sqlite3.connect(HIST_DB)
    cur = conn.cursor()
    results = []
    for code in codes:
        cur.execute(
            "SELECT date,open,high,low,close FROM stock_daily WHERE code=? ORDER BY date",
            (code,),
        )
        rows = cur.fetchall()
        if len(rows) < forward + 1:
            continue
        idx = len(rows) - (forward + 1)
        entry_date, t1_date = rows[idx][0], rows[idx + 1][0]
        fwd = [{"date": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4]}
               for r in rows[idx + 1: idx + 1 + forward]]
        sim = simulate_hold(fwd[0]["open"], fwd)
        sim["code"] = code
        sim["signal_date"] = entry_date
        sim["t1_date"] = t1_date
        results.append(sim)
    conn.close()
    _print_aggregate(results, label="历史库回放(17候选/真实历史/完整规则)")
    return results


def run_universe_sweep(forward: int = 20, step: int = 1):
    """全宇宙滑动回测：对 stock_history.db 中每只股票、每个历史入场点应用完整持仓规则。

    意义：把 P1 的 17 只代理样本扩到数万次模拟，量化『这套 +8%/-5%/峰值-5%/≤20日 规则
    在沪深300 宇宙上的真实期望』。注意：这是规则在宇宙上的表现（每只股票每日都买），
    并非策略『只买其选出的标的』的选择优势——选择优势仍需真实累积选股。但统计力远强于 17 只代理。
    """
    import os
    if not os.path.exists(HIST_DB):
        print(f"  [全宇宙回测] 跳过：{HIST_DB} 不存在")
        return []
    print(f"\n=== [全宇宙回测] stock_history.db × 完整 {forward} 日规则（滑动步长 {step}）===")
    conn = sqlite3.connect(HIST_DB)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT code FROM stock_daily")
    codes = [r[0] for r in cur.fetchall()]
    all_results = []
    for code in codes:
        cur.execute("SELECT date,open,high,low,close FROM stock_daily WHERE code=? ORDER BY date", (code,))
        bars = cur.fetchall()
        if len(bars) < forward + 2:
            continue
        for i in range(0, len(bars) - forward - 1, step):
            fwd = [{"date": b[0], "open": b[1], "high": b[2], "low": b[3], "close": b[4]}
                   for b in bars[i + 1: i + 1 + forward]]
            sim = simulate_hold(fwd[0]["open"], fwd)
            if sim["status"] == "completed":
                all_results.append(sim)
    conn.close()
    _print_aggregate(all_results, label=f"全宇宙回测(N={len(all_results)})")
    return all_results


def _print_aggregate(results, label: str):
    comp = [r for r in results if r["status"] == "completed"]
    if not comp:
        print(f"  [{label}] 无完成样本")
        return
    wins = [r for r in comp if r["outcome"] == "win"]
    losses = [r for r in comp if r["outcome"] == "loss"]
    rets = [r["return_pct"] for r in comp]
    days = [r["exit_day"] for r in comp]
    reasons = defaultdict(int)
    for r in comp:
        reasons[r["exit_reason"]] += 1
    avg = sum(rets) / len(rets)
    import statistics
    print(f"  [{label}] 样本={len(comp)}  胜率={len(wins)/len(comp)*100:.1f}%  "
          f"{len(wins)}胜/{len(losses)}负")
    print(f"    平均收益={avg:.3f}%  中位数={statistics.median(rets):.3f}%  "
          f"平均持仓={sum(days)/len(days):.1f}日")
    print(f"    退出原因: " + "  ".join(f"{k}={v}" for k, v in sorted(reasons.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-selftest", action="store_true", help="跳过历史窗口自测")
    ap.add_argument("--no-real", action="store_true", help="跳过真实 09-02 复盘")
    ap.add_argument("--no-histdb", action="store_true", help="跳过历史库回放")
    ap.add_argument("--sweep", action="store_true", help="跑全宇宙滑动回测（数万次模拟）")
    ap.add_argument("--sweep-step", type=int, default=1, help="全宇宙滑动步长（默认1=每交易日）")
    args = ap.parse_args()

    # 17 只候选（来自 run12_screening_20260902.json）
    codes = ["601328","601318","601658","601939","601601","000001","601169",
             "600036","601398","601288","600015","601988","601818","600016",
             "600030","601628","601668"]

    if not args.no_selftest:
        run_self_test(codes)
    if not args.no_real:
        run_real_picks()
    if not args.no_histdb:
        run_histdb_backtest(codes)
    if args.sweep:
        run_universe_sweep(forward=20, step=args.sweep_step)


if __name__ == "__main__":
    main()
