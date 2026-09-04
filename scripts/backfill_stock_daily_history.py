# -*- coding: utf-8 -*-
"""P2-ETL：历史日K补数，写入独立库 data/stock_history.db（不碰生产 stock_daily 单源治理）。

用途：为 P1 回放器提供真实历史行情，使其在未来真实选股累积后能滚动出统计胜率/收益。
数据源：新浪日K（沙箱可用）；沪深300 成分股清单用 baostock（沙箱可用）。

用法：
  python scripts/backfill_stock_daily_history.py                 # 沪深300，默认 250 交易日
  python scripts/backfill_stock_daily_history.py --days 500      # 更长窗口
  python scripts/backfill_stock_daily_history.py --universe-file codes.txt
  python scripts/backfill_stock_daily_history.py --dry-run       # 只列宇宙不抓取

落库：data/stock_history.db 表 stock_daily（code,date,open,high,low,close,volume,amount,pct_chg,data_source）
幂等：同一 (code,date) 已存在则跳过。
"""
import sys, os, json, argparse, sqlite3, time, urllib.request
from datetime import datetime, date

REPO = r"D:/w-dev/stock/daily_stock_analysis"
HIST_DB = os.path.join(REPO, "data", "stock_history.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS stock_daily (
    code TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL, pct_chg REAL,
    data_source TEXT,
    PRIMARY KEY (code, date)
);
"""


def get_hs300_codes():
    """baostock 取沪深300成分股，返回裸码列表。"""
    import baostock as bs
    bs.login()
    rs = bs.query_hs300_stocks()
    codes = []
    while (rs.error_code == "0") and rs.next():
        row = rs.get_row_data()
        # row = [update_date, code(sh.600000), name]
        c = row[1].split(".")[-1]
        if c:
            codes.append(c)
    bs.logout()
    return codes


def fetch_sina_bars(code: str, days: int):
    sym = "sh" + code if code[0] == "6" else "sz" + code
    url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen={days}")
    raw = json.loads(urllib.request.urlopen(url, timeout=20).read().decode("gbk"))
    out = []
    for b in raw:
        out.append({
            "date": b["day"],
            "open": float(b["open"]), "high": float(b["high"]),
            "low": float(b["low"]), "close": float(b["close"]),
            "volume": float(b.get("volume", 0) or 0),
            "pct_chg": None,
        })
    return out


def ensure_schema(conn):
    conn.executescript(SCHEMA)


def upsert(conn, code, bars):
    cur = conn.cursor()
    n = 0
    for b in bars:
        cur.execute(
            "INSERT OR IGNORE INTO stock_daily "
            "(code,date,open,high,low,close,volume,amount,pct_chg,data_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (code, b["date"], b["open"], b["high"], b["low"], b["close"],
             b["volume"], b.get("amount"), b["pct_chg"], "sina"),
        )
        n += 1
    conn.commit()
    return n


def count_rows(conn, code):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM stock_daily WHERE code=?", (code,))
    return cur.fetchone()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--universe-file", default=None, help="每行一个裸码的文本文件")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 只（调试）")
    args = ap.parse_args()

    if args.universe_file:
        with open(args.universe_file, encoding="utf-8") as f:
            codes = [l.strip() for l in f if l.strip()]
    else:
        print("[宇宙] 取沪深300成分股（baostock）...")
        codes = get_hs300_codes()
    print(f"[宇宙] 共 {len(codes)} 只")

    if args.dry_run:
        print("[dry-run] 退出，不抓取")
        return

    conn = sqlite3.connect(HIST_DB)
    ensure_schema(conn)
    total = 0
    ok = 0
    fail = 0
    for i, code in enumerate(codes):
        if args.limit and i >= args.limit:
            break
        try:
            bars = fetch_sina_bars(code, args.days + 5)
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  ! {code} 抓取失败: {str(e)[:40]}")
            continue
        before = count_rows(conn, code)
        upsert(conn, code, bars)
        after = count_rows(conn, code)
        total += after
        ok += 1
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(codes)} 已处理, 累计 {total} 根K线")
        time.sleep(0.04)
    conn.close()
    print(f"[完成] 成功 {ok} 只 / 失败 {fail} 只 / 累计 {total} 根K线 → {HIST_DB}")


if __name__ == "__main__":
    main()
