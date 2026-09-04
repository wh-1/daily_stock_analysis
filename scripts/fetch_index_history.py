#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""抓取指数历史日K，写入 data/stock_history.db 的 index_daily 表。

用途：离线历史回放时判定 market_state（STRONG/MIX/BEAR），
对应线上 `_detect_market_state()` 的沪深300 MA20/MA60 门控。

不碰生产库 data/stock_analysis.db，符合数据源治理纪律。

用法:
    python scripts/fetch_index_history.py              # 默认抓沪深300
    python scripts/fetch_index_history.py --days 600
    python scripts/fetch_index_history.py --symbol sh000905
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "stock_history.db"

SINA_KLINE = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={datalen}"
)


def fetch_sina_index(symbol: str, datalen: int) -> list[dict]:
    """抓取新浪指数日K。返回 [{date, open, high, low, close, volume}] 升序。"""
    url = SINA_KLINE.format(symbol=symbol, datalen=datalen)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read().decode("gbk", "ignore")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError(f"unexpected payload type: {type(payload)!r}")

    rows: list[dict] = []
    for item in payload:
        day = str(item.get("day", ""))[:10]
        if not day:
            continue
        rows.append(
            {
                "date": day,
                "open": float(item.get("open") or 0),
                "high": float(item.get("high") or 0),
                "low": float(item.get("low") or 0),
                "close": float(item.get("close") or 0),
                "volume": float(item.get("volume") or 0),
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_daily (
            code    TEXT NOT NULL,
            date    TEXT NOT NULL,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  REAL,
            PRIMARY KEY (code, date)
        )
        """
    )
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取指数历史日K到 stock_history.db")
    parser.add_argument("--symbol", default="sh000300", help="新浪指数代码，默认 sh000300（沪深300）")
    parser.add_argument("--code", default="000300", help="落库用的指数代码，默认 000300")
    parser.add_argument("--days", type=int, default=600, help="抓取根数，默认 600")
    parser.add_argument("--db", default=str(DB_PATH), help="历史库路径")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] {args.symbol} datalen={args.days} ...")
    rows = fetch_sina_index(args.symbol, args.days)
    if not rows:
        print("[fetch] 空结果，中止")
        return 1
    print(f"[fetch] 取得 {len(rows)} 根，{rows[0]['date']} → {rows[-1]['date']}")

    conn = sqlite3.connect(db_path)
    try:
        ensure_table(conn)
        conn.executemany(
            """
            INSERT OR IGNORE INTO index_daily (code, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (args.code, r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"])
                for r in rows
            ],
        )
        conn.commit()
        total = conn.execute(
            "SELECT COUNT(*) FROM index_daily WHERE code = ?", (args.code,)
        ).fetchone()[0]
        span = conn.execute(
            "SELECT MIN(date), MAX(date) FROM index_daily WHERE code = ?", (args.code,)
        ).fetchone()
        print(f"[db] {db_path} index_daily code={args.code} 共 {total} 行，区间 {span[0]} → {span[1]}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
