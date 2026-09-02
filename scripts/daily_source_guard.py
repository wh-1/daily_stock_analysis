#!/usr/bin/env python3
"""Audit & repair stock_daily for heterogeneous qfq providers (P0-A).

Problem: historically save_daily_data UPSERTed by (code, date) without checking
the data source, so a stock's history could be stitched from multiple providers'
forward-adjusted (qfq) series (Tencent / AkShare / Baostock / Sina / Tushare /
Efinance). Those factor bases differ and are NOT interchangeable, so a stitched
series is internally inconsistent (small jumps near ex-rights dates, distorted
MA/MACD/RSI).

This tool (peripheral — does NOT touch src/):
  --audit    list codes that have >1 distinct data_source in stock_daily
  --repair   for each mixed code, re-fetch from one canonical source, overwrite
             the row set, and pin that source via daily_source_policy
  --dry-run  with --repair, print intended actions without writing

Env:
  DAILY_CANONICAL_SOURCE  preferred canonical source (default: tencent).
                          Also honored by get_daily_data (P1) and by the
                          save_daily_data guard (P0-B) once a policy is set.

Typical use:
  python scripts/daily_source_guard.py --audit
  python scripts/daily_source_guard.py --repair --dry-run
  python scripts/daily_source_guard.py --repair
"""
import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sqlalchemy import delete, func, select

from src.storage import DailySourcePolicy, StockDaily, get_db
from data_provider.base import DataFetcherManager, resolve_canonical_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("daily_source_guard")

DEFAULT_CANON = "tencent"
DEFAULT_REPAIR_DAYS = 1500


def audit(db):
    """Return [(code, n_distinct_sources), ...] for codes with mixed sources."""
    with db.get_session() as s:
        q = (
            select(StockDaily.code, func.count(func.distinct(StockDaily.data_source)).label("n"))
            .group_by(StockDaily.code)
            .having(func.count(func.distinct(StockDaily.data_source)) > 1)
        )
        return s.execute(q).all()


def _set_env(key, value):
    old = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    return old


def repair(db, canon, dry_run, days):
    mixed = audit(db)
    if not mixed:
        print("audit: no mixed-source codes found")
        return
    print(f"audit: {len(mixed)} mixed-source code(s) found")
    for code, n in mixed:
        existing = db.get_canonical_source(code)
        # Resolve to the canonical fetcher.name (long form) so the policy row and
        # DAILY_CANONICAL_SOURCE pin always match what get_daily_data returns.
        target = resolve_canonical_source(existing or canon) or resolve_canonical_source(DEFAULT_CANON)
        print(f"[repair] {code}: mixed_sources={n} -> prefer_canonical={target} (dry={dry_run})")
        if dry_run:
            continue

        # Pin the preferred source so get_daily_data (P1) prefers it, then fetch.
        old_canon = _set_env("DAILY_CANONICAL_SOURCE", target)
        try:
            mgr = DataFetcherManager()
            df, src = mgr.get_daily_data(code, days=days)
        except Exception as exc:  # noqa: BLE001 - report and move on
            print(f"  !! fetch failed for {code}: {exc}")
            continue
        finally:
            _set_env("DAILY_CANONICAL_SOURCE", old_canon)

        if df is None or df.empty:
            print(f"  !! fetch returned empty for {code} (source={src}), skipped")
            continue

        # Write with whatever source actually won (consistent with itself), and
        # pin THAT as the code's canonical so future fills stay homogeneous.
        actual = src
        db.set_canonical_source(code, actual)
        with db.get_session() as s:
            s.execute(delete(StockDaily).where(StockDaily.code == code))
            s.commit()
        saved = db.save_daily_data(df, code, actual, canonical_source=actual)
        print(f"  ok: {code} refilled {saved} new row(s) from {actual}")


def main():
    ap = argparse.ArgumentParser(description="Audit/repair stock_daily qfq source consistency")
    ap.add_argument("--audit", action="store_true", help="list mixed-source codes")
    ap.add_argument("--repair", action="store_true", help="re-fetch mixed codes from canonical source and overwrite")
    ap.add_argument("--dry-run", action="store_true", help="with --repair, do not write")
    ap.add_argument("--days", type=int, default=DEFAULT_REPAIR_DAYS, help=f"lookback days for re-fetch (default {DEFAULT_REPAIR_DAYS})")
    ap.add_argument(
        "--canonical",
        default=os.getenv("DAILY_CANONICAL_SOURCE", DEFAULT_CANON),
        help=f"canonical source (default {DEFAULT_CANON}; honors DAILY_CANONICAL_SOURCE)",
    )
    args = ap.parse_args()

    db = get_db()
    if args.repair:
        repair(db, args.canonical, args.dry_run, args.days)
        return

    # default action: audit
    rows = audit(db)
    if not rows:
        print("audit: no mixed-source codes found")
        return
    print(f"audit: {len(rows)} mixed-source code(s):")
    for code, n in rows:
        print(f"  {code}: {n} distinct sources")


if __name__ == "__main__":
    main()
