#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把每日选股信号同步到本地台账库，为回测提供真实信号样本。

## 为什么需要这个脚本

GitHub Actions 无状态：runner 跑完即销毁，选股结果只进飞书消息与 artifact
（保留 30 天）。本地 `screening_runs` 表长期为 0 行，**没有任何历史信号**，
回测无从谈起。

配合 `daily-screening.yml` 的「归档信号到 signals 分支」步骤（把信号 JSON
提交到独立的 `signals` 孤儿分支，不污染 dev 历史），本脚本把信号拉回本地
持久化，从而让「Actions 跑选股 + 本地跑回测」成立。

## 产物

`data/signal_ledger.db`（独立库，不污染生产 `stock_analysis.db`）

## 用法

    python scripts/sync_signals_from_git.py                  # 从 origin/signals 同步
    python scripts/sync_signals_from_git.py --local reports/  # 从本地目录导入
    python scripts/sync_signals_from_git.py --show            # 只看台账统计
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "signal_ledger.db"
BRANCH = "signals"
FILE_RE = re.compile(r"screening_(\d{8})\.json$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_runs (
    signal_date     TEXT PRIMARY KEY,
    run_id          TEXT,
    strategy        TEXT,
    market_state    TEXT,
    candidate_count INTEGER,
    snapshot_count  INTEGER,
    llm_model       TEXT,
    sourced_from    TEXT,
    synced_at       TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    signal_date     TEXT NOT NULL,
    code            TEXT NOT NULL,
    name            TEXT,
    rank            INTEGER,
    score           REAL,
    screen_score    REAL,
    llm_score       REAL,
    llm_confidence  REAL,
    price           REAL,
    change_pct      REAL,
    industry        TEXT,
    risk_level      TEXT,
    strategy        TEXT,
    market_state    TEXT,
    reason          TEXT,
    PRIMARY KEY (signal_date, code)
);

CREATE INDEX IF NOT EXISTS idx_signals_code ON signals(code);
"""


def _norm_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


# ---------------------------------------------------------------- git

def git(*args: str, cwd: Optional[Path] = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{proc.stderr.strip()[:300]}")
    return proc.stdout


def list_remote_files() -> List[str]:
    """列出 origin/signals 分支下的信号文件。"""
    git("fetch", "origin", BRANCH)
    out = git("ls-tree", "-r", "--name-only", f"origin/{BRANCH}")
    return [ln.strip() for ln in out.splitlines() if FILE_RE.search(ln.strip())]


def read_remote_file(path: str) -> str:
    return git("show", f"origin/{BRANCH}:{path}")


def list_local_files(directory: Path) -> List[Tuple[str, Path]]:
    files = []
    # 本地报告可能带 run 前缀（如 run12_screening_20260902.json），故用通配前后缀
    for p in sorted(directory.glob("*screening_*.json")):
        m = FILE_RE.search(p.name)
        if m:
            files.append((m.group(1), p))
    return files


# ---------------------------------------------------------------- 解析

def parse_payload(payload: dict, yyyymmdd: str) -> Tuple[dict, List[tuple]]:
    """解析选股 JSON → (run_info, signal_rows)。"""
    signal_date = _norm_date(yyyymmdd)
    run_info = {
        "signal_date": signal_date,
        "run_id": payload.get("run_id"),
        "strategy": payload.get("strategy"),
        "market_state": payload.get("market_state"),
        "candidate_count": payload.get("candidate_count") or len(payload.get("candidates") or []),
        "snapshot_count": payload.get("snapshot_count"),
        "llm_model": payload.get("llm_model_used"),
    }

    rows: List[tuple] = []
    for c in payload.get("candidates") or []:
        code = str(c.get("code") or "").strip()
        if not code:
            continue

        def f(key: str) -> Optional[float]:
            v = c.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        rows.append(
            (
                signal_date,
                code,
                c.get("name"),
                c.get("rank"),
                f("score"),
                f("screen_score"),
                f("llm_score"),
                f("llm_confidence"),
                f("price"),
                f("change_pct"),
                c.get("industry") or c.get("llm_sector"),
                c.get("risk_level"),
                payload.get("strategy"),
                payload.get("market_state"),
                (c.get("reason") or "")[:500],
            )
        )
    return run_info, rows


# ---------------------------------------------------------------- 入库

def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert(conn: sqlite3.Connection, run_info: dict, rows: List[tuple], sourced_from: str) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO signal_runs
            (signal_date, run_id, strategy, market_state, candidate_count,
             snapshot_count, llm_model, sourced_from, synced_at)
        VALUES
            (:signal_date, :run_id, :strategy, :market_state, :candidate_count,
             :snapshot_count, :llm_model, :sourced_from, :synced_at)
        ON CONFLICT(signal_date) DO UPDATE SET
            run_id=excluded.run_id,
            strategy=excluded.strategy,
            market_state=excluded.market_state,
            candidate_count=excluded.candidate_count,
            snapshot_count=excluded.snapshot_count,
            llm_model=excluded.llm_model,
            sourced_from=excluded.sourced_from,
            synced_at=excluded.synced_at
        """,
        {**run_info, "sourced_from": sourced_from, "synced_at": datetime.now().isoformat(timespec="seconds")},
    )
    cur.executemany(
        """
        INSERT INTO signals
            (signal_date, code, name, rank, score, screen_score, llm_score,
             llm_confidence, price, change_pct, industry, risk_level,
             strategy, market_state, reason)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(signal_date, code) DO UPDATE SET
            name=excluded.name,
            rank=excluded.rank,
            score=excluded.score,
            screen_score=excluded.screen_score,
            llm_score=excluded.llm_score,
            llm_confidence=excluded.llm_confidence,
            price=excluded.price,
            change_pct=excluded.change_pct,
            industry=excluded.industry,
            risk_level=excluded.risk_level,
            strategy=excluded.strategy,
            market_state=excluded.market_state,
            reason=excluded.reason
        """,
        rows,
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------- 展示

def show_ledger(conn: sqlite3.Connection) -> None:
    n_runs = conn.execute("SELECT COUNT(*) FROM signal_runs").fetchone()[0]
    n_sig = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    print(f"[台账] {DB_PATH}")
    print(f"  交易日 {n_runs} 天，信号 {n_sig} 条")
    if not n_runs:
        print("  （空：先跑一次同步）")
        return
    print("\n  最近 10 个交易日：")
    for d, st, strat, cnt in conn.execute(
        "SELECT signal_date, market_state, strategy, candidate_count "
        "FROM signal_runs ORDER BY signal_date DESC LIMIT 10"
    ):
        print(f"    {d}  {st or '-':<7} {strat or '-':<20} {cnt} 只")

    print("\n  市场态分布：")
    for st, c in conn.execute(
        "SELECT market_state, COUNT(*) AS c FROM signal_runs "
        "GROUP BY market_state ORDER BY c DESC"
    ):
        print(f"    {st or '-':<10} {c} 天")


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description="同步选股信号到本地台账")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--local", default="", help="从本地目录导入（如 reports/），不走 git")
    parser.add_argument("--show", action="store_true", help="只查看台账统计")
    parser.add_argument("--branch", default=BRANCH)
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)

    try:
        if args.show:
            show_ledger(conn)
            return 0

        total_runs = 0
        total_rows = 0

        if args.local:
            src = Path(args.local)
            if not src.exists():
                print(f"[err] 目录不存在：{src}")
                return 1
            files = list_local_files(src)
            print(f"[local] {src} 发现 {len(files)} 个信号文件")
            for yyyymmdd, path in files:
                payload = json.loads(path.read_text(encoding="utf-8"))
                run_info, rows = parse_payload(payload, yyyymmdd)
                n = upsert(conn, run_info, rows, f"local:{path.name}")
                total_runs += 1
                total_rows += n
                print(f"  ✓ {run_info['signal_date']}  {n} 条信号")
        else:
            print(f"[git] fetch origin/{args.branch} ...")
            files = list_remote_files()
            if not files:
                print(f"[warn] origin/{args.branch} 分支没有信号文件。"
                      "若 Actions 尚未跑过归档步骤，先用 --local 导入已有的 reports/*.json。")
                return 1
            print(f"[git] 发现 {len(files)} 个信号文件")
            for path in files:
                m = FILE_RE.search(path)
                if not m:
                    continue
                payload = json.loads(read_remote_file(path))
                run_info, rows = parse_payload(payload, m.group(1))
                n = upsert(conn, run_info, rows, f"git:{args.branch}")
                total_runs += 1
                total_rows += n
                print(f"  ✓ {run_info['signal_date']}  {n} 条信号")

        print(f"\n[完成] 同步 {total_runs} 个交易日 / {total_rows} 条信号 → {db_path}")
        show_ledger(conn)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
