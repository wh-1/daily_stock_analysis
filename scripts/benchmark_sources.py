#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_sources.py — 数据源接口实测：速度 / 稳定性 / 真实优先级排序

目的
----
原作者默认源序 与 你（主人）在 workflow env + repo Variables 里改过的序 不一致。
本脚本在**真实运行环境**（建议挂在海外 runner 上跑，因 screening 就在那跑）
对每个数据源的真实接口做定时、定次探测，按
    (成功率降序, 平均延迟升序)
排出每条链（快照 / 实时 / 日K）的「真实顺序」，并对比原默认序与当前 env 序。

设计要点
--------
* 用标准库 urllib（零额外依赖），沙箱 / runner / 本机都能直接跑。
* 以「真上游」为测量单元（东财 / 新浪 / 腾讯 / Tushare / Baostock），
  因为这 5 家才是真正后端；akshare / efinance / em_datacenter 等只是封装。
* Tushare / Baostock 走库内调用（需 token / 库），缺失则标记 skipped。
* 非阻塞：任何源失败都不抛异常，最后给结论。

用法
----
    python scripts/benchmark_sources.py                 # 默认 3 票 x 3 次
    python scripts/benchmark_sources.py --iterations 2 --timeout 10
    python scripts/benchmark_sources.py --json out.json
环境变量（可选，用于对比「当前 env 序」）：
    SNAPSHOT_SOURCE_PRIORITY / REALTIME_SOURCE_PRIORITY / DAILY_SOURCE
    TUSHARE_TOKEN（启用 Tushare 实测）
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

# ----------------------------------------------------------------------------
# 工具：代码格式转换（与项目内 _to_tencent_code 同口径）
# ----------------------------------------------------------------------------

def _to_sina_tx(code: str) -> str:
    raw = str(code).strip().zfill(6)
    if raw.startswith(("4", "8", "920")):
        return f"bj{raw}"
    if raw.startswith(("6", "9", "5")):
        return f"sh{raw}"
    return f"sz{raw}"


def _to_em_secid(code: str) -> str:
    raw = str(code).strip().zfill(6)
    if raw.startswith(("6", "9", "5")):
        return f"1.{raw}"          # 上交所
    if raw.startswith(("4", "8", "920")):
        return f"0.{raw}"          # 北交所
    return f"0.{raw}"              # 深交所


# ----------------------------------------------------------------------------
# 端点定义：每个「真上游」按其被用到的链挂若干端点（kind: snapshot/realtime/daily）
# ----------------------------------------------------------------------------

@dataclass
class Endpoint:
    name: str            # 上游名（东财/新浪/腾讯/Tushare/Baostock）
    kind: str            # snapshot | realtime | daily
    url: str             # 中 {code} 占位
    referer: Optional[str] = None
    validator: Optional[Callable[[bytes], bool]] = None


def _nonempty(b: bytes) -> bool:
    return bool(b) and len(b) > 8


def _json_ok(b: bytes) -> bool:
    try:
        json.loads(b.decode("utf-8", "ignore"))
        return True
    except Exception:
        return False


def _sina_realtime_ok(b: bytes) -> bool:
    # hq.sinajs.cn 返回 "var hq_str_sh600519=..."，含 "=" 即可认定成功
    return b"=" in b and b"hq_str" in b


def _em_kline_ok(b: bytes) -> bool:
    return b"klines" in b or b"data" in b


def _tx_kline_ok(b: bytes) -> bool:
    return b"qfqday" in b or b"day" in b


def _build_endpoints() -> list[Endpoint]:
    return [
        # ---------------- 东方财富 ----------------
        Endpoint("东财", "snapshot",
                 "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=1&po=1&np=1"
                 "&fltt=2&invt=2&fid=f3&fs=m:1+t:2&fields=f12,f14,f2",
                 validator=_nonempty),
        Endpoint("东财", "daily",
                 "https://push2his.eastmoney.com/api/qt/stock/kline/get"
                 "?fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
                 "&klt=101&fqt=1&secid={secid}&end=20500101&lmt=10",
                 validator=_em_kline_ok),
        # ---------------- 新浪 ----------------
        Endpoint("新浪", "realtime",
                 "https://hq.sinajs.cn/list={sina}",
                 referer="https://finance.sina.com.cn",
                 validator=_sina_realtime_ok),
        Endpoint("新浪", "snapshot",
                 "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                 "Market_Center.getHQNodeData?page=1&num=1&node=hs_a",
                 validator=_json_ok),
        Endpoint("新浪", "daily",
                 "https://quotes.sina.cn/cn/api/openapi.php/CN_MarketDataService.getKLineData"
                 "?symbol={sina}&scale=240&ma=no&datalen=10",
                 validator=_json_ok),
        # ---------------- 腾讯 ----------------
        Endpoint("腾讯", "realtime",
                 "https://qt.gtimg.cn/q={sina}",
                 validator=_nonempty),
        Endpoint("腾讯", "daily",
                 "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sina},day,,,10,qfq",
                 validator=_tx_kline_ok),
    ]


# ----------------------------------------------------------------------------
# 测量
# ----------------------------------------------------------------------------

@dataclass
class EpStats:
    name: str
    kind: str
    n: int = 0
    ok: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.ok / self.n if self.n else 0.0

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.latencies_ms) if self.latencies_ms else float("inf")

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else float("inf")

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return float("inf")
        s = sorted(self.latencies_ms)
        return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def _http_get(url: str, referer: Optional[str], timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (benchmark-sources)",
        **({"Referer": referer} if referer else {}),
    })
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


def measure_endpoint(ep: Endpoint, codes: list[str], iterations: int, timeout: int) -> EpStats:
    stats = EpStats(ep.name, ep.kind)
    for code in codes:
        sina = _to_sina_tx(code)
        secid = _to_em_secid(code)
        url = ep.url.format(sina=sina, secid=secid)
        for _ in range(iterations):
            stats.n += 1
            t0 = time.perf_counter()
            try:
                body = _http_get(url, ep.referer, timeout)
                dt = (time.perf_counter() - t0) * 1000.0
                if ep.validator and not ep.validator(body):
                    raise ValueError("response failed validation")
                stats.ok += 1
                stats.latencies_ms.append(dt)
            except Exception as e:  # noqa: BLE001
                stats.errors.append(f"{type(e).__name__}: {e}")
    return stats


# ----------------------------------------------------------------------------
# Tushare / Baostock（库内调用，可选）
# ----------------------------------------------------------------------------

def measure_tushare(codes: list[str], iterations: int, timeout: int) -> Optional[EpStats]:
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_API_TOKEN")
    if not token:
        return None
    try:
        import tushare as ts
    except Exception:
        return None
    stats = EpStats("Tushare", "daily")
    pro = ts.pro_api(token)
    for code in codes:
        ts_code = f"{code.zfill(6)}.SH" if code.startswith("6") else f"{code.zfill(6)}.SZ"
        for _ in range(iterations):
            stats.n += 1
            t0 = time.perf_counter()
            try:
                df = pro.daily(ts_code=ts_code,
                               start_date="20240101", end_date="20240131",
                               fields="ts_code,trade_date,close")
                dt = (time.perf_counter() - t0) * 1000.0
                if df is None or df.empty:
                    raise ValueError("empty")
                stats.ok += 1
                stats.latencies_ms.append(dt)
            except Exception as e:  # noqa: BLE001
                stats.errors.append(f"{type(e).__name__}: {e}")
    return stats


def measure_baostock(codes: list[str], iterations: int, timeout: int) -> Optional[EpStats]:
    try:
        import baostock as bs
    except Exception:
        return None
    stats = EpStats("Baostock", "daily")
    try:
        lg = bs.login()
        if lg.error_code != "0":
            stats.errors.append(f"login:{lg.error_msg}")
            return stats
    except Exception as e:  # noqa: BLE001
        stats.errors.append(f"login:{e}")
        return stats
    try:
        for code in codes:
            bcode = f"sh.{code.zfill(6)}" if code.startswith("6") else f"sz.{code.zfill(6)}"
            for _ in range(iterations):
                stats.n += 1
                t0 = time.perf_counter()
                try:
                    rs = bs.query_history_k_data_plus(
                        bcode, "date,close",
                        start_date="2024-01-01", end_date="2024-01-31", frequency="d")
                    dt = (time.perf_counter() - t0) * 1000.0
                    if rs.error_code != "0":
                        raise ValueError(rs.error_msg)
                    stats.ok += 1
                    stats.latencies_ms.append(dt)
                except Exception as e:  # noqa: BLE001
                    stats.errors.append(f"{type(e).__name__}: {e}")
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    return stats


# ----------------------------------------------------------------------------
# 链定义：项目源名 -> (真上游, 偏好的 kind)
# ----------------------------------------------------------------------------

CHAINS: dict[str, list[tuple[str, str, str]]] = {
    "snapshot": [
        ("sina", "新浪", "snapshot"),
        ("efinance", "东财", "snapshot"),
        ("akshare_em", "东财", "snapshot"),
        ("em_datacenter", "东财", "snapshot"),
    ],
    "realtime": [
        ("tencent", "腾讯", "realtime"),
        ("akshare_sina", "新浪", "realtime"),
        ("efinance", "东财", "snapshot"),   # 实时链里东财走 snapshot/clist 端点
        ("akshare_em", "东财", "snapshot"),
    ],
    "daily": [
        ("tushare", "Tushare", "daily"),
        ("tencent", "腾讯", "daily"),
        ("sina", "新浪", "daily"),
        ("akshare", "东财", "daily"),
        ("baostock", "Baostock", "daily"),
    ],
}

# 原作者默认序（无 TUSHARE_TOKEN 时）
ORIGINAL_DEFAULT: dict[str, list[str]] = {
    "snapshot": ["sina", "efinance", "akshare_em", "em_datacenter"],
    "realtime": ["tencent", "akshare_sina", "efinance", "akshare_em"],
    "daily": ["tencent", "sina", "akshare", "baostock"],
}


def _fmt_ms(v: float) -> str:
    return "  n/a" if v == float("inf") else f"{v:6.1f}ms"


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="数据源接口速度/稳定性基准 + 真实排序")
    ap.add_argument("--codes", default="600519,000001,300750,601318",
                    help="测试股票代码，逗号分隔")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=12, help="单请求超时(秒)")
    ap.add_argument("--json", dest="json_path", default="", help="写出 JSON 路径")
    args = ap.parse_args()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    endpoints = _build_endpoints()

    print(f"== 数据源基准测试 ==")
    print(f"环境: {os.environ.get('GITHUB_RUNNER_ENV', os.uname().nodename if hasattr(os, 'uname') else 'local')}")
    print(f"样本票: {codes}  |  每端迭代: {args.iterations}  |  超时: {args.timeout}s\n")

    raw: list[EpStats] = []
    for ep in endpoints:
        st = measure_endpoint(ep, codes, args.iterations, args.timeout)
        raw.append(st)
        print(f"  [{st.name:>4}/{st.kind:<8}] "
              f"成功 {st.ok:>2}/{st.n:<2}  均 {_fmt_ms(st.mean_ms)}  "
              f"中位 {_fmt_ms(st.median_ms)}  p95 {_fmt_ms(st.p95_ms)}")

    # 库内源
    for fn in (measure_tushare, measure_baostock):
        st = fn(codes, args.iterations, args.timeout)
        if st is None:
            print(f"  [{'Tushare' if fn is measure_tushare else 'Baostock':>8}/daily ] skipped (无 token / 未装库)")
        else:
            raw.append(st)
            print(f"  [{st.name:>8}/daily ] 成功 {st.ok:>2}/{st.n:<2}  均 {_fmt_ms(st.mean_ms)}")

    # 归并：上游×kind 的统计量
    def stat_for(upstream: str, kind: str) -> Optional[EpStats]:
        # 优先取该 upstream 该 kind 的端点；否则退回该 upstream 任意端点
        hit = [s for s in raw if s.name == upstream and s.kind == kind and s.n]
        if not hit:
            hit = [s for s in raw if s.name == upstream and s.n]
        if not hit:
            return None
        # 合并多个同名端点
        merged = EpStats(upstream, kind)
        for s in hit:
            merged.n += s.n
            merged.ok += s.ok
            merged.latencies_ms.extend(s.latencies_ms)
            merged.errors.extend(s.errors)
        return merged

    print("\n== 各链真实顺序（按 成功率↓, 平均延迟↑）==")
    report: dict[str, object] = {"codes": codes, "chains": {}}
    for chain, sources in CHAINS.items():
        rows = []
        for src_name, upstream, kind in sources:
            st = stat_for(upstream, kind)
            if st is None:
                rows.append((src_name, 0.0, float("inf"), "未测"))
            else:
                rows.append((src_name, st.success_rate, st.mean_ms,
                             f"{st.ok}/{st.n} ok, {_fmt_ms(st.mean_ms)}"))
        # 排序：成功率降序（用 -r[1]），成功率相同则平均延迟升序（r[2]）
        rows.sort(key=lambda r: (-r[1], r[2]))

        orig = ORIGINAL_DEFAULT[chain]
        cur = os.getenv(f"{chain.upper()}_SOURCE_PRIORITY") or os.getenv("DAILY_SOURCE" if chain == "daily" else "")
        cur_list = [s.strip() for s in cur.split(",")] if cur else None

        print(f"\n--- 链 {chain} ---")
        print(f"  推荐真实序: {' > '.join(r[0] for r in rows)}")
        print(f"  原默认序  : {' > '.join(orig)}")
        if cur_list:
            print(f"  当前 env  : {' > '.join(cur_list)}")
        for r in rows:
            print(f"    {r[0]:<14} 成功率 {r[1]*100:5.1f}%   {r[3]}")

        report["chains"][chain] = {
            "recommended": [r[0] for r in rows],
            "original_default": orig,
            "current_env": cur_list,
            "detail": [
                {"source": r[0], "success_rate": round(r[1], 3),
                 "mean_ms": None if r[2] == float("inf") else round(r[2], 1)}
                for r in rows
            ],
        }

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已写出: {args.json_path}")

    print("\n完成（非阻塞，任何源失败不影响退出）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
