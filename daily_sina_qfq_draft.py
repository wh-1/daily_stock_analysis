# =============================================================================
# _fetch_daily_sina 改 qfq 草稿（NOT APPLIED — 待主人拍板后由本机应用）
# 目标文件：src/services/screening/daily.py
# 替换范围：整个 _fetch_daily_sina 函数（原函数约 line 599-642）
# 设计：改用 akshare 的 stock_zh_a_daily(adjust="qfq") 取新浪前复权。
#       该接口 = 新浪不复权日K × 新浪累计复权因子（客户端相乘）= 前复权，
#       与 data_provider 内 _fetch_stock_data_sina 口径一致，对齐全链 qfq。
# 注意：
#   1. 改动落在 src/ 下，触及 engine-zero-modification 红线边界
#      （daily.py 是支撑模块非 engine 本体，但需主人确认）。
#   2. 日K 富集层 DAILY_ENRICH_ENABLED 默认 False → 该函数当前生产不跑，
#      此改属"一致性/前向准备"，非紧急修复。
#   3. 依赖 akshare + 新浪网络（与本文件 _fetch_daily_akshare 同源依赖）。
# =============================================================================

def _fetch_daily_sina(code: str, *, lookback_days: int) -> pd.DataFrame:
    """Fetch forward-adjusted (qfq) daily history backed by Sina via akshare.

    Sina's raw ``getKLineData`` endpoint only returns unadjusted prices (no
    adjust parameter available), so we route through akshare's
    ``stock_zh_a_daily(adjust="qfq")``, which pulls Sina's non-adjusted bars
    and multiplies them by Sina's cumulative adjustment factor on the client
    side (qfq = unadjusted × factor). This aligns the Sina slot in the ``auto``
    daily chain with the qfq convention used by Tencent/AkShare/Baostock/Tushare,
    so switching daily sources no longer shifts indicator values.
    """
    import akshare as ak

    symbol = _to_tencent_code(code)  # akshare Sina endpoint wants sh/sz/bj prefix
    count = max(int(lookback_days), 30)
    start_date = (datetime.now() - timedelta(days=count * 2)).strftime("%Y%m%d")

    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, adjust="qfq")
    if df is None or df.empty:
        raise RuntimeError(f"sina(qfq) daily history empty for {code}")

    # akshare returns: date, open, high, low, close, volume (no amount for Sina daily)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["amount"] = pd.NA
    df = df.sort_values("date")
    for col in ("open", "close", "high", "low", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.tail(count).copy()


# -----------------------------------------------------------------------------
# 一键替换（unified diff 风格，供主人本机 git apply / 手工替换参考）
# -----------------------------------------------------------------------------
# --- a/src/services/screening/daily.py
# +++ b/src/services/screening/daily.py
# @@ -599,44 +599,21 @@ def _fetch_daily_sina(code: str, *, lookback_days: int) -> pd.DataFrame:
# -    """Fetch unadjusted daily history from Sina's direct K-line API.
# -    ...（旧 docstring，略）..."""
# -    symbol = _to_tencent_code(code)
# -    count = max(int(lookback_days), 30)
# -    response = requests.get(
# -        "https://quotes.sina.cn/cn/api/openapi.php/CN_MarketDataService.getKLineData",
# -        params={"symbol": symbol, "scale": 240, "ma": "no", "datalen": count},
# -        headers={"User-Agent": "Mozilla/5.0"},
# -        timeout=10,
# -    )
# -    response.raise_for_status()
# -    payload = response.json()
# -    data = payload.get("result", {}).get("data") if isinstance(payload, dict) else None
# -    if not isinstance(data, list) or not data:
# -        raise RuntimeError(f"sina daily history empty for {code}")
# -    rows: list[dict[str, object]] = []
# -    for row in data:
# -        if not isinstance(row, dict):
# -            continue
# -        rows.append({
# -            "date": row.get("day") or row.get("date"),
# -            "open": row.get("open"),
# -            "close": row.get("close"),
# -            "high": row.get("high"),
# -            "low": row.get("low"),
# -            "volume": row.get("volume"),
# -            "amount": row.get("amount", pd.NA),
# -        })
# -    if not rows:
# -        raise RuntimeError(f"sina daily history malformed for {code}")
# -    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume", "amount"])
# -    if "date" in df.columns:
# -        df = df.sort_values("date")
# -    for col in ("open", "close", "high", "low", "volume", "amount"):
# -        df[col] = pd.to_numeric(df[col], errors="coerce")
# -    return df.tail(count).copy()
# +    """Fetch forward-adjusted (qfq) daily history backed by Sina via akshare.
# +    ...（见上方新函数体）..."""
# +    import akshare as ak
# +    symbol = _to_tencent_code(code)
# +    count = max(int(lookback_days), 30)
# +    start_date = (datetime.now() - timedelta(days=count * 2)).strftime("%Y%m%d")
# +    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, adjust="qfq")
# +    if df is None or df.empty:
# +        raise RuntimeError(f"sina(qfq) daily history empty for {code}")
# +    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
# +    df["amount"] = pd.NA
# +    df = df.sort_values("date")
# +    for col in ("open", "close", "high", "low", "volume", "amount"):
# +        df[col] = pd.to_numeric(df[col], errors="coerce")
# +    return df.tail(count).copy()
