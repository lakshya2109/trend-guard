"""Live/recent data via Binance REST mirrors, with Bybit fallback for funding/OI when
fapi.binance.com is geo-blocked (BUILD_SPEC.md Section 5.3).

Only closed bars are ever returned to the caller. `now` is always an explicit parameter -- never
read from the wall clock here -- so this stays deterministic and safe to reuse in a backtest.
Optional data (funding, open interest) must never crash the caller: every network path here
catches its own exceptions, logs a warning via logging.getLogger("data"), and returns an empty
DataFrame on failure.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

from trend_guard.data.bars import find_gaps

log = logging.getLogger("data")

_DATA_API_KLINES = "https://data-api.binance.vision/api/v3/klines"
_SPOT_FALLBACK_KLINES = "https://api.binance.com/api/v3/klines"
_FAPI_PING = "https://fapi.binance.com/fapi/v1/ping"
_FAPI_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
_FAPI_OI = "https://fapi.binance.com/futures/data/openInterestHist"
_BYBIT_FUNDING = "https://api.bybit.com/v5/market/funding/history"
_BYBIT_OI = "https://api.bybit.com/v5/market/open-interest"
_TIMEOUT = 10.0

KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]
_EMPTY_FUNDING = pd.DataFrame(columns=["funding_rate"]).set_axis(pd.DatetimeIndex([], name="time", tz="UTC"))
_EMPTY_OI = pd.DataFrame(columns=["open_interest"]).set_axis(pd.DatetimeIndex([], name="time", tz="UTC"))


def probe_fapi_access() -> bool:
    """Cheap GET to fapi.binance.com. fapi rejects US IPs; returns False (never raises) if
    blocked or unreachable, so callers can decide to use the Bybit fallback instead."""
    try:
        r = requests.get(_FAPI_PING, timeout=_TIMEOUT)
        return r.status_code == 200
    except requests.RequestException as e:
        log.warning("fapi probe failed: %s", e)
        return False


def drop_unclosed_bar(
    df: pd.DataFrame, now: datetime, interval: timedelta = timedelta(hours=1)
) -> pd.DataFrame:
    """Drop the forming/most-recent bar: only keep rows whose open_time + interval <= now."""
    if df.empty:
        return df
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return df[(df.index + interval) <= now]


def fetch_recent_klines_1h(symbol: str, now: datetime, limit: int = 1000) -> pd.DataFrame:
    """Recent 1h spot klines for `symbol`, closed bars only.

    Tries the data-api mirror first (not geo-blocked for US IPs), falls back to api.binance.com.
    """
    params = {"symbol": symbol, "interval": "1h", "limit": limit}
    raw = None
    for url in (_DATA_API_KLINES, _SPOT_FALLBACK_KLINES):
        try:
            r = requests.get(url, params=params, timeout=_TIMEOUT)
            r.raise_for_status()
            raw = r.json()
            break
        except (requests.RequestException, ValueError) as e:
            log.warning("klines fetch failed from %s: %s", url, e)
    if not raw:
        return pd.DataFrame(columns=KLINE_COLS[1:-2]).set_axis(
            pd.DatetimeIndex([], name="open_time", tz="UTC")
        )

    df = pd.DataFrame(raw, columns=KLINE_COLS)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = df.drop(columns=["close_time", "ignore"]).set_index("open_time")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return drop_unclosed_bar(df, now, timedelta(hours=1))


def fetch_funding(
    symbol: str, fapi_available: bool, limit: int = 100, bybit_symbol: str | None = None
) -> pd.DataFrame:
    """Recent funding-rate history, indexed by UTC time with a single `funding_rate` column.

    Uses fapi.binance.com when `fapi_available` (from probe_fapi_access), else Bybit. Never
    raises -- returns an empty frame and logs a warning on any failure.
    """
    if fapi_available:
        try:
            r = requests.get(_FAPI_FUNDING, params={"symbol": symbol, "limit": limit}, timeout=_TIMEOUT)
            r.raise_for_status()
            raw = r.json()
            if not raw:
                return _EMPTY_FUNDING.copy()
            df = pd.DataFrame(raw)
            df["time"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
            df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            return df.set_index("time")[["funding_rate"]].sort_index()
        except (requests.RequestException, ValueError, KeyError) as e:
            log.warning("fapi funding fetch failed for %s: %s", symbol, e)
    return _fetch_funding_bybit(bybit_symbol or symbol, limit)


def _fetch_funding_bybit(symbol: str, limit: int) -> pd.DataFrame:
    try:
        r = requests.get(
            _BYBIT_FUNDING,
            params={"category": "linear", "symbol": symbol, "limit": limit},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        rows = r.json().get("result", {}).get("list", [])
        if not rows:
            return _EMPTY_FUNDING.copy()
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df.set_index("time")[["funding_rate"]].sort_index()
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning("bybit funding fetch failed for %s: %s", symbol, e)
        return _EMPTY_FUNDING.copy()


def fetch_open_interest(
    symbol: str, fapi_available: bool, period: str = "1h", limit: int = 200, bybit_symbol: str | None = None
) -> pd.DataFrame:
    """Recent open-interest history, indexed by UTC time with a single `open_interest` column.

    Uses fapi.binance.com when `fapi_available`, else Bybit. Never raises.
    """
    if fapi_available:
        try:
            r = requests.get(
                _FAPI_OI, params={"symbol": symbol, "period": period, "limit": limit}, timeout=_TIMEOUT
            )
            r.raise_for_status()
            raw = r.json()
            if not raw:
                return _EMPTY_OI.copy()
            df = pd.DataFrame(raw)
            df["time"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
            df["open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
            return df.set_index("time")[["open_interest"]].sort_index()
        except (requests.RequestException, ValueError, KeyError) as e:
            log.warning("fapi OI fetch failed for %s: %s", symbol, e)
    return _fetch_oi_bybit(bybit_symbol or symbol, period, limit)


def _fetch_oi_bybit(symbol: str, period: str, limit: int) -> pd.DataFrame:
    interval_map = {"1h": "1h", "4h": "4h", "1d": "1d"}
    try:
        r = requests.get(
            _BYBIT_OI,
            params={
                "category": "linear",
                "symbol": symbol,
                "intervalTime": interval_map.get(period, "1h"),
                "limit": limit,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        rows = r.json().get("result", {}).get("list", [])
        if not rows:
            return _EMPTY_OI.copy()
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["open_interest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        return df.set_index("time")[["open_interest"]].sort_index()
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning("bybit OI fetch failed for %s: %s", symbol, e)
        return _EMPTY_OI.copy()


def check_continuity(df: pd.DataFrame, freq: str = "1h", max_fill: int = 2) -> tuple[pd.DataFrame, bool]:
    """Forward-fill gaps of at most `max_fill` consecutive missing bars; beyond that the asset is
    stale. Returns (filled_df, is_stale) so callers can exclude stale assets from signals."""
    if df.empty:
        return df, True
    idx = pd.DatetimeIndex(df.index).sort_values()
    full_idx = pd.date_range(idx[0], idx[-1], freq=freq)
    is_stale = False
    for gap_start, gap_end in find_gaps(df, freq=freq):
        n_missing = len(pd.date_range(gap_start, gap_end, freq=freq)) - 2  # exclude the two bracketing bars
        if n_missing > max_fill:
            is_stale = True
    out = df.reindex(full_idx).ffill(limit=max_fill)
    return out, is_stale
