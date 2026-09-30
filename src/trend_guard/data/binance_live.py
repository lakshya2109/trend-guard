"""Hourly Binance spot klines over REST, plus an incremental on-disk cache.

Roostoo streams its prices from Binance, so Binance 1h candles are both the backtest data and the
live signal data. `data-api.binance.vision` (a market-data mirror that is not geo-blocked) is tried
first, then api.binance.com. Only CLOSED bars are ever returned: `now` is an explicit argument, so
this module never reads the wall clock and stays deterministic in tests.

Cache layout: data/bars/1h/{SYMBOL}.parquet with columns open, high, low, close, volume, quote_volume,
indexed by bar open time (UTC). The backtest and the live engine read the same files.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from trend_guard.data.bars import find_gaps

log = logging.getLogger("data")

HOSTS = ("https://data-api.binance.vision", "https://api.binance.com")
_KLINES_PATH = "/api/v3/klines"
_TIMEOUT = 15.0
_LIMIT = 1000  # Binance's maximum klines per request
HOUR = timedelta(hours=1)
BAR_COLS = ["open", "high", "low", "close", "volume", "quote_volume"]
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


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLS, dtype=float).set_axis(
        pd.DatetimeIndex([], name="open_time", tz="UTC")
    )


def drop_unclosed_bar(df: pd.DataFrame, now: datetime, interval: timedelta = HOUR) -> pd.DataFrame:
    """Keep only bars whose open_time + interval <= now (the forming bar is dropped)."""
    if df.empty:
        return df
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return df[(df.index + interval) <= now]


def _get(params: dict) -> list:
    """GET klines from the first host that answers. HTTP 400 means the symbol is not listed: []."""
    last: Exception | None = None
    for host in HOSTS:
        for attempt in range(3):
            try:
                r = requests.get(host + _KLINES_PATH, params=params, timeout=_TIMEOUT)
                if r.status_code == 400:
                    return []
                if r.status_code == 429:  # rate limited: back off, then retry
                    time.sleep(5 * 2**attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                last = e
                time.sleep(2**attempt)
        log.warning("klines host failed: %s (%s)", host, last)
    raise RuntimeError(f"klines request failed on every host: {params} ({last})")


def _to_frame(raw: list) -> pd.DataFrame:
    if not raw:
        return empty_bars()
    df = pd.DataFrame(raw, columns=KLINE_COLS)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("open_time")[BAR_COLS].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_klines(symbol: str, start: datetime, now: datetime) -> pd.DataFrame:
    """All closed 1h klines for `symbol` from `start` (inclusive) up to `now`, paginated."""
    frames = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    while cursor < end_ms:
        batch = _get({"symbol": symbol, "interval": "1h", "startTime": cursor, "limit": _LIMIT})
        if not batch:
            break
        frames.append(_to_frame(batch))
        cursor = int(batch[-1][0]) + 3_600_000
        if len(batch) < _LIMIT:
            break
    if not frames:
        return empty_bars()
    df = pd.concat(frames)
    return drop_unclosed_bar(df[~df.index.duplicated(keep="last")].sort_index(), now)


def update_cache(symbol: str, cache_dir: Path, start: datetime, now: datetime) -> pd.DataFrame:
    """Read data/bars/1h/{symbol}.parquet, append any newer closed bars, write it back, return it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol}.parquet"
    cached = pd.read_parquet(path) if path.exists() else empty_bars()
    if len(cached):
        cached = cached[BAR_COLS] if set(BAR_COLS) <= set(cached.columns) else cached
        fetch_from = cached.index[-1] + HOUR
    else:
        fetch_from = pd.Timestamp(start)
    fresh = (
        fetch_klines(symbol, fetch_from.to_pydatetime(), now) if fetch_from + HOUR <= now else empty_bars()
    )
    df = pd.concat([cached, fresh]) if len(cached) else fresh
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(fresh):
        df.to_parquet(path)
    return df


def load_cache(cache_dir: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """{symbol: bars} for every cached symbol in `symbols` (missing files are skipped)."""
    out = {}
    for sym in symbols:
        path = cache_dir / f"{sym}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if len(df):
                out[sym] = df
    return out


def check_continuity(df: pd.DataFrame, freq: str = "1h", max_fill: int = 2) -> tuple[pd.DataFrame, bool]:
    """Forward-fill gaps of at most `max_fill` missing bars; a longer gap marks the asset stale."""
    if df.empty:
        return df, True
    idx = pd.DatetimeIndex(df.index).sort_values()
    full_idx = pd.date_range(idx[0], idx[-1], freq=freq)
    is_stale = False
    for gap_start, gap_end in find_gaps(df, freq=freq):
        n_missing = len(pd.date_range(gap_start, gap_end, freq=freq)) - 2  # exclude the bracketing bars
        if n_missing > max_fill:
            is_stale = True
    out = df.reindex(full_idx).ffill(limit=max_fill)
    return out, is_stale


def panel(bars: dict[str, pd.DataFrame], pair_of: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (time x pair) close and quote_volume frames on one hourly grid from per-symbol bars."""
    frames = {pair_of[s]: df for s, df in bars.items() if s in pair_of}
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    index = pd.date_range(
        min(f.index[0] for f in frames.values()),
        max(f.index[-1] for f in frames.values()),
        freq="1h",
        tz="UTC",
    )
    close = pd.DataFrame({p: f["close"].reindex(index) for p, f in frames.items()})
    qv = pd.DataFrame({p: f["quote_volume"].reindex(index) for p, f in frames.items()})
    return close, qv
