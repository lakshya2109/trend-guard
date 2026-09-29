"""bStock underlying history via yfinance (BUILD_SPEC.md Section 5.2/7.1).

yfinance limits 1h-interval history to roughly the last 730 days: requesting further back does
not error, it just silently truncates the returned range. Callers backfilling bStock signals
should not assume a full multi-year history for intraday intervals.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

log = logging.getLogger("data")

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_OHLCV_COLS).set_axis(pd.DatetimeIndex([], name="time", tz="UTC"))


def download_stock_history(ticker: str, start: date, interval: str = "1h") -> pd.DataFrame:
    """Underlying stock OHLCV history, normalized to the same shape as the Binance loaders:
    columns open/high/low/close/volume, UTC-indexed. Returns an empty frame on any failure --
    a bStock data outage must never crash the caller."""
    try:
        raw = yf.download(ticker, start=start, interval=interval, progress=False, auto_adjust=False)
    except Exception as e:  # yfinance surfaces network/parse failures as varied exception types
        log.warning("yfinance download failed for %s: %s", ticker, e)
        return _empty()

    if raw is None or raw.empty:
        return _empty()

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)  # yf.download nests columns under the ticker
    df = df.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    missing = [c for c in _OHLCV_COLS if c not in df.columns]
    if missing:
        log.warning("yfinance result for %s missing columns %s", ticker, missing)
        return _empty()
    df = df[_OHLCV_COLS]

    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    df.index.name = "time"
    return df
