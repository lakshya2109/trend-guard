"""Shared resampling / alignment / gap-check utilities.

Pure: no I/O, no clock reads. Used by both the historical (binance_vision) and
live (binance_live, ticker_recorder) data modules, and by strategy code.
"""

from __future__ import annotations

import pandas as pd


def to_hourly(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Resample a UTC-indexed OHLCV frame to 1h bars.

    OHLC columns are resampled with the usual first/max/min/last rule when present;
    volume (and any other numeric column) is summed. `price_col` is used to build
    open/high/low/close when the frame only has a single price series.
    """
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df = df.set_axis(idx)

    has_ohlc = {"open", "high", "low", "close"}.issubset(df.columns)
    if has_ohlc:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        out = df.resample("1h").agg(agg)
        # drop hours with no observations at all (close is NaN)
        return out.dropna(subset=["close"])

    # single price series: derive OHLC from price_col
    s = df[price_col]
    out = s.resample("1h").ohlc()
    if "volume" in df.columns:
        out["volume"] = df["volume"].resample("1h").sum()
    return out.dropna(subset=["close"])


def align(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join a dict of {name: DataFrame} on a UTC hour index.

    Each input frame is reduced to its `close` column (if present) or its single
    column, renamed to the dict key. Returns a wide DataFrame, one column per asset.
    """
    series = {}
    for name, df in dfs.items():
        if df.empty:
            series[name] = pd.Series(dtype=float)
            continue
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        col = "close" if "close" in df.columns else df.columns[0]
        s = df[col].copy()
        s.index = idx
        series[name] = s
    return pd.DataFrame(series).sort_index()


def find_gaps(df: pd.DataFrame, freq: str = "1h") -> list[tuple]:
    """Return [(gap_start, gap_end)] for missing periods in a UTC-indexed frame.

    gap_start/gap_end are the timestamps immediately bracketing each missing run,
    i.e. the last observed bar before the gap and the first observed bar after it.
    Empty or single-row frames have no gaps by definition.
    """
    if df.empty or len(df.index) < 2:
        return []
    idx = pd.DatetimeIndex(df.index).sort_values()
    expected = pd.date_range(idx[0], idx[-1], freq=freq)
    missing = expected.difference(idx)
    if missing.empty:
        return []

    gaps: list[tuple] = []
    missing_sorted = missing.sort_values()
    run_start = None
    prev = None
    step = pd.Timedelta(expected.freq) if expected.freq is not None else (expected[1] - expected[0])
    for ts in missing_sorted:
        if run_start is None:
            run_start = ts
        elif prev is not None and ts - prev > step:
            gaps.append((run_start - step, prev + step))
            run_start = ts
        prev = ts
    gaps.append((run_start - step, prev + step))
    return gaps
