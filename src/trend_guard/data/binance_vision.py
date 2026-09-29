"""Bulk historical data download from data.binance.vision (BUILD_SPEC.md Section 5.2).

HTTP 404 on a monthly/daily zip means "no data published for that period", not an error: skip
and continue. Raw zips are cached under data/raw/ so re-runs don't re-download; cleaned parquet
is written under data/bars/ so downstream code (strategy, backtest) never re-parses CSVs.

Monthly zips are tried first for a calendar month; if a monthly zip 404s (Binance has not yet
published it -- true for the current, still-open month) we fall back to per-day daily zips for
that month, clipped to the requested [start, end] range. This needs no wall-clock read: it just
follows whatever 404s the server actually returns.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("data")

_VISION_BASE = "https://data.binance.vision/data"
_TIMEOUT = 30.0
_MICROSECOND_THRESHOLD = 1e14  # magnitude cutoff: a ms epoch never reaches this; a us epoch always does

KLINE_COLUMNS = [
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
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
METRICS_COLUMNS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]


def _month_range(start: date, end: date) -> list[tuple[int, int]]:
    """[(year, month), ...] inclusive of both endpoints' months."""
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return months


def _day_range(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start, end - timedelta(days=1)


def _fetch_zip(url: str, cache_path: Path) -> bytes | None:
    """Download (or read from cache) a zip. Returns None on HTTP 404 -- no data for that period."""
    if cache_path.exists():
        return cache_path.read_bytes()
    resp = requests.get(url, timeout=_TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)
    return resp.content


def _looks_like_timestamp(cell: str) -> bool:
    try:
        int(cell.strip())
        return True
    except ValueError:
        return False


def _read_csv_from_zip(raw: bytes, columns: list[str]) -> pd.DataFrame:
    """Extract the single CSV member from a Binance Vision zip. Some files ship a header row,
    some don't: peek at the first cell of the first row and only skip it if it isn't a plausible
    integer timestamp."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            first_line = f.readline().decode("utf-8", errors="replace")
        first_cell = first_line.split(",")[0]
        has_header = not _looks_like_timestamp(first_cell)
        with zf.open(name) as f:
            df = pd.read_csv(f, header=0 if has_header else None)
    df.columns = columns[: len(df.columns)]
    return df


def _normalize_ts(col: pd.Series) -> pd.Series:
    """Detect ms vs microsecond epoch PER ROW and return UTC datetimes.

    Binance switched spot files to microseconds in Jan 2025, so a concatenated multi-month column
    mixes both units; deciding the unit from one sample silently drops half the history.
    """
    col = col.astype("int64")
    ms = col.where(col.abs() <= _MICROSECOND_THRESHOLD, col // 1000)
    return pd.to_datetime(ms, unit="ms", utc=True)


def _dedupe_sort(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    return df.drop_duplicates(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)


def download_klines_1h(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path = Path("data/raw"),
    out_dir: Path = Path("data/bars/1h"),
) -> pd.DataFrame:
    """Backfill 1h spot klines for `symbol` (a Binance symbol, e.g. BTCUSDT) over [start, end].

    Writes cleaned parquet to out_dir/{symbol}.parquet and returns the same frame, indexed by
    UTC open_time.
    """
    frames: list[pd.DataFrame] = []
    for year, month in _month_range(start, end):
        url = f"{_VISION_BASE}/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{year:04d}-{month:02d}.zip"
        cache_path = (
            cache_dir / "spot" / "monthly" / "klines" / symbol / f"{symbol}-1h-{year:04d}-{month:02d}.zip"
        )
        raw = _fetch_zip(url, cache_path)
        if raw is not None:
            frames.append(_read_csv_from_zip(raw, KLINE_COLUMNS))
            continue
        # monthly not published yet (current/open month): fall back to daily files
        month_start, month_end = _month_bounds(year, month)
        for day in _day_range(max(month_start, start), min(month_end, end)):
            d_str = day.isoformat()
            d_url = f"{_VISION_BASE}/spot/daily/klines/{symbol}/1h/{symbol}-1h-{d_str}.zip"
            d_cache = cache_dir / "spot" / "daily" / "klines" / symbol / f"{symbol}-1h-{d_str}.zip"
            d_raw = _fetch_zip(d_url, d_cache)
            if d_raw is None:
                log.info("no daily klines for %s %s", symbol, d_str)
                continue
            frames.append(_read_csv_from_zip(d_raw, KLINE_COLUMNS))

    if not frames:
        return pd.DataFrame(columns=[c for c in KLINE_COLUMNS if c != "open_time"]).set_axis(
            pd.DatetimeIndex([], name="open_time", tz="UTC")
        )

    df = pd.concat(frames, ignore_index=True)
    df = _dedupe_sort(df, "open_time")
    df["open_time"] = _normalize_ts(df["open_time"])
    df = _dedupe_sort(df, "open_time")  # re-sort: raw ints of mixed units don't sort chronologically
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + timedelta(days=1)
    df = df[(df["open_time"] >= start_ts) & (df["open_time"] < end_ts)]
    df = df.drop(columns=["close_time", "ignore"], errors="ignore").set_index("open_time")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_path = out_dir / f"{symbol}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df


def download_funding(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path = Path("data/raw"),
    out_dir: Path = Path("data/bars/funding"),
) -> pd.DataFrame:
    """Backfill UM futures funding rate history for `symbol` over [start, end] (monthly zips only)."""
    frames: list[pd.DataFrame] = []
    for year, month in _month_range(start, end):
        fname = f"{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
        url = f"{_VISION_BASE}/futures/um/monthly/fundingRate/{symbol}/{fname}"
        cache_path = (
            cache_dir
            / "futures"
            / "monthly"
            / "fundingRate"
            / symbol
            / f"{symbol}-fundingRate-{year:04d}-{month:02d}.zip"
        )
        raw = _fetch_zip(url, cache_path)
        if raw is None:
            log.info("no funding data for %s %04d-%02d", symbol, year, month)
            continue
        frames.append(_read_csv_from_zip(raw, FUNDING_COLUMNS))

    if not frames:
        return pd.DataFrame(columns=[c for c in FUNDING_COLUMNS if c != "calc_time"]).set_axis(
            pd.DatetimeIndex([], name="calc_time", tz="UTC")
        )

    df = pd.concat(frames, ignore_index=True)
    df = _dedupe_sort(df, "calc_time")
    df["calc_time"] = _normalize_ts(df["calc_time"])
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + timedelta(days=1)
    df = df[(df["calc_time"] >= start_ts) & (df["calc_time"] < end_ts)].set_index("calc_time")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_path = out_dir / f"{symbol}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df


def download_open_interest(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path = Path("data/raw"),
    out_dir: Path = Path("data/bars/oi"),
) -> pd.DataFrame:
    """Backfill UM futures open-interest metrics for `symbol` over [start, end] (daily zips only)."""
    frames: list[pd.DataFrame] = []
    for day in _day_range(start, end):
        d_str = day.isoformat()
        url = f"{_VISION_BASE}/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{d_str}.zip"
        cache_path = cache_dir / "futures" / "daily" / "metrics" / symbol / f"{symbol}-metrics-{d_str}.zip"
        raw = _fetch_zip(url, cache_path)
        if raw is None:
            log.info("no OI metrics for %s %s", symbol, d_str)
            continue
        frames.append(_read_csv_from_zip(raw, METRICS_COLUMNS))

    if not frames:
        return pd.DataFrame(columns=[c for c in METRICS_COLUMNS if c != "create_time"]).set_axis(
            pd.DatetimeIndex([], name="create_time", tz="UTC")
        )

    df = pd.concat(frames, ignore_index=True)
    df = _dedupe_sort(df, "create_time")
    df["create_time"] = _normalize_ts(df["create_time"])
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + timedelta(days=1)
    df = df[(df["create_time"] >= start_ts) & (df["create_time"] < end_ts)].set_index("create_time")
    for c in df.columns:
        if c != "symbol":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out_path = out_dir / f"{symbol}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df
