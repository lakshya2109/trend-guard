"""Tests for trend_guard.data.binance_vision. No live network calls -- all HTTP mocked with
`responses` using small synthetic zip/CSV fixtures built in-memory."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import responses

from trend_guard.data import binance_vision as bv


def _zip_csv(rows: list[list], header: list[str] | None = None, filename: str = "data.csv") -> bytes:
    lines = []
    if header is not None:
        lines.append(",".join(header))
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    csv_bytes = ("\n".join(lines) + "\n").encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, csv_bytes)
    return buf.getvalue()


def _kline_row(open_time_ms: int) -> list:
    return [open_time_ms, 100.0, 101.0, 99.0, 100.5, 10.0, open_time_ms + 3_599_999, 1000.0, 5, 4.0, 400.0, 0]


# --- unit-level: timestamp unit detection -----------------------------------------------------


def test_normalize_ts_detects_milliseconds():
    col = pd.Series([1704067200000, 1704070800000])  # 2024-01-01 00:00, 01:00 UTC in ms
    out = bv._normalize_ts(col)
    assert out.iloc[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert out.iloc[1] == pd.Timestamp("2024-01-01 01:00:00", tz="UTC")


def test_normalize_ts_detects_microseconds():
    col = pd.Series([1704067200000000, 1704070800000000])  # same instants, in microseconds
    out = bv._normalize_ts(col)
    assert out.iloc[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert out.iloc[1] == pd.Timestamp("2024-01-01 01:00:00", tz="UTC")


# --- unit-level: header vs no-header CSV parsing -----------------------------------------------


def test_read_csv_from_zip_no_header():
    raw = _zip_csv([_kline_row(1704067200000)], header=None)
    df = bv._read_csv_from_zip(raw, bv.KLINE_COLUMNS)
    assert list(df.columns) == bv.KLINE_COLUMNS
    assert len(df) == 1
    assert float(df.iloc[0]["open"]) == 100.0


def test_read_csv_from_zip_with_header():
    raw = _zip_csv([_kline_row(1704067200000)], header=bv.KLINE_COLUMNS)
    df = bv._read_csv_from_zip(raw, bv.KLINE_COLUMNS)
    assert list(df.columns) == bv.KLINE_COLUMNS
    assert len(df) == 1
    assert float(df.iloc[0]["open"]) == 100.0


# --- download_klines_1h: integration with mocked HTTP ------------------------------------------


@responses.activate
def test_download_klines_1h_monthly_available(tmp_path: Path):
    rows = [_kline_row(1704067200000 + i * 3_600_000) for i in range(3)]  # 3 hourly bars
    zip_bytes = _zip_csv(rows, header=None)
    url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    responses.add(responses.GET, url, body=zip_bytes, status=200)

    df = bv.download_klines_1h(
        "BTCUSDT",
        date(2024, 1, 1),
        date(2024, 1, 31),
        cache_dir=tmp_path / "raw",
        out_dir=tmp_path / "bars",
    )

    assert len(df) == 3
    assert df.index.tz is not None
    assert (tmp_path / "bars" / "BTCUSDT.parquet").exists()
    assert list(df.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]


@responses.activate
def test_download_klines_1h_404_falls_back_to_daily(tmp_path: Path):
    month_url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    responses.add(responses.GET, month_url, status=404)

    day_rows = {
        "2024-01-01": [_kline_row(1704067200000)],
        "2024-01-02": [_kline_row(1704153600000)],
        "2024-01-03": [_kline_row(1704240000000)],
    }
    for d, rows in day_rows.items():
        url = f"https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{d}.zip"
        responses.add(responses.GET, url, body=_zip_csv(rows, header=None), status=200)

    df = bv.download_klines_1h(
        "BTCUSDT",
        date(2024, 1, 1),
        date(2024, 1, 3),
        cache_dir=tmp_path / "raw",
        out_dir=tmp_path / "bars",
    )

    assert len(df) == 3


@responses.activate
def test_download_klines_1h_404_day_is_skipped_not_an_error(tmp_path: Path):
    month_url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    responses.add(responses.GET, month_url, status=404)

    # only day 1 has data; days 2 and 3 404 (e.g. new listing) and must be skipped, not raise
    url1 = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01-01.zip"
    responses.add(responses.GET, url1, body=_zip_csv([_kline_row(1704067200000)], header=None), status=200)
    url2 = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01-02.zip"
    responses.add(responses.GET, url2, status=404)
    url3 = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01-03.zip"
    responses.add(responses.GET, url3, status=404)

    df = bv.download_klines_1h(
        "BTCUSDT",
        date(2024, 1, 1),
        date(2024, 1, 3),
        cache_dir=tmp_path / "raw",
        out_dir=tmp_path / "bars",
    )

    assert len(df) == 1


@responses.activate
def test_download_klines_1h_caches_zip(tmp_path: Path):
    rows = [_kline_row(1704067200000)]
    url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    responses.add(responses.GET, url, body=_zip_csv(rows, header=None), status=200)

    cache_dir = tmp_path / "raw"
    out_dir = tmp_path / "bars"
    bv.download_klines_1h(
        "BTCUSDT", date(2024, 1, 1), date(2024, 1, 31), cache_dir=cache_dir, out_dir=out_dir
    )
    cached = list(cache_dir.rglob("*.zip"))
    assert len(cached) == 1

    # second call should hit the cache only (no more responses registered) and not raise
    responses.reset()
    df2 = bv.download_klines_1h(
        "BTCUSDT", date(2024, 1, 1), date(2024, 1, 31), cache_dir=cache_dir, out_dir=out_dir
    )
    assert len(df2) == 1


# --- download_funding ---------------------------------------------------------------------------


@responses.activate
def test_download_funding_basic(tmp_path: Path):
    rows = [[1704067200000, 8, "0.0001"], [1704096000000, 8, "-0.0002"]]
    url = "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip"
    responses.add(responses.GET, url, body=_zip_csv(rows, header=None), status=200)

    df = bv.download_funding(
        "BTCUSDT",
        date(2024, 1, 1),
        date(2024, 1, 31),
        cache_dir=tmp_path / "raw",
        out_dir=tmp_path / "funding",
    )

    assert len(df) == 2
    assert "last_funding_rate" in df.columns
    assert (tmp_path / "funding" / "BTCUSDT.parquet").exists()


@responses.activate
def test_download_funding_404_is_skipped(tmp_path: Path):
    url = "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip"
    responses.add(responses.GET, url, status=404)

    df = bv.download_funding(
        "BTCUSDT",
        date(2024, 1, 1),
        date(2024, 1, 31),
        cache_dir=tmp_path / "raw",
        out_dir=tmp_path / "funding",
    )

    assert df.empty


# --- download_open_interest -----------------------------------------------------------------------


@responses.activate
def test_download_open_interest_basic(tmp_path: Path):
    row = [1704067200000, "BTCUSDT", 1000.0, 50_000_000.0, 1.1, 1.05, 1.2, 0.95]
    url = "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-01.zip"
    responses.add(responses.GET, url, body=_zip_csv([row], header=None), status=200)

    df = bv.download_open_interest(
        "BTCUSDT", date(2024, 1, 1), date(2024, 1, 1), cache_dir=tmp_path / "raw", out_dir=tmp_path / "oi"
    )

    assert len(df) == 1
    assert "sum_open_interest" in df.columns
