"""Tests for trend_guard.data.bars: pure resample/align/gap-check helpers, no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_guard.data.bars import align, find_gaps, to_hourly


def test_to_hourly_resamples_ohlcv():
    idx = pd.date_range("2024-01-01 00:00", periods=6, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5, 6],
            "high": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "close": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
            "volume": [10, 10, 10, 10, 10, 10],
        },
        index=idx,
    )
    out = to_hourly(df)
    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == 1
    assert first["high"] == 4.5
    assert first["low"] == 0.5
    assert first["close"] == 4.2
    assert first["volume"] == 40
    second = out.iloc[1]
    assert second["open"] == 5
    assert second["volume"] == 20


def test_to_hourly_empty_frame_passthrough():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = to_hourly(df)
    assert out.empty


def test_align_outer_joins_on_utc_hour():
    idx_a = pd.date_range("2024-01-01 00:00", periods=3, freq="1h", tz="UTC")
    idx_b = pd.date_range("2024-01-01 01:00", periods=3, freq="1h", tz="UTC")
    df_a = pd.DataFrame({"close": [1, 2, 3]}, index=idx_a)
    df_b = pd.DataFrame({"close": [10, 20, 30]}, index=idx_b)

    out = align({"A": df_a, "B": df_b})

    assert list(out.columns) == ["A", "B"]
    assert len(out) == 4
    assert out.loc[idx_a[0], "A"] == 1
    assert np.isnan(out.loc[idx_a[0], "B"])
    assert out.loc[idx_b[-1], "B"] == 30
    assert np.isnan(out.loc[idx_b[-1], "A"])


def test_align_handles_empty_frame():
    idx_a = pd.date_range("2024-01-01 00:00", periods=2, freq="1h", tz="UTC")
    df_a = pd.DataFrame({"close": [1, 2]}, index=idx_a)
    out = align({"A": df_a, "B": pd.DataFrame(columns=["close"])})
    assert "B" in out.columns
    assert out["B"].isna().all()


def test_find_gaps_detects_missing_run():
    idx = pd.date_range("2024-01-01 00:00", periods=11, freq="1h", tz="UTC")
    keep = idx[[0, 1, 2, 6, 7, 8, 9, 10]]  # drop hours 3, 4, 5
    df = pd.DataFrame({"close": range(len(keep))}, index=keep)

    gaps = find_gaps(df, freq="1h")

    assert gaps == [(idx[2], idx[6])]


def test_find_gaps_no_gaps_returns_empty():
    idx = pd.date_range("2024-01-01 00:00", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": range(5)}, index=idx)
    assert find_gaps(df, freq="1h") == []


def test_find_gaps_multiple_runs():
    idx = pd.date_range("2024-01-01 00:00", periods=12, freq="1h", tz="UTC")
    keep = idx[[0, 1, 4, 5, 9, 10, 11]]  # gaps: [2,3] and [6,7,8]
    df = pd.DataFrame({"close": range(len(keep))}, index=keep)

    gaps = find_gaps(df, freq="1h")

    assert gaps == [(idx[1], idx[4]), (idx[5], idx[9])]
