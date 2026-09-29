"""Tests for trend_guard.data.binance_live. No live network calls -- HTTP mocked with `responses`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import responses
from responses import registries

from trend_guard.data import binance_live as bl


def _kline_row(open_time_ms: int) -> list:
    return [
        open_time_ms,
        "100.0",
        "101.0",
        "99.0",
        "100.5",
        "10.0",
        open_time_ms + 3_599_999,
        "1000.0",
        5,
        "4.0",
        "400.0",
        "0",
    ]


# --- drop_unclosed_bar: pure, no I/O -----------------------------------------------------------


def test_drop_unclosed_bar_keeps_only_closed():
    idx = pd.date_range("2024-01-01 00:00", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
    now = datetime(2024, 1, 1, 2, 30, tzinfo=UTC)  # bar at hour 2 (closes at 3:00) hasn't closed yet

    out = bl.drop_unclosed_bar(df, now, timedelta(hours=1))

    assert len(out) == 2
    assert idx[2] not in out.index


def test_drop_unclosed_bar_naive_now_treated_as_utc():
    idx = pd.date_range("2024-01-01 00:00", periods=2, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": [1, 2]}, index=idx)
    now = datetime(2024, 1, 1, 1, 30)  # naive, should be treated as UTC
    out = bl.drop_unclosed_bar(df, now, timedelta(hours=1))
    assert len(out) == 1


# --- fetch_recent_klines_1h: mirror + fallback + closed-bar filtering --------------------------


@responses.activate
def test_fetch_recent_klines_uses_data_api_and_drops_forming_bar():
    rows = [_kline_row(1704067200000), _kline_row(1704070800000)]  # 00:00 and 01:00 UTC
    responses.add(responses.GET, bl._DATA_API_KLINES, json=rows, status=200)

    now = datetime(2024, 1, 1, 1, 30, tzinfo=UTC)  # bar at 01:00 (closes 02:00) not yet closed
    df = bl.fetch_recent_klines_1h("BTCUSDT", now)

    assert len(df) == 1
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")


@responses.activate(registry=registries.OrderedRegistry)
def test_fetch_recent_klines_falls_back_to_api_binance_com_on_failure():
    responses.add(responses.GET, bl._DATA_API_KLINES, status=500)
    rows = [_kline_row(1704067200000)]
    responses.add(responses.GET, bl._SPOT_FALLBACK_KLINES, json=rows, status=200)

    now = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)
    df = bl.fetch_recent_klines_1h("BTCUSDT", now)

    assert len(df) == 1


@responses.activate
def test_fetch_recent_klines_both_sources_fail_returns_empty():
    responses.add(responses.GET, bl._DATA_API_KLINES, status=500)
    responses.add(responses.GET, bl._SPOT_FALLBACK_KLINES, status=500)

    df = bl.fetch_recent_klines_1h("BTCUSDT", datetime(2024, 1, 1, tzinfo=UTC))

    assert df.empty


# --- probe_fapi_access ---------------------------------------------------------------------------


@responses.activate
def test_probe_fapi_access_true_on_200():
    responses.add(responses.GET, bl._FAPI_PING, json={}, status=200)
    assert bl.probe_fapi_access() is True


@responses.activate
def test_probe_fapi_access_false_on_connection_error():
    import requests

    responses.add(responses.GET, bl._FAPI_PING, body=requests.ConnectionError("blocked"))
    assert bl.probe_fapi_access() is False


# --- fetch_funding: fapi-blocked falls back to Bybit --------------------------------------------


@responses.activate
def test_fetch_funding_uses_fapi_when_available():
    responses.add(
        responses.GET,
        bl._FAPI_FUNDING,
        json=[{"symbol": "BTCUSDT", "fundingTime": 1704067200000, "fundingRate": "0.0001"}],
        status=200,
    )
    df = bl.fetch_funding("BTCUSDT", fapi_available=True)
    assert len(df) == 1
    assert df["funding_rate"].iloc[0] == 0.0001


@responses.activate
def test_fetch_funding_falls_back_to_bybit_when_fapi_blocked():
    bybit_payload = {
        "result": {
            "list": [
                {"fundingRateTimestamp": "1704067200000", "fundingRate": "0.0002"},
            ]
        }
    }
    responses.add(responses.GET, bl._BYBIT_FUNDING, json=bybit_payload, status=200)

    df = bl.fetch_funding("BTCUSDT", fapi_available=False)

    assert len(df) == 1
    assert df["funding_rate"].iloc[0] == 0.0002


@responses.activate
def test_fetch_funding_fapi_request_fails_falls_back_to_bybit():
    responses.add(responses.GET, bl._FAPI_FUNDING, status=403)
    bybit_payload = {"result": {"list": [{"fundingRateTimestamp": "1704067200000", "fundingRate": "0.0003"}]}}
    responses.add(responses.GET, bl._BYBIT_FUNDING, json=bybit_payload, status=200)

    df = bl.fetch_funding("BTCUSDT", fapi_available=True)

    assert len(df) == 1
    assert df["funding_rate"].iloc[0] == 0.0003


@responses.activate
def test_fetch_funding_never_raises_on_total_failure():
    responses.add(responses.GET, bl._FAPI_FUNDING, status=500)
    responses.add(responses.GET, bl._BYBIT_FUNDING, status=500)

    df = bl.fetch_funding("BTCUSDT", fapi_available=True)

    assert df.empty


# --- fetch_open_interest: same fapi/Bybit pattern ------------------------------------------------


@responses.activate
def test_fetch_open_interest_uses_fapi_when_available():
    responses.add(
        responses.GET,
        bl._FAPI_OI,
        json=[{"symbol": "BTCUSDT", "sumOpenInterest": "1000.5", "timestamp": 1704067200000}],
        status=200,
    )
    df = bl.fetch_open_interest("BTCUSDT", fapi_available=True)
    assert len(df) == 1
    assert df["open_interest"].iloc[0] == 1000.5


@responses.activate
def test_fetch_open_interest_falls_back_to_bybit_when_fapi_blocked():
    bybit_payload = {"result": {"list": [{"timestamp": "1704067200000", "openInterest": "2000.0"}]}}
    responses.add(responses.GET, bl._BYBIT_OI, json=bybit_payload, status=200)

    df = bl.fetch_open_interest("BTCUSDT", fapi_available=False)

    assert len(df) == 1
    assert df["open_interest"].iloc[0] == 2000.0


# --- check_continuity: forward-fill <=2 gaps, mark stale beyond that ----------------------------


def test_check_continuity_fills_short_gap_not_stale():
    idx = pd.date_range("2024-01-01 00:00", periods=5, freq="1h", tz="UTC")
    keep = idx[[0, 1, 4]]  # missing hours 2, 3 (a 2-bar gap)
    df = pd.DataFrame({"close": [1.0, 2.0, 5.0]}, index=keep)

    out, is_stale = bl.check_continuity(df, freq="1h", max_fill=2)

    assert is_stale is False
    assert len(out) == 5
    assert out.loc[idx[2], "close"] == 2.0  # forward-filled from hour 1
    assert out.loc[idx[3], "close"] == 2.0


def test_check_continuity_marks_stale_beyond_max_fill():
    idx = pd.date_range("2024-01-01 00:00", periods=6, freq="1h", tz="UTC")
    keep = idx[[0, 1, 5]]  # missing hours 2, 3, 4 (a 3-bar gap) -- beyond max_fill=2
    df = pd.DataFrame({"close": [1.0, 2.0, 6.0]}, index=keep)

    out, is_stale = bl.check_continuity(df, freq="1h", max_fill=2)

    assert is_stale is True


def test_check_continuity_empty_frame_is_stale():
    df = pd.DataFrame(columns=["close"])
    out, is_stale = bl.check_continuity(df)
    assert is_stale is True
    assert out.empty
