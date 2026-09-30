"""Binance REST klines + cache. HTTP is mocked with `responses`; no live network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import responses

from trend_guard.data import binance_live as bl

T0 = datetime(2024, 1, 1, tzinfo=UTC)
MS = 3_600_000
URL = bl.HOSTS[0] + "/api/v3/klines"
FALLBACK = bl.HOSTS[1] + "/api/v3/klines"


def kline(open_ms: int, close: float = 100.5) -> list:
    return [open_ms, "100", "101", "99", str(close), "10", open_ms + MS - 1, "1005", 5, "5", "500", "0"]


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


@responses.activate
def test_fetch_klines_drops_the_forming_bar():
    responses.add(responses.GET, URL, json=[kline(ms(T0)), kline(ms(T0) + MS)])
    now = T0 + timedelta(hours=1, minutes=30)  # the 01:00 bar closes at 02:00: still forming
    df = bl.fetch_klines("BTCUSDT", T0, now)
    assert list(df.index) == [pd.Timestamp(T0)]
    assert list(df.columns) == bl.BAR_COLS and df["quote_volume"].iloc[0] == 1005.0


@responses.activate
def test_fetch_klines_paginates_until_a_short_batch(monkeypatch):
    monkeypatch.setattr(bl, "_LIMIT", 2)
    responses.add(responses.GET, URL, json=[kline(ms(T0)), kline(ms(T0) + MS)])
    responses.add(responses.GET, URL, json=[kline(ms(T0) + 2 * MS)])
    df = bl.fetch_klines("BTCUSDT", T0, T0 + timedelta(hours=10))
    assert len(df) == 3 and len(responses.calls) == 2
    assert "startTime=" + str(ms(T0) + 2 * MS) in responses.calls[1].request.url


@responses.activate
def test_fetch_klines_falls_back_to_second_host(monkeypatch):
    monkeypatch.setattr(bl.time, "sleep", lambda s: None)
    for _ in range(3):
        responses.add(responses.GET, URL, status=500)
    responses.add(responses.GET, FALLBACK, json=[kline(ms(T0))])
    assert len(bl.fetch_klines("BTCUSDT", T0, T0 + timedelta(hours=5))) == 1


@responses.activate
def test_unknown_symbol_returns_empty():
    responses.add(responses.GET, URL, status=400, json={"code": -1121, "msg": "Invalid symbol."})
    assert bl.fetch_klines("NOPEUSDT", T0, T0 + timedelta(hours=5)).empty


@responses.activate
def test_update_cache_only_fetches_new_bars(tmp_path):
    responses.add(responses.GET, URL, json=[kline(ms(T0)), kline(ms(T0) + MS)])
    first = bl.update_cache("BTCUSDT", tmp_path, T0, T0 + timedelta(hours=2))
    assert len(first) == 2
    responses.add(responses.GET, URL, json=[kline(ms(T0) + 2 * MS)])
    second = bl.update_cache("BTCUSDT", tmp_path, T0, T0 + timedelta(hours=3))
    assert len(second) == 3
    assert "startTime=" + str(ms(T0) + 2 * MS) in responses.calls[-1].request.url
    assert len(bl.load_cache(tmp_path, ["BTCUSDT", "MISSING"])["BTCUSDT"]) == 3


def test_check_continuity_fills_short_gaps_and_flags_long_ones():
    idx = pd.date_range(T0, periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": range(10)}, index=idx, dtype=float)
    filled, stale = bl.check_continuity(df.drop(idx[[3, 4]]))
    assert not stale and filled.loc[idx[4], "close"] == 2.0
    _, stale = bl.check_continuity(df.drop(idx[[3, 4, 5]]))
    assert stale


def test_panel_aligns_pairs_on_one_hourly_grid():
    a = pd.DataFrame(
        {"close": [1.0, 2.0], "quote_volume": [5.0, 6.0]},
        index=pd.date_range(T0, periods=2, freq="1h", tz="UTC"),
    )
    b = pd.DataFrame(
        {"close": [3.0], "quote_volume": [7.0]},
        index=pd.date_range(T0 + timedelta(hours=1), periods=1, freq="1h", tz="UTC"),
    )
    close, qv = bl.panel({"AUSDT": a, "BUSDT": b}, {"AUSDT": "A/USD", "BUSDT": "B/USD"})
    assert list(close.columns) == ["A/USD", "B/USD"] and len(close) == 2
    assert pd.isna(close.iloc[0]["B/USD"]) and qv.iloc[1]["B/USD"] == 7.0
