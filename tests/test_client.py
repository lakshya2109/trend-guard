"""Roostoo client tests. BUILD_SPEC.md Section 4.6."""

from decimal import Decimal

import pytest
import requests
import responses

from trend_guard.roostoo.client import RoostooClient
from trend_guard.roostoo.errors import OrderStateUnknownError, RoostooAPIError, ShortsDisabledError

BASE = "https://mock-api.roostoo.com"


def make_client() -> RoostooClient:
    return RoostooClient(BASE, api_key="USEAPIKEYASMYID", secret_key="testsecret", rate_limiter=None)


def test_signature_matches_documented_example():
    """Docs vector: pair=BNB/USD, quantity=2000, side=BUY, timestamp=1580774512000, type=MARKET,
    secret S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep
    -> 20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff"""
    client = RoostooClient(
        BASE,
        api_key="USEAPIKEYASMYID",
        secret_key="S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep",
    )
    params = {
        "pair": "BNB/USD",
        "quantity": "2000",
        "side": "BUY",
        "timestamp": "1580774512000",
        "type": "MARKET",
    }
    total = client.build_total_params(params)
    assert total == "pair=BNB/USD&quantity=2000&side=BUY&timestamp=1580774512000&type=MARKET"
    assert client.sign(total) == "20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff"


@responses.activate
def test_missing_numeric_fields_parse_as_zero():
    responses.add(
        responses.GET,
        f"{BASE}/v3/ticker",
        json={
            "Success": True,
            "ErrMsg": "",
            "ServerTime": 1,
            "Data": {"BTC/USD": {"MaxBid": 9318.45, "MinAsk": 9319.42, "LastPrice": 9319.35}},
        },
        status=200,
    )
    client = make_client()
    tickers = client.get_ticker("BTC/USD")
    assert tickers["BTC/USD"].change == 0
    assert tickers["BTC/USD"].coin_trade_value == 0


@responses.activate
@pytest.mark.parametrize("key", ["SpotWallet", "Wallet"])
def test_balance_reads_spot_wallet_and_legacy_wallet(key: str):
    # the live API returns "SpotWallet"; the published docs still say "Wallet"
    responses.add(
        responses.GET,
        f"{BASE}/v3/balance",
        json={"Success": True, "ErrMsg": "", key: {"USD": {"Free": 50000, "Lock": 0, "ShortCollateral": 0}}},
        status=200,
    )
    bal = make_client().get_balance()
    assert bal["USD"].free == 50000


@responses.activate
def test_pending_count_empty_is_not_an_error():
    responses.add(
        responses.GET,
        f"{BASE}/v3/pending_count",
        json={
            "Success": False,
            "ErrMsg": "no pending order under this account",
            "TotalPending": 0,
            "OrderPairs": {},
        },
        status=200,
    )
    client = make_client()
    total, pairs = client.pending_count()
    assert total == 0
    assert pairs == {}


@responses.activate
def test_query_order_no_match_is_not_an_error():
    responses.add(
        responses.POST,
        f"{BASE}/v3/query_order",
        json={"Success": False, "ErrMsg": "no order matched"},
        status=200,
    )
    client = make_client()
    result = client.query_order(pair="BTC/USD")
    assert result == []


def test_post_timeout_raises_order_state_unknown_and_is_not_retried(monkeypatch):
    client = make_client()
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        raise requests.Timeout("boom")

    monkeypatch.setattr(client.session, "post", fake_post)
    with pytest.raises(OrderStateUnknownError):
        client.place_order("BTC/USD", "BUY", "MARKET", Decimal("1"))
    assert calls["n"] == 1  # never retried


@responses.activate
def test_shorts_disabled_raises():
    responses.add(
        responses.POST,
        f"{BASE}/v6/short_open",
        json={"Success": False, "ErrMsg": "this competition does not allow short positions"},
        status=200,
    )
    client = make_client()
    with pytest.raises(ShortsDisabledError):
        client.short_open("BTC/USD", Decimal("100"))


@responses.activate
def test_cancel_order_requires_explicit_cancel_all_flag():
    client = make_client()
    with pytest.raises(ValueError):
        client.cancel_order()


@responses.activate
def test_cancel_order_all_when_flagged():
    responses.add(
        responses.POST,
        f"{BASE}/v3/cancel_order",
        json={"Success": True, "ErrMsg": "", "CanceledList": [20, 35]},
        status=200,
    )
    client = make_client()
    assert client.cancel_order(cancel_all=True) == [20, 35]


@responses.activate
def test_place_order_failure_raises_api_error():
    responses.add(
        responses.POST,
        f"{BASE}/v3/place_order",
        json={"Success": False, "ErrMsg": "insufficient balance"},
        status=200,
    )
    client = make_client()
    with pytest.raises(RoostooAPIError):
        client.place_order("BTC/USD", "BUY", "MARKET", Decimal("1"))


@responses.activate
def test_sync_clock_sets_offset():
    responses.add(
        responses.GET,
        f"{BASE}/v3/serverTime",
        json={"ServerTime": 1_700_000_000_000},
        status=200,
    )
    client = make_client()
    client.sync_clock()
    # offset should roughly bridge "now" (year ~2026) back to the mocked server time (2023-ish)
    assert client.offset_ms < 0
