"""LiveBroker: the code-level live-order gate and exposure aggregation, with a fake client."""

from decimal import Decimal as D

import pytest

from trend_guard.execution.live_broker import LiveBroker
from trend_guard.roostoo.models import OrderDetail, ShortPosition, Ticker, WalletEntry
from trend_guard.roostoo.rules import TradingRule

RULES = {"BTC/USD": TradingRule(pair="BTC/USD", price_precision=2, amount_precision=5, mini_order=D("1"))}


class FakeClient:
    def __init__(self):
        self.orders = []

    def get_balance(self):
        return {"USD": WalletEntry(Free=50000, Lock=1000), "BTC": WalletEntry(Free=0.5, Lock=0)}

    def get_ticker(self, pair=None):
        return {"BTC/USD": Ticker(MaxBid=60000, MinAsk=60002, LastPrice=60001)}

    def short_positions(self):
        return [ShortPosition(Pair="ETH/USD", ShortQty=2, Collateral=5000, PositionValue=5200)]

    def place_order(self, pair, side, type_, quantity, price=None):
        self.orders.append((pair, side, type_, quantity, price))
        return OrderDetail(
            Pair=pair, OrderID=7, Status="PENDING", Role="MAKER", Side=side, Quantity=float(quantity)
        )


def test_orders_refused_without_allow_live_orders():
    b = LiveBroker(FakeClient(), RULES)  # default: gate closed
    with pytest.raises(PermissionError):
        b.place_limit("BTC/USD", "BUY", D("0.1"), D("60000"))
    with pytest.raises(PermissionError):
        b.short_open("BTC/USD", D("100"))


def test_precision_applied_before_sending():
    client = FakeClient()
    LiveBroker(client, RULES, allow_live_orders=True).place_limit(
        "BTC/USD", "BUY", D("0.1234567"), D("60000.129")
    )
    assert client.orders[0][3] == D("0.12345") and client.orders[0][4] == D("60000.12")


def test_exposure_combines_cash_longs_and_short_value():
    exp = LiveBroker(FakeClient(), RULES).get_exposure()
    assert exp.cash == D("50000")
    assert exp.long_qty == {"BTC/USD": D("0.5")}
    assert exp.short_collateral == {"ETH/USD": D("5000")}
    assert exp.equity == D("51000") + D("0.5") * D("60001") + D("5200")
