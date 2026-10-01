"""LiveBroker: the code-level live-order gate and exposure aggregation, with a fake client."""

from decimal import Decimal as D

import pytest

from trend_guard.execution.live_broker import LiveBroker
from trend_guard.roostoo.models import OrderDetail, ShortOpenResult, ShortPosition, Ticker, WalletEntry
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


class QueryClient(FakeClient):
    """query_order answers copied from the live mock exchange (Oct 2026)."""

    def __init__(self, detail: dict):
        super().__init__()
        self.detail = detail

    def query_order(self, order_id=None, pair=None, pending_only=False, limit=None):
        return [OrderDetail.model_validate(self.detail)]


def test_canceled_limit_echoing_full_quantity_is_not_a_fill():
    # Roostoo reports FilledQuantity = Quantity on an untouched order, with FilledAverPrice 0
    detail = {"OrderID": 3411817, "Pair": "SEI/USD", "Status": "CANCELED", "Role": "MAKER", "Side": "BUY",
              "Type": "LIMIT", "Price": 0.07158, "Quantity": 32971.8, "FilledQuantity": 32971.8,
              "FilledAverPrice": 0, "CommissionCoin": "SEI", "CommissionChargeValue": 0}
    st = LiveBroker(QueryClient(detail), RULES).query("3411817")
    assert st.filled_qty == 0 and st.avg_price == 0 and st.fee == 0


def test_filled_limit_reports_quantity_price_and_usd_fee():
    detail = {"OrderID": 3412171, "Pair": "SEI/USD", "Status": "FILLED", "Role": "MAKER", "Side": "BUY",
              "Type": "LIMIT", "Price": 0.07128, "Quantity": 33549.4, "FilledQuantity": 33549.4,
              "FilledAverPrice": 0.07128, "CommissionCoin": "USD", "CommissionChargeValue": 1.1957}
    st = LiveBroker(QueryClient(detail), RULES).query("3412171")
    assert st.filled_qty == D("33549.4") and st.avg_price == D("0.07128") and st.fee == D("1.1957")


def test_fee_charged_in_base_coin_is_converted_to_usd():
    detail = {"OrderID": 1, "Pair": "BTC/USD", "Status": "FILLED", "Role": "MAKER", "Side": "BUY",
              "Type": "LIMIT", "Price": 84784, "Quantity": 0.01695, "FilledQuantity": 0.01695,
              "FilledAverPrice": 84784, "CommissionCoin": "BTC", "CommissionChargeValue": 0.0000085}
    st = LiveBroker(QueryClient(detail), RULES).query("1")
    assert st.fee == D("0.0000085") * D("84784")


class MergingShortClient(FakeClient):
    """Adding to an existing short: Roostoo answers with the merged position, not the trade."""

    def short_positions(self):
        return [ShortPosition(Pair="ARB/USD", ShortQty=6114.7, EntryPrice=0.2026, Collateral=1238.84)]

    def short_open(self, pair, collateral):
        total = 6114.7 + 4709.9
        merged_px = (6114.7 * 0.2026 + 4709.9 * 0.2009) / total
        return ShortOpenResult(ID=9, Pair=pair, OrderType="MARKET", EntryPrice=merged_px, ShortQty=total,
                               Collateral=1238.84 + 946.22, OpenFee=0.946, Status="OPEN")


def test_short_open_onto_existing_position_reports_only_the_added_trade():
    ack = LiveBroker(MergingShortClient(), RULES, allow_live_orders=True).short_open("ARB/USD", D("946.22"))
    assert ack.short_qty == pytest.approx(D("4709.9"))
    assert ack.entry_price == pytest.approx(D("0.2009"))
    assert ack.collateral == pytest.approx(D("946.22"))


class LiveShapedClient(FakeClient):
    """Balance as the live API returns it: USD Lock = pending-order cash 1000 + short collateral 5000."""

    def get_balance(self):
        return {
            "USD": WalletEntry(Free=50000, Lock=6000, ShortCollateral=5000),
            "BTC": WalletEntry(Free=0.5, Lock=0),
        }


def test_short_collateral_is_counted_once():
    # collateral sits in USD Lock AND inside the short's PositionValue; equity must hold it once
    b = LiveBroker(LiveShapedClient(), RULES)
    expected = D("50000") + D("1000") + D("0.5") * D("60001") + D("5200")
    assert b.get_exposure().equity == expected
    assert b.get_equity() == expected


def test_exposure_combines_cash_longs_and_short_value():
    exp = LiveBroker(FakeClient(), RULES).get_exposure()
    assert exp.cash == D("50000")
    assert exp.long_qty == {"BTC/USD": D("0.5")}
    assert exp.short_collateral == {"ETH/USD": D("5000")}
    assert exp.equity == D("51000") + D("0.5") * D("60001") + D("5200")
