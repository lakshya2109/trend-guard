"""Order planning, maker pricing, ambiguous-order matching and executor reprice logic. Section 8."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from trend_guard.clock import SimClock
from trend_guard.config import ExecutionConfig
from trend_guard.execution.broker import Exposure, OrderAck, OrderStatus
from trend_guard.execution.order_manager import (
    OrderExecutor,
    PlannedOrder,
    maker_price,
    match_unknown_order,
    plan_orders,
)
from trend_guard.roostoo.errors import OrderStateUnknownError
from trend_guard.roostoo.models import OrderDetail, Ticker
from trend_guard.roostoo.rules import TradingRule

NOW = datetime(2026, 10, 5, 12, 3, tzinfo=UTC)
CFG = ExecutionConfig()  # deadband 0.02, min_hold 3h, reprice 60s, 4 tries
D = Decimal


def exposure(cash="100000", long_qty=None, shorts=None, prices=None) -> Exposure:
    long_qty = {k: D(v) for k, v in (long_qty or {}).items()}
    shorts = {k: D(v) for k, v in (shorts or {}).items()}
    prices = {k: D(v) for k, v in (prices or {"BTC/USD": "50000", "ETH/USD": "2000"}).items()}
    equity = D(cash) + sum((q * prices[p] for p, q in long_qty.items()), D(0)) + sum(shorts.values(), D(0))
    return Exposure(
        equity=equity,
        cash=D(cash),
        prices=prices,
        long_qty=long_qty,
        short_collateral=shorts,
        short_value=shorts,
    )


def plan(targets, exp, act_all=True, prev=None, rules=None):
    return plan_orders(targets, exp, CFG, act_all, prev or {}, rules)


def test_small_resize_is_skipped_but_open_goes_through():
    exp = exposure(cash="90000", long_qty={"BTC/USD": "0.2"})  # 10k long = 10% of equity
    assert plan({"BTC/USD": 0.11}, exp) == []  # 1% < max(1%, 25% x 11%)
    assert plan({"BTC/USD": 0.14}, exp)[0].action == "BUY"  # 4% > 3.5%
    assert plan({"BTC/USD": 0.10, "ETH/USD": 0.005}, exposure())[1].action == "BUY"  # opens always


def test_unchanged_target_outside_decision_hour_is_left_to_ride():
    exp = exposure(cash="90000", long_qty={"BTC/USD": "0.2"})
    assert plan({"BTC/USD": 0.20}, exp, act_all=False, prev={"BTC/USD": 0.20}) == []
    assert plan({"BTC/USD": 0.20}, exp, act_all=False, prev={"BTC/USD": 0.15})[0].action == "BUY"


def test_buy_when_target_above_current():
    plans = plan({"BTC/USD": 0.10}, exposure())
    assert len(plans) == 1 and plans[0].action == "BUY"
    assert abs(plans[0].notional - D("10000")) < D("1")


def test_reductions_come_before_additions():
    exp = exposure(cash="80000", long_qty={"BTC/USD": "0.4"})  # 20k BTC long
    plans = plan({"BTC/USD": 0.0, "ETH/USD": 0.15}, exp)
    assert [p.action for p in plans] == ["SELL", "BUY"]
    assert plans[0].reducing and not plans[1].reducing


def test_sign_flip_closes_at_market_then_opens_in_same_cycle():
    exp = exposure(cash="80000", long_qty={"BTC/USD": "0.4"})
    plans = plan({"BTC/USD": -0.10}, exp)
    assert [(p.action, p.urgent) for p in plans] == [("SELL", True), ("SHORT_OPEN", False)]
    exp2 = exposure(cash="90000", shorts={"BTC/USD": "10000"})
    plans2 = plan({"BTC/USD": 0.10}, exp2)
    assert [(p.action, p.close_pct) for p in plans2][0] == ("SHORT_CLOSE", D(100))
    assert plans2[1].action == "BUY"


def test_short_open_and_partial_close_pct():
    plans = plan({"BTC/USD": -0.10}, exposure())
    assert plans[0].action == "SHORT_OPEN" and plans[0].collateral == D("10000.00")
    exp = exposure(cash="80000", shorts={"BTC/USD": "20000"})
    plans = plan({"BTC/USD": -0.10}, exp)
    assert plans[0].action == "SHORT_CLOSE" and plans[0].close_pct == D("50.00")


def test_additions_never_exceed_cash_no_leverage():
    exp = exposure(cash="5000", long_qty={"ETH/USD": "47.5"})  # 95k ETH, only 5k cash
    plans = plan({"ETH/USD": 0.95, "BTC/USD": 0.30}, exp)
    spend = sum((p.notional for p in plans if not p.reducing), D(0))
    assert spend <= D("5000")


def test_mini_order_drops_tiny_orders():
    rules = {
        "BTC/USD": TradingRule(pair="BTC/USD", price_precision=2, amount_precision=6, mini_order=D("5000"))
    }
    assert plan({"BTC/USD": 0.03}, exposure(), rules=rules) == []  # $3k < $5k MiniOrder


def test_quantities_are_rounded_to_amount_precision():
    rules = {"ETH/USD": TradingRule(pair="ETH/USD", price_precision=2, amount_precision=2, mini_order=D("1"))}
    plans = plan({"ETH/USD": 0.1234567}, exposure(), rules=rules)
    assert plans[0].qty == plans[0].qty.quantize(D("0.01"))


def test_maker_price_at_touch_and_inside_wide_spread():
    rule = TradingRule(pair="BTC/USD", price_precision=2, amount_precision=6, mini_order=D("1"))
    tight = Ticker(MaxBid=100.00, MinAsk=100.01, LastPrice=100.0)
    wide = Ticker(MaxBid=100.00, MinAsk=100.10, LastPrice=100.0)
    assert maker_price("BUY", tight, rule) == D("100.00")
    assert maker_price("SELL", tight, rule) == D("100.01")
    assert maker_price("BUY", wide, rule) == D("100.01")  # one tick inside, still below the ask
    assert maker_price("SELL", wide, rule) == D("100.09")


def test_match_unknown_order_by_side_qty_and_time():
    sent = 1_780_000_000_000
    cands = [
        OrderDetail(Pair="BTC/USD", OrderID=1, Side="SELL", Quantity=0.5, CreateTimestamp=sent),
        OrderDetail(Pair="BTC/USD", OrderID=2, Side="BUY", Quantity=0.5, CreateTimestamp=sent - 500_000),
        OrderDetail(Pair="BTC/USD", OrderID=3, Side="BUY", Quantity=0.5, CreateTimestamp=sent + 1_000),
    ]
    assert match_unknown_order("BTC/USD", "BUY", D("0.5"), sent, cands).order_id == 3
    assert match_unknown_order("BTC/USD", "BUY", D("0.7"), sent, cands) is None


class FakeBroker:
    """Scriptable broker: limits rest unfilled, markets fill, queries return the scripted status."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.status: dict[str, OrderStatus] = {}
        self.raise_unknown = False
        self.recent: list[OrderDetail] = []
        self._n = 0

    def get_tickers(self):
        return {"BTC/USD": Ticker(MaxBid=50000, MinAsk=50001, LastPrice=50000)}

    def place_limit(self, pair, side, qty, price):
        self.calls.append(("LIMIT", pair, side, qty, price))
        if self.raise_unknown:
            raise OrderStateUnknownError("/v3/place_order")
        self._n += 1
        oid = str(self._n)
        self.status[oid] = OrderStatus(
            order_id=oid,
            status="PENDING",
            role="MAKER",
            filled_qty=D(0),
            avg_price=D(0),
            fee=D(0),
            qty=qty,
            pair=pair,
            side=side,
        )
        return OrderAck(
            order_id=oid, status="PENDING", role="MAKER", filled_qty=D(0), avg_price=D(0), fee=D(0)
        )

    def place_market(self, pair, side, qty):
        self.calls.append(("MARKET", pair, side, qty))
        return OrderAck(
            order_id="m", status="FILLED", role="TAKER", filled_qty=qty, avg_price=D(50000), fee=qty * 50
        )

    def cancel(self, order_id):
        self.calls.append(("CANCEL", order_id))

    def query(self, order_id):
        return self.status[order_id]

    def recent_orders(self, pair, limit=20):
        return self.recent


def _executor(broker, clock, orders, fills):
    return OrderExecutor(
        broker=broker,
        exec_cfg=CFG,
        clock=clock,
        rules={},
        on_order=orders.append,
        on_fill=lambda *a: fills.append(a),
        sleep=lambda s: None,
    )


def test_executor_reprices_then_escalates_to_market_after_max_tries():
    clock = SimClock(NOW)
    broker, orders, fills = FakeBroker(), [], []
    ex = _executor(broker, clock, orders, fills)
    ex.execute([PlannedOrder("BTC/USD", "BUY", False, False, qty=D("0.1"), notional=D("5000"))], "c1")
    for _ in range(CFG.limit_max_tries):
        clock.advance(timedelta(seconds=CFG.limit_reprice_s))
        ex.service()
    kinds = [c[0] for c in broker.calls]
    assert kinds.count("LIMIT") == CFG.limit_max_tries
    assert kinds[-1] == "MARKET" and not ex.working
    assert fills and fills[-1][1] == "BUY"


def test_executor_partial_fill_reprices_only_the_residual():
    clock = SimClock(NOW)
    broker, orders, fills = FakeBroker(), [], []
    ex = _executor(broker, clock, orders, fills)
    ex.execute([PlannedOrder("BTC/USD", "BUY", False, False, qty=D("1.0"), notional=D("50000"))], "c1")
    broker.status["1"] = broker.status["1"].model_copy(update={"filled_qty": D("0.4"), "status": "PARTIAL"})
    clock.advance(timedelta(seconds=CFG.limit_reprice_s))
    ex.service()
    assert broker.calls[-1][0] == "LIMIT" and broker.calls[-1][3] == D("0.6")
    assert fills[0][2] == D("0.4")  # the partial fill was reported once


def test_ambiguous_send_is_reconciled_not_resubmitted():
    clock = SimClock(NOW)
    broker, orders, fills = FakeBroker(), [], []
    broker.raise_unknown = True
    ex = _executor(broker, clock, orders, fills)
    ex.execute([PlannedOrder("BTC/USD", "BUY", False, False, qty=D("0.1"), notional=D("5000"))], "c1")
    assert [c[0] for c in broker.calls] == ["LIMIT"]  # exactly one send, never retried
    assert orders[0]["status"] == "UNKNOWN"
