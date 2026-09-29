"""SimBroker fill model (Section 11.2) and short accounting."""

from datetime import UTC, datetime
from decimal import Decimal as D

import pytest

from trend_guard.clock import SimClock
from trend_guard.config import BacktestConfig
from trend_guard.execution.sim_broker import OHLCBar, SimBroker

T = datetime(2026, 1, 1, tzinfo=UTC)
CFG = BacktestConfig(slippage_bps=0)  # zero slippage keeps the arithmetic exact


def bar(o, h, low, c) -> OHLCBar:
    return OHLCBar(ts=T, open=D(o), high=D(h), low=D(low), close=D(c))


def broker() -> SimBroker:
    b = SimBroker(D("100000"), CFG, SimClock(T))
    b.advance_bar("BTC/USD", bar(100, 100, 100, 100))  # sets the mark
    return b


def test_maker_limit_fills_at_limit_with_maker_fee_when_touched():
    b = broker()
    b.submit_limit("BTC/USD", "BUY", D("10"), D("99"))
    fills = b.advance_bar("BTC/USD", bar(100, 101, 98, 100))
    assert fills[0].role == "MAKER" and fills[0].avg_price == D("99")
    assert fills[0].fee == D("10") * D("99") * D(str(CFG.maker_fee))


def test_untouched_limit_falls_back_to_taker_at_next_close():
    b = broker()
    b.submit_limit("BTC/USD", "BUY", D("10"), D("95"))
    fills = b.advance_bar("BTC/USD", bar(100, 103, 99, 102))
    assert fills[0].role == "TAKER" and fills[0].avg_price == D("102")


def test_short_locks_collateral_so_equity_only_drops_by_the_fee():
    """Regression: collateral must leave free cash, or equity double counts it."""
    b = broker()
    ack = b.short_open("BTC/USD", D("10000"))
    assert b.cash == D("100000") - D("10000") - ack.open_fee
    assert b.get_equity() == D("100000") - ack.open_fee


def test_short_pnl_and_close_fee_on_closed_value():
    b = broker()
    b.short_open("BTC/USD", D("10000"))  # 100 units at 100
    b.advance_bar("BTC/USD", bar(90, 90, 90, 90))
    close = b.short_close("BTC/USD", D("100"))
    assert close.realized_pnl == D("1000")  # 100 * (100 - 90)
    assert close.close_fee == close.closed_qty * D("90") * D(str(CFG.short_fee))  # ClosedQty*ClosePrice*0.1%
    assert close.fully_closed and "BTC/USD" not in b.shorts


def test_short_loss_capped_at_collateral():
    b = broker()
    b.short_open("BTC/USD", D("10000"))
    b.advance_bar("BTC/USD", bar(300, 300, 300, 300))  # price triples
    assert b.get_exposure().short_value["BTC/USD"] == D("0")


def test_close_without_short_raises():
    with pytest.raises(ValueError):
        broker().short_close("ETH/USD", D("100"))
