"""Broker protocol and the small models it speaks. BUILD_SPEC.md Section 8.1.

`LiveBroker` (live_broker.py) and `SimBroker` (sim_broker.py) both implement this Protocol so
the order manager and engine loop never need to know which one they are talking to. We reuse
`trend_guard.roostoo.models.Ticker` rather than redefining it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Protocol

from pydantic import BaseModel

from trend_guard.roostoo.models import Ticker

Side = Literal["BUY", "SELL"]
PositionSide = Literal["LONG", "SHORT"]
Role = Literal["MAKER", "TAKER", ""]
# NEW: created locally, not yet sent. PENDING: sent, resting on the book. FILLED/PARTIAL/CANCELED:
# terminal or partially-terminal exchange states. UNKNOWN: OrderStateUnknownError was raised and
# reconciliation has not resolved it yet. REJECTED: exchange refused the order outright.
OrderStatusValue = Literal["NEW", "PENDING", "FILLED", "PARTIAL", "CANCELED", "UNKNOWN", "REJECTED"]


class Position(BaseModel):
    """A single pair's net exposure. Positive signed_qty is long, negative is short."""

    pair: str
    signed_qty: Decimal
    avg_price: Decimal
    side: PositionSide


class OrderAck(BaseModel):
    """Immediate response to place_limit/place_market."""

    order_id: str
    status: OrderStatusValue
    role: Role
    filled_qty: Decimal
    avg_price: Decimal
    fee: Decimal


class OrderStatus(BaseModel):
    """Response to query(order_id). Same shape as OrderAck plus enough to tell fill states apart."""

    order_id: str
    status: OrderStatusValue
    role: Role
    filled_qty: Decimal
    avg_price: Decimal
    fee: Decimal
    qty: Decimal  # originally requested quantity, so the caller can derive remaining_qty
    pair: str = ""
    side: Side | None = None


class ShortAck(BaseModel):
    """Mirrors roostoo.models.ShortOpenResult."""

    position_id: str
    pair: str
    order_type: str
    entry_price: Decimal
    short_qty: Decimal
    collateral: Decimal
    open_fee: Decimal
    status: str


class ShortCloseAck(BaseModel):
    """Mirrors roostoo.models.ShortCloseResult."""

    close_price: Decimal
    realized_pnl: Decimal
    close_fee: Decimal
    return_amount: Decimal
    closed_qty: Decimal
    fully_closed: bool
    remaining_qty: Decimal
    remaining_collateral: Decimal


class Exposure(BaseModel):
    """One consistent snapshot of the book, kept per instrument rather than netted.

    A spot long and a v6 short on the same pair are different instruments (different fees, close
    paths and collateral rules), so the order manager needs them separately.
    """

    equity: Decimal  # cash + longs at mid + short PositionValue
    cash: Decimal  # free USD available for new buys / short collateral
    prices: dict[str, Decimal]  # pair -> mid used for valuation
    long_qty: dict[str, Decimal]  # pair -> coin quantity held (free + lock)
    short_collateral: dict[str, Decimal]  # pair -> USD collateral locked in the short
    short_value: dict[str, Decimal]  # pair -> PositionValue (collateral + unrealised PnL)

    def long_notional(self, pair: str) -> Decimal:
        return self.long_qty.get(pair, Decimal("0")) * self.prices.get(pair, Decimal("0"))

    def gross_used(self) -> Decimal:
        """Long notional plus short collateral: must never exceed equity (no leverage)."""
        longs = sum((self.long_notional(p) for p in self.long_qty), Decimal("0"))
        return longs + sum(self.short_collateral.values(), Decimal("0"))


class Broker(Protocol):
    def get_exposure(self) -> Exposure: ...  # single-pass snapshot used by the order manager

    def get_equity(self) -> Decimal: ...  # USD free+lock + longs at mid + short PositionValue

    def get_positions(self) -> dict[str, Position]: ...  # pair -> signed qty, avg price, side

    def get_tickers(self) -> dict[str, Ticker]: ...

    def place_limit(self, pair: str, side: str, qty: Decimal, price: Decimal) -> OrderAck: ...

    def place_market(self, pair: str, side: str, qty: Decimal) -> OrderAck: ...

    def cancel(self, order_id: str) -> None: ...

    def query(self, order_id: str) -> OrderStatus: ...

    def short_open(self, pair: str, collateral: Decimal) -> ShortAck: ...

    def short_close(self, pair: str, pct: Decimal) -> ShortCloseAck: ...
