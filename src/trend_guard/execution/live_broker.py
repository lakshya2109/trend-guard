"""LiveBroker: wraps a RoostooClient + TradingRules as a Broker (BUILD_SPEC.md Section 8.1).

Precision handling: place_limit/place_market always round price/qty through the pair's
TradingRule *before* calling client.place_order(...). An order that fails TradingRule.meets_min
after rounding is never sent: we raise ValueError with a clear message instead (documented
sentinel choice -- callers, i.e. order_manager, are expected to catch ValueError and skip/log the
dropped order rather than propagate it as a hard failure).

ShortsDisabledError and OrderStateUnknownError are raised by the underlying RoostooClient and are
left to propagate uncaught here: the caller (order_manager) is responsible for handling both, per
BUILD_SPEC Section 8.3.

Known limitation: Roostoo's /v3/balance endpoint does not report a cost basis for spot holdings,
so get_positions() reports the current ticker mid as avg_price for long positions rather than a
true entry price. Short positions do have a real entry_price from /v6/short_positions and use it.
"""

from __future__ import annotations

from decimal import Decimal

from trend_guard.execution.broker import (
    Exposure,
    OrderAck,
    OrderStatus,
    OrderStatusValue,
    Position,
    Role,
    ShortAck,
    ShortCloseAck,
)
from trend_guard.roostoo.client import RoostooClient
from trend_guard.roostoo.models import OrderDetail, ShortCloseResult, ShortOpenResult, Ticker, WalletEntry
from trend_guard.roostoo.rules import TradingRule

# Roostoo's raw OrderDetail.Status strings, normalised to our OrderStatusValue. Unrecognised raw
# values map to UNKNOWN rather than raising, so a wording change upstream degrades gracefully.
_STATUS_MAP: dict[str, OrderStatusValue] = {
    "NEW": "NEW",
    "PENDING": "PENDING",
    "OPEN": "PENDING",
    "FILLED": "FILLED",
    "FULLY_FILLED": "FILLED",
    "PART_FILLED": "PARTIAL",
    "PARTIALLY_FILLED": "PARTIAL",
    "PARTIAL": "PARTIAL",
    "CANCELED": "CANCELED",
    "CANCELLED": "CANCELED",
    "REJECTED": "REJECTED",
}


def _normalise_status(raw: str) -> OrderStatusValue:
    return _STATUS_MAP.get(raw.upper(), "UNKNOWN")


def _normalise_role(raw: str) -> Role:
    r = raw.upper()
    if r in ("MAKER", "TAKER"):
        return r  # type: ignore[return-value]
    return ""


def _order_ack_from_detail(detail: OrderDetail) -> OrderAck:
    return OrderAck(
        order_id=str(detail.order_id),
        status=_normalise_status(detail.status),
        role=_normalise_role(detail.role),
        filled_qty=Decimal(str(detail.filled_quantity)),
        avg_price=Decimal(str(detail.filled_aver_price)),
        fee=Decimal(str(detail.commission_charge_value)),
    )


def _order_status_from_detail(detail: OrderDetail) -> OrderStatus:
    return OrderStatus(
        order_id=str(detail.order_id),
        status=_normalise_status(detail.status),
        role=_normalise_role(detail.role),
        filled_qty=Decimal(str(detail.filled_quantity)),
        avg_price=Decimal(str(detail.filled_aver_price)),
        fee=Decimal(str(detail.commission_charge_value)),
        qty=Decimal(str(detail.quantity)),
        pair=detail.pair,
        side=detail.side if detail.side in ("BUY", "SELL") else None,
    )


def _short_ack_from_result(result: ShortOpenResult) -> ShortAck:
    return ShortAck(
        position_id=str(result.id),
        pair=result.pair,
        order_type=result.order_type,
        entry_price=Decimal(str(result.entry_price)),
        short_qty=Decimal(str(result.short_qty)),
        collateral=Decimal(str(result.collateral)),
        open_fee=Decimal(str(result.open_fee)),
        status=result.status,
    )


def _short_close_ack_from_result(result: ShortCloseResult) -> ShortCloseAck:
    return ShortCloseAck(
        close_price=Decimal(str(result.close_price)),
        realized_pnl=Decimal(str(result.realized_pnl)),
        close_fee=Decimal(str(result.close_fee)),
        return_amount=Decimal(str(result.return_amount)),
        closed_qty=Decimal(str(result.closed_qty)),
        fully_closed=result.fully_closed,
        remaining_qty=Decimal(str(result.remaining_qty)),
        remaining_collateral=Decimal(str(result.remaining_collateral)),
    )


class LiveBroker:
    """Broker implementation backed by a real RoostooClient. See module docstring."""

    def __init__(self, client: RoostooClient, rules: dict[str, TradingRule], allow_live_orders: bool = False):
        self.client = client
        self.rules = rules
        # second, code-level gate (the CLI checks ALLOW_LIVE_ORDERS too): every order-creating
        # method refuses unless this is True, so no code path can place a live order by accident
        self.allow_live_orders = allow_live_orders

    def _require_live(self) -> None:
        if not self.allow_live_orders:
            raise PermissionError("live orders disabled: set ALLOW_LIVE_ORDERS=1 to trade")

    def _rule(self, pair: str) -> TradingRule:
        try:
            return self.rules[pair]
        except KeyError:
            raise ValueError(f"no TradingRule loaded for {pair}") from None

    # --- valuation -----------------------------------------------------------------
    @staticmethod
    def _locked_usd(usd: WalletEntry | None) -> Decimal:
        """USD locked in pending orders only. Roostoo's USD Lock also holds short collateral, which is
        already counted in each short's PositionValue; adding it here would count it twice."""
        if usd is None:
            return Decimal("0")
        return max(Decimal(str(usd.lock)) - Decimal(str(usd.short_collateral)), Decimal("0"))

    def get_equity(self) -> Decimal:
        """USD free + pending-order USD + long spot holdings at ticker mid + short PositionValue."""
        balance = self.client.get_balance()
        usd = balance.get("USD")
        cash = Decimal(str(usd.free)) + self._locked_usd(usd) if usd else Decimal("0")

        tickers = self.client.get_ticker()
        longs_value = Decimal("0")
        for coin, entry in balance.items():
            if coin == "USD":
                continue
            qty = Decimal(str(entry.free)) + Decimal(str(entry.lock))
            if qty <= 0:
                continue
            t = tickers.get(f"{coin}/USD")
            if t is not None:
                longs_value += qty * Decimal(str(t.mid))

        shorts_value = sum(
            (Decimal(str(sp.position_value)) for sp in self.client.short_positions()), Decimal("0")
        )
        return cash + longs_value + shorts_value

    def get_positions(self) -> dict[str, Position]:
        balance = self.client.get_balance()
        tickers = self.client.get_ticker()
        positions: dict[str, Position] = {}
        for coin, entry in balance.items():
            if coin == "USD":
                continue
            qty = Decimal(str(entry.free)) + Decimal(str(entry.lock))
            if qty <= 0:
                continue
            pair = f"{coin}/USD"
            t = tickers.get(pair)
            mid = Decimal(str(t.mid)) if t is not None else Decimal("0")
            positions[pair] = Position(pair=pair, signed_qty=qty, avg_price=mid, side="LONG")

        for sp in self.client.short_positions():
            qty = Decimal(str(sp.short_qty))
            if qty <= 0:
                continue
            pair = sp.pair
            entry_price = Decimal(str(sp.entry_price))
            existing = positions.get(pair)
            if existing is None:
                positions[pair] = Position(pair=pair, signed_qty=-qty, avg_price=entry_price, side="SHORT")
            else:
                # Rare edge case: simultaneous spot long + short margin position on the same pair.
                # Net them so callers see a single signed exposure per pair.
                net_qty = existing.signed_qty - qty
                side: str = "SHORT" if net_qty < 0 else "LONG"
                price = entry_price if net_qty < 0 else existing.avg_price
                positions[pair] = Position(pair=pair, signed_qty=net_qty, avg_price=price, side=side)  # type: ignore[arg-type]
        return positions

    def get_tickers(self) -> dict[str, Ticker]:
        return self.client.get_ticker()

    def get_exposure(self) -> Exposure:
        """One balance + one ticker + one short_positions call (3 requests) for a consistent book."""
        balance = self.client.get_balance()
        tickers = self.client.get_ticker()
        shorts = self.client.short_positions()
        usd = balance.get("USD")
        free_usd = Decimal(str(usd.free)) if usd else Decimal("0")
        locked_usd = self._locked_usd(usd)
        prices = {p: Decimal(str(t.mid)) for p, t in tickers.items() if t.mid}
        long_qty: dict[str, Decimal] = {}
        for coin, entry in balance.items():
            qty = Decimal(str(entry.free)) + Decimal(str(entry.lock))
            if coin != "USD" and qty > 0:
                long_qty[f"{coin}/USD"] = qty
        short_collateral = {sp.pair: Decimal(str(sp.collateral)) for sp in shorts if sp.short_qty > 0}
        short_value = {sp.pair: Decimal(str(sp.position_value)) for sp in shorts if sp.short_qty > 0}
        longs_value = sum((q * prices.get(p, Decimal("0")) for p, q in long_qty.items()), Decimal("0"))
        equity = free_usd + locked_usd + longs_value + sum(short_value.values(), Decimal("0"))
        return Exposure(
            equity=equity,
            cash=free_usd,
            prices=prices,
            long_qty=long_qty,
            short_collateral=short_collateral,
            short_value=short_value,
        )

    def recent_orders(self, pair: str, limit: int = 20) -> list[OrderDetail]:
        """Raw recent orders for a pair, used only by ambiguous-order reconciliation (Section 8.3)."""
        return self.client.query_order(pair=pair, pending_only=False, limit=limit)

    # --- orders -----------------------------------------------------------------
    def place_limit(self, pair: str, side: str, qty: Decimal, price: Decimal) -> OrderAck:
        self._require_live()
        rule = self._rule(pair)
        q = rule.clamp_qty(qty)
        p = rule.clamp_price(price)
        if not rule.meets_min(p, q):
            raise ValueError(f"{pair} {side} LIMIT qty={q} price={p} fails MiniOrder={rule.mini_order}")
        detail = self.client.place_order(pair, side, "LIMIT", q, p)
        return _order_ack_from_detail(detail)

    def place_market(self, pair: str, side: str, qty: Decimal) -> OrderAck:
        self._require_live()
        rule = self._rule(pair)
        q = rule.clamp_qty(qty)
        # MARKET orders carry no price param; use the current mid purely as a MiniOrder pre-check.
        ticker = self.client.get_ticker(pair).get(pair)
        ref_price = Decimal(str(ticker.mid)) if ticker is not None else Decimal("0")
        if not rule.meets_min(ref_price, q):
            raise ValueError(f"{pair} {side} MARKET qty={q} fails MiniOrder={rule.mini_order}")
        detail = self.client.place_order(pair, side, "MARKET", q, None)
        return _order_ack_from_detail(detail)

    def cancel(self, order_id: str) -> None:
        self._require_live()
        self.client.cancel_order(order_id=order_id)

    def query(self, order_id: str) -> OrderStatus:
        matches = self.client.query_order(order_id=order_id)
        if not matches:
            return OrderStatus(
                order_id=order_id,
                status="UNKNOWN",
                role="",
                filled_qty=Decimal("0"),
                avg_price=Decimal("0"),
                fee=Decimal("0"),
                qty=Decimal("0"),
            )
        return _order_status_from_detail(matches[0])

    # --- shorts: collateral/qty are already USD-denominated, no TradingRule rounding needed ---
    def short_open(self, pair: str, collateral: Decimal) -> ShortAck:
        self._require_live()
        result = self.client.short_open(pair, collateral)
        return _short_ack_from_result(result)

    def short_close(self, pair: str, pct: Decimal) -> ShortCloseAck:
        self._require_live()
        result = self.client.short_close(pair, close_pct=pct)
        return _short_close_ack_from_result(result)
