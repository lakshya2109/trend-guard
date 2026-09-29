"""SimBroker: simulated fills for both paper mode and backtests (BUILD_SPEC.md Section 8.1, 11.2).

SimBroker implements the same `Broker` protocol as LiveBroker but never touches the network.
It is driven in one of two mutually exclusive modes, chosen at construction via `mode`:

  * mode="backtest": price only moves when `advance_bar(pair, bar)` is called. `submit_limit`
    and `submit_market` never fill synchronously -- they QUEUE a `PendingOrder` against `pair`
    and return an OrderAck with status "PENDING" and zero fill. The queued order is resolved
    *in full* (no partial fills -- hourly bars give no finer granularity to model a partial) the
    next time `advance_bar` is called for that pair, using BUILD_SPEC Section 11.2's rules:

      - queued LIMIT BUY: fills at `order.price` (maker_fee) if `bar.low <= order.price`;
        otherwise fills at `bar.close * (1 + slippage)` (taker_fee).
      - queued LIMIT SELL: fills at `order.price` (maker_fee) if `bar.high >= order.price`;
        otherwise fills at `bar.close * (1 - slippage)` (taker_fee).
      - queued MARKET (submit_market; used for risk-reducing protective/overlay exits): fills at
        `bar.open * (1 + slippage)` for BUY / `bar.open * (1 - slippage)` for SELL, taker_fee.

    This means an order decided from bar t's close is never filled using information from bar t
    itself (no look-ahead): the caller submits at bar t, then calls advance_bar with bar t+1,
    which is the earliest point the fill is resolved. `advance_bar` returns the list of OrderAcks
    that resolved on that call (empty if nothing was queued for `pair`).

  * mode="paper": price only moves when `mark_to_market(tickers)` is called (fed by
    RoostooClient.get_ticker() polling -- there is no bar t+1 to wait for). `submit_limit`
    queues a resting order exactly like backtest mode, but it is checked for a fill on every
    subsequent `mark_to_market` call rather than on a fixed next-bar boundary: a BUY fills
    (maker_fee) once `ticker.last_price <= order.price`, a SELL fills once
    `ticker.last_price >= order.price`. Reprice/retry/escalate-to-market timing (limit_reprice_s,
    limit_max_tries) is entirely the order_manager's job, identical to live mode -- SimBroker only
    ever resolves what it is asked to resolve. `submit_market` fills immediately against the last
    mark (BUY at min_ask*(1+slippage), SELL at max_bid*(1-slippage), taker_fee), since paper mode
    has real ticks and no reason to defer a market order. `advance_bar` is unused in this mode.

Shorts (`short_open`/`short_close`) are market-only in both modes and always resolve immediately
against the current mark price, charging `backtest_cfg.short_fee` on the collateral amount on
*both* open and close (BUILD_SPEC Section 1: "flat 0.10% on open and close regardless of order
type").
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from trend_guard.clock import Clock
from trend_guard.config import BacktestConfig
from trend_guard.execution.broker import (
    Exposure,
    OrderAck,
    OrderStatus,
    Position,
    Role,
    ShortAck,
    ShortCloseAck,
    Side,
)
from trend_guard.roostoo.models import Ticker


class OHLCBar(BaseModel):
    """One hourly bar fed into advance_bar(). `ts` is the bar's own close/label timestamp."""

    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


class PendingOrder(BaseModel):
    """A resting order tracked internally by SimBroker. Returned to callers only via query()."""

    order_id: str
    pair: str
    side: Side
    kind: Literal["LIMIT", "MARKET"]
    qty: Decimal
    price: Decimal | None = None  # None for MARKET orders
    placed_ts_ms: int


class _ShortBookEntry(BaseModel):
    qty: Decimal
    collateral: Decimal
    entry_price: Decimal


def _slippage_frac(cfg: BacktestConfig) -> Decimal:
    return Decimal(str(cfg.slippage_bps)) / Decimal(10000)


class SimBroker:
    def __init__(
        self,
        starting_cash: Decimal,
        backtest_cfg: BacktestConfig,
        clock: Clock,
        mode: Literal["backtest", "paper"] = "backtest",
    ):
        self.cash = starting_cash
        self.cfg = backtest_cfg
        self.clock = clock
        self.mode = mode

        self.longs: dict[str, Decimal] = {}  # pair -> qty held
        self.long_cost: dict[str, Decimal] = {}  # pair -> running avg entry price
        self.shorts: dict[str, _ShortBookEntry] = {}  # pair -> short book entry
        self.last_price: dict[str, Decimal] = {}  # mark price: last bar close, or last ticker mid
        self._tickers: dict[str, Ticker] = {}  # paper mode only, for bid/ask-aware fills

        self._pending: dict[str, list[PendingOrder]] = {}  # pair -> resting orders (FIFO)
        self._orders: dict[str, OrderStatus] = {}  # order_id -> latest known status
        self._seq = 0

    # --- helpers -----------------------------------------------------------------
    def _next_id(self) -> str:
        self._seq += 1
        return f"SIM-{self._seq}"

    def _maker_fee(self) -> Decimal:
        return Decimal(str(self.cfg.maker_fee))

    def _taker_fee(self) -> Decimal:
        return Decimal(str(self.cfg.taker_fee))

    def _short_fee(self) -> Decimal:
        return Decimal(str(self.cfg.short_fee))

    def _apply_fill_to_book(self, pair: str, side: Side, qty: Decimal, price: Decimal) -> None:
        """Update cash and the long inventory book for a spot fill (not shorts)."""
        if side == "BUY":
            self.longs[pair] = self.longs.get(pair, Decimal("0")) + qty
            prev_qty = self.longs[pair] - qty
            prev_cost = self.long_cost.get(pair, price)
            # weighted-average cost basis
            total_cost = prev_cost * prev_qty + price * qty
            self.long_cost[pair] = total_cost / self.longs[pair] if self.longs[pair] > 0 else price
            self.cash -= qty * price
        else:  # SELL
            self.longs[pair] = self.longs.get(pair, Decimal("0")) - qty
            if self.longs[pair] <= 0:
                self.longs.pop(pair, None)
                self.long_cost.pop(pair, None)
            self.cash += qty * price

    def _record_fill(
        self, order_id: str, pair: str, side: Side, qty: Decimal, price: Decimal, fee: Decimal, role: Role
    ) -> OrderAck:
        self.cash -= fee
        status = OrderStatus(
            order_id=order_id,
            status="FILLED",
            role=role,
            filled_qty=qty,
            avg_price=price,
            fee=fee,
            qty=qty,
            pair=pair,
            side=side,
        )
        self._orders[order_id] = status
        return OrderAck(
            order_id=order_id, status="FILLED", role=role, filled_qty=qty, avg_price=price, fee=fee
        )

    # --- Broker protocol: valuation -----------------------------------------------------------------
    def get_equity(self) -> Decimal:
        longs_value = sum(
            (qty * self.last_price.get(pair, Decimal("0")) for pair, qty in self.longs.items()),
            Decimal("0"),
        )
        shorts_value = sum((self._short_position_value(pair) for pair in self.shorts), Decimal("0"))
        return self.cash + longs_value + shorts_value

    def _short_position_value(self, pair: str) -> Decimal:
        entry = self.shorts[pair]
        mark = self.last_price.get(pair, entry.entry_price)
        unrealized = entry.qty * (entry.entry_price - mark)  # profit when price falls
        return max(Decimal("0"), entry.collateral + unrealized)  # loss capped at collateral

    def get_positions(self) -> dict[str, Position]:
        positions: dict[str, Position] = {}
        for pair, qty in self.longs.items():
            if qty <= 0:
                continue
            positions[pair] = Position(
                pair=pair, signed_qty=qty, avg_price=self.long_cost.get(pair, Decimal("0")), side="LONG"
            )
        for pair, entry in self.shorts.items():
            if entry.qty <= 0:
                continue
            existing = positions.get(pair)
            if existing is None:
                positions[pair] = Position(
                    pair=pair, signed_qty=-entry.qty, avg_price=entry.entry_price, side="SHORT"
                )
            else:
                net_qty = existing.signed_qty - entry.qty
                side: str = "SHORT" if net_qty < 0 else "LONG"
                price = entry.entry_price if net_qty < 0 else existing.avg_price
                positions[pair] = Position(pair=pair, signed_qty=net_qty, avg_price=price, side=side)  # type: ignore[arg-type]
        return positions

    def get_exposure(self) -> Exposure:
        return Exposure(
            equity=self.get_equity(),
            cash=self.cash,
            prices=dict(self.last_price),
            long_qty={p: q for p, q in self.longs.items() if q > 0},
            short_collateral={p: e.collateral for p, e in self.shorts.items()},
            short_value={p: self._short_position_value(p) for p in self.shorts},
        )

    def fill_now(self, pair: str, side: Side, qty: Decimal, price: Decimal) -> OrderAck:
        """Backtest only: immediate taker fill at an explicit price (protective stops evaluated on a
        bar's high/low fill at the stop level, or the open if the bar gapped through it)."""
        slip = _slippage_frac(self.cfg)
        px = price * (1 + slip) if side == "BUY" else price * (1 - slip)
        fee = qty * px * self._taker_fee()
        self._apply_fill_to_book(pair, side, qty, px)
        return self._record_fill(self._next_id(), pair, side, qty, px, fee, "TAKER")

    def get_tickers(self) -> dict[str, Ticker]:
        if self.mode == "paper":
            return dict(self._tickers)
        # backtest: synthesize a zero-spread ticker from the last bar close (documented in the
        # module docstring -- order_manager's spread-aware touch pricing degrades to a single
        # price in backtest, which is fine since backtest limit prices are computed by the
        # caller from half_spread, not from a simulated order book).
        return {
            pair: Ticker(max_bid=float(p), min_ask=float(p), last_price=float(p))
            for pair, p in self.last_price.items()
        }

    # --- Broker protocol: orders -----------------------------------------------------------------
    def place_limit(self, pair: str, side: str, qty: Decimal, price: Decimal) -> OrderAck:
        return self.submit_limit(pair, side, qty, price)  # type: ignore[arg-type]

    def place_market(self, pair: str, side: str, qty: Decimal) -> OrderAck:
        return self.submit_market(pair, side, qty)  # type: ignore[arg-type]

    def submit_limit(self, pair: str, side: Side, qty: Decimal, price: Decimal) -> OrderAck:
        """Queue (backtest) or rest (paper) a LIMIT order. Never fills synchronously."""
        order_id = self._next_id()
        order = PendingOrder(
            order_id=order_id,
            pair=pair,
            side=side,
            kind="LIMIT",
            qty=qty,
            price=price,
            placed_ts_ms=self.clock.now_ms(),
        )
        self._pending.setdefault(pair, []).append(order)
        status = OrderStatus(
            order_id=order_id,
            status="PENDING",
            role="MAKER",
            filled_qty=Decimal("0"),
            avg_price=Decimal("0"),
            fee=Decimal("0"),
            qty=qty,
            pair=pair,
            side=side,
        )
        self._orders[order_id] = status
        if self.mode == "paper":
            self._try_fill_resting(pair)  # in case the current mark already crosses it
        return OrderAck(
            order_id=order_id,
            status="PENDING",
            role="MAKER",
            filled_qty=Decimal("0"),
            avg_price=Decimal("0"),
            fee=Decimal("0"),
        )

    def submit_market(self, pair: str, side: Side, qty: Decimal) -> OrderAck:
        """Backtest: queue, resolved next advance_bar() at bar.open + slippage. Paper: fill now."""
        if self.mode == "backtest":
            order_id = self._next_id()
            order = PendingOrder(
                order_id=order_id,
                pair=pair,
                side=side,
                kind="MARKET",
                qty=qty,
                price=None,
                placed_ts_ms=self.clock.now_ms(),
            )
            self._pending.setdefault(pair, []).append(order)
            status = OrderStatus(
                order_id=order_id,
                status="PENDING",
                role="TAKER",
                filled_qty=Decimal("0"),
                avg_price=Decimal("0"),
                fee=Decimal("0"),
                qty=qty,
                pair=pair,
                side=side,
            )
            self._orders[order_id] = status
            return OrderAck(
                order_id=order_id,
                status="PENDING",
                role="TAKER",
                filled_qty=Decimal("0"),
                avg_price=Decimal("0"),
                fee=Decimal("0"),
            )
        # paper: fill immediately against the last mark, taker side of the book, with slippage.
        t = self._tickers.get(pair)
        slip = _slippage_frac(self.cfg)
        if t is not None:
            ref = Decimal(str(t.min_ask)) if side == "BUY" else Decimal(str(t.max_bid))
        else:
            ref = self.last_price.get(pair, Decimal("0"))
        price = ref * (1 + slip) if side == "BUY" else ref * (1 - slip)
        fee = qty * price * self._taker_fee()
        order_id = self._next_id()
        self._apply_fill_to_book(pair, side, qty, price)
        return self._record_fill(order_id, pair, side, qty, price, fee, "TAKER")

    def cancel(self, order_id: str) -> None:
        for _pair, orders in self._pending.items():
            for o in list(orders):
                if o.order_id == order_id:
                    orders.remove(o)
                    st = self._orders.get(order_id)
                    if st is not None:
                        self._orders[order_id] = st.model_copy(update={"status": "CANCELED"})
                    return
        # already filled or unknown: no-op, mirrors cancelling a non-pending order on the exchange

    def query(self, order_id: str) -> OrderStatus:
        if order_id not in self._orders:
            return OrderStatus(
                order_id=order_id,
                status="UNKNOWN",
                role="",
                filled_qty=Decimal("0"),
                avg_price=Decimal("0"),
                fee=Decimal("0"),
                qty=Decimal("0"),
            )
        return self._orders[order_id]

    # --- shorts: market-only, short_fee both sides, no TradingRule rounding -----------------
    def short_open(self, pair: str, collateral: Decimal) -> ShortAck:
        mark = self.last_price.get(pair) or (
            Decimal(str(self._tickers[pair].mid)) if pair in self._tickers else None
        )
        if mark is None or mark <= 0:
            raise ValueError(f"no mark price available for {pair}; cannot short_open")
        mark = mark * (1 - _slippage_frac(self.cfg))  # opens at MaxBid on the exchange: pay slippage
        fee = collateral * self._short_fee()
        qty = collateral / mark
        existing = self.shorts.get(pair)
        if existing is None:
            self.shorts[pair] = _ShortBookEntry(qty=qty, collateral=collateral, entry_price=mark)
        else:
            new_qty = existing.qty + qty
            new_collateral = existing.collateral + collateral
            # size-weighted average entry price across the two opens
            new_entry = (existing.entry_price * existing.qty + mark * qty) / new_qty
            self.shorts[pair] = _ShortBookEntry(qty=new_qty, collateral=new_collateral, entry_price=new_entry)
        # collateral is locked out of free cash (it comes back via short_close's return_amount);
        # get_equity adds it back through the short's PositionValue, so equity is unchanged
        # apart from the fee
        self.cash -= collateral + fee
        entry = self.shorts[pair]
        return ShortAck(
            position_id=f"SIM-SHORT-{pair}",
            pair=pair,
            order_type="MARKET",
            entry_price=entry.entry_price,
            short_qty=entry.qty,
            collateral=entry.collateral,
            open_fee=fee,
            status="OPEN",
        )

    def short_close(self, pair: str, pct: Decimal) -> ShortCloseAck:
        entry = self.shorts.get(pair)
        if entry is None:
            raise ValueError(f"no open short on {pair} to close")
        mark = self.last_price.get(pair) or (
            Decimal(str(self._tickers[pair].mid)) if pair in self._tickers else entry.entry_price
        )
        mark = mark * (1 + _slippage_frac(self.cfg))  # closes at MinAsk on the exchange: pay slippage
        frac = min(pct, Decimal(100)) / Decimal(100)
        closed_qty = entry.qty * frac
        closed_collateral = entry.collateral * frac
        realized_pnl = closed_qty * (entry.entry_price - mark)
        fee = closed_qty * mark * self._short_fee()  # docs: ClosedQty * ClosePrice * 0.001
        return_amount = max(Decimal("0"), closed_collateral + realized_pnl) - fee
        self.cash += return_amount

        remaining_qty = entry.qty - closed_qty
        remaining_collateral = entry.collateral - closed_collateral
        fully_closed = pct >= Decimal(100) or remaining_qty <= 0
        if fully_closed:
            self.shorts.pop(pair, None)
            remaining_qty = Decimal("0")
            remaining_collateral = Decimal("0")
        else:
            self.shorts[pair] = _ShortBookEntry(
                qty=remaining_qty, collateral=remaining_collateral, entry_price=entry.entry_price
            )
        return ShortCloseAck(
            close_price=mark,
            realized_pnl=realized_pnl,
            close_fee=fee,
            return_amount=return_amount,
            closed_qty=closed_qty,
            fully_closed=fully_closed,
            remaining_qty=remaining_qty,
            remaining_collateral=remaining_collateral,
        )

    # --- driving interfaces -----------------------------------------------------------------
    def mark_to_market(self, tickers: dict[str, Ticker]) -> None:
        """Paper mode: feed a fresh RoostooClient.get_ticker() snapshot. Updates the mark price
        book and checks every resting limit order for a fill (BUY when last_price <= order.price,
        SELL when last_price >= order.price)."""
        self._tickers.update(tickers)
        for pair, t in tickers.items():
            self.last_price[pair] = Decimal(str(t.last_price)) if t.last_price else Decimal(str(t.mid))
        for pair in list(tickers):
            self._try_fill_resting(pair)

    def _try_fill_resting(self, pair: str) -> None:
        last = self.last_price.get(pair)
        if last is None:
            return
        still_open: list[PendingOrder] = []
        for o in self._pending.get(pair, []):
            if o.kind != "LIMIT" or o.price is None:
                still_open.append(o)
                continue
            touched = (o.side == "BUY" and last <= o.price) or (o.side == "SELL" and last >= o.price)
            if not touched:
                still_open.append(o)
                continue
            fee = o.qty * o.price * self._maker_fee()
            self._apply_fill_to_book(pair, o.side, o.qty, o.price)
            self._record_fill(o.order_id, pair, o.side, o.qty, o.price, fee, "MAKER")
        self._pending[pair] = still_open

    def advance_bar(self, pair: str, bar: OHLCBar) -> list[OrderAck]:
        """Backtest mode: advance simulated time by one bar for `pair`. Resolves every order
        queued for `pair` (see module docstring for the exact fill rules) and updates the mark
        price to `bar.close`. Returns the list of fills that happened on this call (empty if
        nothing was pending). No look-ahead: only `bar`'s own OHLC is used."""
        fills: list[OrderAck] = []
        slip = _slippage_frac(self.cfg)
        for o in self._pending.pop(pair, []):
            if o.kind == "LIMIT":
                assert o.price is not None
                if o.side == "BUY":
                    touched = bar.low <= o.price
                    fallback = bar.close * (1 + slip)
                else:
                    touched = bar.high >= o.price
                    fallback = bar.close * (1 - slip)
                price = o.price if touched else fallback
                fee_rate = self._maker_fee() if touched else self._taker_fee()
                role: Role = "MAKER" if touched else "TAKER"
            else:  # queued MARKET: risk-reducing exit, fills at next bar's open + slippage
                price = bar.open * (1 + slip) if o.side == "BUY" else bar.open * (1 - slip)
                fee_rate = self._taker_fee()
                role = "TAKER"
            fee = o.qty * price * fee_rate
            self._apply_fill_to_book(pair, o.side, o.qty, price)
            fills.append(self._record_fill(o.order_id, pair, o.side, o.qty, price, fee, role))
        self.last_price[pair] = bar.close
        return fills
