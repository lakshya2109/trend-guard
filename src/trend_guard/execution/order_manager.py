"""Target weights -> orders. BUILD_SPEC.md Section 8.2-8.3.

Two layers:
  * `plan_orders` is PURE: given target weights and an Exposure snapshot it returns the ordered list
    of PlannedOrders. The backtest engine and the live engine call exactly this function, so the
    deadband, min-hold, sign-flip, no-leverage and MiniOrder rules are identical in both.
  * `OrderExecutor` (live/paper only) turns plans into broker calls with maker-first pricing,
    reprices resting limits every `limit_reprice_s`, escalates to MARKET after `limit_max_tries`,
    and reconciles ambiguous sends via `match_unknown_order` instead of ever resubmitting blindly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Literal

from trend_guard.clock import Clock
from trend_guard.config import ExecutionConfig
from trend_guard.execution.broker import Broker, Exposure
from trend_guard.roostoo.errors import OrderStateUnknownError, RoostooAPIError, ShortsDisabledError
from trend_guard.roostoo.models import OrderDetail, Ticker
from trend_guard.roostoo.rules import TradingRule

log = logging.getLogger("engine")

Action = Literal["BUY", "SELL", "SHORT_OPEN", "SHORT_CLOSE"]
ZERO = Decimal("0")
MIN_SHORT_COLLATERAL = Decimal("1")  # /v6/short_open rejects collateral below $1 (API docs)
CASH_BUFFER = Decimal("0.995")  # keep a sliver of cash so fees never push a buy over balance
UNKNOWN_MATCH_WINDOW_MS = 120_000  # Section 8.3: match an ambiguous send within 120 s


@dataclass
class PlannedOrder:
    pair: str
    action: Action
    reducing: bool  # True when it shrinks exposure (sells of longs, short closes)
    urgent: bool  # risk exit: straight to MARKET, bypasses min-hold
    qty: Decimal = ZERO  # coin qty for BUY/SELL
    collateral: Decimal = ZERO  # USD for SHORT_OPEN
    close_pct: Decimal = ZERO  # (0, 100] for SHORT_CLOSE
    notional: Decimal = ZERO  # USD size, for logging and cash budgeting
    reason: str = ""


def _held_too_briefly(pair: str, last_trade: dict[str, datetime], now: datetime, min_hold_h: int) -> bool:
    t = last_trade.get(pair)
    return t is not None and now - t < timedelta(hours=min_hold_h)


def _round_qty(pair: str, qty: Decimal, rules: dict[str, TradingRule] | None) -> Decimal:
    if rules is not None and pair in rules:
        return rules[pair].clamp_qty(qty)
    return qty


def _meets_min(pair: str, price: Decimal, qty: Decimal, rules: dict[str, TradingRule] | None) -> bool:
    if qty <= 0:
        return False
    if rules is not None and pair in rules:
        return rules[pair].meets_min(price, qty)
    return True


def plan_orders(
    target_w: dict[str, float],
    exp: Exposure,
    exec_cfg: ExecutionConfig,
    now: datetime,
    last_trade: dict[str, datetime],
    rules: dict[str, TradingRule] | None = None,
    urgent_pairs: frozenset[str] = frozenset(),
) -> list[PlannedOrder]:
    """Diff targets against the current book and return orders, reductions first.

    Rules (Section 8.2):
      - |target - current| < deadband * equity with matching signs -> skip.
      - A pair traded less than `min_hold_h` ago is left alone unless the move is urgent or takes the
        position fully to zero (exits are always allowed: holding a position the strategy no longer
        wants is itself a risk).
      - Sign flips close fully this cycle; the other side opens on a later cycle.
      - Additions are scaled so long notional + short collateral never exceeds cash on hand (no
        leverage): budget = free cash + proceeds of immediate reductions (short closes, urgent sells).
    """
    equity = exp.equity
    if equity <= 0:
        return []
    deadband = Decimal(str(exec_cfg.deadband)) * equity
    pairs = set(target_w) | set(exp.long_qty) | set(exp.short_collateral)
    reductions: list[PlannedOrder] = []
    additions: list[PlannedOrder] = []

    for pair in sorted(pairs):
        w = target_w.get(pair, 0.0)
        w = 0.0 if w != w else w  # NaN -> flat
        target = Decimal(str(w)) * equity
        price = exp.prices.get(pair, ZERO)
        long_qty = exp.long_qty.get(pair, ZERO)
        long_n = long_qty * price
        short_c = exp.short_collateral.get(pair, ZERO)
        urgent = pair in urgent_pairs
        recent = _held_too_briefly(pair, last_trade, now, exec_cfg.min_hold_h)

        if target >= 0:
            if short_c > 0:  # wrong side (or flat target): close the short fully first
                reductions.append(
                    PlannedOrder(
                        pair,
                        "SHORT_CLOSE",
                        True,
                        urgent,
                        close_pct=Decimal(100),
                        notional=short_c,
                        reason="short->flat/long",
                    )
                )
                continue
            diff = target - long_n
            if target > 0 and long_n > 0 and abs(diff) < deadband:
                continue  # inside the deadband, same sign
            if diff == 0 or price <= 0:
                continue
            if diff < 0:
                full_exit = target == 0
                if recent and not (urgent or full_exit):
                    continue
                qty = long_qty if full_exit else _round_qty(pair, -diff / price, rules)
                qty = _round_qty(pair, min(qty, long_qty), rules)
                if _meets_min(pair, price, qty, rules):
                    reductions.append(
                        PlannedOrder(
                            pair, "SELL", True, urgent, qty=qty, notional=qty * price, reason="reduce long"
                        )
                    )
            else:
                if recent and not urgent:
                    continue
                qty = _round_qty(pair, diff / price, rules)
                if _meets_min(pair, price, qty, rules):
                    additions.append(
                        PlannedOrder(
                            pair, "BUY", False, False, qty=qty, notional=qty * price, reason="add long"
                        )
                    )
        else:
            target_c = -target
            if long_qty > 0:  # wrong side: sell the long fully first
                if _meets_min(pair, price, long_qty, rules):
                    reductions.append(
                        PlannedOrder(
                            pair, "SELL", True, urgent, qty=long_qty, notional=long_n, reason="long->short"
                        )
                    )
                continue
            diff = target_c - short_c
            if short_c > 0 and abs(diff) < deadband:
                continue
            if diff < 0:
                if recent and not urgent:
                    continue
                pct = min(Decimal(100), (-diff / short_c * 100)).quantize(Decimal("0.01"), ROUND_DOWN)
                if pct > 0:
                    reductions.append(
                        PlannedOrder(
                            pair,
                            "SHORT_CLOSE",
                            True,
                            urgent,
                            close_pct=pct,
                            notional=-diff,
                            reason="reduce short",
                        )
                    )
            elif diff > 0:
                if recent and not urgent:
                    continue
                coll = diff.quantize(Decimal("0.01"), ROUND_DOWN)
                if coll >= MIN_SHORT_COLLATERAL:
                    additions.append(
                        PlannedOrder(
                            pair,
                            "SHORT_OPEN",
                            False,
                            False,
                            collateral=coll,
                            notional=coll,
                            reason="add short",
                        )
                    )

    # no-leverage budget: additions only spend cash we have now or will get back immediately
    immediate = sum((o.notional for o in reductions if o.action == "SHORT_CLOSE" or o.urgent), ZERO)
    budget = (exp.cash + immediate) * CASH_BUFFER
    wanted = sum((o.notional for o in additions), ZERO)
    if wanted > budget and wanted > 0:
        scale = max(budget, ZERO) / wanted
        scaled: list[PlannedOrder] = []
        for o in additions:
            price = exp.prices.get(o.pair, ZERO)
            if o.action == "BUY":
                o.qty = _round_qty(o.pair, o.qty * scale, rules)
                o.notional = o.qty * price
                if _meets_min(o.pair, price, o.qty, rules):
                    scaled.append(o)
            else:
                o.collateral = (o.collateral * scale).quantize(Decimal("0.01"), ROUND_DOWN)
                o.notional = o.collateral
                if o.collateral >= MIN_SHORT_COLLATERAL:
                    scaled.append(o)
        additions = scaled

    # short closes first (they return cash immediately), then sells, then additions
    reductions.sort(key=lambda o: (o.action != "SHORT_CLOSE", o.pair))
    return reductions + additions


def match_unknown_order(
    pair: str, side: str, qty: Decimal, sent_ms: int, candidates: list[OrderDetail]
) -> OrderDetail | None:
    """Section 8.3: find the order an ambiguous send may have created. Pure: the caller supplies the
    candidate list from query_order(pair=..., pending_only=FALSE, limit=20)."""
    for c in candidates:
        if c.pair != pair or c.side != side:
            continue
        if abs(Decimal(str(c.quantity)) - qty) > qty * Decimal("0.0001"):
            continue
        if abs(c.create_ts - sent_ms) <= UNKNOWN_MATCH_WINDOW_MS:
            return c
    return None


def maker_price(side: str, ticker: Ticker, rule: TradingRule | None) -> Decimal:
    """Touch price for a maker-first limit: BUY at MaxBid (one tick better if the spread is wider than
    2 ticks), SELL at MinAsk (one tick better likewise). Never crosses the spread."""
    bid, ask = Decimal(str(ticker.max_bid)), Decimal(str(ticker.min_ask))
    tick = Decimal(1).scaleb(-rule.price_precision) if rule is not None else ZERO
    wide = tick > 0 and (ask - bid) > 2 * tick
    if side == "BUY":
        px = bid + tick if wide else bid
    else:
        px = ask - tick if wide else ask
    return rule.clamp_price(px) if rule is not None else px


@dataclass
class WorkingOrder:
    plan: PlannedOrder
    order_id: str
    placed_at: datetime
    sent_qty: Decimal  # quantity of THIS order (the residual after a reprice, not the plan qty)
    reported_filled: Decimal  # fills already passed to on_fill for this order
    tries: int = 1


@dataclass
class OrderExecutor:
    """Live/paper execution of PlannedOrders (never used by the backtest, which fills on bars).

    `on_fill(pair, side, qty, price, fee, role, reason)` and `on_order(record: dict)` are callbacks
    into the state store / fill log so this class stays free of SQL.
    """

    broker: Broker
    exec_cfg: ExecutionConfig
    clock: Clock
    rules: dict[str, TradingRule]
    on_order: Callable[[dict], None]
    on_fill: Callable[[str, str, Decimal, Decimal, Decimal, str, str], None]
    sleep: Callable[[float], None] = time.sleep
    working: dict[str, WorkingOrder] = field(default_factory=dict)
    shorts_enabled: bool = True

    def execute(self, plans: list[PlannedOrder], cycle_id: str) -> None:
        tickers: dict[str, Ticker] = {}
        for plan in plans:
            try:
                if plan.action in ("SHORT_OPEN", "SHORT_CLOSE"):
                    self._short(plan, cycle_id)
                    continue
                if not tickers:
                    tickers = self.broker.get_tickers()
                if plan.urgent:
                    self._market(plan, plan.qty, cycle_id)
                else:
                    self._limit(plan, plan.qty, tickers.get(plan.pair), cycle_id)
            except ShortsDisabledError:
                raise  # engine disables the short sleeve for the session
            except (RoostooAPIError, ValueError) as e:
                log.warning("order rejected", extra={"pair": plan.pair, "action": plan.action, "err": str(e)})
                self.on_order(
                    {
                        "pair": plan.pair,
                        "side": plan.action,
                        "status": "REJECTED",
                        "reason": f"{plan.reason}: {e}",
                        "cycle_id": cycle_id,
                    }
                )

    def _short(self, plan: PlannedOrder, cycle_id: str) -> None:
        if not self.shorts_enabled:
            return
        sent_ms = self.clock.now_ms()
        try:
            if plan.action == "SHORT_OPEN":
                ack = self.broker.short_open(plan.pair, plan.collateral)
                self.on_order(
                    {
                        "pair": plan.pair,
                        "side": "SHORT_OPEN",
                        "kind": "SHORT_OPEN",
                        "collateral": str(plan.collateral),
                        "status": ack.status,
                        "avg_price": str(ack.entry_price),
                        "fee": str(ack.open_fee),
                        "exchange_order_id": ack.position_id,
                        "reason": plan.reason,
                        "cycle_id": cycle_id,
                    }
                )
                self.on_fill(
                    plan.pair,
                    "SHORT_OPEN",
                    ack.short_qty,
                    ack.entry_price,
                    ack.open_fee,
                    "TAKER",
                    plan.reason,
                )
            else:
                ack = self.broker.short_close(plan.pair, plan.close_pct)
                self.on_order(
                    {
                        "pair": plan.pair,
                        "side": "SHORT_CLOSE",
                        "kind": "SHORT_CLOSE",
                        "qty": str(ack.closed_qty),
                        "status": "FILLED",
                        "avg_price": str(ack.close_price),
                        "fee": str(ack.close_fee),
                        "reason": plan.reason,
                        "cycle_id": cycle_id,
                    }
                )
                self.on_fill(
                    plan.pair,
                    "SHORT_CLOSE",
                    ack.closed_qty,
                    ack.close_price,
                    ack.close_fee,
                    "TAKER",
                    plan.reason,
                )
        except OrderStateUnknownError:
            # shorts have no order id to query by qty; record and let the 5-min reconcile read
            # /v6/short_positions, which is the source of truth. Never resubmit here.
            log.error(
                "short request ambiguous; will reconcile from short_positions",
                extra={"pair": plan.pair, "action": plan.action, "sent_ms": sent_ms},
            )
            self.on_order(
                {
                    "pair": plan.pair,
                    "side": plan.action,
                    "kind": plan.action,
                    "status": "UNKNOWN",
                    "reason": plan.reason,
                    "cycle_id": cycle_id,
                }
            )

    def _send(self, plan: PlannedOrder, qty: Decimal, price: Decimal | None, cycle_id: str):
        """Send one spot order. On an ambiguous outcome, reconcile before anything else happens."""
        sent_ms = self.clock.now_ms()
        side = plan.action
        try:
            if price is None:
                return self.broker.place_market(plan.pair, side, qty)
            return self.broker.place_limit(plan.pair, side, qty, price)
        except OrderStateUnknownError:
            self.on_order(
                {
                    "pair": plan.pair,
                    "side": side,
                    "qty": str(qty),
                    "status": "UNKNOWN",
                    "reason": plan.reason,
                    "cycle_id": cycle_id,
                }
            )
            self.sleep(3.0)  # Section 8.3: give the exchange a moment before querying
            recent = getattr(self.broker, "recent_orders", None)
            candidates = recent(plan.pair) if recent else []
            match = match_unknown_order(plan.pair, side, qty, sent_ms, candidates)
            if match is None:
                log.warning(
                    "ambiguous order not found; treating as not placed",
                    extra={"pair": plan.pair, "side": side},
                )
                return None
            log.info("ambiguous order adopted", extra={"pair": plan.pair, "order_id": match.order_id})
            return self.broker.query(str(match.order_id))

    def _market(self, plan: PlannedOrder, qty: Decimal, cycle_id: str) -> None:
        ack = self._send(plan, qty, None, cycle_id)
        if ack is not None:
            self._record(plan, ack, "MARKET", cycle_id, qty)

    def _limit(
        self, plan: PlannedOrder, qty: Decimal, ticker: Ticker | None, cycle_id: str, tries: int = 1
    ) -> None:
        if ticker is None:
            log.warning("no ticker for limit order; skipping", extra={"pair": plan.pair})
            return
        price = maker_price(plan.action, ticker, self.rules.get(plan.pair))
        ack = self._send(plan, qty, price, cycle_id)
        if ack is None:
            return
        self._record(plan, ack, "LIMIT", cycle_id, qty, price=price)
        if ack.status != "FILLED":
            self.working[ack.order_id] = WorkingOrder(
                plan, ack.order_id, self.clock.now(), qty, ack.filled_qty, tries
            )

    def _record(
        self, plan: PlannedOrder, ack, kind: str, cycle_id: str, qty: Decimal, price: Decimal | None = None
    ) -> None:
        self.on_order(
            {
                "pair": plan.pair,
                "side": plan.action,
                "kind": kind,
                "qty": str(qty),
                "price": str(price) if price is not None else None,
                "exchange_order_id": ack.order_id,
                "status": ack.status,
                "role": ack.role,
                "filled_qty": str(ack.filled_qty),
                "avg_price": str(ack.avg_price),
                "fee": str(ack.fee),
                "reason": plan.reason,
                "cycle_id": cycle_id,
            }
        )
        if ack.filled_qty > 0:
            self.on_fill(
                plan.pair, plan.action, ack.filled_qty, ack.avg_price, ack.fee, ack.role, plan.reason
            )

    def service(self, cycle_id: str = "service") -> None:
        """Call every loop tick: reprice stale resting limits, escalate to MARKET after max tries."""
        now = self.clock.now()
        due = [
            w
            for w in self.working.values()
            if now - w.placed_at >= timedelta(seconds=self.exec_cfg.limit_reprice_s)
        ]
        if not due:
            return
        tickers = self.broker.get_tickers()
        for w in due:
            self.working.pop(w.order_id, None)
            status = self.broker.query(w.order_id)
            newly_filled = status.filled_qty - w.reported_filled
            if newly_filled > 0:
                self.on_fill(
                    w.plan.pair,
                    w.plan.action,
                    newly_filled,
                    status.avg_price,
                    status.fee,
                    status.role,
                    w.plan.reason,
                )
            if status.status == "FILLED":
                continue
            try:
                self.broker.cancel(w.order_id)
            except RoostooAPIError as e:
                log.warning("cancel failed", extra={"order_id": w.order_id, "err": str(e)})
                continue  # leave it; next reconcile will pick up whatever state it is in
            residual = _round_qty(w.plan.pair, w.sent_qty - status.filled_qty, self.rules)
            if residual <= 0:
                continue
            price = Decimal(str(tickers[w.plan.pair].mid)) if w.plan.pair in tickers else ZERO
            if not _meets_min(w.plan.pair, price, residual, self.rules):
                continue
            if w.tries >= self.exec_cfg.limit_max_tries:
                self._market(w.plan, residual, cycle_id)
            else:
                self._limit(w.plan, residual, tickers.get(w.plan.pair), cycle_id, tries=w.tries + 1)

    def cancel_all_working(self) -> None:
        """HALT path: cancel only the orders this process created (never cancel_all on the account)."""
        for oid in list(self.working):
            try:
                self.broker.cancel(oid)
            except RoostooAPIError as e:
                log.warning("cancel failed during halt", extra={"order_id": oid, "err": str(e)})
            self.working.pop(oid, None)
