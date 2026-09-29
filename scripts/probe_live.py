"""Prep-window probes against the LIVE Roostoo mock API (BUILD_SPEC.md Section 15).

This script places real orders on the competition account. It is in .claude/settings.json's deny
list: Claude Code must never run this. THE HUMAN RUNS THIS, not the assistant. Before running it,
confirm with the organizers that prep-period trades on the competition key are acceptable
(Screen 1 checks for manually-called APIs) -- if not allowed, run the same checks through the bot
itself in live mode at risk_scale 0.1 instead.

Requires ALLOW_LIVE_ORDERS=1 in the environment and the --i-understand flag. Appends a
human-readable record of every probe result to docs/PROBES.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from trend_guard.config import load_env
from trend_guard.roostoo.client import RoostooClient
from trend_guard.roostoo.errors import RoostooAPIError, ShortsDisabledError
from trend_guard.roostoo.ratelimit import RateLimiter

PROBES_MD = Path("docs/PROBES.md")


def _append(lines: list[str]) -> None:
    PROBES_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(PROBES_MD, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def probe_maker_behaviour(client: RoostooClient) -> None:
    """BTC/USD BUY LIMIT at MaxBid for ~$20, poll 5 min for Role and fill time."""
    ticker = client.get_ticker("BTC/USD")["BTC/USD"]
    qty = Decimal(str(round(20 / ticker.max_bid, 6)))
    price = Decimal(str(ticker.max_bid))
    order = client.place_order("BTC/USD", "BUY", "LIMIT", qty, price)
    start = time.time()
    role, filled = order.role, order.status == "FILLED"
    while time.time() - start < 300 and not filled:
        time.sleep(15)
        matched = client.query_order(order_id=str(order.order_id))
        if matched:
            role, filled = matched[0].role, matched[0].status == "FILLED"
            if filled:
                break
    _append(
        [
            "## Maker behaviour probe",
            f"- order_id: {order.order_id}, role: {role}, filled: {filled}, "
            f"elapsed_s: {time.time() - start:.1f}",
        ]
    )


def probe_market_order(client: RoostooClient) -> None:
    """BUY then SELL ~$20 of ETH; compare fill price to ticker and record fee."""
    ticker = client.get_ticker("ETH/USD")["ETH/USD"]
    qty = Decimal(str(round(20 / ticker.last_price, 6)))
    buy = client.place_order("ETH/USD", "BUY", "MARKET", qty)
    sell = client.place_order("ETH/USD", "SELL", "MARKET", qty)
    _append(
        [
            "## Market order probe (ETH/USD)",
            f"- pre-trade last_price: {ticker.last_price}",
            f"- BUY: fill={buy.filled_aver_price} fee={buy.commission_charge_value} "
            f"pct={buy.commission_percent}",
            f"- SELL: fill={sell.filled_aver_price} fee={sell.commission_charge_value} "
            f"pct={sell.commission_percent}",
        ]
    )


def probe_shorts(client: RoostooClient) -> None:
    """short_open $10 collateral on BTC, then short_close 100%. Records whether shorts are enabled."""
    try:
        opened = client.short_open("BTC/USD", Decimal("10"))
    except ShortsDisabledError as e:
        _append(["## Shorts probe", f"- DISABLED: {e}"])
        return
    closed = client.short_close("BTC/USD", close_pct=Decimal("100"))
    _append(
        [
            "## Shorts probe",
            f"- open: id={opened.id} entry={opened.entry_price} fee={opened.open_fee}",
            f"- close: pnl={closed.realized_pnl} fee={closed.close_fee} fully_closed={closed.fully_closed}",
        ]
    )


def probe_cancel(client: RoostooClient) -> None:
    """Place a far-away limit order, cancel by id, confirm it worked."""
    ticker = client.get_ticker("BTC/USD")["BTC/USD"]
    far_price = Decimal(str(round(ticker.max_bid * 0.5, 2)))  # 50% below touch: never fills
    qty = Decimal("0.001")
    order = client.place_order("BTC/USD", "BUY", "LIMIT", qty, far_price)
    canceled = client.cancel_order(order_id=str(order.order_id))
    _append(
        [
            "## Cancel probe",
            f"- order_id: {order.order_id}, canceled_list: {canceled}, worked: {order.order_id in canceled}",
        ]
    )


def probe_rate_limit(client: RoostooClient, n: int = 40) -> None:
    """Poll ticker at ~30s cadence intent but back-to-back here to observe any throttling errors."""
    errors = 0
    for _ in range(n):
        try:
            client.get_ticker()
        except RoostooAPIError:
            errors += 1
        time.sleep(1)
    _append(["## Rate limit probe", f"- requests: {n}, errors: {errors}"])


PROBES = {
    "maker": probe_maker_behaviour,
    "market": probe_market_order,
    "shorts": probe_shorts,
    "cancel": probe_cancel,
    "rate_limit": probe_rate_limit,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-understand", action="store_true", required=True)
    parser.add_argument("--probes", default="maker,market,shorts,cancel", help="comma-separated probe names")
    args = parser.parse_args()

    env = load_env()
    if not env.allow_live_orders:
        print("ALLOW_LIVE_ORDERS is not 1. Refusing to run.", file=sys.stderr)
        return 1
    if not (env.roostoo_api_key and env.roostoo_secret_key):
        print("Missing API keys.", file=sys.stderr)
        return 1

    client = RoostooClient(
        env.roostoo_base_url, env.roostoo_api_key, env.roostoo_secret_key, rate_limiter=RateLimiter()
    )
    client.sync_clock()
    _append([f"# Probe run {datetime.now(UTC).isoformat()}"])

    for name in args.probes.split(","):
        fn = PROBES.get(name.strip())
        if fn is None:
            print(f"unknown probe: {name}", file=sys.stderr)
            continue
        print(f"running probe: {name}")
        fn(client)

    print(f"results appended to {PROBES_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
