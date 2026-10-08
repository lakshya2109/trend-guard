"""CLI entrypoint. BUILD_SPEC.md Section 10.3.

  python -m trend_guard.main --mode paper                  # live Roostoo prices, simulated fills
  python -m trend_guard.main --mode live                   # real orders; needs ALLOW_LIVE_ORDERS=1
  python -m trend_guard.main --mode paper --once           # one hourly cycle, then exit (debugging)

Live mode refuses to start unless ALLOW_LIVE_ORDERS=1 and both API keys are present. Paper mode
needs no keys: tickers and exchangeInfo are public/timestamp-only endpoints.
"""

from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal
from pathlib import Path

from trend_guard.clock import RealClock
from trend_guard.config import load_env, load_strategy_config, load_symbols
from trend_guard.engine import Engine
from trend_guard.execution.live_broker import LiveBroker
from trend_guard.execution.sim_broker import SimBroker
from trend_guard.logging_setup import setup_logging
from trend_guard.roostoo.client import RoostooClient
from trend_guard.roostoo.ratelimit import RateLimiter
from trend_guard.state.store import Store

EC2_HALT = Path("/etc/trend-guard/HALT")
LOCAL_HALT = Path("HALT")
PAPER_START_CASH = Decimal("100000")


def halt_path() -> Path:
    """/etc/trend-guard/HALT on EC2 (where that directory exists), ./HALT locally."""
    return EC2_HALT if EC2_HALT.parent.exists() else LOCAL_HALT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--config", default="config/strategy.yaml")
    ap.add_argument("--once", action="store_true", help="run one hourly cycle then exit")
    ap.add_argument("--db", default=None, help="SQLite path (default state/<mode>.db)")
    args = ap.parse_args(argv)

    setup_logging()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    env = load_env()
    cfg = load_strategy_config(args.config)
    symbols = load_symbols()
    clock = RealClock()
    limiter = RateLimiter(cfg.execution.requests_per_min, cfg.execution.min_order_gap_s)
    client = RoostooClient(
        env.roostoo_base_url, env.roostoo_api_key, env.roostoo_secret_key, rate_limiter=limiter
    )

    if args.mode == "live":
        if not env.allow_live_orders:
            print("refusing to start live mode: ALLOW_LIVE_ORDERS is not 1", file=sys.stderr)
            return 2
        if not (env.roostoo_api_key and env.roostoo_secret_key):
            print(
                "refusing to start live mode: ROOSTOO_API_KEY / ROOSTOO_SECRET_KEY missing", file=sys.stderr
            )
            return 2
        from trend_guard.roostoo.rules import parse_trading_rules

        broker = LiveBroker(client, parse_trading_rules(client.exchange_info()), allow_live_orders=True)
    else:
        broker = SimBroker(PAPER_START_CASH, cfg.backtest, clock, mode="paper")

    store = Store(args.db or f"state/{args.mode}.db", clock)
    # paper fills go to their own folder: logs/fills is the live trade log the organisers check
    fills_dir = Path("logs/fills") if args.mode == "live" else Path("logs/paper/fills")
    engine = Engine(
        cfg, symbols, env, client, broker, store, clock, halt_path(), args.config, fills_dir=fills_dir
    )

    if args.once:
        engine.startup()
        engine.hourly_cycle(force_act=True)
        return 0
    engine.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
