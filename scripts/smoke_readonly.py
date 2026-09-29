"""Read-only smoke test against the Roostoo mock API: no orders are ever placed.

Prints server time, pair count and the BTC/USD ticker. If API keys are present in the
environment, also prints wallet balance. Safe to run at any time (BUILD_SPEC.md Milestone 1).
"""

from __future__ import annotations

import sys

from trend_guard.config import load_env
from trend_guard.roostoo.client import RoostooClient
from trend_guard.roostoo.ratelimit import RateLimiter


def main() -> int:
    env = load_env()
    client = RoostooClient(
        base_url=env.roostoo_base_url,
        api_key=env.roostoo_api_key,
        secret_key=env.roostoo_secret_key,
        rate_limiter=RateLimiter(),
    )

    offset = client.sync_clock()
    print(f"clock offset (server - local): {offset} ms")

    server_time = client.server_time()
    print(f"server time: {server_time}")

    info = client.exchange_info()
    pairs = info.get("TradePairs", {})
    print(f"pair count: {len(pairs)}")

    if "BTC/USD" in pairs:
        tickers = client.get_ticker("BTC/USD")
        btc = tickers.get("BTC/USD")
        print(f"BTC/USD ticker: last={btc.last_price} bid={btc.max_bid} ask={btc.min_ask}")
    else:
        print("BTC/USD not listed in exchangeInfo")

    if env.roostoo_api_key and env.roostoo_secret_key:
        balance = client.get_balance()
        usd = balance.get("USD")
        if usd:
            print(f"USD balance: free={usd.free} lock={usd.lock}")
        else:
            print("no USD entry in wallet")
    else:
        print("no API keys set: skipping balance check")

    return 0


if __name__ == "__main__":
    sys.exit(main())
