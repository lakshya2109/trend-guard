"""Download or top up hourly Binance history for every crypto pair listed on Roostoo.

    python scripts/download_history.py                        # all Roostoo crypto pairs since 2023-06-01
    python scripts/download_history.py --pairs BTC/USD ETH/USD --start 2025-01-01

Pairs come from Roostoo's public /v3/exchangeInfo (no API keys needed). Bars are cached as
data/bars/1h/{BINANCE_SYMBOL}.parquet; re-running only fetches bars newer than the cache.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from trend_guard.config import load_env, load_symbols
from trend_guard.data.binance_live import update_cache
from trend_guard.roostoo.client import RoostooClient
from trend_guard.universe import binance_symbol, tradable_pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-06-01")
    ap.add_argument("--pairs", nargs="*", help="Roostoo pairs; default: every tradable crypto pair")
    ap.add_argument("--out", default="data/bars/1h")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    symbols_cfg = load_symbols()
    if args.pairs:
        pairs = args.pairs
    else:
        env = load_env()
        info = RoostooClient(env.roostoo_base_url, "", "").exchange_info()
        pairs = tradable_pairs(info, symbols_cfg)
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    now = datetime.now(UTC)
    out = Path(args.out)

    def one(pair: str) -> str:
        sym = binance_symbol(pair, symbols_cfg)
        try:
            df = update_cache(sym, out, start, now)
        except RuntimeError as e:  # keep going: one bad symbol must not stop the rest
            return f"{pair:16s} FAILED {e}"
        if df.empty:
            return f"{pair:16s} not on Binance"
        return f"{pair:16s} {len(df):6d} bars  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d %H:%M}"

    with ThreadPoolExecutor(args.workers) as pool:
        for line in pool.map(one, sorted(pairs)):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
