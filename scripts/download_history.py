"""CLI: backfill 1h spot klines into data/bars/1h/{SYM}.parquet (BUILD_SPEC.md Section 5.2/M2).

Usage:
    python scripts/download_history.py --symbols BTCUSDT,ETHUSDT --start 2024-01-01 --end 2026-09-30
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from trend_guard.data.binance_vision import download_klines_1h

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
]


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill Binance 1h spot klines to data/bars/1h/")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="comma-separated Binance spot symbols (default: a reasonable major-crypto list)",
    )
    parser.add_argument(
        "--start", default="2024-01-01", type=_parse_date, help="YYYY-MM-DD, default 2024-01-01"
    )
    parser.add_argument(
        "--end", default=date.today().isoformat(), type=_parse_date, help="YYYY-MM-DD, default today"
    )
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for symbol in symbols:
        print(f"downloading {symbol} {args.start} -> {args.end} ...")
        df = download_klines_1h(symbol, args.start, args.end)
        print(f"  {symbol}: {len(df)} bars written to data/bars/1h/{symbol}.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
