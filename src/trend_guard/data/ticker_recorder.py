"""Roostoo ticker recorder (BUILD_SPEC.md Section 5.4).

Runs as its own systemd service (deploy/recorder.service) from day 1: poll /v3/ticker (no pair =
all pairs) every `poll_s` seconds and append raw snapshots to UTC-day parquet files. Used for
bStock bars, the Roostoo-vs-Binance basis check, the live protective loop and the universe spread
filter.

`poll_once`/`build_ohlc`/`compute_spread` are split out from `run_forever` so the bar-building
and spread math can be unit tested with no I/O and no sleeping.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trend_guard.roostoo.client import RoostooClient

log = logging.getLogger("data")

_FLUSH_EVERY = 20  # snapshots buffered before an incremental parquet write (~10 min at 30s poll)
SNAPSHOT_COLUMNS = [
    "ts",
    "pair",
    "max_bid",
    "min_ask",
    "last_price",
    "change",
    "coin_trade_value",
    "unit_trade_value",
]


def poll_once(client: RoostooClient, now: datetime) -> pd.DataFrame:
    """One /v3/ticker call for all pairs, flattened into snapshot rows. Never raises: an API
    failure is logged and an empty frame is returned so the recorder loop keeps running."""
    try:
        tickers = client.get_ticker()
    except Exception as e:  # the recorder must survive any transient API failure
        log.warning("ticker poll failed: %s", e)
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    rows = [
        {
            "ts": now,
            "pair": pair,
            "max_bid": t.max_bid,
            "min_ask": t.min_ask,
            "last_price": t.last_price,
            "change": t.change,
            "coin_trade_value": t.coin_trade_value,
            "unit_trade_value": t.unit_trade_value,
        }
        for pair, t in tickers.items()
    ]
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


def day_path(out_dir: Path, day: str) -> Path:
    return out_dir / f"{day}.parquet"


def flush_snapshots(rows: pd.DataFrame, path: Path) -> None:
    """Append snapshot rows to a day's parquet file (read-concat-write; files are small)."""
    if rows.empty:
        return
    if path.exists():
        existing = pd.read_parquet(path)
        rows = pd.concat([existing, rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path)


def run_forever(client: RoostooClient, poll_s: int = 30, out_dir: Path = Path("data/roostoo_ticks")) -> None:
    """Poll forever, rotating the output file at UTC midnight. Intended to run as a long-lived
    service process; there is no exit condition short of process termination."""
    out_dir.mkdir(parents=True, exist_ok=True)
    current_day: str | None = None
    buffer = pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    while True:
        now = datetime.now(UTC)
        day = now.strftime("%Y-%m-%d")
        if current_day is not None and day != current_day:
            flush_snapshots(buffer, day_path(out_dir, current_day))  # rotate: flush the old day
            buffer = pd.DataFrame(columns=SNAPSHOT_COLUMNS)
        current_day = day

        snap = poll_once(client, now)
        buffer = pd.concat([buffer, snap], ignore_index=True)
        if len(buffer) >= _FLUSH_EVERY:
            flush_snapshots(buffer, day_path(out_dir, current_day))
            buffer = pd.DataFrame(columns=SNAPSHOT_COLUMNS)

        time.sleep(poll_s)


def build_ohlc(snapshots: pd.DataFrame, freq: str) -> pd.DataFrame:
    """1-minute/1-hour OHLC built from LastPrice snapshots (pure, no I/O).

    If `snapshots` has a `pair` column with more than one distinct pair, returns one OHLC block
    per pair under a (pair, time) MultiIndex; otherwise a single-asset frame indexed by time.
    """
    empty = pd.DataFrame(columns=["open", "high", "low", "close"])
    if snapshots.empty:
        return empty

    def _one(df: pd.DataFrame) -> pd.DataFrame:
        idx = pd.DatetimeIndex(df["ts"])
        s = pd.Series(df["last_price"].to_numpy(), index=idx).sort_index()
        out = s.resample(freq).ohlc()
        return out.dropna(subset=["close"])

    if "pair" in snapshots.columns and snapshots["pair"].nunique() > 1:
        parts = []
        for pair, g in snapshots.groupby("pair"):
            o = _one(g)
            o["pair"] = pair
            parts.append(o)
        if not parts:
            return empty
        combined = pd.concat(parts)
        return combined.set_index("pair", append=True).swaplevel(0, 1).sort_index()

    return _one(snapshots)


def compute_spread(snapshots: pd.DataFrame) -> pd.Series:
    """spread = (MinAsk - MaxBid) / mid, row-aligned with `snapshots` (pure, no I/O)."""
    mid = (snapshots["max_bid"] + snapshots["min_ask"]) / 2
    spread = (snapshots["min_ask"] - snapshots["max_bid"]) / mid
    return spread.where(mid != 0)


def main() -> None:
    """Entry point for deploy/recorder.service: `python -m trend_guard.data.ticker_recorder`."""
    from trend_guard.config import load_env, load_strategy_config
    from trend_guard.logging_setup import setup_logging
    from trend_guard.roostoo.ratelimit import RateLimiter

    setup_logging(categories=("api", "recorder"))
    env = load_env()
    cfg = load_strategy_config()
    client = RoostooClient(
        env.roostoo_base_url,
        env.roostoo_api_key,
        env.roostoo_secret_key,
        rate_limiter=RateLimiter(cfg.execution.requests_per_min),
    )
    client.sync_clock()
    run_forever(client, poll_s=cfg.execution.ticker_poll_s)


if __name__ == "__main__":
    main()
