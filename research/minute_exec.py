"""Live execution policy on real 1-minute Binance bars: how long should a maker limit rest before MARKET?

Replays order_manager's logic at the bot's decision times (HH:03 UTC): post at the touch (mid -/+ half
spread), re-post at the new touch every minute, fill as MAKER (5 bp) if the next minute trades through the
limit by `thru`, else after N minutes cross at the touch as TAKER (10 bp). Cost is measured against the
mid when the order was first sent; signal decay over N minutes is ignored (the book's signals are daily).
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "extra" / "minute_events.parquet"
URL = "https://data-api.binance.vision/api/v3/klines"


def fetch(n_events: int = 600) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    syms = sorted(p.stem for p in (ROOT / "data/bars/1h").glob("*.parquet") if p.stem != "PAXGUSDT")
    rng = random.Random(7)
    hours = pd.date_range("2026-01-01 00:03", "2026-09-29 12:03", freq="12h", tz="UTC")
    rows = []
    s = requests.Session()
    for k in range(n_events):
        sym, t0 = rng.choice(syms), rng.choice(hours)
        r = s.get(
            URL,
            params={"symbol": sym, "interval": "1m", "startTime": int(t0.timestamp() * 1000), "limit": 62},
            timeout=20,
        )
        if r.status_code != 200 or len(r.json()) < 62:
            continue
        for j, b in enumerate(r.json()):
            rows.append((k, sym, j, float(b[1]), float(b[2]), float(b[3]), float(b[4])))
    df = pd.DataFrame(rows, columns=["event", "sym", "m", "open", "high", "low", "close"])
    df.to_parquet(CACHE)
    return df


def cost_bp(
    bars: np.ndarray,
    side: int,
    patience: int,
    hs: float,
    thru: float,
    maker: float,
    taker: float,
    reprice: bool,
) -> tuple[float, bool]:
    """bars: (62, 4) open/high/low/close per minute. side +1 buy / -1 sell. Returns (cost in bp, maker?)."""
    mid0 = bars[0, 0]
    lim = mid0 * (1 - side * hs)
    for k in range(patience):
        hi, lo = bars[k, 1], bars[k, 2]
        if (side > 0 and lo <= lim * (1 - thru)) or (side < 0 and hi >= lim * (1 + thru)):
            return (side * (lim - mid0) / mid0 + maker) * 1e4, True
        if reprice:
            lim = bars[k, 3] * (1 - side * hs)
    px = bars[patience - 1, 3] * (1 + side * hs) if patience > 0 else mid0 * (1 + side * hs)
    return (side * (px - mid0) / mid0 + taker) * 1e4, False


if __name__ == "__main__":
    df = fetch()
    ev = [g[["open", "high", "low", "close"]].to_numpy() for _, g in df.groupby("event")]
    print(f"{len(ev)} events (random coin, random 2026 decision time), buys and sells")
    rows = []
    for hs in (0.00025, 0.0005):
        for thru in (0.0, 0.0002):
            for N in (0, 4, 10, 15, 30, 60):
                for reprice in (True, False) if N else (True,):
                    c = [cost_bp(b, s, N, hs, thru, 0.0005, 0.001, reprice) for b in ev for s in (1, -1)]
                    cs = np.array([x[0] for x in c])
                    rows.append(
                        {
                            "half_spread_bp": hs * 1e4,
                            "thru_bp": thru * 1e4,
                            "minutes": N,
                            "reprice": reprice,
                            "cost_bp": cs.mean(),
                            "se": cs.std() / np.sqrt(len(cs)),
                            "maker%": np.mean([x[1] for x in c]) * 100,
                        }
                    )
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(pd.DataFrame(rows).to_string(index=False))
