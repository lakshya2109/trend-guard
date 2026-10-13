"""Idle cash into BTC + ETH: should the share of equity the book leaves in cash be filled with the majors?

The live book (normal book + net hedge, bear mode off) is often below full gross: the trend sleeve scales
down when volatility is high and only holds coins in an uptrend, so after the October crash ~20% of equity
sat in cash. Proposal (team, 14 Oct): put that cash into BTC + ETH 50/50. Variants fill all or half of the
unused gross with BTC / ETH (netted with the book's own BTC / ETH weights, gross capped at 1). Same tests as
round 25: continuous run, contest windows, contest replay, the hours left from starts like now.

    python research/cash_fill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.breaker_now import sim  # noqa: E402
from research.paired import paired  # noqa: E402
from research.steady_book import H_LEFT, act, br, ex, lab, live, now_like, row_metrics  # noqa: E402

btc, eth = lab.cols.index("BTC/USD"), lab.cols.index("ETH/USD")


def fill(W: np.ndarray, share: float) -> np.ndarray:
    """Add `share` of the unused gross as BTC / ETH 50/50, then cap gross at 1."""
    spare = np.clip(1.0 - np.abs(W).sum(axis=1), 0.0, None) * share
    out = W.copy()
    out[:, btc] += 0.5 * spare
    out[:, eth] += 0.5 * spare
    return out / np.maximum(np.abs(out).sum(axis=1, keepdims=True), 1.0)


V = {"deployed (hedged)": live, "fill all idle cash BTC/ETH": fill(live, 1.0), "fill half": fill(live, 0.5)}
base = lab.daily(live)
rows = []
for k, W in V.items():
    row = row_metrics(k, W)
    r = (
        sim(
            np.ascontiguousarray(W),
            lab.px,
            act,
            now_like,
            H_LEFT,
            lab.cost,
            ex.min_trade_weight,
            ex.min_trade_frac,
            0.06,
            br.dd_level,
            br.dd_mult,
            br.dd_recover,
        )
        - 1.0
    )
    row |= {
        "left P>5%": float(np.mean(r > 0.05)),
        "left P<-5%": float(np.mean(r < -0.05)),
        "gross now": float(np.abs(W[-1]).sum()),
        "net now": float(W[-1].sum()),
    }
    if k != "deployed (hedged)":
        p = paired(base, lab.daily(W))
        row |= {
            "dSharpe": p["dSR"],
            "P(better)": p["P(dSR>0)"],
            "dSR 24-25": p["dSR 2024-25"],
            "dSR 2026": p["dSR 2026"],
        }
    rows.append(row)
keep = [
    "net",
    "gross now",
    "net now",
    "SR 24-25",
    "SR 2026",
    "ret/yr",
    "day vol",
    "maxDD",
    "score med",
    "contest",
    "left P>2.5%",
    "left P>5%",
    "left mean",
    "left P<-3%",
    "left P<-5%",
    "dSharpe",
    "P(better)",
    "dSR 24-25",
    "dSR 2026",
]
df = pd.DataFrame(rows).set_index("book")[keep]
print(f"{len(now_like)} starts like now, {H_LEFT} h left, from 6% under the peak")
print(df.T.to_string(float_format=lambda x: f"{x:+.3f}"))
