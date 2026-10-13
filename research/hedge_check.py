"""Net-exposure hedge on the live book: paired test, neighbours, and simpler hedge baskets.

Follows research/steady_book.py (hedge 50% of net: Sharpe 1.95 / 2.65 -> 2.03 / 2.82). Hedge baskets: every
liquid coin equal weight (EW), the 10 most liquid coins (fewer, larger shorts), BTC + ETH. Same test as every
candidate since round 18: paired stationary bootstrap of the daily Sharpe vs the live book, both periods.

    python research/hedge_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.paired import paired  # noqa: E402
from research.steady_book import CONTEST, basket, hedged, lab, live, ok, row_metrics  # noqa: E402

cols = lab.cols
dollar_vol = (lab.qv.rolling(168).sum() / 7).to_numpy()
top10 = np.zeros_like(basket)
for t in range(len(top10)):
    v = np.where(ok[t], np.nan_to_num(dollar_vol[t]), -1.0)
    pick = np.argsort(v)[-10:]
    pick = pick[v[pick] > 0]
    top10[t, pick] = 1.0 / max(len(pick), 1)
majors = np.zeros_like(basket)
majors[:, cols.index("BTC/USD")] = majors[:, cols.index("ETH/USD")] = 0.5


def hedge_with(W: np.ndarray, frac: float, bask: np.ndarray) -> np.ndarray:
    out = W - frac * W.sum(axis=1, keepdims=True) * bask
    return out / np.maximum(np.abs(out).sum(axis=1, keepdims=True), 1.0)


V = {
    "EW 25%": hedged(live, 0.25),
    "EW 50%": hedged(live, 0.5),
    "EW 75%": hedged(live, 0.75),
    "top-10 liquid 50%": hedge_with(live, 0.5, top10),
    "BTC+ETH 50%": hedge_with(live, 0.5, majors),
}
base = lab.daily(live)
print("vs the live book (paired bootstrap of the daily Sharpe, 2024 -> now):")
for k, W in V.items():
    p = paired(base, lab.daily(W))
    c = lab.continuous(W, CONTEST)
    n_short = int(np.median((W[-24 * 30 :] < -1e-6).sum(axis=1)))
    print(
        f"  {k:<18} dSharpe {p['dSR']:+.2f}  P(better) {p['P(dSR>0)']:.0%}  2024-25 {p['dSR 2024-25']:+.2f}  "
        f"2026 {p['dSR 2026']:+.2f}  d return/yr {p['d_ann_ret']:+.1%}  "
        f"contest {c.iloc[-1] / c.iloc[0] - 1:+.1%}  "
        f"shorts held ~{n_short}"
    )
rows = [
    row_metrics(k, W)
    for k, W in (("live 4x", live), ("EW 25%", V["EW 25%"]), ("top-10 liquid 50%", V["top-10 liquid 50%"]))
]
print(pd.DataFrame(rows).set_index("book").T.to_string(float_format=lambda x: f"{x:+.3f}"))
