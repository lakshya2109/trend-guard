"""Filter: no NEW long entries in a coin within `hold` hours after a 1h jump above `thr` (held longs stay)."""

import sys
from pathlib import Path

import numba as nb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.harness import Lab  # noqa: E402


@nb.njit(cache=True)
def apply(W, r1, thr, hold, also_short):
    T, N = W.shape
    out = W.copy()
    since = np.full(N, 10**9)
    for t in range(T):
        for i in range(N):
            if r1[t, i] > thr:
                since[i] = 0
            elif since[i] < 10**9:
                since[i] += 1
            if since[i] <= hold:
                prev = out[t - 1, i] if t > 0 else 0.0
                if out[t, i] > 0 and prev <= 0:
                    out[t, i] = 0.0  # blocked entry
                elif out[t, i] > prev > 0:
                    out[t, i] = prev  # no adding either
    return out


if __name__ == "__main__":
    lab = Lab()
    lp = np.log(lab.close.ffill(limit=3)).to_numpy()
    r1 = np.vstack([np.zeros((1, lp.shape[1])), np.nan_to_num(np.diff(lp, axis=0))])
    W = np.nan_to_num(lab.book)
    books = {"book": W}
    for thr in (0.07, 0.10, 0.15):
        for hold in (24, 72):
            books[f"no new longs {hold}h after +{thr:.0%}"] = apply(W, r1, np.log1p(thr), hold, False)
    lab.report(books)
