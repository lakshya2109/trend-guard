"""Partial rebalancing: move a fraction lam of the way to the target each hour (Garleanu-Pedersen style).
Closes (target 0) and sign flips go all the way; opens go all the way if open_full."""

import sys
from pathlib import Path

import numba as nb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.harness import Lab  # noqa: E402


@nb.njit(cache=True)
def smooth(W, lam, open_full):
    T, N = W.shape
    out = np.zeros_like(W)
    cur = np.zeros(N)
    for t in range(T):
        for i in range(N):
            w = W[t, i]
            c = cur[i]
            if w == 0.0 or c == 0.0 or np.sign(w) != np.sign(c):
                cur[i] = w if (w == 0.0 or open_full or c * w < 0) else lam * w
            else:
                cur[i] = c + lam * (w - c)
        out[t] = cur
    return out


if __name__ == "__main__":
    lab = Lab()
    W = np.nan_to_num(lab.book)
    books = {"book": W}
    for lam in (0.1, 0.25, 0.5):
        books[f"lam={lam}"] = smooth(W, lam, True)
    books["lam=0.25 open partial"] = smooth(W, 0.25, False)
    lab.report(books)
