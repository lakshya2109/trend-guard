"""Evaluate prediction panels as sleeves, alone and added to the live book.

python research/eval_alpha.py price base
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.alpha_book import alpha_sleeve  # noqa: E402
from research.features import build  # noqa: E402
from research.harness import Lab  # noqa: E402


def load(tag: str) -> np.ndarray:
    return np.load(ROOT / "data" / "extra" / f"pred_{tag}.npz")["pred"]


def context(lab: Lab) -> dict:
    d = build(lab.close, lab.qv)
    r = np.log(lab.close.ffill(limit=3)).diff().to_numpy()
    return {
        "ret": r,
        "vol_d": d["vol_d"].to_numpy(),
        "beta": d["beta"].to_numpy(),
        "hours": lab.idx.hour.to_numpy(),
    }


def sleeve(pred: np.ndarray, c: dict, **kw) -> np.ndarray:
    return alpha_sleeve(pred, c["ret"], c["vol_d"], c["beta"], c["hours"], **kw)


if __name__ == "__main__":
    lab = Lab()
    c = context(lab)
    books = {"book": lab.book}
    for tag in sys.argv[1:]:
        p = load(tag)
        for kw in ({}, {"smooth": 3}, {"k": 8}, {"k": 8, "smooth": 3}):
            name = f"{tag} " + ",".join(f"{a}={b}" for a, b in kw.items())
            books[name] = sleeve(p, c, target_vol=0.01, **kw)
    df = lab.report(books)
    base = lab.daily(lab.book)
    for k, W in books.items():
        print(f"corr with book {k}: {lab.daily(W).corr(base):.2f}")
