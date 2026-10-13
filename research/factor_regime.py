"""Momentum vs short-term reversal: are their regimes persistent enough to time?

Contest so far (4-13 Oct 2026): the bear-mode rebound book (long the 10 worst 3-day coins) made ~+11% while
every momentum / trend sleeve lost. Factor books, all on the same liquid universe as the rebound sleeve:
  MOM     the live v2 momentum sleeve alone (long/short, re-ranked daily)
  REV_L   long the 10 worst 3-day coins, equal weight, gross 1, re-picked daily at 00 UTC (the rebound book)
  REV_LS  long the 10 worst / short the 10 best 3-day coins, gross 1 (0.5 a side)
  MKT     every liquid coin, equal weight, gross 1
  NORMAL  the live book without bear mode;  LIVE  the live book (bear mode on)
Persistence test: does the trailing L-day return spread between two books predict the next 5 days' spread?

    python research/factor_regime.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.harness import Lab  # noqa: E402
from trend_guard.strategy import book as B  # noqa: E402

lab = Lab()
b, bm = lab.cfg.book, lab.cfg.book.bear_mode
T, N = lab.px.shape
off = b.model_copy(update={"bear_mode": bm.model_copy(update={"enabled": False})})
px = lab.close.ffill(limit=3)
liq = (lab.qv.rolling(168).sum() / 7) > bm.min_liquidity_usd
age = lab.close.notna().cumsum().to_numpy()
ok = liq.to_numpy() & (age >= bm.rebound_min_age_h) & np.isfinite(px.to_numpy())
for pair in bm.rebound_exclude:
    ok[:, lab.cols.index(pair)] = False
lp = np.log(px.to_numpy())
pick_hour = lab.idx.hour == bm.rebound_hour_utc


def reversal(k: int, look: int, short_side: bool) -> np.ndarray:
    """Long the k worst `look`-hour coins (and short the k best), re-picked daily, gross 1."""
    ret = np.full((T, N), np.nan)
    ret[look:] = lp[look:] - lp[:-look]
    W, cur = np.zeros((T, N)), np.zeros(N)
    for t in range(T):
        if pick_hour[t]:
            r = np.where(ok[t] & np.isfinite(ret[t]), ret[t], np.nan)
            valid = np.flatnonzero(np.isfinite(r))
            cur = np.zeros(N)
            if len(valid) >= 2 * k:
                order = valid[np.argsort(r[valid])]
                side = 0.5 if short_side else 1.0
                cur[order[:k]] = side / k
                if short_side:
                    cur[order[-k:]] = -side / k
        W[t] = cur
    return W


def books() -> dict[str, np.ndarray]:
    mom_cfg = off.model_copy(
        update={"v2_share": 1.0, "team_share": 0.0, "trend_share": 0.0, "momentum_share": 1.0}
    )
    return {
        "MOM": np.nan_to_num(sum(B.book_components(lab.close, lab.qv, mom_cfg, True).values())),
        "REV_L": reversal(bm.rebound_k, bm.rebound_lookback_h, False),
        "REV_LS": reversal(bm.rebound_k, bm.rebound_lookback_h, True),
        "MKT": ok / np.maximum(ok.sum(axis=1, keepdims=True), 1),
        "NORMAL": np.nan_to_num(sum(B.book_components(lab.close, lab.qv, off, True).values())),
        "LIVE": np.nan_to_num(lab.book),
    }


if __name__ == "__main__":
    BK = books()
    D = pd.DataFrame({k: lab.daily(W) for k, W in BK.items()})  # daily returns after costs, 12:00 marks
    print(f"daily returns {D.index[0]:%Y-%m-%d} -> {D.index[-1]:%Y-%m-%d}, after costs")
    half = D.index.year.astype(str) + np.where(D.index.month <= 6, "H1", "H2")
    tab = D.groupby(half).apply(lambda x: (1 + x).prod() - 1)
    with pd.option_context("display.width", 200, "display.float_format", "{:+.1%}".format):
        print("\nreturn by half-year")
        print(tab)
        print("\ncorrelation with the market (daily)")
        print(D.corr()["MKT"].round(2).to_string())
    print("\npersistence: corr(trailing L-day spread, next 5-day spread), hit rate of its sign")
    for a, c in (("REV_L", "NORMAL"), ("REV_LS", "MOM"), ("REV_L", "MKT"), ("MOM", "MKT")):
        s = np.log1p(D[a]) - np.log1p(D[c])
        nxt = s[::-1].rolling(5).sum()[::-1].shift(-1)  # next 5 days, starting tomorrow
        for L in (7, 14, 28):
            tr = s.rolling(L).sum()
            m = tr.notna() & nxt.notna()
            hit = (np.sign(tr[m]) == np.sign(nxt[m])).mean()
            print(f"  {a:>6} - {c:<6} L={L:>2}d  corr {np.corrcoef(tr[m], nxt[m])[0, 1]:+.3f}  hit {hit:.0%}")
