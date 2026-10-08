"""Bear mode that follows the market's direction: short while it keeps falling, long while it rises.

Inside the same bear regime, every 6 h: if the liquid-coin market's trailing L-hour return is down, hold a
short book, else a long book. Short = the 10 worst 3-day coins (continuation) or an equal-weight basket of
liquid coins; long = the rebound book or the basket. Same 699 post-crash starts, 220 h left, live breaker
(0.20, starting 7% under the peak). Also split by what the market did over the 220 h that followed.

    python research/market_timing.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.breaker_now import sim  # noqa: E402
from research.harness import Lab  # noqa: E402
from trend_guard.strategy import book as B  # noqa: E402

H, DD0, EVERY = 220, 0.07, 6
lab = Lab()
b, bm, br, ex = lab.cfg.book, lab.cfg.book.bear_mode, lab.cfg.breaker, lab.cfg.execution
off = b.model_copy(update={"bear_mode": bm.model_copy(update={"enabled": False})})
normal = np.nan_to_num(sum(B.book_components(lab.close, lab.qv, off, True).values()))
bear = B.bear_regime(lab.close, lab.qv, bm)[:, None]
rebound = np.nan_to_num(B.rebound_sleeve(lab.close, lab.qv, bm))
px = lab.close.ffill(limit=3)
liq = (lab.qv.rolling(168).sum() / 7) > bm.min_liquidity_usd
ok = (
    liq.to_numpy()
    & (lab.close.notna().cumsum().to_numpy() >= bm.rebound_min_age_h)
    & np.isfinite(px.to_numpy())
)
for pair in bm.rebound_exclude:
    ok[:, lab.cols.index(pair)] = False
basket = ok / np.maximum(ok.sum(axis=1, keepdims=True), 1)  # equal weight, every liquid coin, gross 1
ewc = np.log(px).diff().where(liq).mean(axis=1).fillna(0).cumsum().to_numpy()


def market_up(look: int) -> np.ndarray:
    """True where the market's trailing `look`-hour return is positive, decided every EVERY hours."""
    up = np.r_[np.zeros(look), ewc[look:] - ewc[:-look]] > 0
    held = up.copy()
    for t in range(1, len(up)):
        held[t] = up[t] if t % EVERY == 0 else held[t - 1]
    return held[:, None]


up24, up72 = market_up(24), market_up(72)
V = {
    "rebound (live)": np.where(bear, rebound, normal),
    "follow market 24h: rebound / short 10 worst": np.where(bear, np.where(up24, rebound, -rebound), normal),
    "follow market 72h: rebound / short 10 worst": np.where(bear, np.where(up72, rebound, -rebound), normal),
    "follow market 24h: long / short basket": np.where(bear, np.where(up24, basket, -basket), normal),
    "follow market 72h: long / short basket": np.where(bear, np.where(up72, basket, -basket), normal),
    "short 10 worst only if 24h AND 72h down": np.where(
        bear, np.where(~up24 & ~up72, -rebound, rebound), normal
    ),
    "old bot (its own momentum shorts)": normal,
}
act = np.isin(lab.idx.hour, ex.decision_hours_utc)
first = lab.idx.get_loc(pd.Timestamp("2024-01-01 18:00", tz="UTC"))
starts = np.arange(first, len(lab.idx) - H - 1, 6)
prior = np.array([np.exp(ewc[s] - ewc[s - 120]) - 1 for s in starts])
idx_m = starts[prior < bm.enter]
fwd = np.array([np.exp(ewc[s + H] - ewc[s]) - 1 for s in idx_m])  # what the average coin did next
bins = {
    "kept falling": fwd < -0.05,
    "drifted down": (fwd >= -0.05) & (fwd < -0.02),
    "sideways": (fwd >= -0.02) & (fwd <= 0.02),
    "drifted up": (fwd > 0.02) & (fwd <= 0.05),
    "bounced": fwd > 0.05,
}
rows = []
for vn, W in V.items():
    r = (
        sim(
            np.ascontiguousarray(W),
            lab.px,
            act,
            idx_m,
            H,
            lab.cost,
            ex.min_trade_weight,
            ex.min_trade_frac,
            DD0,
            br.dd_level,
            br.dd_mult,
            br.dd_recover,
        )
        - 1.0
    )
    row = {
        "bear book": vn,
        "P>4.5%": f"{np.mean(r > 0.045):.0%}",
        "median": f"{np.median(r):+.1%}",
        "P<-8%": f"{np.mean(r < -0.08):.0%}",
    }
    for bn, m in bins.items():
        row[bn] = f"{np.mean(r[m] > 0.045):.0%}"
    rows.append(row)
print(f"{len(idx_m)} post-crash starts, {H} h left, breaker {br.dd_level} from {DD0:.0%} under the peak")
print("market next 220 h: " + ", ".join(f"{bn} {m.mean():.0%}" for bn, m in bins.items()))
print("columns after P<-8%: P>4.5% within each of those outcomes")
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(pd.DataFrame(rows).to_string(index=False))
