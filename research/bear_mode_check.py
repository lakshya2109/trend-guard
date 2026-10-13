"""Re-check of the bear-mode rebound switch (rounds 19-20) on the continuous run, not just post-crash tails.

Rounds 19-20 adopted it on P(+4.5% in the 225 h after a crash), an objective that rewards variance. Here:
  1. continuous daily returns from 2024, live book (bear mode on) vs the same book without it, paired
     stationary bootstrap of the Sharpe difference (research/paired.py) and returns by year;
  2. what the switch did on the days it was on;
  3. after past crashes: MEAN as well as median of the next 225 h (the median hid the left tail);
  4. the contest's remaining ~114 h from every start and from starts like 13 Oct 2026 (market down > 5%
     over the previous 9 days), with the live breaker.

    python research/bear_mode_check.py
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
from research.harness import Lab  # noqa: E402
from research.paired import paired  # noqa: E402
from trend_guard.strategy import book as B  # noqa: E402

lab = Lab()
b, bm, br, ex = lab.cfg.book, lab.cfg.book.bear_mode, lab.cfg.breaker, lab.cfg.execution
off = b.model_copy(update={"bear_mode": bm.model_copy(update={"enabled": False})})
normal = np.nan_to_num(sum(B.book_components(lab.close, lab.qv, off, True).values()))
live = np.nan_to_num(lab.book)
bear = B.bear_regime(lab.close, lab.qv, bm)
books = {"bear mode on (live)": live, "bear mode off": normal, "bear mode off, half size": 0.5 * normal}

# 1. continuous run
D = pd.DataFrame({k: lab.daily(W) for k, W in books.items()})
print(f"continuous daily returns {D.index[0]:%Y-%m-%d} -> {D.index[-1]:%Y-%m-%d}, after costs")
rows = {}
for k in D:
    r = D[k]
    row = {f"ret {y}": (1 + r[r.index.year == y]).prod() - 1 for y in (2024, 2025, 2026)}
    row |= {
        f"SR {y}": r[r.index.year == y].mean() / r[r.index.year == y].std() * np.sqrt(365)
        for y in (2024, 2025, 2026)
    }
    eq = (1 + r).cumprod()
    row["maxDD"] = float((1 - eq / eq.cummax()).max())
    rows[k] = row
with pd.option_context("display.width", 220, "display.float_format", "{:+.2f}".format):
    print(pd.DataFrame(rows).T)
p = paired(D["bear mode on (live)"], D["bear mode off"])
print(
    "paired bootstrap, off vs on: dSharpe {:+.2f}  P(off better) {:.1%}  2024-25 {:+.2f}  2026 {:+.2f}  "
    "d return/yr {:+.1%}".format(p["dSR"], p["P(dSR>0)"], p["dSR 2024-25"], p["dSR 2026"], p["d_ann_ret"])
)

# 2. the days it was on
on_share = pd.Series(bear, index=lab.idx).resample("24h", offset="12h").mean()  # contest days, 12:00 UTC
on_day = on_share.reindex(D.index, method="ffill") > 0.5
diff = D["bear mode on (live)"] - D["bear mode off"]
t_on = diff[on_day].mean() / diff[on_day].std() * np.sqrt(on_day.sum())
print(
    f"\nbear mode on {on_day.mean():.0%} of days; on those days the switch changed the daily return by "
    f"{diff[on_day].mean():+.2%} on average (t {t_on:+.1f})"
)

# 3/4. windows, live breaker
px = lab.close.ffill(limit=3)
liq = (lab.qv.rolling(168).sum() / 7) > bm.min_liquidity_usd
ewc = np.log(px).diff().where(liq).mean(axis=1).fillna(0).cumsum().to_numpy()
act = np.isin(lab.idx.hour, ex.decision_hours_utc)
first = lab.idx.get_loc(pd.Timestamp("2024-01-01 18:00", tz="UTC"))


def outcomes(H: int, starts: np.ndarray, dd0: float) -> dict[str, np.ndarray]:
    return {
        k: sim(
            np.ascontiguousarray(W),
            lab.px,
            act,
            starts,
            H,
            lab.cost,
            ex.min_trade_weight,
            ex.min_trade_frac,
            dd0,
            br.dd_level,
            br.dd_mult,
            br.dd_recover,
        )
        - 1.0
        for k, W in books.items()
    }


starts = np.arange(first, len(lab.idx) - 226, 6)
prior5 = np.array([np.exp(ewc[s] - ewc[s - 120]) - 1 for s in starts])
res = outcomes(225, starts[prior5 < bm.enter], 0.0)
print(f"\nafter past crashes ({(prior5 < bm.enter).sum()} starts), next 225 h:")
for k, r in res.items():
    print(f"  {k:<26} mean {r.mean():+.2%}  median {np.median(r):+.2%}  P(< -8%) {np.mean(r < -0.08):.0%}")

H = 114
starts = np.arange(first, len(lab.idx) - H - 1, 6)
prior9 = np.array([np.exp(ewc[s] - ewc[s - 216]) - 1 for s in starts])
for name, m in (
    ("every start", np.ones(len(starts), bool)),
    ("market down > 5% over 9 days", prior9 < -0.05),
):
    res = outcomes(H, starts[m], 0.06)
    print(f"\n{H} h left, {name} ({m.sum()} starts), from 6% under the peak:")
    for k, r in res.items():
        print(
            f"  {k:<26} P(> +2.5%) {np.mean(r > 0.025):.0%}  P(> +3.5%) {np.mean(r > 0.035):.0%}  "
            f"mean {r.mean():+.2%}  median {np.median(r):+.2%}  P(< -3%) {np.mean(r < -0.03):.0%}"
        )
