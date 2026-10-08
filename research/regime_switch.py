"""Regime switch: run a bear-mode book while the market is falling, the normal book otherwise.

Regime (causal, hourly): bear while the equal-weight liquid-coin market's trailing `look`-hour return is
below `enter`; back to normal once it is above `exit_` (hysteresis so it does not flip every hour).
Scored on (a) every past start and (b) starts right after a crash like this one, for the time left.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from backtest.contest import _simulate  # noqa: E402
from research.harness import Lab  # noqa: E402
from trend_guard.strategy.book import book_components  # noqa: E402

H = int(sys.argv[1]) if len(sys.argv) > 1 else 230
lab = Lab()
b = lab.cfg.book
px = lab.close.ffill(limit=3)
liq = (lab.qv.rolling(168).sum() / 7) > 5e6
ewc = np.log(px).diff().where(liq).mean(axis=1).fillna(0).cumsum().to_numpy()


def book(**upd) -> np.ndarray:
    mom = upd.pop("momentum", {})
    cfg = b.model_copy(update={**upd, "momentum": b.momentum.model_copy(update=mom)})
    return np.nan_to_num(sum(book_components(lab.close, lab.qv, cfg, True).values()))


def regime(look: int, enter: float, exit_: float) -> np.ndarray:
    bear = np.zeros(len(ewc), bool)
    on = False
    for t in range(look, len(ewc)):
        r = np.exp(ewc[t] - ewc[t - look]) - 1
        on = (r < enter) if not on else (r < exit_)
        bear[t] = on
    return bear


normal = np.nan_to_num(lab.book)
longs_cut = np.where(normal > 0, 0.0, normal)
longs_half = np.where(normal > 0, normal * 0.5, normal)
net_short = book(
    v2_share=1.0, team_share=0.0, trend_share=0.0, momentum_share=1.0, momentum={"short_share": 0.7}
)
mom_only = book(v2_share=1.0, team_share=0.0, trend_share=0.0, momentum_share=1.0)

V = {"current (no switch)": normal}
for look, enter, exit_ in ((72, -0.05, 0.0), (120, -0.08, -0.02)):
    bear = regime(look, enter, exit_)[:, None]
    tag = f"{look // 24}d<{enter:.0%}"
    V[f"bear->cut longs [{tag}]"] = np.where(bear, longs_cut, normal)
    V[f"bear->half longs [{tag}]"] = np.where(bear, longs_half, normal)
    V[f"bear->net short [{tag}]"] = np.where(bear, net_short, normal)
    V[f"bear->momentum only [{tag}]"] = np.where(bear, mom_only, normal)
    print(f"regime {tag}: bear {bear.mean():.0%} of hours since 2023; bear now? {bool(bear[-1, 0])}")

ex = lab.cfg.execution
act = np.isin(lab.idx.hour, ex.decision_hours_utc)
first = lab.idx.get_loc(pd.Timestamp("2024-01-01 18:00", tz="UTC"))
starts = np.arange(first, len(lab.idx) - H - 1, 6)
prior = np.array([np.exp(ewc[s] - ewc[s - 120]) - 1 for s in starts])
yr = lab.idx[starts].year
for name, m in {
    "all starts": np.ones(len(starts), bool),
    "after avg coin fell > 8% in 5 days": prior < -0.08,
}.items():
    rows = {}
    for vn, W in V.items():
        eq, _, _ = _simulate(
            np.ascontiguousarray(W),
            lab.px,
            act,
            starts[m],
            H,
            lab.cost,
            ex.min_trade_weight,
            ex.min_trade_frac,
            1.0,
            9.0,
            1.0,
            9.0,
        )
        r = eq[:, -1] - 1
        rows[vn] = {
            "median": np.median(r),
            "P(>+4.5%)": np.mean(r > 0.045),
            "P(>+6%)": np.mean(r > 0.06),
            "P(<-5%)": np.mean(r < -0.05),
            "P(>+4.5%) 2026": np.mean(r[yr[m] == 2026] > 0.045),
        }
    print(f"\n{name}: {m.sum()} starts, {H} hours each")
    with pd.option_context(
        "display.width", 250, "display.max_columns", 20, "display.float_format", "{:+.3f}".format
    ):
        print(pd.DataFrame(rows).T)
