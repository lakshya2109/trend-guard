"""Faster adaptation: re-rank momentum / rebalance trend every `cad` hours instead of once a day,
always or only in a bear regime, alone or with momentum-only. Objective: P(gain > gap) for the time left."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from backtest.contest import _simulate  # noqa: E402
from research.harness import Lab  # noqa: E402
from trend_guard.strategy import book as B  # noqa: E402

H = int(sys.argv[1]) if len(sys.argv) > 1 else 230
lab = Lab()
b = lab.cfg.book
px = lab.close.ffill(limit=3)
liq = (lab.qv.rolling(168).sum() / 7) > 5e6
ewc = np.log(px).diff().where(liq).mean(axis=1).fillna(0).cumsum().to_numpy()
orig_features = B.features


class CadIndex:
    """Stands in for the DatetimeIndex: .hour hits the sleeve's decision hour every `cad` hours."""

    def __init__(self, idx, cad):
        self.idx, self.cad = idx, cad

    @property
    def hour(self):
        h = self.idx.hour.to_numpy()
        return np.where(h % self.cad == 0, -1, -2)  # -1 marks a decision hour (configs below use -1)


def book(cad: int | None = None, **upd) -> np.ndarray:
    mom = upd.pop("momentum", {})
    tr = upd.pop("trend", {})
    if cad:
        mom = {**mom, "select_hour_utc": -1}
        tr = {**tr, "rebalance_hour_utc": -1}
        B.features = lambda *a, **k: dataclasses.replace(orig_features(*a, **k), index=CadIndex(lab.idx, cad))
    try:
        cfg = b.model_copy(
            update={
                **upd,
                "momentum": b.momentum.model_copy(update=mom),
                "trend": b.trend.model_copy(update=tr),
            }
        )
        return np.nan_to_num(sum(B.book_components(lab.close, lab.qv, cfg, True).values()))
    finally:
        B.features = orig_features


def regime(look=120, enter=-0.08, exit_=-0.02) -> np.ndarray:
    bear, on = np.zeros(len(ewc), bool), False
    for t in range(look, len(ewc)):
        r = np.exp(ewc[t] - ewc[t - look]) - 1
        on = (r < enter) if not on else (r < exit_)
        bear[t] = on
    return bear[:, None]


normal = np.nan_to_num(lab.book)
fast6 = book(cad=6)
fast12 = book(cad=12)
mom_fast6 = book(cad=6, v2_share=1.0, team_share=0.0, trend_share=0.0, momentum_share=1.0)
mom_only = book(v2_share=1.0, team_share=0.0, trend_share=0.0, momentum_share=1.0)
bear = regime()
V = {
    "current": normal,
    "fast 12h always": fast12,
    "fast 6h always": fast6,
    "bear -> fast 6h": np.where(bear, fast6, normal),
    "bear -> momentum only": np.where(bear, mom_only, normal),
    "bear -> momentum only, fast 6h": np.where(bear, mom_fast6, normal),
}
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
            "P(>+3%)": np.mean(r > 0.03),
            "P(>+4.5%)": np.mean(r > 0.045),
            "P(>+6%)": np.mean(r > 0.06),
            "P(>+4.5%) 2026": np.mean(r[yr[m] == 2026] > 0.045),
            "P(>+4.5%) 24-25": np.mean(r[yr[m] < 2026] > 0.045),
            "median": np.median(r),
        }
    print(f"\n{name}: {m.sum()} starts, {H} hours each")
    with pd.option_context(
        "display.width", 250, "display.max_columns", 20, "display.float_format", "{:+.3f}".format
    ):
        print(pd.DataFrame(rows).T)
