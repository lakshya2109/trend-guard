"""Mid-contest decision: for the time left, how often does each book gain at least X? (all past starts)"""

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

H = int(sys.argv[1]) if len(sys.argv) > 1 else 260  # hours left in the contest
lab = Lab()
b = lab.cfg.book


def book(**upd) -> np.ndarray:
    mom = upd.pop("momentum", {})
    cfg = b.model_copy(update={**upd, "momentum": b.momentum.model_copy(update=mom)})
    return np.nan_to_num(sum(book_components(lab.close, lab.qv, cfg, True).values()))


V = {
    "current (4x blend)": np.nan_to_num(lab.book),
    "current at half risk": np.nan_to_num(lab.book) * 0.5,
    "v2 only (no team book)": book(v2_share=1.0, team_share=0.0),
    "momentum only, 6/side": book(v2_share=1.0, team_share=0.0, trend_share=0.0, momentum_share=1.0),
    "momentum only, 3/side": book(
        v2_share=1.0, team_share=0.0, trend_share=0.0, momentum_share=1.0, momentum={"k": 3, "buffer": 6}
    ),
    "trend only": book(v2_share=1.0, team_share=0.0, trend_share=1.0, momentum_share=0.0),
    "blend, momentum short 60%": book(momentum={"short_share": 0.6}),
}
ex = lab.cfg.execution
act = np.isin(lab.idx.hour, ex.decision_hours_utc)
starts = np.array(
    [i for i in range(lab.idx.get_loc(pd.Timestamp("2024-01-01 16:00", tz="UTC")), len(lab.idx) - H - 1, 24)]
)
yr = lab.idx[starts].year
rows = {}
for name, W in V.items():
    eq, _, _ = _simulate(
        np.ascontiguousarray(W),
        lab.px,
        act,
        starts,
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
    row = {f"P(>{g:+.0%})": np.mean(r > g) for g in (0.0, 0.02, 0.03, 0.04, 0.06)}
    row["P(<-5%)"] = np.mean(r < -0.05)
    row["median"] = np.median(r)
    row["P(>+3%) 2026"] = np.mean(r[yr == 2026] > 0.03)
    rows[name] = row
print(f"{len(starts)} past starts, {H} hours each (what is left of the contest), fees included")
with pd.option_context(
    "display.width", 250, "display.max_columns", 20, "display.float_format", "{:.2f}".format
):
    print(pd.DataFrame(rows).T)
