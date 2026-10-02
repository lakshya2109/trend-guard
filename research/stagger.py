"""Staggered sub-books: average each v2 sleeve over several decision hours (removes timing luck)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.harness import Lab  # noqa: E402
from trend_guard.strategy.book import momentum_sleeve, trend_sleeve  # noqa: E402

lab = Lab()
b = lab.cfg.book
mult = b.risk_multiplier
team = lab.comp["team_trend"] + lab.comp["team_momentum"]


def assemble(tr: np.ndarray, mo: np.ndarray) -> np.ndarray:
    """Same capital shares and risk dial as book_components, then the per-hour gross cap of 1."""
    W = team / _cap + mult * b.v2_share * (b.trend_share * tr + b.momentum_share * mo)
    g = np.abs(W).sum(1, keepdims=True)
    return W / np.maximum(g, b.max_gross)


# team components in lab.comp already include the hourly cap scale; undo it so we can re-cap
raw_v2 = (
    mult
    * b.v2_share
    * (
        b.trend_share * trend_sleeve(lab.close, lab.qv, b.trend)
        + b.momentum_share * momentum_sleeve(lab.close, lab.qv, b.momentum, True)
    )
)
raw_team_cap = np.abs(lab.comp["v2_trend"] + lab.comp["v2_momentum"]).sum(1)
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(np.abs(raw_v2).sum(1) > 0, raw_team_cap / np.abs(raw_v2).sum(1), 1.0)
_cap = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)[:, None]

hours = [0, 6, 12, 18]
tr = {h: trend_sleeve(lab.close, lab.qv, b.trend.model_copy(update={"rebalance_hour_utc": h})) for h in hours}
mo = {
    h: momentum_sleeve(lab.close, lab.qv, b.momentum.model_copy(update={"select_hour_utc": h}), True)
    for h in hours
}
books = {
    "book (live)": lab.book,
    "rebuilt live": assemble(tr[12], mo[0]),
    "stagger momentum": assemble(tr[12], np.mean([mo[h] for h in hours], axis=0)),
    "stagger trend": assemble(np.mean([tr[h] for h in hours], axis=0), mo[0]),
    "stagger both": assemble(
        np.mean([tr[h] for h in hours], axis=0), np.mean([mo[h] for h in hours], axis=0)
    ),
    "stagger both 0/12": assemble((tr[0] + tr[12]) / 2, (mo[0] + mo[12]) / 2),
}
lab.report(books)
