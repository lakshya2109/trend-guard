"""Bear-mode shorts vs the rebound longs: same regime, same objective (P gain > 4.5% in the ~225h left)."""

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

H = 225
lab = Lab()
b = lab.cfg.book
off = b.model_copy(update={"bear_mode": b.bear_mode.model_copy(update={"enabled": False})})
normal = np.nan_to_num(sum(B.book_components(lab.close, lab.qv, off, True).values()))
bear = B.bear_regime(lab.close, lab.qv, b.bear_mode)[:, None]
rebound = B.rebound_sleeve(lab.close, lab.qv, b.bear_mode)  # long the 10 worst 3-day performers
cols = list(lab.close.columns)
btc, eth = cols.index("BTC/USD"), cols.index("ETH/USD")
majors_short = np.zeros_like(normal)
majors_short[:, btc] = majors_short[:, eth] = -0.5
btc_short = np.zeros_like(normal)
btc_short[:, btc] = -1.0
V = {
    "old bot (no switch)": normal,
    "bear -> LONG 10 worst (deployed)": np.where(bear, rebound, normal),
    "bear -> SHORT 10 worst": np.where(bear, -rebound, normal),
    "bear -> SHORT BTC+ETH": np.where(bear, majors_short, normal),
    "bear -> long 10 worst 50% + short BTC 50%": np.where(bear, 0.5 * rebound + 0.5 * btc_short, normal),
}
px = lab.close.ffill(limit=3)
liq = (lab.qv.rolling(168).sum() / 7) > 5e6
ewc = np.log(px).diff().where(liq).mean(axis=1).fillna(0).cumsum().to_numpy()
ex = lab.cfg.execution
act = np.isin(lab.idx.hour, ex.decision_hours_utc)
first = lab.idx.get_loc(pd.Timestamp("2024-01-01 18:00", tz="UTC"))
starts = np.arange(first, len(lab.idx) - H - 1, 6)
prior = np.array([np.exp(ewc[s] - ewc[s - 120]) - 1 for s in starts])
m = prior < -0.08
yr = lab.idx[starts[m]].year
idx_m = starts[m]
ep = np.cumsum(np.r_[1, np.diff(idx_m) > 72])
rows, res = {}, {}
for vn, W in V.items():
    eq, _, _ = _simulate(
        np.ascontiguousarray(W),
        lab.px,
        act,
        idx_m,
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
    res[vn] = r
    rows[vn] = {
        "P>4.5%": np.mean(r > 0.045),
        "P>6%": np.mean(r > 0.06),
        "24-25": np.mean(r[yr < 2026] > 0.045),
        "2026": np.mean(r[yr == 2026] > 0.045),
        "median": np.median(r),
        "P<-8%": np.mean(r < -0.08),
    }
print(f"after the market fell > 8% in 5 days: {m.sum()} starts in {ep.max()} crash episodes, {H} hours each")
with pd.option_context(
    "display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format
):
    print(pd.DataFrame(rows).T)
wins = dict.fromkeys(V, 0)
for e in np.unique(ep):
    s = ep == e
    wins[max(V, key=lambda vn: np.mean(res[vn][s] > 0.045) + 1e-6 * np.median(res[vn][s]))] += 1
print("crash episodes where each was best at clearing +4.5%:", wins)
