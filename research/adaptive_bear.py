"""Adaptive bear book: inside the same bear regime, hold whichever crash book is working NOW.

Candidates (all gross 1): rebound longs (deployed on this branch), shorts on the same 10 worst coins
(continuation), fast momentum long/short, short BTC+ETH, cash. Adaptive designs, fixed before any result:
  FTL L/menu    every day at 00:00 UTC hold the candidate with the best trailing L-hour return (after costs)
  gate 24h      rebound longs once the market's own 24h return turned positive, else an alternative
Objective as rounds 19-20: P(return over the hours left > +4.5%), after past crashes, grouped by episode.

    python research/adaptive_bear.py [hours_left]
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
from trend_guard.strategy import book as B  # noqa: E402

H = int(sys.argv[1]) if len(sys.argv) > 1 else 220
lab = Lab()
b = lab.cfg.book
bm = b.bear_mode
off = b.model_copy(update={"bear_mode": bm.model_copy(update={"enabled": False})})
normal = np.nan_to_num(sum(B.book_components(lab.close, lab.qv, off, True).values()))
bear = B.bear_regime(lab.close, lab.qv, bm)[:, None]
T, N = normal.shape
cols = list(lab.close.columns)
rebound = np.nan_to_num(B.rebound_sleeve(lab.close, lab.qv, bm))
mom_cfg = off.model_copy(
    update={
        "v2_share": 1.0,
        "team_share": 0.0,
        "trend_share": 0.0,
        "momentum_share": 1.0,
        "momentum": off.momentum.model_copy(update={"select_every_h": bm.select_every_h}),
    }
)
momentum = np.nan_to_num(sum(B.book_components(lab.close, lab.qv, mom_cfg, True).values()))
majors_short = np.zeros((T, N))
majors_short[:, cols.index("BTC/USD")] = majors_short[:, cols.index("ETH/USD")] = -0.5
CAND = {
    "rebound": rebound,
    "short10": -rebound,
    "momentum": momentum,
    "shortBTCETH": majors_short,
    "cash": np.zeros((T, N)),
}

# realised hourly return of each candidate at bar t (weights held from t-1), minus turnover cost
ret = np.nan_to_num(lab.px[1:] / lab.px[:-1] - 1.0)
cand_r = {}
for k, W in CAND.items():
    r = np.zeros(T)
    r[1:] = (W[:-1] * ret).sum(axis=1)
    r[1:] -= lab.cost * np.abs(np.diff(W, axis=0)).sum(axis=1)
    cand_r[k] = np.cumsum(r)
sel_hour = lab.idx.hour == bm.rebound_hour_utc


def ftl(menu: list[str], look: int) -> np.ndarray:
    """At each 00:00 UTC hold the menu candidate with the best trailing `look`-hour return."""
    R = np.stack([cand_r[k] for k in menu])  # (n, T) cumulative
    out, cur = np.zeros((T, N)), 0
    for t in range(T):
        if sel_hour[t] and t >= look:
            cur = int(np.argmax(R[:, t] - R[:, t - look]))
        out[t] = CAND[menu[cur]][t]
    return out


px = lab.close.ffill(limit=3)
liq = (lab.qv.rolling(168).sum() / 7) > bm.min_liquidity_usd
ewc = np.log(px).diff().where(liq).mean(axis=1).fillna(0).cumsum().to_numpy()
mkt24 = np.r_[np.zeros(24), ewc[24:] - ewc[:-24]][:, None]  # liquid-coin market, last 24h log return
V = {
    "old bot (no switch)": normal,
    "bear -> rebound (branch)": np.where(bear, rebound, normal),
    "bear -> FTL 48h {rebound, short10}": np.where(bear, ftl(["rebound", "short10"], 48), normal),
    "bear -> FTL 72h {rebound, short10}": np.where(bear, ftl(["rebound", "short10"], 72), normal),
    "bear -> FTL 48h all 5": np.where(bear, ftl(list(CAND), 48), normal),
    "bear -> FTL 72h all 5": np.where(bear, ftl(list(CAND), 72), normal),
    "bear -> gate 24h: rebound else short10": np.where(bear, np.where(mkt24 > 0, rebound, -rebound), normal),
    "bear -> gate 24h: rebound else cash": np.where(bear, np.where(mkt24 > 0, rebound, 0.0), normal),
    "bear -> gate 24h: rebound else momentum": np.where(bear, np.where(mkt24 > 0, rebound, momentum), normal),
}

ex = lab.cfg.execution
act = np.isin(lab.idx.hour, ex.decision_hours_utc)
first = lab.idx.get_loc(pd.Timestamp("2024-01-01 18:00", tz="UTC"))
starts = np.arange(first, len(lab.idx) - H - 1, 6)
prior = np.array([np.exp(ewc[s] - ewc[s - 120]) - 1 for s in starts])
m = prior < bm.enter
idx_m = starts[m]
yr = lab.idx[idx_m].year
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
print("crash episodes where each was best at clearing +4.5%:")
for vn, w in wins.items():
    print(f"  {w:3d}  {vn}")
# episode-level P>4.5%: each crash episode counts once (no long episode dominates)
print("episode-weighted P>4.5%:")
for vn in V:
    print(f"  {np.mean([np.mean(res[vn][ep == e] > 0.045) for e in np.unique(ep)]):.3f}  {vn}")

# the current crash: what each candidate did over the last 24 / 72 hours, and the regime now
print(
    f"\nnow ({lab.idx[-1]:%Y-%m-%d %H:00} UTC): bear regime {'ON' if bear[-1, 0] else 'off'}, "
    f"market 5d {np.exp(ewc[-1] - ewc[-121]) - 1:+.1%}, 24h {np.exp(mkt24[-1, 0]) - 1:+.1%}"
)
for k in CAND:
    d24, d72 = cand_r[k][-1] - cand_r[k][-25], cand_r[k][-1] - cand_r[k][-73]
    print(f"  {k:<12} last 24h {d24:+.2%}   last 72h {d72:+.2%}")
