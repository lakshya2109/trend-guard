"""Bear-mode candidates aimed at the upside: oversold-rebound longs, BTC/ETH beta, concentrated fast momentum.

Same regime as the deployed switch (liquid-coin market < -8% over 5 days, leave above -2%). Objective:
P(return over the ~230 hours left > +4.5% / +6%), after past crashes and over all starts, by period.
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

H = int(sys.argv[1]) if len(sys.argv) > 1 else 230
lab = Lab()  # config now has bear_mode enabled: lab.book IS the deployed switch
b = lab.cfg.book
off = b.model_copy(update={"bear_mode": b.bear_mode.model_copy(update={"enabled": False})})
px = lab.close.ffill(limit=3)
liq = (lab.qv.rolling(168).sum() / 7) > 5e6
ewc = np.log(px).diff().where(liq).mean(axis=1).fillna(0).cumsum().to_numpy()
bear = B.bear_regime(lab.close, lab.qv, b.bear_mode)[:, None]
normal = np.nan_to_num(sum(B.book_components(lab.close, lab.qv, off, True).values()))
T, N = normal.shape
cols = list(lab.close.columns)
age = lab.close.notna().cumsum().to_numpy()
ok = liq.to_numpy() & (age >= 720) & np.isfinite(px.to_numpy())
ok[:, cols.index("PAXG/USD")] = False
lp = np.log(px.to_numpy())
vol = pd.DataFrame(np.diff(lp, axis=0, prepend=np.nan)).rolling(
    168, min_periods=72
).std().to_numpy() * np.sqrt(24)


def rebound(k: int, look: int, every: int, inv_vol: bool) -> np.ndarray:
    """Long the k worst `look`-hour performers among liquid coins, re-picked every `every` hours, gross 1."""
    W, cur = np.zeros((T, N)), np.zeros(N)
    ret = np.full((T, N), np.nan)
    ret[look:] = lp[look:] - lp[:-look]
    for t in range(T):
        if t % every == 0:
            r = np.where(ok[t] & np.isfinite(ret[t]), ret[t], np.inf)
            pick = np.argsort(r)[:k]
            pick = pick[np.isfinite(r[pick])]
            cur = np.zeros(N)
            if len(pick):
                w = 1 / np.nan_to_num(vol[t, pick], nan=0.05) if inv_vol else np.ones(len(pick))
                cur[pick] = w / w.sum()
        W[t] = cur
    return W


def majors() -> np.ndarray:
    W = np.zeros((T, N))
    W[:, cols.index("BTC/USD")] = 0.5
    W[:, cols.index("ETH/USD")] = 0.5
    return W


mom_fast = np.nan_to_num(
    sum(
        B.book_components(
            lab.close,
            lab.qv,
            off.model_copy(
                update={
                    "v2_share": 1.0,
                    "team_share": 0.0,
                    "trend_share": 0.0,
                    "momentum_share": 1.0,
                    "momentum": off.momentum.model_copy(update={"select_every_h": 6}),
                }
            ),
            True,
        ).values()
    )
)
mom_fast3 = np.nan_to_num(
    sum(
        B.book_components(
            lab.close,
            lab.qv,
            off.model_copy(
                update={
                    "v2_share": 1.0,
                    "team_share": 0.0,
                    "trend_share": 0.0,
                    "momentum_share": 1.0,
                    "momentum": off.momentum.model_copy(update={"select_every_h": 6, "k": 3, "buffer": 6}),
                }
            ),
            True,
        ).values()
    )
)
rb5 = rebound(5, 120, 24, False)
rb10 = rebound(10, 120, 24, False)
rb5v = rebound(5, 120, 24, True)
rb10_72 = rebound(10, 72, 24, False)
V = {
    "no switch (old bot)": normal,
    "DEPLOY-READY: bear -> fast momentum": np.nan_to_num(lab.book),
    "bear -> fast momentum 3/side": np.where(bear, mom_fast3, normal),
    "bear -> BTC+ETH 50/50": np.where(bear, majors(), normal),
    "bear -> rebound 5 worst (5d)": np.where(bear, rb5, normal),
    "bear -> rebound 10 worst (5d)": np.where(bear, rb10, normal),
    "bear -> rebound 5 worst, inv-vol": np.where(bear, rb5v, normal),
    "bear -> rebound 10 worst (3d)": np.where(bear, rb10_72, normal),
    "bear -> 50% fast momentum + 50% rebound 10": np.where(bear, 0.5 * mom_fast + 0.5 * rb10, normal),
}
ex = lab.cfg.execution
act = np.isin(lab.idx.hour, ex.decision_hours_utc)
first = lab.idx.get_loc(pd.Timestamp("2024-01-01 18:00", tz="UTC"))
starts = np.arange(first, len(lab.idx) - H - 1, 6)
prior = np.array([np.exp(ewc[s] - ewc[s - 120]) - 1 for s in starts])
yr = lab.idx[starts].year
sets = {"all starts": np.ones(len(starts), bool), "after avg coin fell > 8% in 5 days": prior < -0.08}
for name, m in sets.items():
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
        y = yr[m]
        rows[vn] = {
            "P>4.5%": np.mean(r > 0.045),
            "P>6%": np.mean(r > 0.06),
            "P>8%": np.mean(r > 0.08),
            "P>4.5% 24-25": np.mean(r[y < 2026] > 0.045),
            "P>4.5% 2026": np.mean(r[y == 2026] > 0.045),
            "median": np.median(r),
            "P<-8%": np.mean(r < -0.08),
        }
    print(f"\n{name}: {m.sum()} starts, {H} hours each")
    with pd.option_context(
        "display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format
    ):
        print(pd.DataFrame(rows).T)


# ---- episode view: post-crash starts grouped into separate crash episodes (gaps > 72h start a new one)
m = prior < -0.08
idx_m = starts[m]
ep = np.cumsum(np.r_[1, np.diff(idx_m) > 72])
keep = [
    "no switch (old bot)",
    "DEPLOY-READY: bear -> fast momentum",
    "bear -> rebound 10 worst (3d)",
    "bear -> rebound 10 worst (5d)",
]
res = {}
for vn in keep:
    eq, _, _ = _simulate(
        np.ascontiguousarray(V[vn]),
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
    res[vn] = eq[:, -1] - 1
rows = []
for e in np.unique(ep):
    s = ep == e
    row = {"episode start": f"{lab.idx[idx_m[s][0]]:%Y-%m-%d}", "starts": int(s.sum())}
    for vn in keep:
        row[vn.split(":")[0].replace("bear -> ", "")[:22]] = (
            f"{np.median(res[vn][s]):+.1%} ({np.mean(res[vn][s] > 0.045):.0%})"
        )
    rows.append(row)
print("\nper crash episode: median return over the next 225h (share of starts that gained > 4.5%)")
with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 100):
    print(pd.DataFrame(rows).to_string(index=False))
wins = {vn: 0 for vn in keep}
for e in np.unique(ep):
    s = ep == e
    best = max(keep, key=lambda vn: np.mean(res[vn][s] > 0.045) + 1e-6 * np.median(res[vn][s]))
    wins[best] += 1
print("episodes where each was best at clearing +4.5%:", wins)
