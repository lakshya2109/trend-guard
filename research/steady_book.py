"""Steadier versions of the live normal book: is the swinging the price of its return, or fixable?

Live (13 Oct 2026): the normal book at 4x is ~100% invested and ~60% net long altcoins, so it moves with the
altcoin market (-8% in the contest). Pre-declared variants, all with the same costs and trade rule:
  live 4x           the book as deployed (bear mode off)
  2x / 1x           the whole book scaled down (the risk dial, STRATEGY.md 3c)
  hedge 50% / 100%  short an equal-weight basket of liquid coins against 50% / 100% of the book's net
                    exposure, then rescale so long notional + short collateral <= equity (no leverage)
  momentum only     the v2 momentum long/short sleeve alone, at the gross the live book gives it
Each is scored on: the continuous run from 2024 (Sharpe, daily vol, worst day, max drawdown, return), all
contest windows (composite score, P(fortnight < 0)), a replay of this contest (4 Oct 12:00 -> now) and
the hours left from starts like now (market down > 5% over 9 days, 6% under the peak, live breaker).

    python research/steady_book.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from backtest.contest import contest_score, simulate  # noqa: E402
from research.breaker_now import sim  # noqa: E402
from research.harness import Lab  # noqa: E402
from trend_guard.strategy import book as B  # noqa: E402

H_LEFT = 114
CONTEST = "2026-10-04 12:00"
lab = Lab()
b, bm, br, ex = lab.cfg.book, lab.cfg.book.bear_mode, lab.cfg.breaker, lab.cfg.execution
assert not bm.enabled, "this compares versions of the normal book (bear mode off)"
live = np.nan_to_num(lab.book)
px = lab.close.ffill(limit=3)
liq = (lab.qv.rolling(168).sum() / 7) > bm.min_liquidity_usd
ok = (
    liq.to_numpy()
    & (lab.close.notna().cumsum().to_numpy() >= bm.rebound_min_age_h)
    & np.isfinite(px.to_numpy())
)
for pair in bm.rebound_exclude:
    ok[:, lab.cols.index(pair)] = False
basket = ok / np.maximum(ok.sum(axis=1, keepdims=True), 1)


def hedged(W: np.ndarray, frac: float) -> np.ndarray:
    """Short `frac` of the net exposure as an equal-weight basket, netted per pair, gross capped at 1."""
    out = W - frac * W.sum(axis=1, keepdims=True) * basket
    gross = np.abs(out).sum(axis=1, keepdims=True)
    return out / np.maximum(gross, 1.0)


mom_cfg = b.model_copy(update={"trend_share": 0.0, "team_share": 0.0, "v2_share": 1.0, "momentum_share": 1.0})
mom = np.nan_to_num(sum(B.book_components(lab.close, lab.qv, mom_cfg, True).values()))
V = {
    "live 4x": live,
    "2x": 0.5 * live,
    "1x": 0.25 * live,
    "hedge 50% of net": hedged(live, 0.5),
    "hedge 100% of net": hedged(live, 1.0),
    "momentum only": mom,
}

ewc = np.log(px).diff().where(liq).mean(axis=1).fillna(0).cumsum().to_numpy()
act = np.isin(lab.idx.hour, ex.decision_hours_utc)
first = lab.idx.get_loc(pd.Timestamp("2024-01-01 18:00", tz="UTC"))
starts = np.arange(first, len(lab.idx) - H_LEFT - 1, 6)
prior9 = np.array([np.exp(ewc[s] - ewc[s - 216]) - 1 for s in starts])
now_like = starts[prior9 < -0.05]
bt = lab.cfg.backtest.model_copy(update={"fee": lab.cost, "slippage": 0.0})


def row_metrics(name: str, W: np.ndarray) -> dict:
    """Every column of the table for one book."""
    d = lab.daily(W)
    eq = (1 + d).cumprod()
    y = d.index.year
    row = {
        "book": name,
        "net": float(W[lab.idx >= pd.Timestamp("2024-01-01", tz="UTC")].sum(axis=1).mean()),
        "SR 24-25": d[y < 2026].mean() / d[y < 2026].std() * np.sqrt(365),
        "SR 2026": d[y == 2026].mean() / d[y == 2026].std() * np.sqrt(365),
        "ret/yr": (1 + d.mean()) ** 365 - 1,
        "day vol": d.std(),
        "worst day": d.min(),
        "maxDD": float((1 - eq / eq.cummax()).max()),
    }
    weq, _ = simulate(np.nan_to_num(W), lab.close, lab.starts, bt, ex, br)
    fn = weq[:, -1] / weq[:, 0] - 1
    row["score med"] = float(np.median(contest_score(weq)))
    row["P(fn<0)"] = float(np.mean(fn < 0))
    row["worst fn"] = float(fn.min())
    c = lab.continuous(W, CONTEST)
    cd = c.iloc[::24].pct_change().dropna()
    row["contest"] = float(c.iloc[-1] / c.iloc[0] - 1)
    row["contest day vol"] = float(cd.std())
    r = (
        sim(
            np.ascontiguousarray(W),
            lab.px,
            act,
            now_like,
            H_LEFT,
            lab.cost,
            ex.min_trade_weight,
            ex.min_trade_frac,
            0.06,
            br.dd_level,
            br.dd_mult,
            br.dd_recover,
        )
        - 1.0
    )
    row["left P>2.5%"] = float(np.mean(r > 0.025))
    row["left mean"] = float(r.mean())
    row["left P<-3%"] = float(np.mean(r < -0.03))
    return row


def main() -> None:
    rows = [row_metrics(name, W) for name, W in V.items()]

    df = pd.DataFrame(rows).set_index("book")
    pct = [
        "ret/yr",
        "day vol",
        "worst day",
        "maxDD",
        "P(fn<0)",
        "worst fn",
        "contest",
        "contest day vol",
        "left P>2.5%",
        "left mean",
        "left P<-3%",
    ]
    show = df.copy()
    for c in pct:
        show[c] = show[c].map("{:+.1%}".format)
    for c in ("net", "SR 24-25", "SR 2026", "score med"):
        show[c] = show[c].map("{:+.3f}".format)
    print(
        f"continuous 2024 -> {lab.idx[-1]:%Y-%m-%d %H:00}; {len(lab.starts)} contest windows; "
        f"contest replay from {CONTEST}; {len(now_like)} starts like now, {H_LEFT} h left"
    )
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(show.T.to_string())


if __name__ == "__main__":
    main()
