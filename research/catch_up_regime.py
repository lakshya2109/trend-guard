"""Remaining-contest odds per book, conditional on the market having just fallen (prior 5 days)."""

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
ew = np.log(px).diff().where(liq).mean(axis=1).cumsum()
now = lab.idx[-1]
past5 = np.exp(ew.iloc[-1] - ew.iloc[-121]) - 1
btc5 = px["BTC/USD"].iloc[-1] / px["BTC/USD"].iloc[-121] - 1
print(f"now {now:%m-%d %H:%M}: last 5 days BTC {btc5:+.2%}, average coin {past5:+.2%}")


def book(**upd) -> np.ndarray:
    mom = upd.pop("momentum", {})
    cfg = b.model_copy(update={**upd, "momentum": b.momentum.model_copy(update=mom)})
    return np.nan_to_num(sum(book_components(lab.close, lab.qv, cfg, True).values()))


V = {
    "current (4x blend)": np.nan_to_num(lab.book),
    "momentum only, 6/side": book(v2_share=1.0, team_share=0.0, trend_share=0.0, momentum_share=1.0),
    "momentum only, 3/side": book(
        v2_share=1.0, team_share=0.0, trend_share=0.0, momentum_share=1.0, momentum={"k": 3, "buffer": 6}
    ),
    "momentum only, 50/50 L/S": book(
        v2_share=1.0, team_share=0.0, trend_share=0.0, momentum_share=1.0, momentum={"short_share": 0.5}
    ),
    "trend only": book(v2_share=1.0, team_share=0.0, trend_share=1.0, momentum_share=0.0),
}
ex = lab.cfg.execution
act = np.isin(lab.idx.hour, ex.decision_hours_utc)
first = lab.idx.get_loc(pd.Timestamp("2024-01-01 18:00", tz="UTC"))
starts = np.arange(first, len(lab.idx) - H - 1, 6)
prior = np.array([np.exp(ew.iloc[s] - ew.iloc[s - 120]) - 1 for s in starts])
sel = {"all starts": np.ones(len(starts), bool), "after avg coin fell > 8% in 5 days": prior < -0.08}
for name, m in sel.items():
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
            "mean": r.mean(),
            **{f"P(>{g:+.0%})": np.mean(r > g) for g in (0.0, 0.03, 0.045, 0.06)},
            "P(<-5%)": np.mean(r < -0.05),
        }
    print(f"\n{name}: {m.sum()} starts (every 6h), {H} hours each")
    with pd.option_context(
        "display.width", 250, "display.max_columns", 20, "display.float_format", "{:+.3f}".format
    ):
        print(pd.DataFrame(rows).T)
