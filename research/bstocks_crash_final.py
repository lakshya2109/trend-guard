"""Final checks of the shipped crash hedge (book.crash_hedge via book_weights, the live code path), on the
underlying-price panel: back-to-back fortnights, and a cost stress test (3x slippage)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
import stocks_sleeve_experiment as sx  # noqa: E402
from backtest.contest import contest_score, simulate  # noqa: E402
from trend_guard.strategy.book import book_weights  # noqa: E402

b = sx.cfg.book
qv = pd.concat([sx.c_qv, sx.s_qv], axis=1)
cols = list(sx.c_close.columns) + list(b.crash_hedge.instruments)
C, Q = sx.close[cols], qv[cols]
idx = C.index


def weights(on):
    cfg = b.model_copy(update={"crash_hedge": b.crash_hedge.model_copy(update={"enabled": on})})
    return book_weights(C, Q, cfg, b.allow_short).to_numpy()


def fortnights(first, last, step):
    return np.array([idx.get_loc(s) for s in pd.date_range(f"{first} 12:00", f"{last} 12:00", freq=f"{step}D", tz="UTC")
                     if s in idx and idx.get_loc(s) + 336 < len(idx)], dtype=np.int64)


def row(W, starts, bt):
    eq, _ = simulate(W, C, starts, bt, sx.cfg.execution, sx.cfg.breaker)
    r, s = eq[:, -1] / eq[:, 0] - 1, contest_score(eq)
    return f"{np.median(s):7.3f}{np.median(r):+8.2%}{(r > -0.01).mean():6.0%}{(r > 0).mean():6.0%}{r.min():+8.1%}{np.mean(r):+8.2%}"


if __name__ == "__main__":
    off, on = weights(False), weights(True)
    fn = fortnights("2024-02-15", "2026-09-15", 14)
    views = {f"back-to-back fortnights, all ({len(fn)})": (fn, sx.cfg.backtest),
             f"back-to-back fortnights, 2026 ({(idx[fn] >= pd.Timestamp('2026-01-01', tz='UTC')).sum()})": (fn[idx[fn] >= pd.Timestamp("2026-01-01", tz="UTC")], sx.cfg.backtest),
             "944 daily-start windows, 3x slippage (15 bp)": (sx.starts, sx.cfg.backtest.model_copy(update={"slippage": 0.0015})),
             "944 daily-start windows, normal costs": (sx.starts, sx.cfg.backtest)}
    print(f"{'':52s}{'score':>7s}{'median':>8s}{'>-1%':>6s}{'pos':>6s}{'worst':>8s}{'mean':>8s}")
    for v, (starts, bt) in views.items():
        print(v)
        print(f"  {'hedge off (main)':50s}{row(off, starts, bt)}")
        print(f"  {'hedge on':50s}{row(on, starts, bt)}")
