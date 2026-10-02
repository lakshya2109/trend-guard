"""Book performance under execution policies (research/exec_sim.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.contest import non_overlapping  # noqa: E402
from research.exec_sim import simulate_exec  # noqa: E402
from research.harness import Lab  # noqa: E402


def evaluate(lab, W, name, **kw):
    s0 = lab.idx.get_loc(pd.Timestamp("2024-01-01 12:00", tz="UTC"))
    H = len(lab.idx) - 1 - s0
    eq, to, fees, ms = simulate_exec(lab, W, np.array([s0]), H, **kw)
    e = pd.Series(eq[0], index=lab.idx[s0 : s0 + H + 1])
    d = e.iloc[::24].pct_change().dropna()
    out = {"name": name}
    for y in (2024, 2025, 2026):
        r = d[d.index.year == y]
        out[f"SR{y % 100}"] = r.mean() / r.std() * np.sqrt(365)
    out["SRall"] = d.mean() / d.std() * np.sqrt(365)
    out["ann_ret"] = (1 + d.mean()) ** 365 - 1
    out["maxDD"] = float((1 - e / e.cummax()).max())
    out["fee%/yr"] = fees[0] / H * 24 * 365 * 100 / e.mean()
    out["maker%"] = ms[0]
    eqw, tow, _, _ = simulate_exec(lab, W, lab.starts, 14 * 24, **kw)
    ret = eqw[:, -1] / eqw[:, 0] - 1
    sel = non_overlapping(lab.idx[lab.starts], 14)
    marks = eqw[:, ::24]
    r = marks[:, 1:] / marks[:, :-1] - 1
    out["fn_med"] = np.median(ret[sel])
    out["fn_mean"] = ret[sel].mean()
    out["P<-5%"] = (ret[sel] < -0.05).mean()
    out["fn_sharpe"] = np.median(r.mean(1) / r.std(1))
    out["win_pos"] = (ret > 0).mean()
    return out


if __name__ == "__main__":
    lab = Lab()
    W = lab.book
    rows = [evaluate(lab, W, "market 15bp (old model)", patience=0, taker=0.001, short_fee=0.001)]
    for pat in (1, 2, 3, 6):
        for off in (0.0, 0.001, 0.002):
            rows.append(evaluate(lab, W, f"limit pat={pat} off={off * 1e4:.0f}bp", patience=pat, offset=off))
    rows.append(evaluate(lab, W, "pat=3 off=10bp thru=30bp (harsh)", patience=3, offset=0.001, thru=0.003))
    rows.append(evaluate(lab, W, "pat=3 off=0 thru=0 (lenient)", patience=3, offset=0.0, thru=0.0))
    with pd.option_context(
        "display.max_columns", 30, "display.width", 250, "display.float_format", "{:.3f}".format
    ):
        print(pd.DataFrame(rows).set_index("name"))
