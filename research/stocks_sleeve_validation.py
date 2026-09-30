"""Two checks on the stock sleeve before trusting research/stocks_sleeve_experiment.py:
1. Selection bias: Roostoo chose these 21 stocks in mid-2026, largely past winners. How much did the
   basket itself outperform over 2024-26 (buy and hold, equal weight)?
2. Out-of-selection test: windows after the tokens listed (1 Jul - 15 Sep 2026). Signals come from the
   underlying (long history, as the live bot can do), but the account is valued and traded at the real
   24/7 Binance token prices, so off-hours moves and the token's own path are in the result.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
import stocks_sleeve_experiment as sx  # noqa: E402
from backtest.contest import contest_score, simulate, window_starts  # noqa: E402

# 1. basket drift by year (proxy, names that existed at the start of each year)
print("equal-weight buy-and-hold of the stocks listed at each year start (underlying prices):")
for y in (2024, 2025, 2026):
    a, b = pd.Timestamp(f"{y}-01-02", tz="UTC"), min(pd.Timestamp(f"{y}-12-31", tz="UTC"), sx.s_close.index[-1])
    px = sx.s_close.loc[a:b]; live = px.columns[px.iloc[0].notna()]
    rets = px[live].iloc[-1] / px[live].iloc[0] - 1
    btc = sx.c_close["BTC/USD"].loc[a:b]
    print(f"  {y}: {len(live)} names, basket {rets.mean():+.0%} (median name {rets.median():+.0%}),  BTC {btc.iloc[-1]/btc.iloc[0]-1:+.0%}")

# 2. token-priced windows after listing
tok = {}
for f in sorted((ROOT / "data/bars_stocks/1h").glob("*.parquet")):
    d = pd.read_parquet(f)["close"]; d.index = pd.to_datetime(d.index, utc=True)
    tok[f.stem.replace("USDT", "") + "/USD"] = d
tok = pd.DataFrame(tok).reindex(sx.close.index)[sx.s_close.columns]
close_tok = pd.concat([sx.c_close, tok], axis=1)
bt = sx.cfg.backtest.model_copy(update={"first_window": "2026-07-01"})
starts = window_starts(close_tok.index, bt)
held_ok = lambda W: np.where(np.isfinite(close_tok.to_numpy()), W, 0.0)  # no position in a token before it lists
print(f"\n{len(starts)} windows from 1 Jul 2026, valued at real token prices (signals from the underlying):")
print(f"{'':40s}{'score':>7s}{'median':>8s}{'>-1%':>6s}{'worst':>8s}   vs same windows priced with the proxy")
base_W = held_ok(sx.assemble(np.zeros(sx.s_close.shape), 0.0, "add"))
cands = {"live book (crypto only)": base_W}
S_mom15, S_mom10, S_tr = sx.stock_momentum(0.015), sx.stock_momentum(0.010), sx.stock_trend(0.004)
for name, S in (("momentum 1.5%/d", S_mom15), ("momentum 1.0%/d", S_mom10), ("trend 0.4%/d", S_tr)):
    for s in (0.1, 0.2):
        cands[f"{name} carve {s:g}"] = held_ok(sx.assemble(S, s, "carve"))
for name, W in cands.items():
    eq, _ = simulate(W, close_tok, starts, sx.cfg.backtest, sx.cfg.execution, sx.cfg.breaker)
    eqp, _ = simulate(W, sx.close, starts, sx.cfg.backtest, sx.cfg.execution, sx.cfg.breaker)
    r, rp = eq[:, -1] / eq[:, 0] - 1, eqp[:, -1] / eqp[:, 0] - 1
    print(f"{name:40s}{np.median(contest_score(eq)):7.3f}{np.median(r):+8.2%}{(r > -0.01).mean():6.0%}{r.min():+8.1%}"
          f"   proxy: {np.median(contest_score(eqp)):.3f} {np.median(rp):+.2%}")
