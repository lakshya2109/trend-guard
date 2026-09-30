"""Should the book trade Roostoo's tokenized stocks (bStocks)? Per-sleeve inclusion test.

Stocks have Binance history only since 11 Jun - 13 Jul 2026, so windows start 1 Jul 2026 (earliest
date the trend sleeves' 14-day history rule admits them). Every run uses the same code as the live
book (sleeve functions, risk multiplier, gross cap, simulator, contest score); only the set of
sleeves that may see stocks changes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.contest import contest_score, simulate, window_starts  # noqa: E402
from backtest.run_backtest import load_panel  # noqa: E402
from trend_guard.config import load_strategy_config, load_symbols  # noqa: E402
from trend_guard.strategy.book import momentum_sleeve, trend_sleeve  # noqa: E402
from trend_guard.strategy.team_book import team_momentum, team_trend  # noqa: E402

cfg = load_strategy_config(ROOT / "config/strategy.yaml"); sym = load_symbols(ROOT / "config/symbols.yaml")
c_close, c_qv = load_panel(ROOT / "data/bars/1h", sym.excluded, sym.overrides)
s_close, s_qv = load_panel(ROOT / "data/bars_stocks/1h", sym.excluded, sym.overrides)
stocks = list(s_close.columns)
close = pd.concat([c_close, s_close.reindex(c_close.index)], axis=1)
qv = pd.concat([c_qv, s_qv.reindex(c_qv.index)], axis=1)
crypto_cols = list(c_close.columns)
b = cfg.book

def sleeve(name, with_stocks):
    cols = list(close.columns) if with_stocks else crypto_cols
    C, Q = close[cols], qv[cols]
    if name == "v2_trend": W = trend_sleeve(C, Q, b.trend)
    elif name == "v2_momentum": W = momentum_sleeve(C, Q, b.momentum, b.allow_short)
    elif name == "team_trend": W = team_trend(C, Q)
    else: W = team_momentum(C, Q, b.team.momentum_vol_target, b.allow_short)
    out = np.zeros(close.shape); idx = [close.columns.get_loc(c) for c in cols]; out[:, idx] = W
    return out

share = {"v2_trend": b.v2_share * b.trend_share, "v2_momentum": b.v2_share * b.momentum_share,
         "team_trend": b.team_share * b.team.trend_share, "team_momentum": b.team_share * b.team.momentum_share}
cache = {}
def book(stock_sleeves):
    total = np.zeros(close.shape)
    for k, s in share.items():
        key = (k, k in stock_sleeves)
        if key not in cache: cache[key] = sleeve(k, k in stock_sleeves)
        total += s * cache[key]
    total *= b.risk_multiplier
    gross = np.abs(total).sum(axis=1, keepdims=True)
    return total * np.where(gross > b.max_gross, b.max_gross / np.maximum(gross, 1e-12), 1.0)

bt = cfg.backtest.model_copy(update={"first_window": "2026-07-01"})
starts = window_starts(close.index, bt); st = close.index[starts]
variants = {
    "crypto only (current main)": set(),
    "stocks in v2 trend": {"v2_trend"},
    "stocks in both trend sleeves": {"v2_trend", "team_trend"},
    "stocks in all four sleeves": set(share),
}
stock_idx = [close.columns.get_loc(c) for c in stocks]
print(f"{len(starts)} windows {st[0]:%Y-%m-%d} -> {st[-1]:%Y-%m-%d} (12:00 UTC starts), {len(stocks)} tokenized stocks, {len(crypto_cols)} crypto pairs")
print(f"{'':32s}{'score':>7s}{'median':>8s}{'>-1%':>6s}{'pos':>6s}{'worst':>8s}{'p10':>7s}{'best':>7s}{'turn':>6s}{'stock gross':>12s}{'stock names':>12s}")
rows = {}
for name, ss in variants.items():
    W = book(ss)
    eq, turn = simulate(W, close, starts, cfg.backtest, cfg.execution, cfg.breaker)
    ret = eq[:, -1] / eq[:, 0] - 1; sc = contest_score(eq)
    Wn = W[starts[0]:]; sg = np.abs(Wn[:, stock_idx]).sum(1); sn = (np.abs(Wn[:, stock_idx]) > 1e-6).sum(1)
    rows[name] = ret
    print(f"{name:32s}{np.median(sc):7.3f}{np.median(ret):+8.2%}{(ret > -0.01).mean():6.0%}{(ret > 0).mean():6.0%}{ret.min():+8.1%}"
          f"{np.quantile(ret, .1):+7.1%}{ret.max():+7.1%}{np.median(turn):6.1f}{sg.mean():12.3f}{sn.mean():12.1f}")
base = rows["crypto only (current main)"]
for name, r in rows.items():
    if name != "crypto only (current main)":
        print(f"  {name}: beats crypto-only in {(r > base).mean():.0%} of windows, mean diff {np.mean(r - base):+.2%}")
