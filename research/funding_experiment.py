"""Funding-rate crowding filters on the live book (untested so far: no funding history in the bar cache).

Feature: trailing 3-day mean of Binance USD-M perpetual funding (8-hourly), known at each settlement,
forward-filled to hourly bars (causal). A position is dropped (or halved) while its coin's funding is
extreme in the crowded direction: longs when funding > +thr (longs paying to stay in), shorts when
funding < -thr (shorts crowded, squeeze risk).

Adoption rule, fixed before running (docs/STRATEGY.md round 8): median score up in BOTH 2024-25 and
2026, share of fortnights above -1% down by at most 2 points, worst fortnight no worse.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.contest import simulate, summarize, window_starts  # noqa: E402
from backtest.run_backtest import PERIODS, load_panel  # noqa: E402
from trend_guard.config import load_strategy_config, load_symbols  # noqa: E402
from trend_guard.strategy.book import book_components  # noqa: E402

cfg = load_strategy_config(ROOT / "config/strategy.yaml"); sym = load_symbols(ROOT / "config/symbols.yaml")
close, qv = load_panel(ROOT / "data/bars/1h", sym.excluded, sym.overrides)
fund = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
for pair in close.columns:
    f = ROOT / "data/funding" / f"{pair.split('/')[0]}USDT.csv"
    if f.exists():
        d = pd.read_csv(f); s = pd.Series(d.fundingRate.values, index=pd.to_datetime(d.fundingTime, unit="ms", utc=True))
        s = s.rolling(9, min_periods=3).mean()                      # trailing 3 days of settlements
        s.index = s.index.floor("h")
        s = s[~s.index.duplicated(keep="last")]
        fund[pair] = s.reindex(close.index.union(s.index)).ffill().reindex(close.index)  # value at t: settlements <= t
F = fund.to_numpy()
comp = book_components(close, qv, cfg.book, cfg.book.allow_short)
starts = window_starts(close.index, cfg.backtest); st = close.index[starts]
MOM, TREND = ("v2_momentum", "team_momentum"), ("v2_trend", "team_trend")


def variant(thr, longs_in=(), shorts_in=(), mult=0.0):
    W = np.zeros(close.shape)
    for k, c in comp.items():
        c = c.copy()
        if k in longs_in:
            c = np.where((c > 0) & (F > thr), c * mult, c)
        if k in shorts_in:
            c = np.where((c < 0) & (F < -thr), c * mult, c)
        W += c
    return np.nan_to_num(W)


def row(W):
    eq, turn = simulate(W, close, starts, cfg.backtest, cfg.execution, cfg.breaker)
    t = summarize(eq, turn, st, PERIODS)
    return {p: (t.loc[p, "score"], t.loc[p, "median_return"], t.loc[p, "above_-1%"], t.loc[p, "worst"]) for p in ("2024-25", "2026", "all")} | {"turn": float(np.median(turn))}


def main():
    base = row(variant(1.0))
    tests = {}
    for thr in (0.0003, 0.0005):
        b = f"{thr * 1e4:.0f}bp"
        tests[f"momentum: no shorts if funding < -{b}"] = variant(thr, shorts_in=MOM)
        tests[f"momentum: no longs if funding > +{b}"] = variant(thr, longs_in=MOM)
        tests[f"momentum: both ({b})"] = variant(thr, longs_in=MOM, shorts_in=MOM)
        tests[f"all sleeves: both ({b})"] = variant(thr, longs_in=MOM + TREND, shorts_in=MOM)
        tests[f"momentum: both, halve not drop ({b})"] = variant(thr, longs_in=MOM, shorts_in=MOM, mult=0.5)
    cov = np.isfinite(F[-24 * 365:]).mean()
    share_hot = np.nanmean((np.abs(F[-24 * 365:]) > 0.0003))
    print(f"funding coverage last year {cov:.0%} of pair-hours; |3d funding| > 3bp in {share_hot:.1%} of them\n")
    fmt = lambda r, p: f"{r[p][0]:.3f} {r[p][1]:+.2%} {r[p][2]:.0%} {r[p][3]:+.1%}"
    print(f"{'':44s}{'2024-25: score med >-1% worst':>32s}{'2026: score med >-1% worst':>32s}{'all: score med >-1% worst':>32s}  turn  rule")
    print(f"{'live book (main)':44s}{fmt(base,'2024-25'):>32s}{fmt(base,'2026'):>32s}{fmt(base,'all'):>32s}  {base['turn']:4.1f}")
    for name, W in tests.items():
        r = row(W)
        ok = (r["2024-25"][0] > base["2024-25"][0] and r["2026"][0] > base["2026"][0]
              and all(r[p][2] >= base[p][2] - 0.02 for p in ("2024-25", "2026")) and r["all"][3] >= base["all"][3] - 1e-9)
        print(f"{name:44s}{fmt(r,'2024-25'):>32s}{fmt(r,'2026'):>32s}{fmt(r,'all'):>32s}  {r['turn']:4.1f}  {'PASS' if ok else 'fail'}", flush=True)


if __name__ == "__main__":
    main()
