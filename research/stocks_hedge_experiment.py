"""Tokenized stocks as a hedge or a defensive holding for the live crypto book.

  beta hedge       short COINB + MSTRB (crypto-beta stocks) worth h x the book's net long crypto
                   exposure, reset daily at 12:00 UTC; compared with the same hedge done by shorting BTC
  conditional      the beta hedge only while BTC is below its 20-day average (risk-off)
  defensive park   when the book's gross is under 0.5x, put part of the idle capital in an equal-weight
                   MSFT / GOOGL / META basket (large, lower-volatility names)
Prices: US underlyings (research/fetch_stock_proxies.py) for 944 windows, then the real tokens for the
77 windows after listing (research/stocks_sleeve_validation.py explains why both are needed: the
listed basket is a set of past winners, which flatters longs and penalises shorts).
Adoption rule (round 8): median score up in 2024-25 and 2026, above -1% down <= 2 pts, worst no worse.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
import stocks_sleeve_experiment as sx  # noqa: E402  (panels, live book, evaluate)
from backtest.contest import contest_score, simulate, window_starts  # noqa: E402

cols = list(sx.close.columns); ci = {c: i for i, c in enumerate(cols)}; ncr = sx.ncr
idx = sx.close.index
book = sx.book
net_crypto = book.sum(1)                                   # signed crypto exposure, fraction of equity
gross = np.abs(book).sum(1)
daily = pd.Series(idx.hour == 12, index=idx)


def hold_daily(x: np.ndarray) -> np.ndarray:
    """Hold a daily-set value between 12:00 UTC resets (so the hedge is not traded every hour)."""
    s = pd.Series(np.where(daily.values, x, np.nan), index=idx).ffill().fillna(0.0)
    return s.to_numpy()


btc = sx.c_close["BTC/USD"]
risk_off = (btc < btc.rolling(20 * 24, min_periods=240).mean()).to_numpy()


def with_overlay(extra: dict[str, np.ndarray]) -> np.ndarray:
    W = np.zeros(sx.close.shape); W[:, :ncr] = book
    for pair, w in extra.items():
        W[:, ci[pair]] += w
    g = np.abs(W).sum(1, keepdims=True)
    return np.nan_to_num(W * np.where(g > 1.0, 1.0 / np.maximum(g, 1e-12), 1.0))


def beta_hedge(h, via="stocks", only_risk_off=False):
    size = hold_daily(np.clip(net_crypto, 0, None) * h * (risk_off if only_risk_off else 1.0))
    if via == "btc":
        return with_overlay({"BTC/USD": -size})
    live = {p: sx.s_close[p].notna().to_numpy() for p in ("COINB/USD", "MSTRB/USD")}
    n = live["COINB/USD"].astype(float) + live["MSTRB/USD"].astype(float)
    return with_overlay({p: -size * live[p] / np.maximum(n, 1) for p in live})


def defensive(frac):
    names = ["MSFTB/USD", "GOOGLB/USD", "METAB/USD"]
    idle = hold_daily(np.clip(0.5 - gross, 0, None) * 2 * frac)   # 0 at gross >= 0.5, up to `frac` when flat
    return with_overlay({p: idle / len(names) for p in names})


variants = {"live book": with_overlay({})}
for h in (0.25, 0.5):
    variants[f"beta hedge {h:g} via COINB/MSTRB"] = beta_hedge(h)
    variants[f"beta hedge {h:g} via BTC short"] = beta_hedge(h, "btc")
    variants[f"beta hedge {h:g} stocks, risk-off only"] = beta_hedge(h, only_risk_off=True)
for f in (0.25, 0.5):
    variants[f"defensive park {f:g} (MSFT/GOOGL/META)"] = defensive(f)

if __name__ == "__main__":
    base = sx.evaluate(variants["live book"])
    fmt = lambda r, p: f"{r[p][0]:.3f} {r[p][1]:+.2%} {r[p][2]:.0%} {r[p][3]:+.1%}"
    print(f"{'944 windows, underlying prices':44s}{'2024-25: score med >-1% worst':>31s}{'2026':>31s}{'all':>31s}  rule")
    for name, W in variants.items():
        r = sx.evaluate(W)
        ok = (r["2024-25"][0] > base["2024-25"][0] and r["2026"][0] > base["2026"][0]
              and all(r[p][2] >= base[p][2] - 0.02 for p in ("2024-25", "2026")) and r["all"][3] >= base["all"][3] - 1e-9)
        print(f"{name:44s}{fmt(r,'2024-25'):>31s}{fmt(r,'2026'):>31s}{fmt(r,'all'):>31s}  {'' if name == 'live book' else ('PASS' if ok else 'fail')}", flush=True)
    # post-listing check at real token prices
    tok = {}
    for f in sorted((ROOT / "data/bars_stocks/1h").glob("*.parquet")):
        d = pd.read_parquet(f)["close"]; d.index = pd.to_datetime(d.index, utc=True)
        tok[f.stem.replace("USDT", "") + "/USD"] = d
    close_tok = pd.concat([sx.c_close, pd.DataFrame(tok).reindex(idx)[sx.s_close.columns]], axis=1)
    starts = window_starts(close_tok.index, sx.cfg.backtest.model_copy(update={"first_window": "2026-07-01"}))
    print(f"\n{len(starts)} windows after listing (1 Jul - 15 Sep 2026), real token prices:   score  median  >-1%  worst")
    for name, W in variants.items():
        W = np.where(np.isfinite(close_tok.to_numpy()), W, 0.0)
        eq, _ = simulate(W, close_tok, starts, sx.cfg.backtest, sx.cfg.execution, sx.cfg.breaker)
        r = eq[:, -1] / eq[:, 0] - 1
        print(f"  {name:44s}{np.median(contest_score(eq)):7.3f}{np.median(r):+8.2%}{(r > -0.01).mean():6.0%}{r.min():+7.1%}")
