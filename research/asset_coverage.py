"""Which Roostoo pairs does the book actually trade, and why are the others left out?

Classifies every pair in Roostoo's live /v3/exchangeInfo: tokenized stock, excluded in
config/symbols.yaml, no Binance history, or crypto in the panel. For crypto in the panel it reports,
over 2026, how often each sleeve's liquidity/history rule admits it and how often the live book
(book_weights, the exact live code) holds it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_backtest import load_panel  # noqa: E402
from trend_guard.config import load_strategy_config, load_symbols  # noqa: E402
from trend_guard.strategy.book import book_components  # noqa: E402

cfg = load_strategy_config(ROOT / "config/strategy.yaml"); sym = load_symbols(ROOT / "config/symbols.yaml")
info = requests.get("https://mock-api.roostoo.com/v3/exchangeInfo", timeout=20).json()["TradePairs"]
ticker = requests.get("https://mock-api.roostoo.com/v3/ticker", params={"timestamp": 0}, timeout=20).json().get("Data", {})
close, qv = load_panel(ROOT / "data/bars/1h", sym.excluded, sym.overrides)
comp = book_components(close, qv, cfg.book, cfg.book.allow_short)
W = sum(comp.values())
m26 = close.index >= pd.Timestamp("2026-01-01", tz="UTC")
liq = (qv.rolling(168, min_periods=72).sum() / 7)[m26]
age = close.notna().cumsum()[m26]
rows = []
for pair, v in sorted(info.items()):
    t = ticker.get(pair, {})
    row = {"pair": pair, "type": v.get("AssetType"), "roostoo_24h_usd": t.get("UnitTradeValue", 0.0)}
    if v.get("AssetType") != "crypto":
        row["status"] = "tokenized stock: not traded (research/stocks_experiment.py)"
    elif pair in sym.excluded:
        row["status"] = "excluded in symbols.yaml"
    elif pair not in close:
        row["status"] = "no Binance history"
    else:
        j = close.columns.get_loc(pair); held = W[m26, j]
        row.update(trend_ok=float(((liq[pair] >= cfg.book.trend.min_liquidity_usd) & (age[pair] >= cfg.book.trend.min_age_h)).mean()),
                   momentum_ok=float(((liq[pair] >= cfg.book.momentum.min_liquidity_usd) & (age[pair] >= cfg.book.momentum.min_age_h)).mean())
                   if pair not in cfg.book.momentum.exclude else 0.0,
                   held_long=float((held > 1e-6).mean()), held_short=float((held < -1e-6).mean()),
                   status="crypto in the book's universe")
        for k, c in comp.items():
            row[k] = float((np.abs(c[m26, j]) > 1e-6).mean())
    rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(df.groupby("status").size().to_string(), "\n")
cr = df[df.status == "crypto in the book's universe"].sort_values("roostoo_24h_usd", ascending=False)
cols = ["pair", "roostoo_24h_usd", "trend_ok", "momentum_ok", "held_long", "held_short", "v2_trend", "v2_momentum", "team_trend", "team_momentum"]
print("2026 share of hours: eligible per sleeve rule, and held by the live book")
print(cr[cols].round(2).to_string(index=False))
never = cr[(cr.held_long == 0) & (cr.held_short == 0)]
print(f"\ncrypto never held in 2026: {len(never)} -> {', '.join(never.pair)}")
print("\nexcluded / missing:"); print(df[df.status.isin(['excluded in symbols.yaml', 'no Binance history'])][['pair', 'status', 'roostoo_24h_usd']].to_string(index=False))
