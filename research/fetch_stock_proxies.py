"""Hourly history of the US stocks behind Roostoo's tokenized stocks (Yahoo Finance), saved in the bar
cache format as data/bars_proxy/1h/{TOKEN}USDT.parquet so the backtest can treat them like the tokens.

Yahoo bars cover US trading hours only; they are placed at their UTC hour and forward-filled across
nights and weekends (the underlying does not trade then). research/stocks_proxy_check.py measures how
closely this proxy tracks the real 24/7 Binance token where both exist.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bars_proxy/1h"
UNDERLYING = {"AMDB": "AMD", "CBRSB": "CBRS", "COINB": "COIN", "CRCLB": "CRCL", "GLWB": "GLW", "GOOGLB": "GOOGL",
              "INTCB": "INTC", "LITEB": "LITE", "METAB": "META", "MSFTB": "MSFT", "MSTRB": "MSTR", "MUB": "MU",
              "NBISB": "NBIS", "NVDAB": "NVDA", "PLTRB": "PLTR", "QCOMB": "QCOM", "SKHYB": "SKHY", "SNDKB": "SNDK",
              "SPCXB": "SPCX", "TSLAB": "TSLA", "WDCB": "WDC"}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for tok, u in UNDERLYING.items():
        h = yf.Ticker(u).history(period="730d", interval="1h", auto_adjust=False)
        if h.empty:
            print(f"{tok}: no data"); continue
        h.index = h.index.tz_convert("UTC").floor("h")
        df = pd.DataFrame({"open": h.Open, "high": h.High, "low": h.Low, "close": h.Close, "volume": h.Volume,
                           "quote_volume": h.Volume * h.Close})
        df = df[~df.index.duplicated(keep="last")]
        df.to_parquet(OUT / f"{tok}USDT.parquet")
        print(f"{tok:7s} {u:5s} {len(df):5d} bars {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d %H:%M}", flush=True)
