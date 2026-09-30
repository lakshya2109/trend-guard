"""Download 8-hourly perpetual funding-rate history (Binance USD-M, via www.binance.com/fapi) for every
pair in the bar cache. Saves data/funding/{SYMBOL}.csv (fundingTime ms, fundingRate)."""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.binance.com/fapi/v1/fundingRate"
START = int(pd.Timestamp("2023-05-01", tz="UTC").timestamp() * 1000)
# sub-cent coins trade as 1000x contracts on the futures venue; funding rates are unaffected by the scaling
FUTURES_SYMBOL = {"PEPEUSDT": "1000PEPEUSDT", "SHIBUSDT": "1000SHIBUSDT", "BONKUSDT": "1000BONKUSDT",
                  "FLOKIUSDT": "1000FLOKIUSDT"}


def one(sym: str) -> str:
    if (ROOT / "data/funding" / f"{sym}.csv").exists():
        return f"{sym}: cached"
    fut = FUTURES_SYMBOL.get(sym, sym)
    rows, cur = [], START
    while True:
        for attempt in range(4):
            r = requests.get(URL, params={"symbol": fut, "startTime": cur, "limit": 1000}, timeout=20)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            break
        if r.status_code != 200:
            return f"{sym}: HTTP {r.status_code} (no perpetual?)"
        batch = r.json()
        if not batch:
            break
        rows += batch
        cur = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    if not rows:
        return f"{sym}: no funding history"
    df = pd.DataFrame(rows)[["fundingTime", "fundingRate"]].astype({"fundingTime": "int64", "fundingRate": "float64"})
    df.to_csv(ROOT / "data/funding" / f"{sym}.csv", index=False)
    return f"{sym}: {len(df)} rows from {pd.to_datetime(df.fundingTime.iloc[0], unit='ms'):%Y-%m-%d}"


syms = sorted(p.stem for p in (ROOT / "data/bars/1h").glob("*.parquet"))
with ThreadPoolExecutor(4) as pool:
    for line in pool.map(one, syms):
        print(line, flush=True)
