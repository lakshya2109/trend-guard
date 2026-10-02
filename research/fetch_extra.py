"""Download the extra market data the alpha research needs (not used by the live bot directly).

    python research/fetch_extra.py            # all kinds
    python research/fetch_extra.py spot fut   # a subset

kinds:
  spot     hourly spot klines with taker-buy volume and trade count (data-api.binance.vision REST)
  fut      hourly USD-M perpetual klines (futures taker flow) from data.binance.vision monthly/daily zips
  premium  hourly premium-index klines (perp basis vs spot index), same source
  funding  8-hourly funding rate settlements, same source

Sub-cent coins trade on Binance futures as 1000x contracts (1000PEPEUSDT); the 1000x symbol is tried
when the plain one has no data, and prices are left in contract units (only ratios are used).
Writes data/extra/<kind>/<SPOT_SYMBOL>.parquet.
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "extra"
RAW = ROOT / "data" / "raw_extra"  # zip cache so re-runs are cheap
VISION = "https://data.binance.vision/data/futures/um"
SPOT_REST = "https://data-api.binance.vision/api/v3/klines"
START = date(2023, 6, 1)
END = date(2026, 10, 2)
KCOLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]
SESSION = requests.Session()


def symbols() -> list[str]:
    return sorted(p.stem for p in (ROOT / "data/bars/1h").glob("*.parquet"))


def _get(url: str, params: dict | None = None, tries: int = 4) -> requests.Response | None:
    for k in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r
            time.sleep(2**k)  # 429 / 5xx: back off
        except requests.RequestException:
            time.sleep(2**k)
    return None


# ----------------------------------------------------------------------------- spot REST
def spot_klines(sym: str) -> pd.DataFrame | None:
    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(END, tz="UTC").timestamp() * 1000)
    rows: list[list] = []
    while start_ms < end_ms:
        r = _get(SPOT_REST, {"symbol": sym, "interval": "1h", "startTime": start_ms, "limit": 1000})
        if r is None:
            break
        chunk = r.json()
        if not chunk:
            break
        rows += chunk
        start_ms = chunk[-1][0] + 3_600_000
    if not rows:
        return None
    return _frame(pd.DataFrame(rows, columns=KCOLS))


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["ignore", "close_time"], errors="ignore")
    ts = pd.to_numeric(df["open_time"])
    ts = ts.where(ts < 1e14, ts // 1000)  # 2025+ vision files use microseconds
    df.index = pd.to_datetime(ts, unit="ms", utc=True)
    df = df.drop(columns=["open_time"]).apply(pd.to_numeric, errors="coerce")
    return df[~df.index.duplicated(keep="last")].sort_index()


# ----------------------------------------------------------------------------- futures zips
def _zip_csv(url: str, cache: Path) -> pd.DataFrame | None:
    if cache.exists():
        raw = cache.read_bytes()
        if not raw:
            return None
    else:
        r = _get(url)
        raw = r.content if r is not None else b""
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)  # empty file marks a 404 so it is not re-requested
        if not raw:
            return None
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode()
    except zipfile.BadZipFile:  # truncated by an interrupted run: drop it so the next run refetches
        cache.unlink(missing_ok=True)
        return None
    first = text.split("\n", 1)[0]
    header = 0 if not first[:1].isdigit() else None
    return pd.read_csv(io.StringIO(text), header=header)


def _months() -> list[tuple[int, int]]:
    out, y, m = [], START.year, START.month
    while (y, m) <= (END.year, END.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _vision_series(kind: str, fsym: str) -> pd.DataFrame | None:
    """kind in {klines, premiumIndexKlines, fundingRate}; monthly zips, daily zips for the open month."""
    parts = []
    sub = "1h/" if kind != "fundingRate" else ""
    suffix = "-1h" if kind != "fundingRate" else ""
    for y, m in _months():
        name = f"{fsym}{suffix}-{y}-{m:02d}" if kind != "fundingRate" else f"{fsym}-fundingRate-{y}-{m:02d}"
        url = f"{VISION}/monthly/{kind}/{fsym}/{sub}{name}.zip"
        df = _zip_csv(url, RAW / kind / fsym / f"{name}.zip")
        if df is None and kind != "fundingRate" and (y, m) >= (END.year, END.month - 1):
            d = date(y, m, 1)  # month not published yet: daily files
            while d.month == m and d <= END:
                dn = f"{fsym}{suffix}-{d:%Y-%m-%d}"
                dd = _zip_csv(f"{VISION}/daily/{kind}/{fsym}/{sub}{dn}.zip", RAW / kind / fsym / f"{dn}.zip")
                if dd is not None:
                    parts.append(dd)
                d += timedelta(days=1)
        elif df is not None:
            parts.append(df)
    if not parts:
        return None
    if kind == "fundingRate":
        df = pd.concat([p.set_axis(["calc_time", "interval_h", "rate"][: p.shape[1]], axis=1) for p in parts])
        ts = pd.to_numeric(df["calc_time"])
        out = pd.DataFrame(
            {"rate": pd.to_numeric(df["rate"]).to_numpy()},
            index=pd.to_datetime(ts.where(ts < 1e14, ts // 1000), unit="ms", utc=True),
        )
        return out[~out.index.duplicated(keep="last")].sort_index()
    df = pd.concat([p.iloc[:, :12].set_axis(KCOLS, axis=1) for p in parts])
    return _frame(df)


def fut_symbol(sym: str, kind: str) -> tuple[str, pd.DataFrame | None]:
    for fsym in (sym, "1000" + sym, "1000000" + sym):
        df = _vision_series(kind, fsym)
        if df is not None and len(df):
            return fsym, df
    return "", None


def run(kind: str) -> None:
    out = OUT / kind
    out.mkdir(parents=True, exist_ok=True)

    def one(sym: str) -> str:
        dest = out / f"{sym}.parquet"
        if dest.exists():
            return f"{sym}: cached"
        if kind == "spot":
            df = spot_klines(sym)
        else:
            vk = {"fut": "klines", "premium": "premiumIndexKlines", "funding": "fundingRate"}[kind]
            _, df = fut_symbol(sym, vk)
        if df is None:
            return f"{sym}: none"
        df.to_parquet(dest)
        return f"{sym}: {len(df)} rows {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}"

    with ThreadPoolExecutor(8 if kind != "spot" else 4) as pool:
        for msg in pool.map(one, symbols()):
            print(kind, msg, flush=True)


# ----------------------------------------------------------------------------- OI / positioning metrics
MCOLS = ["create_time", "symbol", "oi", "oi_value", "top_count_ls", "count_ls", "top_sum_ls", "taker_ls"]


def _prefetch_metrics(syms: list[str]) -> None:
    """Warm the zip cache with many requests in flight (one per symbol-day); parsing happens later."""
    jobs = []
    for sym in syms:
        for fsym in (sym, "1000" + sym):
            d = START
            while d <= END:
                name = f"{fsym}-metrics-{d:%Y-%m-%d}"
                jobs.append(
                    (f"{VISION}/daily/metrics/{fsym}/{name}.zip", RAW / "metrics" / fsym / f"{name}.zip")
                )
                d += timedelta(days=1)
    todo = [j for j in jobs if not j[1].exists()]
    print(f"metrics prefetch: {len(todo)} of {len(jobs)} files to fetch", flush=True)

    def get(job: tuple[str, Path]) -> None:
        url, cache = job
        r = _get(url)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content if r is not None else b"")

    with ThreadPoolExecutor(64) as pool:
        for n, _ in enumerate(pool.map(get, todo)):
            if n % 5000 == 0:
                print(f"metrics prefetch {n}/{len(todo)}", flush=True)


def metrics(sym: str) -> pd.DataFrame | None:
    """Daily 5-minute metrics zips -> hourly (last snapshot in each hour). Tries 1000x contracts too."""
    for fsym in (sym, "1000" + sym, "1000000" + sym):
        parts, d, misses = [], START, 0
        while d <= END:
            name = f"{fsym}-metrics-{d:%Y-%m-%d}"
            df = _zip_csv(f"{VISION}/daily/metrics/{fsym}/{name}.zip", RAW / "metrics" / fsym / f"{name}.zip")
            if df is not None:
                parts.append(df.iloc[:, :8].set_axis(MCOLS, axis=1))
                misses = 0
            else:
                misses += 1
                if not parts and misses > 40:  # contract not listed yet: jump ahead a month
                    d += timedelta(days=30)
                    misses = 0
            d += timedelta(days=1)
        if parts:
            df = pd.concat(parts)
            df.index = pd.to_datetime(df["create_time"], utc=True)
            df = df.drop(columns=["create_time", "symbol"]).apply(pd.to_numeric, errors="coerce")
            df = df[~df.index.duplicated(keep="last")].sort_index()
            return df.resample("1h", label="left", closed="right").last()
        if fsym == sym and (RAW / "metrics" / fsym).exists() and not parts:
            continue
    return None


def run_metrics() -> None:
    _prefetch_metrics(symbols())
    out = OUT / "metrics"
    out.mkdir(parents=True, exist_ok=True)

    def one(sym: str) -> str:
        dest = out / f"{sym}.parquet"
        if dest.exists():
            return f"{sym}: cached"
        df = metrics(sym)
        if df is None:
            return f"{sym}: none"
        df.to_parquet(dest)
        return f"{sym}: {len(df)} rows {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}"

    with ThreadPoolExecutor(16) as pool:
        for msg in pool.map(one, symbols()):
            print("metrics", msg, flush=True)


if __name__ == "__main__":
    for k in sys.argv[1:] or ["spot", "funding", "premium", "fut", "metrics"]:
        run_metrics() if k == "metrics" else run(k)
