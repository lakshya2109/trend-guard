"""Read-only Streamlit dashboard over the trend-guard SQLite state file. BUILD_SPEC.md Section 13.1.

Bind to 127.0.0.1 only (see docs/RUNBOOK.md): `streamlit run monitor/dashboard.py --server.address 127.0.0.1`.
Never open this port publicly; reach it via an SSH tunnel.

This reads the DB directly with sqlite3 rather than importing trend_guard.state.store, so the
dashboard has zero write path into the bot's state -- it is structurally read-only.

`decisions.json` (written by the engine each hourly cycle) is expected to have this shape:
{
  "regime": "RISK_ON" | "NEUTRAL" | "RISK_OFF",
  "signals": {"BTC/USD": 0.42, ...},
  "weights_raw": {"BTC/USD": 0.18, ...},
  "weights_final": {"BTC/USD": 0.15, ...},
  "targets_notional": {"BTC/USD": 15000.0, ...},
  "multipliers": {"dd_governor": [1.0, "..."], "daily_brake": [0.5, "..."], ...},
}
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from trend_guard.config import load_symbols
from trend_guard.metrics import all_definitions

DB_PATH = os.environ.get("TREND_GUARD_DB", "state/trend_guard.db")
LOGS_DIR = Path(os.environ.get("TREND_GUARD_LOGS", "logs"))


@st.cache_resource
def _connect(db_path: str) -> sqlite3.Connection:
    # read-only URI connection: this process can never write to the bot's live DB
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def _read_sql(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn)
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame()


def _equity_curve(conn: sqlite3.Connection) -> pd.DataFrame:
    df = _read_sql(conn, "SELECT * FROM equity ORDER BY ts_utc")
    if df.empty:
        return df
    for col in ("equity", "cash", "long_notional", "short_collateral", "peak", "dd"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    return df.set_index("ts_utc")


def _latest_decision(conn: sqlite3.Connection) -> dict:
    df = _read_sql(conn, "SELECT * FROM decisions ORDER BY ts_utc DESC LIMIT 1")
    if df.empty:
        return {}
    try:
        return json.loads(df.iloc[0]["json"])
    except (json.JSONDecodeError, TypeError):
        return {}


def _exposure_by_sleeve(decision: dict) -> pd.DataFrame:
    symbols = load_symbols()
    weights = decision.get("weights_final", {})
    rows = []
    for pair, w in weights.items():
        if pair in symbols.bstocks:
            sleeve = "bStocks"
        elif "PAXG" in pair:
            sleeve = "PAXG"
        elif w >= 0:
            sleeve = "Crypto long"
        else:
            sleeve = "Crypto short"
        rows.append({"sleeve": sleeve, "pair": pair, "weight": w})
    if not rows:
        return pd.DataFrame(columns=["sleeve", "weight"])
    df = pd.DataFrame(rows)
    return df.groupby("sleeve")["weight"].apply(lambda s: s.abs().sum()).reset_index()


def _api_error_rate(logs_dir: Path, hours: int = 24) -> float:
    api_dir = logs_dir / "api"
    if not api_dir.exists():
        return 0.0
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    total, errors = 0, 0
    for f in sorted(api_dir.glob("*.jsonl"))[-2:]:  # today + yesterday covers any `hours` <= 24
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    ts = datetime.fromisoformat(rec["ts"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
                if ts < cutoff:
                    continue
                total += 1
                if rec.get("ok") is False:
                    errors += 1
    return errors / total if total else 0.0


def main() -> None:
    st.set_page_config(page_title="trend-guard", layout="wide")
    st.title("trend-guard")

    conn = _connect(DB_PATH)
    equity_df = _equity_curve(conn)
    decision = _latest_decision(conn)

    col1, col2, col3 = st.columns(3)
    if not equity_df.empty:
        latest = equity_df.iloc[-1]
        col1.metric("Equity", f"${latest['equity']:,.0f}")
        col2.metric("Drawdown", f"{latest['dd']:.2%}")
        col3.metric("Regime", decision.get("regime", "n/a"))
    else:
        st.info("No equity snapshots yet -- the bot has not run a cycle.")

    st.subheader("Equity curve and drawdown")
    if not equity_df.empty:
        st.line_chart(equity_df[["equity"]])
        st.area_chart(equity_df[["dd"]])
    else:
        st.write("no data")

    st.subheader("Live composite (since start)")
    kv_df = _read_sql(conn, "SELECT * FROM kv WHERE k IN ('start_ts', 'start_equity')")
    if not equity_df.empty and len(equity_df) > 2:
        daily = equity_df["equity"].resample("1D").last().dropna().pct_change().dropna()
        hourly = equity_df["equity"].resample("1h").last().dropna().pct_change().dropna()
        if len(daily) > 1 and len(hourly) > 1:
            results = all_definitions(daily, hourly)
            comp_df = pd.DataFrame(
                {
                    name: {
                        "sharpe": r.sharpe,
                        "sortino": r.sortino,
                        "calmar": r.calmar,
                        "composite": r.composite,
                        "max_dd": r.max_drawdown,
                        "red_share": r.red_share,
                    }
                    for name, r in results.items()
                }
            ).T
            st.dataframe(comp_df)
        else:
            st.write("not enough history yet")
    else:
        st.write("not enough history yet")
    if not kv_df.empty:
        st.caption(f"start: {dict(zip(kv_df['k'], kv_df['v'], strict=False))}")

    st.subheader("Exposure by sleeve")
    exposure_df = _exposure_by_sleeve(decision)
    if not exposure_df.empty:
        st.bar_chart(exposure_df.set_index("sleeve"))
    else:
        st.write("no decision logged yet")

    st.subheader("Current multipliers")
    mults = decision.get("multipliers", {})
    if mults:
        mult_df = pd.DataFrame([{"overlay": k, "multiplier": v[0], "reason": v[1]} for k, v in mults.items()])
        st.dataframe(mult_df)
    else:
        st.write("no decision logged yet")

    st.subheader("Positions")
    positions_df = _read_sql(conn, "SELECT * FROM positions_meta")
    st.dataframe(positions_df if not positions_df.empty else pd.DataFrame(columns=["pair"]))

    st.subheader("Last 50 orders")
    orders_df = _read_sql(conn, "SELECT * FROM orders ORDER BY created_utc DESC LIMIT 50")
    cols = ["created_utc", "pair", "side", "kind", "qty", "price", "status", "role", "fee"]
    if not orders_df.empty:
        st.dataframe(orders_df[[c for c in cols if c in orders_df.columns]])
    else:
        st.write("no orders yet")

    st.subheader("API error rate (last 24h)")
    st.metric("Error rate", f"{_api_error_rate(LOGS_DIR):.2%}")


if __name__ == "__main__":
    main()
