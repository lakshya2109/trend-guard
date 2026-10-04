"""Gunek's portfolio view (overnight/monitor_lib.py) for the LIVE contest bot, strictly read-only.

    python scripts/watch_account.py --repo /home/ssm-user/trend-guard      # with the bot's venv python

Unlike overnight/run_overnight.py it never starts, stops or restarts the bot and never trades: it only
reads /v3/balance, /v3/ticker, /v6/short_positions, /v3/pending_count, /v3/query_order (Roostoo), the
bot's log files, and the bot's state/live.db (opened read-only). Every 10 minutes (hh:00, :10 ...) it
redraws Gunek's snapshot, portfolio table included, from the top of the pane so equity and PnL % stay
visible (--scroll appends like run_overnight.py instead). PnL is measured from the contest start
equity and the contest start time in config/strategy.yaml; peak and max drawdown come from the bot's
own per-minute equity record, so they cover the whole contest, not just the time this window has been
open. Output also goes to logs/monitor/<start>/ (monitor_log.txt, snapshots.jsonl).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

SNAPSHOT_EVERY_MIN = 10
HOLDINGS_MINUTE = 10


def bot_status() -> dict:
    """Is a trend_guard.main process running? Reads /proc (Linux); never touches the process."""
    for d in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "trend_guard.main" in cmd and "watch_account" not in cmd:
            return {"alive": True, "pid": int(d.name), "restarts": 0}
    return {"alive": False, "pid": None, "restarts": 0}


def contest_risk(db: Path, since: datetime) -> tuple[float, float, str, float] | None:
    """(peak, max drawdown, when, lowest equity) since `since`, from the bot's equity table (read-only)."""
    if not db.exists():
        return None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    try:
        rows = con.execute(
            "SELECT ts_utc, equity FROM equity WHERE ts_utc >= ? ORDER BY ts_utc", (since.isoformat()[:16],)
        ).fetchall()
    finally:
        con.close()
    peak, max_dd, at, low = 0.0, 0.0, "", float("inf")
    for ts, eq in rows:
        e = float(eq)
        peak = max(peak, e)
        low = min(low, e)
        dd = 1 - e / peak if peak else 0.0
        if dd > max_dd:
            max_dd, at = dd, ts
    return (peak, max_dd, at, low) if rows else None


def next_boundary(minutes: int) -> float:
    now = datetime.now(UTC)
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes - now.minute % minutes)
    return max(1.0, (nxt - now).total_seconds())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="the bot's checkout (has .env, logs/, state/, overnight/)")
    ap.add_argument("--start-equity", type=float, default=100_000.0, help="contest starting equity")
    ap.add_argument("--db", default="state/live.db", help="the bot's state database, relative to --repo")
    ap.add_argument(
        "--scroll", action="store_true", help="append snapshots like run_overnight.py (no redraw)"
    )
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo / "overnight"))
    sys.path.insert(0, str(repo / "src"))
    from monitor_lib import Monitor, RunState, render_snapshot  # Gunek's reader and renderer, unmodified

    from trend_guard.config import load_strategy_config

    start = datetime.fromisoformat(
        load_strategy_config(repo / "config/strategy.yaml").competition.start_utc.replace("Z", "+00:00")
    )
    run_dir = repo / "logs" / "monitor" / datetime.now(UTC).strftime("%Y%m%d_%H%M_UTC")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(run_dir=run_dir, start=start, start_equity=a.start_equity)
    mon = Monitor(state)
    print(
        f"read-only portfolio view; contest since {start:%Y-%m-%d %H:%M} UTC from ${a.start_equity:,.0f}; "
        f"next update on the next 10-minute mark. Ctrl-C stops this view only.",
        flush=True,
    )

    def snap(kind: str) -> None:
        risk = contest_risk(repo / a.db, start)
        if risk:  # seed with the whole contest's history so drawdown is not just "since this window opened"
            state.peak = max(state.peak, risk[0])
            if risk[1] > state.max_dd:
                state.max_dd, state.max_dd_at = risk[1], risk[2]
            state.min_equity = min(state.min_equity, risk[3])
        s = mon.snapshot(kind, bot_status())
        if a.scroll:  # Gunek's console: appended, portfolio table on the hourly snapshot only
            print("\n" + render_snapshot(s, with_holdings=kind in ("start", "hourly")), flush=True)
        else:  # a tmux pane: redraw from the top so Equity / PnL % stay visible, table every time
            print("\033[2J\033[H" + render_snapshot(s, with_holdings=True), flush=True)

    snap("start")
    while True:
        time.sleep(next_boundary(SNAPSHOT_EVERY_MIN))
        kind = "hourly" if datetime.now(UTC).minute == HOLDINGS_MINUTE else "10min"
        try:
            snap(kind)
        except Exception as e:  # noqa: BLE001 -- a failed read must not end the view
            print(f"!! snapshot failed: {e!r} (retrying at the next 10-minute mark)", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
