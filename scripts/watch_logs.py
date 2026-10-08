"""Live, human-readable view of the running bot's logs. Read-only: it never touches the bot's process.

    python scripts/watch_logs.py                 # from the repo directory the bot runs in
    python scripts/watch_logs.py --logs /opt/trend-guard/logs --api

Follows today's files (switching at UTC midnight) and prints one line per event:
  CYCLE     every hourly cycle: equity, gross / net exposure, orders planned, breaker
  FILL      every executed trade (logs/fills/<date>.csv)
  ACTIVITY  the activity rule's forced trade, when a window had no fill
  WARN/ERR  engine warnings and errors (logs/engine/<date>.jsonl)
  api       raw exchange calls, only with --api (noisy)
Stdlib only and Python 3.8+, so it runs with the system python3 on the box (3.9 on Amazon Linux 2023).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc  # noqa: UP017 -- datetime.UTC needs 3.11; the box runs this on 3.9
COLOR = {
    "CYCLE": "\033[36m",
    "FILL": "\033[32m",
    "ACTIVITY": "\033[35m",
    "WARN": "\033[33m",
    "ERR": "\033[31m",
    "INFO": "\033[37m",
    "api": "\033[90m",
}
RESET = "\033[0m"


def stamp(ts: str) -> str:
    """ISO timestamp -> 'MM-DD HH:MM:SS UTC'."""
    try:
        return datetime.fromisoformat(ts).astimezone(UTC).strftime("%m-%d %H:%M:%S")
    except ValueError:
        return ts[:19]


def out(kind: str, ts: str, text: str) -> None:
    print(f"{COLOR.get(kind, '')}{stamp(ts)}  {kind:<8} {text}{RESET}", flush=True)


def fmt_decision(d: dict) -> tuple[str, str]:
    orders = d.get("orders_planned") or []
    mult = (d.get("breaker") or {}).get("mult", 1.0) if isinstance(d.get("breaker"), dict) else 1.0
    text = (
        f"equity ${float(d.get('equity', 0)):,.2f}  gross {float(d.get('gross', 0)):.0%}  "
        f"net {float(d.get('net', 0)):+.0%}  orders {len(orders)}  "
        f"{'decision hour' if d.get('act_all') else 'hourly'}  bar {str(d.get('bar', ''))[5:16]}"
    )
    if d.get("bear_mode"):
        text += "  BEAR MODE (fast momentum only)"
    if mult != 1.0:
        text += f"  BREAKER x{mult}"
    return "CYCLE", text


def fmt_engine(d: dict) -> tuple[str, str] | None:
    lvl = d.get("level", "INFO")
    if lvl == "INFO" and d.get("msg") not in ("startup", "universe"):
        return None  # routine info lines stay in the file
    extra = {k: v for k, v in d.items() if k not in ("ts", "level", "logger", "msg")}
    kind = {"WARNING": "WARN", "ERROR": "ERR", "CRITICAL": "ERR"}.get(lvl, "INFO")
    if str(d.get("msg", "")).startswith("activity rule"):
        kind = "ACTIVITY"  # the forced trade that keeps the contest day active
    return kind, f"{d.get('msg', '')}  {json.dumps(extra, default=str)[:300] if extra else ''}"


class Follower:
    """Tails one per-day file (logs/<cat>/<UTC date>.<ext>), reopening when the date rolls over."""

    def __init__(self, folder: Path, ext: str, backlog: int, fixed_day: str = ""):
        self.folder, self.ext, self.backlog, self.fixed_day = folder, ext, backlog, fixed_day
        self.day = ""
        self.pos = 0
        self.header: list[str] | None = None

    def lines(self) -> list[str]:
        day = self.fixed_day or datetime.now(UTC).strftime("%Y-%m-%d")
        path = self.folder / f"{day}.{self.ext}"
        if day != self.day:  # new day (or first call): start at the end, minus a short backlog
            first = not self.day
            self.day, self.pos, self.header = day, 0, None
            if first and path.exists():
                raw = path.read_bytes()
                raw = raw[: raw.rfind(b"\n") + 1]  # whole lines only
                self.pos = len(raw)  # a byte offset, so the next poll starts exactly here
                text = raw.decode("utf-8", errors="replace").splitlines()
                if self.ext == "csv" and text:
                    self.header = next(csv.reader([text[0]]))
                    text = text[1:]
                return text[-self.backlog :] if self.backlog else []
        if not path.exists():
            return []
        with open(path, "rb") as f:
            f.seek(self.pos)
            chunk = f.read()
        if not chunk.endswith(b"\n"):  # leave a half-written line for the next poll
            chunk = chunk[: chunk.rfind(b"\n") + 1]
        self.pos += len(chunk)
        rows = chunk.decode("utf-8", errors="replace").splitlines()
        if self.ext == "csv" and rows and self.header is None:
            self.header = next(csv.reader([rows[0]]))
            rows = rows[1:]
        return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs", help="the bot's logs directory")
    ap.add_argument("--backlog", type=int, default=5, help="recent lines to show per stream on start")
    ap.add_argument("--api", action="store_true", help="also show raw API calls")
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--day", default="", help="replay one UTC day (YYYY-MM-DD) instead of following today")
    a = ap.parse_args()
    base = Path(a.logs).resolve()
    streams = {
        "decisions": Follower(base / "decisions", "jsonl", a.backlog, a.day),
        "fills": Follower(base / "fills", "csv", a.backlog * 4, a.day),
        "engine": Follower(base / "engine", "jsonl", a.backlog, a.day),
    }
    if a.api:
        streams["api"] = Follower(base / "api", "jsonl", a.backlog, a.day)
    print(f"watching {base}  (Ctrl-C stops this viewer only; the bot keeps running)", flush=True)
    while True:
        for name, fol in streams.items():
            for line in fol.lines():
                try:
                    if name == "fills":
                        row = dict(zip(fol.header or [], next(csv.reader(io.StringIO(line)))))  # noqa: B905
                        qty, px = float(row.get("qty", 0)), float(row.get("price", 0))
                        out(
                            "FILL",
                            row.get("ts_utc", ""),
                            f"{row.get('side', ''):<11} {row.get('pair', ''):<11} "
                            f"${qty * px:>10,.2f} @ {px:<12g} fee ${float(row.get('fee', 0)):.2f} "
                            f"{row.get('role', '')}  {row.get('reason', '')}",
                        )
                        continue
                    d = json.loads(line)
                    if name == "decisions":
                        kind, text = fmt_decision(d)
                        out(kind, d.get("ts", ""), text)
                    elif name == "engine":
                        r = fmt_engine(d)
                        if r:
                            out(r[0], d.get("ts", ""), r[1])
                    else:
                        out("api", d.get("ts", ""), f"{d.get('msg', '')} {d.get('path', d.get('url', ''))}")
                except (ValueError, KeyError, StopIteration):
                    out("INFO", "", line[:200])
        time.sleep(a.poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
