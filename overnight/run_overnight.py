"""Run trend-guard live on the Roostoo MOCK account overnight, with monitoring and a final report.

    .venv\\Scripts\\python overnight\\run_overnight.py

- starts the bot:  python -m trend_guard.main --mode live   (keys + ALLOW_LIVE_ORDERS from .env)
- every 10 minutes (hh:00, :10, :20 ...): account snapshot -> console + monitor_log.txt
- every hour at hh:10 (7 min after the bot's hh:03 rebalance): snapshot + full portfolio table
- restarts the bot automatically if it crashes (it resumes from state/live.db)
- keeps Windows awake while running
- Ctrl+C: stops the bot, takes a final snapshot, writes overnight_report_*.txt in the run folder
"""

from __future__ import annotations

import ctypes
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor_lib import REPO, Monitor, RunState, make_report, now_utc, render_snapshot  # noqa: E402

SNAPSHOT_EVERY_MIN = 10
HOLDINGS_MINUTE = 10            # the hourly portfolio table is printed on the hh:10 snapshot
MAX_RESTARTS = 20
BOT_CMD = [sys.executable, "-m", "trend_guard.main", "--mode", "live"]


def keep_awake(on: bool) -> None:
    """Stop Windows from sleeping while the test runs (no effect on other OSes)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def next_boundary(minutes: int) -> float:
    """Seconds until the next wall-clock multiple of `minutes`."""
    now = now_utc()
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes - now.minute % minutes)
    return max(1.0, (nxt - now).total_seconds())


class BotProcess:
    def __init__(self, console_path: Path):
        self.console_path = console_path
        self.proc: subprocess.Popen | None = None
        self.restarts = 0

    def start(self) -> None:
        log = open(self.console_path, "a", encoding="utf-8")
        log.write(f"\n===== bot start {now_utc().isoformat()} =====\n")
        log.flush()
        self.proc = subprocess.Popen(BOT_CMD, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)

    def status(self) -> dict:
        alive = self.proc is not None and self.proc.poll() is None
        return {"alive": alive, "pid": self.proc.pid if self.proc else None, "restarts": self.restarts,
                "exit_code": None if alive or not self.proc else self.proc.returncode}

    def ensure_running(self, say) -> None:
        if self.proc and self.proc.poll() is not None:
            code = self.proc.returncode
            if self.restarts >= MAX_RESTARTS:
                say(f"!! bot exited with code {code}; restart limit reached, not restarting")
                return
            self.restarts += 1
            say(f"!! bot exited with code {code}; restarting (restart #{self.restarts}) in 30 s")
            time.sleep(30)
            self.start()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.wait(timeout=15)  # Ctrl+C in the same console already reached the bot
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()


def main() -> int:
    if not (REPO / ".env").exists():
        print("missing .env in the repo root", file=sys.stderr)
        return 2
    start = now_utc()
    run_dir = REPO / "overnight" / "runs" / start.strftime("%Y%m%d_%H%M_UTC")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(run_dir=run_dir, start=start)
    bot = BotProcess(run_dir / "bot_console.log")

    def say(msg: str) -> None:
        line = f"[{now_utc():%Y-%m-%d %H:%M:%S} UTC] {msg}"
        print(line, flush=True)
        with open(run_dir / "monitor_log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    keep_awake(True)
    say(f"overnight test starting; run folder {run_dir}")
    say("press Ctrl+C once to stop: the bot stops, a final snapshot is taken and the report is written")
    mon = Monitor(state)
    snap = mon.snapshot("start", bot.status())  # baseline before the bot places anything
    print(render_snapshot(snap, with_holdings=True), flush=True)
    bot.start()
    say(f"bot started (pid {bot.proc.pid}); console output -> {bot.console_path.name}")
    try:
        while True:
            time.sleep(next_boundary(SNAPSHOT_EVERY_MIN))
            bot.ensure_running(say)
            state.restarts = bot.restarts
            kind = "hourly" if now_utc().minute == HOLDINGS_MINUTE else "10min"
            try:
                snap = mon.snapshot(kind, bot.status())
                print(render_snapshot(snap, with_holdings=kind == "hourly"), flush=True)
            except Exception as e:  # noqa: BLE001 -- a failed read must not end the test
                say(f"!! snapshot failed: {e!r} (will retry at the next 10-minute mark)")
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # finish the report even if Ctrl+C is pressed again
        say("Ctrl+C received: stopping the bot and writing the report ...")
    finally:
        bot.stop()
        state.restarts = bot.restarts
        try:
            snap = mon.snapshot("final", bot.status())
            print(render_snapshot(snap, with_holdings=True), flush=True)
        except Exception as e:  # noqa: BLE001
            say(f"!! final snapshot failed: {e!r}")
        report = make_report(run_dir)
        keep_awake(False)
        say(f"REPORT WRITTEN: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
