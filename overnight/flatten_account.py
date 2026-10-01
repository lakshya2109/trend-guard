"""Reset the Roostoo MOCK test account to all cash before a fresh overnight run.

    .venv\\Scripts\\python overnight\\flatten_account.py             # dry run: shows what it would do
    .venv\\Scripts\\python overnight\\flatten_account.py --execute   # cancel orders, close shorts, sell coins

MOCK ACCOUNT ONLY. Never run this against the competition account: manual API calls there would
break the contest's "no traces of manually called APIs" rule (Screen 1).

Steps: refuse if the bot is running -> cancel pending orders -> close every short -> sell every
coin balance at MARKET (dust below the pair's MiniOrder is left) -> print the final account ->
move the bot's state/live.db into state/archive/ so the next run starts with fresh state.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trend_guard.config import load_env  # noqa: E402
from trend_guard.roostoo.client import RoostooClient  # noqa: E402
from trend_guard.roostoo.rules import parse_trading_rules  # noqa: E402


def bot_running() -> bool:
    """True if a trend-guard bot or overnight monitor process is alive (Windows)."""
    if sys.platform != "win32":
        return False
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match 'trend_guard.main|run_overnight' } | Measure-Object | "
          "Select-Object -ExpandProperty Count")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    return out.stdout.strip() not in ("", "0")


def account(client: RoostooClient) -> tuple[dict, list, dict]:
    bal = client.get_balance()
    shorts = [sp for sp in client.short_positions() if sp.short_qty > 0]
    tick = client.get_ticker()
    return bal, shorts, tick


def show(bal: dict, shorts: list, tick: dict) -> None:
    usd = bal.get("USD")
    coins = {c: float(e.free) + float(e.lock) for c, e in bal.items() if c != "USD" and (e.free or e.lock)}
    long_value = sum(q * tick[f"{c}/USD"].mid for c, q in coins.items() if f"{c}/USD" in tick)
    short_value = sum(sp.position_value for sp in shorts)
    usd_cash = float(usd.free) + max(float(usd.lock) - float(usd.short_collateral), 0.0) if usd else 0.0
    print(f"  USD free {float(usd.free) if usd else 0:,.2f}   longs {long_value:,.2f} ({len(coins)} coins)   "
          f"shorts value {short_value:,.2f} ({len(shorts)} positions)")
    print(f"  equity {usd_cash + long_value + short_value:,.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="actually send the orders (default: dry run)")
    args = ap.parse_args()

    if bot_running():
        print("A trend-guard bot or overnight monitor is still running. Stop it (Ctrl+C) first.")
        return 2
    env = load_env(REPO / ".env")
    client = RoostooClient(env.roostoo_base_url, env.roostoo_api_key, env.roostoo_secret_key)
    client.sync_clock()
    rules = parse_trading_rules(client.exchange_info())

    bal, shorts, tick = account(client)
    print("BEFORE:")
    show(bal, shorts, tick)
    pending, pairs = client.pending_count()

    sells, dust = [], []
    for coin, e in bal.items():
        if coin == "USD":
            continue
        pair = f"{coin}/USD"
        qty = Decimal(str(e.free)) + Decimal(str(e.lock))
        if qty <= 0 or pair not in rules:
            continue
        q = rules[pair].clamp_qty(qty)
        px = Decimal(str(tick[pair].mid)) if pair in tick else Decimal("0")
        (sells if q > 0 and rules[pair].meets_min(px, q) else dust).append((pair, q, px))

    print(f"\nPLAN: cancel {pending} pending order(s) {pairs or ''}; close {len(shorts)} short(s); "
          f"market-sell {len(sells)} coin(s); leave {len(dust)} dust balance(s)")
    for sp in shorts:
        print(f"  SHORT_CLOSE {sp.pair:<12} qty {sp.short_qty:,.6f}  value ${sp.position_value:,.2f}")
    for pair, q, px in sells:
        print(f"  SELL        {pair:<12} qty {q:,.6f}  ~${q * px:,.2f}")
    for pair, q, px in dust:
        print(f"  (dust)      {pair:<12} qty {q}  ~${q * px:,.4f}")
    if not args.execute:
        print("\nDry run only. Re-run with --execute to send these orders.")
        return 0

    print("\nEXECUTING ...")
    if pending:
        print("  cancelled:", client.cancel_order(cancel_all=True))
    for sp in shorts:
        r = client.short_close(sp.pair)
        print(f"  closed short {sp.pair}: qty {r.closed_qty} at {r.close_price}"
              f"  realized {r.realized_pnl:+.2f}")
        time.sleep(1.0)  # stay well inside the API rate limit
    # coin balances unlock after the cancel; re-read before selling
    bal = client.get_balance()
    for pair, _, _ in sells:
        e = bal.get(pair.split("/")[0])
        q = rules[pair].clamp_qty(Decimal(str(e.free))) if e else Decimal("0")
        if q <= 0:
            continue
        d = client.place_order(pair, "SELL", "MARKET", q)
        print(f"  sold {pair}: {d.filled_quantity} at {d.filled_aver_price}  status {d.status}")
        time.sleep(1.0)

    time.sleep(2.0)
    bal, shorts, tick = account(client)
    print("\nAFTER:")
    show(bal, shorts, tick)

    db = REPO / "state" / "live.db"
    if db.exists():
        dest = REPO / "state" / "archive" / f"live_{datetime.now(UTC):%Y%m%d_%H%M}_before_flatten.db"
        dest.parent.mkdir(parents=True, exist_ok=True)
        db.replace(dest)
        print(f"\nBot state moved to {dest.relative_to(REPO)} (next run starts fresh).")
    print("Account flattened. Start the overnight run as usual.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
