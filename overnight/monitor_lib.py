"""Overnight test monitor: account snapshots from Roostoo (read-only) plus a readable final report.

Everything here only READS: /v3/balance, /v3/ticker, /v6/short_positions, /v3/pending_count and the
bot's own log files. It never places or cancels orders.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from trend_guard.config import load_env
from trend_guard.roostoo.client import RoostooClient

REPO = Path(__file__).resolve().parents[1]
MARKET_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD"]


def now_utc() -> datetime:
    return datetime.now(UTC)


def fmt_local(ts: datetime) -> str:
    return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def signed_money(x: float) -> str:
    return f"+${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"


def pct(x: float, signed: bool = True) -> str:
    return f"{x:+.2%}" if signed else f"{x:.2%}"


# ----------------------------------------------------------------------------------------------
# bot log readers (logs/ is written by the trend-guard engine)
# ----------------------------------------------------------------------------------------------
def read_fills(since: datetime) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(REPO / "logs" / "fills" / "*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ts = datetime.fromisoformat(r["ts_utc"])
                if ts >= since:
                    r["_ts"] = ts
                    rows.append(r)
    return sorted(rows, key=lambda r: r["_ts"])


def read_jsonl(category: str, since: datetime) -> list[dict]:
    out = []
    for path in sorted(glob.glob(str(REPO / "logs" / category / "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if datetime.fromisoformat(d["ts"]) >= since:
                        out.append(d)
                except (ValueError, KeyError):
                    continue
    return out


def exchange_trades(client: RoostooClient, until: datetime | None = None) -> list[dict]:
    """Every order that actually traded on the account, from Roostoo's own order history.

    Roostoo echoes FilledQuantity = Quantity on cancelled/pending limit orders with FilledAverPrice 0,
    so only rows with a real average price count. Fees charged in the coin are converted to USD.
    """
    rows, offset, page = [], 0, 1000
    while True:
        params = {"limit": page} if offset == 0 else {"limit": page, "offset": offset}  # offset=0 is rejected
        d = client._post("/v3/query_order", params, retry_safe=True)
        if not d.get("Success") and d.get("ErrMsg") != "no order matched":
            raise RuntimeError(f"query_order failed: {d.get('ErrMsg')}")
        batch = d.get("OrderMatched") or []
        for o in batch:
            qty, px = float(o.get("FilledQuantity") or 0), float(o.get("FilledAverPrice") or 0)
            if qty <= 0 or px <= 0:
                continue
            fee = float(o.get("CommissionChargeValue") or 0)
            if o.get("CommissionCoin") not in (None, "", "USD"):
                fee *= px
            ts = datetime.fromtimestamp((o.get("FinishTimestamp") or o["CreateTimestamp"]) / 1000, UTC)
            if until is None or ts <= until:
                rows.append({"_ts": ts, "ts_utc": ts.isoformat(), "pair": o["Pair"], "side": o["Side"],
                             "qty": qty, "price": px, "fee": fee, "role": o.get("Role", ""),
                             "type": o.get("Type", ""), "order_id": o.get("OrderID"), "reason": ""})
        if len(batch) < page:
            break
        offset += page
    return sorted(rows, key=lambda r: r["_ts"])


def cost_basis(fills: list[dict], since: datetime | None = None) -> tuple[dict, dict, float]:
    """Average-cost book over the account's whole history: (open lots, realized PnL per pair, fees).

    Lots use every fill so positions carried over from an earlier run keep their true cost; realized
    PnL and fees only count fills at or after `since` (the run start).
    """
    lots: dict[str, dict] = {}
    realized: dict[str, float] = {}
    fees = 0.0
    for r in fills:
        pair, side = r["pair"], r["side"]
        q, p = float(r["qty"]), float(r["price"])
        in_run = since is None or r["_ts"] >= since
        if p <= 0:  # a phantom fill (see exchange_trades) carries no information
            continue
        if in_run:
            fees += float(r["fee"] or 0)
        lot = lots.setdefault(pair, {"qty": 0.0, "avg": 0.0})
        if side == "BUY":
            tot = lot["qty"] + q
            lot["avg"] = (lot["qty"] * lot["avg"] + q * p) / tot if tot else 0.0
            lot["qty"] = tot
        elif side == "SELL":
            if in_run:
                realized[pair] = realized.get(pair, 0.0) + q * (p - lot["avg"])
            lot["qty"] = max(0.0, lot["qty"] - q)
        elif side == "SHORT_OPEN":
            s = lots.setdefault(pair + "#short", {"qty": 0.0, "avg": 0.0})
            tot = s["qty"] + q
            s["avg"] = (s["qty"] * s["avg"] + q * p) / tot if tot else 0.0
            s["qty"] = tot
        elif side == "SHORT_CLOSE":
            s = lots.setdefault(pair + "#short", {"qty": 0.0, "avg": 0.0})
            if in_run:
                realized[pair] = realized.get(pair, 0.0) + q * (s["avg"] - p)
            s["qty"] = max(0.0, s["qty"] - q)
    return lots, realized, fees


# ----------------------------------------------------------------------------------------------
# snapshots
# ----------------------------------------------------------------------------------------------
@dataclass
class RunState:
    run_dir: Path
    start: datetime
    start_equity: float | None = None
    start_prices: dict = field(default_factory=dict)
    peak: float = 0.0
    max_dd: float = 0.0
    max_dd_at: str = ""
    min_equity: float = float("inf")
    last_fill_count: int = 0
    last_log_check: datetime | None = None
    restarts: int = 0

    def save(self) -> None:
        d = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in self.__dict__.items()}
        d["run_dir"] = str(self.run_dir)
        (self.run_dir / "run_state.json").write_text(json.dumps(d, indent=1, default=str))


class Monitor:
    def __init__(self, state: RunState):
        env = load_env(REPO / ".env")
        self.client = RoostooClient(env.roostoo_base_url, env.roostoo_api_key, env.roostoo_secret_key)
        self.client.sync_clock()
        self.s = state
        self.snap_path = state.run_dir / "snapshots.jsonl"
        self.text_path = state.run_dir / "monitor_log.txt"

    def snapshot(self, kind: str, bot_status: dict) -> dict:
        """Read the account and build one snapshot. Valuation uses mid prices, like the bot."""
        s = self.s
        ts = now_utc()
        self.client.sync_clock()
        bal = self.client.get_balance()
        tick = self.client.get_ticker()
        try:
            shorts = self.client.short_positions()
        except Exception:  # noqa: BLE001 -- shorts endpoint may be disabled for the account
            shorts = []
        pending, _ = self.client.pending_count()

        mid = {p: t.mid for p, t in tick.items() if t.mid}
        usd = bal.get("USD")
        usd_free = float(usd.free) if usd else 0.0
        usd_lock = float(usd.lock) if usd else 0.0
        history = exchange_trades(self.client)  # authoritative fills, whole account
        lots, realized, fees = cost_basis(history, since=s.start)
        fills = [r for r in history if r["_ts"] >= s.start]

        holdings, long_value = [], 0.0
        for coin, e in bal.items():
            qty = float(e.free) + float(e.lock)
            pair = f"{coin}/USD"
            if coin == "USD" or qty <= 0:
                continue
            px = mid.get(pair, 0.0)
            val = qty * px
            long_value += val
            avg = lots.get(pair, {}).get("avg", 0.0)
            upnl = (px - avg) * qty if avg else 0.0
            holdings.append({"pair": pair, "side": "LONG", "qty": qty, "price": px, "value": val,
                             "avg_cost": avg, "upnl": upnl, "upnl_pct": (px / avg - 1) if avg else 0.0})
        short_col = short_upnl = short_value = 0.0
        for sp in shorts:
            if sp.short_qty <= 0:
                continue
            short_col += sp.collateral
            short_upnl += sp.unrealized_pnl
            short_value += sp.position_value
            holdings.append({"pair": sp.pair, "side": "SHORT", "qty": sp.short_qty, "price": sp.current_price,
                             "value": sp.short_qty * sp.current_price, "avg_cost": sp.entry_price,
                             "upnl": sp.unrealized_pnl, "upnl_pct": sp.unrealized_pnl_pct,
                             "collateral": sp.collateral})

        # Roostoo's USD "Lock" = pending-order cash + short collateral; the collateral is already inside
        # each short's PositionValue, so only the pending-order part of Lock is added here
        equity = usd_free + max(usd_lock - short_col, 0.0) + long_value + short_value
        if s.start_equity is None:
            s.start_equity = equity
            s.start_prices = {p: mid.get(p) for p in MARKET_PAIRS}
        s.peak = max(s.peak, equity)
        dd = 1 - equity / s.peak if s.peak else 0.0
        if dd > s.max_dd:
            s.max_dd, s.max_dd_at = dd, ts.isoformat()
        s.min_equity = min(s.min_equity, equity)
        for h in holdings:
            h["weight"] = h["value"] / equity if equity else 0.0
        holdings.sort(key=lambda h: -h["value"])
        short_notional = sum(h["value"] for h in holdings if h["side"] == "SHORT")

        new_fills = fills[s.last_fill_count:]
        s.last_fill_count = len(fills)
        # rows in the bot's own fill log that carry no price (phantom fills from older bot code)
        bot_phantoms = sum(float(r["price"] or 0) <= 0 for r in read_fills(s.start))
        since = s.last_log_check or s.start
        eng = read_jsonl("engine", since)
        s.last_log_check = ts
        warns = [e for e in eng if e.get("level") == "WARNING"]
        errs = [e for e in eng if e.get("level") in ("ERROR", "CRITICAL")]
        cycles = [d for d in read_jsonl("decisions", s.start) if d.get("msg") == "cycle"]
        last_cycle = cycles[-1] if cycles else None

        snap = {
            "ts_utc": ts.isoformat(), "ts_local": fmt_local(ts), "kind": kind,
            "equity": equity, "usd_free": usd_free, "usd_lock": usd_lock, "long_value": long_value,
            "short_collateral": short_col, "short_upnl": short_upnl, "short_value": short_value,
            "short_notional": short_notional,
            "gross": (long_value + short_notional) / equity if equity else 0.0,
            "net": (long_value - short_notional) / equity if equity else 0.0,
            "start_equity": s.start_equity, "pnl": equity - s.start_equity,
            "pnl_pct": equity / s.start_equity - 1 if s.start_equity else 0.0,
            "peak": s.peak, "drawdown": dd, "max_drawdown": s.max_dd, "min_equity": s.min_equity,
            "n_long": sum(h["side"] == "LONG" for h in holdings),
            "n_short": sum(h["side"] == "SHORT" for h in holdings),
            "pending_orders": pending,
            "realized_pnl": sum(realized.values()), "fees_total": fees,
            "fills_total": len(fills),
            "market": {p: {"price": mid.get(p), "chg": (mid[p] / s.start_prices[p] - 1)
                           if mid.get(p) and s.start_prices.get(p) else None} for p in MARKET_PAIRS},
            "holdings": holdings,
            "activity": {
                "new_fills": len(new_fills),
                "new_fill_fees": sum(float(r["fee"] or 0) for r in new_fills),
                "new_fill_notional": sum(float(r["qty"]) * float(r["price"]) for r in new_fills),
                "bot_phantom_fills": bot_phantoms,
                "warnings": len(warns), "errors": len(errs),
                "last_error": errs[-1]["msg"] + " " + str(errs[-1].get("err", "")) if errs else "",
                "cycles_total": len(cycles),
                "last_cycle": None if not last_cycle else {
                    "cycle_id": last_cycle.get("cycle_id"), "gross": last_cycle.get("gross"),
                    "net": last_cycle.get("net"), "orders": len(last_cycle.get("orders_planned", [])),
                    "multiplier": last_cycle.get("multiplier")},
            },
            "bot": bot_status,
        }
        with open(self.snap_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap, default=str) + "\n")
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(render_snapshot(snap, with_holdings=kind in ("start", "hourly", "final")) + "\n")
        s.save()
        return snap


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
def render_market(m: dict) -> str:
    parts = []
    for p, v in m.items():
        if v.get("price"):
            chg = f" ({pct(v['chg'])})" if v.get("chg") is not None else ""
            parts.append(f"{p.split('/')[0]} {v['price']:,.2f}{chg}")
    return " | ".join(parts)


def render_holdings(snap: dict) -> str:
    rows = snap["holdings"]
    if not rows:
        return "    (no open positions: 100% cash)\n"
    out = [f"    {'Asset':<12}{'Side':<6}{'Qty':>16}{'Price':>14}{'Value':>14}{'Weight':>9}"
           f"{'Avg cost':>14}{'Unreal. PnL':>14}{'PnL %':>9}"]
    for h in rows:
        out.append(f"    {h['pair']:<12}{h['side']:<6}{h['qty']:>16,.6f}{h['price']:>14,.6g}"
                   f"{money(h['value']):>14}{h['weight']:>9.2%}{h['avg_cost']:>14,.6g}"
                   f"{signed_money(h['upnl']):>14}{pct(h['upnl_pct']):>9}")
    return "\n".join(out) + "\n"


def render_snapshot(snap: dict, with_holdings: bool) -> str:
    s, a, b = snap, snap["activity"], snap["bot"]
    lc = a.get("last_cycle") or {}
    pending_cash = max(s["usd_lock"] - s["short_collateral"], 0.0)
    unrealized = sum(h["upnl"] for h in s["holdings"])
    phantoms = a.get("bot_phantom_fills")
    status = "RUNNING" if b.get("alive") else "STOPPED"
    lines = [
        "=" * 100,
        f"[{s['kind'].upper()}] {s['ts_local']}   ({s['ts_utc'][:19]} UTC)",
        f"  Equity      {money(s['equity']):>16}   PnL {signed_money(s['pnl'])} ({pct(s['pnl_pct'])})"
        f"   start {money(s['start_equity'])}",
        f"  Cash (USD)  {money(s['usd_free']):>16}   in pending orders {money(pending_cash)}",
        f"  Longs       {money(s['long_value']):>16}   Shorts collateral {money(s['short_collateral'])}"
        f"  short uPnL {signed_money(s['short_upnl'])}",
        "  Check       equity = cash + pending-order cash + longs + short collateral + short uPnL",
        f"  Exposure    gross {s['gross']:.1%}  net {s['net']:+.1%}   positions {s['n_long']} long /"
        f" {s['n_short']} short   pending orders {s['pending_orders']}",
        f"  Risk        peak {money(s['peak'])}  drawdown {s['drawdown']:.2%}"
        f"  max DD {s['max_drawdown']:.2%}  min equity {money(s['min_equity'])}",
        f"  PnL detail  realized this run {signed_money(s['realized_pnl'])}"
        f"  fees this run {money(s['fees_total'])}"
        f"  unrealized on open positions (vs their cost) {signed_money(unrealized)}",
        f"  Fills       this run {s['fills_total']} (+{a['new_fills']} since last,"
        f" {money(a['new_fill_notional'])} traded; from Roostoo order history)"
        + (f"   bot log has {phantoms} phantom fills (no price)" if phantoms else ""),
        f"  Market      {render_market(s['market'])}",
        f"  Bot         {status} (pid {b.get('pid')}, restarts {b.get('restarts', 0)})"
        f"   cycles {a['cycles_total']}  last cycle {lc.get('cycle_id', '-')}"
        f" gross {lc.get('gross') or 0:.1%} orders {lc.get('orders', '-')}"
        f"   warnings +{a['warnings']} errors +{a['errors']}",
    ]
    if a.get("last_error"):
        lines.append(f"  Last error  {a['last_error'][:160]}")
    if with_holdings:
        lines.append("  Portfolio:")
        lines.append(render_holdings(snap).rstrip("\n"))
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# final report
# ----------------------------------------------------------------------------------------------
def normalize(snaps: list[dict], start_equity: float) -> dict:
    """Recompute equity and every derived figure from the raw fields, counting short collateral once.

    Snapshots written before this fix added Roostoo's USD Lock (which contains the short collateral)
    and each short's PositionValue (collateral + PnL), so the collateral was counted twice.
    """
    peak, max_dd, max_dd_at, low = start_equity, 0.0, "", float("inf")
    for s in snaps:
        pending_cash = max(s["usd_lock"] - s["short_collateral"], 0.0)
        eq = s["usd_free"] + pending_cash + s["long_value"] + s["short_value"]
        s["equity"], s["start_equity"] = eq, start_equity
        s["pnl"], s["pnl_pct"] = eq - start_equity, eq / start_equity - 1
        if s["kind"] != "start":  # the start read can land mid-way through the bot's first orders
            peak = max(peak, eq)
        s["peak"], s["drawdown"] = peak, 1 - eq / peak
        if s["drawdown"] > max_dd:
            max_dd, max_dd_at = s["drawdown"], s["ts_utc"]
        s["max_drawdown"] = max_dd
        low = min(low, eq)
        s["min_equity"] = low
        s["gross"] = (s["long_value"] + s["short_notional"]) / eq if eq else 0.0
        s["net"] = (s["long_value"] - s["short_notional"]) / eq if eq else 0.0
        for h in s["holdings"]:
            h["weight"] = h["value"] / eq if eq else 0.0
    return {"max_dd": max_dd, "max_dd_at": max_dd_at}


def make_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    snaps = [json.loads(line) for line in open(run_dir / "snapshots.jsonl", encoding="utf-8")]
    if not snaps:
        raise SystemExit("no snapshots recorded yet")
    rs = json.loads((run_dir / "run_state.json").read_text())
    # baseline: this run's own pre-bot snapshot (run_state.json); "baseline_equity" overrides it when
    # that snapshot was known to be wrong. state/live.db is not used: it belongs to the latest run only.
    baseline = rs.get("baseline_equity") or rs.get("start_equity")
    if not baseline:
        f = snaps[0]
        baseline = (f["usd_free"] + max(f["usd_lock"] - f["short_collateral"], 0.0)
                    + f["long_value"] + f["short_value"])
    rs.update(normalize(snaps, float(baseline)))
    start = datetime.fromisoformat(rs["start"])
    first, last = snaps[0], snaps[-1]
    end = datetime.fromisoformat(last["ts_utc"])
    env = load_env(REPO / ".env")
    client = RoostooClient(env.roostoo_base_url, env.roostoo_api_key, env.roostoo_secret_key)
    client.sync_clock()
    history = exchange_trades(client, until=end)  # authoritative fills, whole account
    for s in snaps:  # re-cost every stored long from the exchange history as of that snapshot
        t = datetime.fromisoformat(s["ts_utc"])
        lots_t, _, _ = cost_basis([r for r in history if r["_ts"] <= t])
        for h in s["holdings"]:
            if h["side"] == "LONG":
                avg = lots_t.get(h["pair"], {}).get("avg", 0.0)
                h["avg_cost"] = avg
                h["upnl"] = (h["price"] - avg) * h["qty"] if avg else 0.0
                h["upnl_pct"] = h["price"] / avg - 1 if avg else 0.0
    lots, realized, fees = cost_basis(history, since=start)
    fills = [r for r in history if r["_ts"] >= start]
    phantoms = [r for r in read_fills(start) if float(r["price"] or 0) <= 0]
    cycles = [d for d in read_jsonl("decisions", start) if d.get("msg") == "cycle"]
    eng = read_jsonl("engine", start)
    warns = [e for e in eng if e.get("level") == "WARNING"]
    errs = [e for e in eng if e.get("level") in ("ERROR", "CRITICAL")]
    startup = next((e for e in eng if e.get("msg") == "startup"), {})
    btc0, btc1 = first["market"]["BTC/USD"]["price"], last["market"]["BTC/USD"]["price"]
    dur = end - start

    L: list[str] = []
    w = L.append
    w("TREND-GUARD OVERNIGHT TEST REPORT (Roostoo mock account)")
    w("=" * 100)
    w(f"Run folder     : {run_dir}")
    w(f"Started        : {fmt_local(start)}  ({start:%Y-%m-%d %H:%M} UTC)")
    w(f"Ended          : {fmt_local(end)}  ({end:%Y-%m-%d %H:%M} UTC)")
    w(f"Duration       : {dur.total_seconds() / 3600:.2f} hours")
    w(f"Bot commit     : {startup.get('commit', 'unknown')}   config hash {startup.get('config_hash', '?')}"
      f"   mode {startup.get('mode', '?')}   shorts enabled {startup.get('shorts_enabled', '?')}")
    w(f"Bot restarts   : {rs.get('restarts', 0)}")
    w("")
    w("SUMMARY")
    w("-" * 100)
    w(f"Start equity   : {money(first['start_equity'])}")
    w(f"End equity     : {money(last['equity'])}")
    w(f"PnL            : {signed_money(last['pnl'])}  ({pct(last['pnl_pct'])})")
    unrealized = sum(h["upnl"] for h in last["holdings"])
    w(f"  realized this run {signed_money(sum(realized.values()))}   fees this run {money(fees)}"
      f"   unrealized on open positions vs their cost {signed_money(unrealized)}")
    w("  (PnL is equity now minus equity at the start. Unrealized includes moves on positions bought before")
    w("   this run, so it does not add up to the run PnL when the run started with open positions.)")
    w(f"Peak equity    : {money(last['peak'])}")
    w(f"Lowest equity  : {money(min(s['equity'] for s in snaps))}")
    w(f"Max drawdown   : {rs.get('max_dd', 0):.2%}  (at {rs.get('max_dd_at', '-')[:16]} UTC)")
    if btc0 and btc1:
        w(f"BTC same period: {pct(btc1 / btc0 - 1)}  ({btc0:,.2f} -> {btc1:,.2f})")
    buys = [r for r in fills if r["side"] in ("BUY", "SHORT_CLOSE")]
    sells = [r for r in fills if r["side"] in ("SELL", "SHORT_OPEN")]
    notional = sum(float(r["qty"]) * float(r["price"]) for r in fills)
    by_side = {k: sum(r["side"] == k for r in fills) for k in ("BUY", "SELL", "SHORT_OPEN", "SHORT_CLOSE")}
    w(f"Trades (fills) : {len(fills)}  ("
      + ", ".join(f"{k} {v}" for k, v in by_side.items())
      + f")   buys {len(buys)} / sells {len(sells)}")
    w(f"Traded notional: {money(notional)}  = {notional / first['start_equity']:.2f}x start equity"
      if first["start_equity"] else "")
    makers = sum(r.get("role") == "MAKER" for r in fills)
    w(f"Maker fills    : {makers} of {len(fills)}" + (f" ({makers / len(fills):.0%})" if fills else ""))
    w(f"Hourly cycles  : {len(cycles)}   engine warnings {len(warns)}   errors {len(errs)}")
    w(f"Final exposure : gross {last['gross']:.1%}  net {last['net']:+.1%}  "
      f"{last['n_long']} long / {last['n_short']} short   cash {money(last['usd_free'])}")
    w("")

    w("EQUITY TIMELINE (every 10 minutes)")
    w("-" * 100)
    w(f"{'Local time':<22}{'Equity':>14}{'PnL':>13}{'PnL %':>9}{'Cash':>14}{'Longs':>13}{'Shorts':>12}"
      f"{'Gross':>8}{'Net':>8}{'DD':>7}{'Pos':>7}{'BTC':>12}")
    for s in snaps:
        btc = s["market"]["BTC/USD"]["price"] or 0
        w(f"{s['ts_local'][:19]:<22}{money(s['equity']):>14}{signed_money(s['pnl']):>13}{pct(s['pnl_pct']):>9}"
          f"{money(s['usd_free']):>14}{money(s['long_value']):>13}{money(s['short_collateral']):>12}"
          f"{s['gross']:>8.1%}{s['net']:>+8.1%}{s['drawdown']:>7.2%}{s['n_long']:>3}/{s['n_short']:<3}{btc:>12,.0f}")
    w("")

    w("PORTFOLIO EVERY HOUR")
    w("-" * 100)
    for s in snaps:
        if s["kind"] in ("start", "hourly", "final"):
            w(f"{s['kind'].upper()}  {s['ts_local']}   equity {money(s['equity'])}"
              f"  PnL {signed_money(s['pnl'])} ({pct(s['pnl_pct'])})  cash {money(s['usd_free'])}"
              f"  gross {s['gross']:.1%} net {s['net']:+.1%}")
            w(render_holdings(s))

    w("REALIZED PnL BY ASSET (average-cost, before fees)")
    w("-" * 100)
    for pair, v in sorted(realized.items(), key=lambda x: x[1]):
        w(f"  {pair:<14}{signed_money(v):>14}")
    if not realized:
        w("  (no positions closed yet)")
    w("")

    w("ALL TRADES (fills, from Roostoo's order history)")
    w("-" * 100)
    w(f"{'Local time':<22}{'Pair':<12}{'Side':<12}{'Qty':>16}{'Price':>14}{'Notional':>14}"
      f"{'Fee $':>10}{'Role':>7}  Type")
    for r in fills:
        n = float(r["qty"]) * float(r["price"])
        w(f"{fmt_local(r['_ts'])[:19]:<22}{r['pair']:<12}{r['side']:<12}{float(r['qty']):>16,.6f}"
          f"{float(r['price']):>14,.6g}{money(n):>14}{float(r['fee'] or 0):>10.4f}"
          f"{r.get('role', ''):>7}  {r.get('type', '')}")
    if not fills:
        w("  (no fills)")
    w("")

    w(f"BOT FILL LOG CHECK ({len(phantoms)} phantom rows in logs/fills with no price)")
    w("-" * 100)
    w("  Roostoo reports FilledQuantity = Quantity on cancelled limit orders that never traded;")
    w("  older bot code logged those as fills. They are ignored above (exchange history is used).")
    for r in phantoms:
        w(f"  {r['ts_utc'][:19]}  {r['pair']:<12}{r['side']:<12}{float(r['qty']):>16,.6f}"
          f"  price {r['price']}  fee {r['fee']}")
    w("")

    w("HOURLY DECISION CYCLES")
    w("-" * 100)
    for d in cycles:
        tw = d.get("weights_final", {}) or {}
        biggest = sorted(tw.items(), key=lambda x: -abs(x[1]))[:8]
        top = ", ".join(f"{k.split('/')[0]} {v:+.1%}" for k, v in biggest)
        w(f"  {d.get('cycle_id')}  equity {money(float(d.get('equity', 0)))}"
          f"  gross {float(d.get('gross', 0)):.1%}"
          f"  net {float(d.get('net', 0)):+.1%}  mult {d.get('multiplier')}  act_all {d.get('act_all')}"
          f"  orders {len(d.get('orders_planned', []))}")
        if top:
            w(f"      top targets: {top}")
    if not cycles:
        w("  (no hourly cycle ran)")
    w("")

    w(f"WARNINGS AND ERRORS FROM THE ENGINE ({len(warns)} warnings, {len(errs)} errors)")
    w("-" * 100)
    for e in (warns + errs)[-60:]:
        extra = {k: v for k, v in e.items() if k not in ("ts", "level", "logger", "msg")}
        w(f"  {e['ts'][:19]} {e['level']:<8}{e['msg']}  {json.dumps(extra, default=str)[:180]}")
    if not (warns or errs):
        w("  (none)")
    w("")

    console = run_dir / "bot_console.log"
    if console.exists():
        tail = console.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        w("BOT CONSOLE (last 40 lines)")
        w("-" * 100)
        L.extend("  " + t for t in tail)
    out = run_dir / f"overnight_report_{start:%Y%m%d_%H%M}_to_{end:%Y%m%d_%H%M}_UTC.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def latest_run_dir() -> Path | None:
    runs = sorted((REPO / "overnight" / "runs").glob("*"))
    return runs[-1] if runs else None


if __name__ == "__main__":
    print(os.getcwd())
