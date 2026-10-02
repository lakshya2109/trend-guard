"""Simulated session: how the live bot would have traded between two dates, with every trade logged.

    python scripts/simulate.py --start 2026-09-01 --end 2026-09-15
    python scripts/simulate.py --start "2026-09-01 12:00" --end "2026-09-15 12:00" --capital 100000

Times are UTC. The session starts flat in cash at --start and is valued at --end (open positions are
marked, not closed). No bar after --end is ever loaded, and a leakage check recomputes every target with
the later half of the session deleted to prove no decision used future data (backtest/session.py).
Prints the ratios and metrics, and writes a full trade log to sim_<start>_<end>.txt (or --out).
Run scripts/download_history.py first so data/bars/1h covers the window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # import trend_guard even without `pip install -e .`

import pandas as pd  # noqa: E402

from backtest.run_backtest import load_panel  # noqa: E402
from backtest.session import metrics, run_session  # noqa: E402
from trend_guard.config import config_hash, load_strategy_config, load_symbols  # noqa: E402

SLEEVE_NAMES = {
    "v2_trend": "v2 trend",
    "v2_momentum": "v2 momentum",
    "team_trend": "Gunek trend",
    "team_momentum": "swyam momentum",
    "jump_guard": "jump guard",
    "crash_hedge": "crash hedge",
}


def _ts(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def format_report(m: dict, res, start: pd.Timestamp, end: pd.Timestamp, cfg_hash: str, risk: float) -> str:
    L = []
    L.append("=" * 100)
    L.append(
        f"SIMULATED SESSION  {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC   "
        f"(config {cfg_hash}, risk x{risk:g})"
    )
    L.append("=" * 100)
    L.append(f"Data leakage check: {'PASS' if res.leakage_ok else 'FAIL'} - {res.leakage_detail}")
    L.append("")
    L.append("RETURNS")
    L.append(f"  start equity        ${m['start_equity']:>12,.2f}")
    L.append(f"  end equity          ${m['end_equity']:>12,.2f}")
    L.append(f"  total return        {m['total_return']:>+12.2%}")
    L.append(
        f"  best / worst day    {m['best_day']:>+12.2%} / {m['worst_day']:+.2%}   "
        f"positive days {m['positive_days']}"
    )
    L.append("")
    L.append("CONTEST RATIOS (daily marks from the start, non-annualised)")
    L.append(f"  score = 0.4 Sortino + 0.3 Sharpe + 0.3 Calmar   {m['score']:>9.4f}")
    L.append(f"  Sortino             {m['sortino']:>12.4f}   (annualised ~{m['sortino_annualised']:.2f})")
    L.append(f"  Sharpe              {m['sharpe']:>12.4f}   (annualised ~{m['sharpe_annualised']:.2f})")
    L.append(f"  Calmar              {m['calmar']:>12.4f}   (mean daily return / max daily drawdown)")
    L.append("")
    L.append("PREVIOUS EDITION'S LEADERBOARD VIEW (backtest/benchmark.py)")
    w = m["winner"]
    L.append(
        f"  Calmar (total return / max drawdown) {m['calmar_total']:>8.3f}   "
        f"last year's winner {w['calmar_total']:.2f}"
    )
    L.append(
        f"  Sharpe / Sortino (daily)             {m['sharpe']:>8.3f} / {m['sortino']:.3f}   winner "
        f"{w['sharpe']:.2f} / {w['sortino']:.2f}"
    )
    L.append(f"  active days (dates with a trade)     {m['active_dates']:>8d}   score = active days x ratios")
    L.append(f"  estimated rank among their 52 teams  {m['rank_vs_field']:>8d}")
    L.append("")
    L.append("RISK AND ACTIVITY")
    L.append(
        f"  max drawdown        {m['max_drawdown_daily']:>12.2%} daily marks, "
        f"{m['max_drawdown_hourly']:.2%} hourly"
    )
    L.append(
        f"  exposure            avg gross {m['avg_gross_exposure']:.0%}, "
        f"avg net {m['avg_net_exposure']:+.0%}, "
        f"max gross {m['max_gross_exposure']:.0%}"
    )
    L.append(
        f"  trades              {m['trades']:>12d}   active trading days {m['active_trading_days']} "
        f"(contest needs >= 8 of 14)"
    )
    L.append(
        f"  fees + slippage     ${m['fees_paid']:>12,.2f}   turnover {m['turnover_x_capital']:.2f}x capital"
    )
    L.append("")
    L.append("DAILY RETURNS")
    for d, v in m["daily_returns"].items():
        L.append(f"  {d:%Y-%m-%d %H:%M}  {v:+.3%}")
    L.append("")
    L.append("P&L BY PAIR (after fees)")
    for pair, v in res.pnl_by_pair.items():
        L.append(f"  {pair:14s} ${v:>+11,.2f}")
    L.append("")
    L.append(
        f"TRADE LOG ({len(res.trades)} trades)  weights are fractions of equity (+ long, - short collateral)"
    )
    L.append(
        f"  {'time (UTC)':16s} {'pair':12s} {'action':6s} {'reason':7s} {'qty':>14s} {'price':>12s} "
        f"{'notional $':>11s} {'fee $':>8s} {'weight':>17s}  driven by"
    )
    for t in res.trades:
        drivers = ", ".join(
            f"{SLEEVE_NAMES[k]} {v:+.3f}" for k, v in sorted(t.drivers.items(), key=lambda x: -abs(x[1]))
        )
        L.append(
            f"  {t.time:%Y-%m-%d %H:%M} {t.pair:12s} {t.action:6s} {t.reason:7s} {t.qty:>14.6g} "
            f"{t.price:>12.6g} {t.notional:>11,.2f} {t.fee:>8,.2f} "
            f"{t.weight_before:>+7.3f} -> {t.weight_after:>+7.3f}"
            f"  {drivers}"
        )
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="UTC, e.g. 2026-09-01 or '2026-09-01 12:00'")
    ap.add_argument("--end", required=True, help="UTC, exclusive of any later data")
    ap.add_argument("--capital", type=float, default=None, help="default: backtest.capital from the config")
    ap.add_argument("--config", default=str(ROOT / "config" / "strategy.yaml"))
    ap.add_argument("--bars", default=str(ROOT / "data" / "bars" / "1h"))
    ap.add_argument("--out", default=None, help="trade log path (default: sim_<start>_<end>.txt here)")
    args = ap.parse_args(argv)

    start, end = _ts(args.start), _ts(args.end)
    if end <= start:
        print("--end must be after --start", file=sys.stderr)
        return 2
    cfg = load_strategy_config(args.config)
    sym = load_symbols(ROOT / "config" / "symbols.yaml")
    close, qv = load_panel(Path(args.bars), sym.excluded, sym.overrides)
    # no leakage: drop every bar after `end`; keep only the history the live bot would have loaded
    hist_start = start - pd.Timedelta(hours=cfg.execution.history_hours)
    keep = (close.index >= hist_start) & (close.index + pd.Timedelta(hours=1) <= end)
    close, qv = close[keep], qv[keep]
    if close.empty or close.index[0] > start or close.index[-1] + pd.Timedelta(hours=1) < end:
        print(f"bar cache does not cover {start} -> {end}: run scripts/download_history.py", file=sys.stderr)
        return 1
    # the session's first decision is the bar that closes at `start`
    first = close.index[close.index + pd.Timedelta(hours=1) >= start][0]
    capital = args.capital or cfg.backtest.capital
    res = run_session(close, qv, cfg, first, end, capital)
    m = metrics(res, capital)
    report = format_report(m, res, start, end, config_hash(args.config), cfg.book.risk_multiplier)
    out = Path(args.out) if args.out else Path.cwd() / f"sim_{start:%Y%m%d_%H%M}_{end:%Y%m%d_%H%M}.txt"
    out.write_text(report, encoding="utf-8")
    head = report.split("TRADE LOG")[0]
    print(head)
    print(f"full trade log ({len(res.trades)} trades) written to {out}")
    return 0 if res.leakage_ok else 3


if __name__ == "__main__":
    sys.exit(main())
