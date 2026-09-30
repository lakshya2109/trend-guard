# trend-guard

Autonomous trading bot for the Roostoo Quant Hackathon. BUILD_SPEC.md is the source of truth for infrastructure, execution safety and compliance. Strategy v2 (docs/STRATEGY.md, src/trend_guard/strategy/book.py) supersedes BUILD_SPEC Sections 5.5-5.6, 6, 7 and 11 (the v1 signal, overlays and backtest failed after costs; see the research log).

## Rules
- Strategy and risk code are pure functions: no network, no disk, no clock reads.
- All thresholds come from config/strategy.yaml via src/trend_guard/config.py. No magic numbers.
- Never place live orders unless ALLOW_LIVE_ORDERS=1. Never run live mode or scripts/probe_live.py yourself: ask the human to run them.
- Never retry an order-creating request blindly. Reconcile via query_order.
- Never read or print .env or /etc/trend-guard/env. Never commit secrets.
- Short inline comments in all code. Type hints everywhere. Python 3.11.
- Conventional commits. Use `strat:` for any strategy/parameter change with a one-line reason in the body.
- Commits are authored by lakshya2109 only: never add `Co-Authored-By` or any AI attribution trailer.
- Do not copy code from other public Roostoo hackathon repos (anti-plagiarism).
- Stop at every milestone checkpoint in BUILD_SPEC.md Section 14 and wait for approval.
- If the spec is ambiguous or conflicts with docs/vendor/roostoo-api, ask. Known unknowns: BUILD_SPEC.md Section 16.

## Commands
- Tests: `pytest -q`   Lint: `ruff check .`
- Data: `python scripts/download_history.py`
- Backtest: `python backtest/run_backtest.py --config config/strategy.yaml`   Design search + PBO: `python backtest/sweep.py`
- Paper mode: `python -m trend_guard.main --mode paper`

## Progress log
(Append one line per completed milestone: date, milestone, commit hash, notes.)
