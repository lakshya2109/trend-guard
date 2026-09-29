# trend-guard

Autonomous trading bot for the Roostoo Quant Hackathon. BUILD_SPEC.md is the source of truth: read the relevant section before any change.

## Rules
- Strategy and risk code are pure functions: no network, no disk, no clock reads.
- All thresholds come from config/strategy.yaml via src/trend_guard/config.py. No magic numbers.
- Never place live orders unless ALLOW_LIVE_ORDERS=1. Never run live mode or scripts/probe_live.py yourself: ask the human to run them.
- Never retry an order-creating request blindly. Reconcile via query_order.
- Never read or print .env or /etc/trend-guard/env. Never commit secrets.
- Short inline comments in all code. Type hints everywhere. Python 3.11.
- Conventional commits. Use `strat:` for any strategy/parameter change with a one-line reason in the body.
- Do not copy code from other public Roostoo hackathon repos (anti-plagiarism).
- Stop at every milestone checkpoint in BUILD_SPEC.md Section 14 and wait for approval.
- If the spec is ambiguous or conflicts with docs/vendor/roostoo-api, ask. Known unknowns: BUILD_SPEC.md Section 16.

## Commands
- Tests: `pytest -q`   Lint: `ruff check .`
- Backtest: `python backtest/run_backtest.py --config config/strategy.yaml`
- Paper mode: `python -m trend_guard.main --mode paper`

## Progress log
(Append one line per completed milestone: date, milestone, commit hash, notes.)
