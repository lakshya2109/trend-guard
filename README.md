# trend-guard

**TREND-GUARD: vol-targeted long/short trend ensemble with a drawdown governor.**

Autonomous trading bot for the Roostoo HK vs AU vs IN Quant Trading Hackathon (live 4 to 17 Oct 2026).

## 1. Overview

_TODO: one-paragraph strategy summary._

## 2. Why this design

See [`docs/STRATEGY.md`](docs/STRATEGY.md) for the full research rationale and citations.

## 3. Architecture

_TODO: Mermaid architecture diagram and module map._

## 4. Rule compliance table

| Rule | Enforced in |
|---|---|
| No leverage (long notional + short collateral <= equity) | _TODO_ |
| Never place live orders unless `ALLOW_LIVE_ORDERS=1` | _TODO_ |
| No manual API calls with the competition key | _TODO_ |
| No blind retries on order-creating requests | _TODO_ |
| Signals only on closed hourly bars | _TODO_ |

## 5. Risk management

_TODO: every overlay, with defaults, once implemented._

## 6. Execution

_TODO: maker-first, reconciliation, no blind retries._

## 7. Backtest methodology and results

_TODO: see `reports/` once generated._

## 8. How to run

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env        # fill in keys

pytest -q                                            # tests
ruff check .                                         # lint
python backtest/run_backtest.py --config config/strategy.yaml   # backtest
python -m trend_guard.main --mode paper              # paper trading
```

## 9. Deployment

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## 10. Logs and audit trail

_TODO: how trades tie to commits._

## 11. Limitations and honest caveats

_TODO._
