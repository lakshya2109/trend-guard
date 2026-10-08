# RUNBOOK

Operational runbook for trend-guard during the competition window (4 Oct 12:00 UTC to 18 Oct 12:00 UTC, 2026). See
BUILD_SPEC.md for the full spec; this is the day-to-day "how do I..." reference.

## First-time EC2 setup

1. On the **dev machine**: generate a read-only deploy key and register it with the private repo.
   ```bash
   ssh-keygen -t ed25519 -f deploy_key -N ""
   gh repo deploy-key add deploy_key.pub --title ec2 -R <owner>/trend-guard
   ```
   Copy `deploy_key` (the private half) to the EC2 instance at `~/.ssh/id_ed25519`, mode 600.
2. On **EC2**: edit `deploy/setup_ec2.sh` to set `REPO_SSH`, then run it:
   ```bash
   bash deploy/setup_ec2.sh
   ```
3. Fill in `/etc/trend-guard/env` with real `ROOSTOO_API_KEY` / `ROOSTOO_SECRET_KEY`. Leave
   `ALLOW_LIVE_ORDERS=0` until you have deliberately decided to go live.
4. Verify read-only access: `sudo -u trendguard /opt/trend-guard/.venv/bin/python scripts/smoke_readonly.py`
5. Download the hourly bar history the strategy needs (public Binance data, a few minutes):
   `sudo -u trendguard /opt/trend-guard/.venv/bin/python scripts/download_history.py`
   The engine tops this cache up every hour afterwards.
6. Dry-run one hourly cycle in paper mode (live prices, simulated fills, no orders):
   `sudo -u trendguard /opt/trend-guard/.venv/bin/python -m trend_guard.main --mode paper --once`
7. Start the ticker recorder (safe, read-only, useful as an audit of Roostoo prices):
   `sudo systemctl start trend-guard-recorder`
8. Only when ready to trade: set `ALLOW_LIVE_ORDERS=1` in `/etc/trend-guard/env`, then
   `sudo systemctl start trend-guard`.

## Redeploying a new version

```bash
# on the dev machine, after tests pass and the change is committed
git tag -a v1.0.x -m "reason"
git push --tags
ssh ec2 'cd /opt/trend-guard && git fetch --tags && git checkout v1.0.x && sudo systemctl restart trend-guard'
```

Redeploys are allowed at any point during the competition (organizers confirmed teams may stop,
restart, and redeploy). Always tag before deploying so every live run maps to an exact commit --
this is also what Screen 1 (commit-history traceability) checks.

## Kill switch (HALT)

The engine checks for a HALT file every loop iteration (BUILD_SPEC.md Section 10.3). While it
exists, the bot cancels its own pending orders and does nothing else -- it keeps running (so the
service doesn't flap) but places no new risk.

```bash
# halt
ssh ec2 'sudo touch /etc/trend-guard/HALT'
# resume
ssh ec2 'sudo rm /etc/trend-guard/HALT'
```

Locally (paper mode), the HALT path is `./HALT` in the working directory.

## Checking the bot is healthy

```bash
ssh ec2 'sudo systemctl status trend-guard trend-guard-recorder'
ssh ec2 'sudo journalctl -u trend-guard -n 200 --no-pager'
ssh ec2 'tail -n 50 /opt/trend-guard/logs/decisions/$(date -u +%F).jsonl'
```

## Live log view in tmux

The bot writes JSONL / CSV logs only (`logs/decisions`, `logs/fills`, `logs/engine`, `logs/api`).
`scripts/watch_logs.py` follows them read-only and prints one readable line per hourly cycle, fill,
activity-rule trade, warning and error. To add it under a bot already running in tmux, from a normal
SSH shell (not inside tmux), without touching the bot's checkout:

```bash
mkdir -p ~/tg-tools && cd <the bot's repo directory>
git fetch origin trend-guard-final-v2
git show origin/trend-guard-final-v2:scripts/watch_logs.py > ~/tg-tools/watch_logs.py
git show origin/trend-guard-final-v2:deploy/add_log_pane.sh > ~/tg-tools/add_log_pane.sh
bash ~/tg-tools/add_log_pane.sh            # or: bash ~/tg-tools/add_log_pane.sh <session-name>
```

It splits the bot's window and runs the viewer below the bot (restarting it if stopped); a plain
`tmux attach` then shows both. Only tmux layout commands are used. Do NOT add tmux hooks: a
`client-attached` hook crashed the tmux server on the contest box (tmux 3.2a) and stopped the bot. In tmux: `Ctrl-b o` switches panes,
`Ctrl-b z` zooms one pane, `Ctrl-b d` detaches. Never press Ctrl-C in the bot's pane.

## Dashboard access

The Streamlit dashboard binds to `127.0.0.1:8501` on the EC2 host only -- never expose it
publicly. Reach it via an SSH tunnel:

```bash
ssh -L 8501:localhost:8501 ubuntu@<ec2-host>
# then open http://localhost:8501 on your own machine
# on the EC2 host, run: TREND_GUARD_DB=state/live.db streamlit run monitor/dashboard.py --server.address 127.0.0.1
```

## Going public before 14 Oct

1. `pip install detect-secrets` (or `gitleaks detect`) over the full history; if anything is
   found, rotate the key and rewrite history.
2. Confirm `.env`, `data/`, `logs/`, `state/*.db` were never committed (`git log --all --full-history -- .env`).
3. `gh repo edit --visibility public --accept-visibility-change-consequences`.
4. Submit the repo link.

## Incident checklist

- **Order state unknown / API timeout on an order**: the client already raises
  `OrderStateUnknownError` and never blind-resubmits (BUILD_SPEC.md Section 8.3). Check
  `logs/api/*.jsonl` for the ambiguous request and confirm the reconciliation loop matched or
  discarded it via `query_order`.
- **Shorts got disabled mid-competition**: `ShortsDisabledError` is caught once by the engine,
  which must permanently disable the short sleeve for the session and log it. Confirm in
  `logs/decisions/*.jsonl` and consider a `strat:` commit to set short budgets to 0 explicitly if
  this happens.
- **No fills for a full UTC day**: the engine logs a warning at 18:00 UTC (BUILD_SPEC.md Section
  8.4). Two days in a row -> raise it and consider lowering the deadband via a `strat:` commit
  with a one-line rationale. Never add fake trades.
- **Binance klines unreachable from the host**: the engine logs `bars unavailable` per pair and
  trades only the pairs it has fresh data for; `data-api.binance.vision` is tried before
  `api.binance.com`. If every pair fails, the cycle is skipped (no trades on stale data).
- **Breaker engaged** (`breaker.mult` 0.5 in the decisions log): the account is 20% below its peak.
  It releases automatically once the drawdown recovers inside 5%.
