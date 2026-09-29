# TREND-GUARD: Build Spec for Claude Code

Autonomous trading bot for the Roostoo HK vs AU vs IN Quant Trading Hackathon (live 4 to 17 Oct 2026). This file is the single source of truth.

**How to start:** put this file in an empty folder named `trend-guard`, run `claude` in that folder, and send:

> Read BUILD_SPEC.md fully. Then execute Milestone 0 exactly as written, including creating CLAUDE.md and .claude/. Stop at each checkpoint and wait for me.

Tips: use plan mode for each milestone so you can approve the plan before code is written; run `/checkpoint` (created in Milestone 0) at the end of each milestone; start a fresh session per milestone if context gets long, since CLAUDE.md and this spec carry the state.

---

## 0. Instructions to the coding agent

1. Read the whole document before writing code.
2. Execute milestones **in order** (Section 14). After each milestone, run the tests, commit, ask before pushing, and **stop at the checkpoint** with a short summary of what was built, what was tested and any open questions.
3. **Never send a live order** unless the environment variable `ALLOW_LIVE_ORDERS=1` is set. Default behaviour is paper mode.
4. **Never commit secrets.** API keys live in `.env` (gitignored) locally and `/etc/trend-guard/env` on EC2.
5. Every threshold comes from `config/strategy.yaml`. No magic numbers in code.
6. Strategy logic must be pure (no I/O). Backtest and live share the exact same strategy and risk code.
7. Write short inline comments in all code. Type hints everywhere. Python 3.11.
8. Commits follow Conventional Commits (`feat:`, `fix:`, `test:`, `chore:`, `docs:`, `refactor:`, `strat:` for strategy parameter changes). Each `strat:` commit body states the reason in one line.
9. If the spec is ambiguous or conflicts with the official Roostoo docs, **ask**, do not guess. Items already known to be uncertain are in Section 16.
10. Do not copy code from other public hackathon repos (anti-plagiarism rule). Write it from this spec and the official docs.

---

## 1. Context in one page

- Mock exchange: `https://mock-api.roostoo.com`. Official docs: https://github.com/roostoo/Roostoo-API-Documents (clone it into `docs/vendor/` for reference, do not modify).
- $100,000 starting USD. Spot only, "1x long and short", **no leverage**.
- Fees: 0.10% taker (MARKET), 0.05% maker (LIMIT that rests). Shorts: flat 0.10% on open and close regardless of order type.
- Banned: HFT, market making, arbitrage (including cross-venue latency arbitrage), manual trading, manual API calls with the competition key.
- Must run autonomously on the provided AWS EC2. At least 8 active trading days (active = at least 1 trade that day).
- Scoring: Screen 1 compliance (trade log integrity, commit history) then Screen 2 top 20 per region by return, then Screen 3 composite `0.4 Sortino + 0.3 Sharpe + 0.3 Calmar`, then Screen 4 code review (clean repo, clear logic, runs continuously).
- Repo must be **open source** by submission (before 14 Oct). We keep it **private until then**.

**Strategy summary:** long/short trend ensemble (Donchian position + volatility-normalised momentum over 6 lookbacks) on hourly bars, gated by a market regime, sized by inverse volatility, scaled to a portfolio volatility target, then multiplied down by risk overlays (drawdown governor, daily brake, shock filter, crowding filter from perp funding/OI, end-game lock-in, optional LLM news veto). Maker-first execution.

---

## 2. Milestone 0: repo bootstrap (run these)

Prerequisites on the dev machine: `git`, `gh` (GitHub CLI), Python 3.11.

```bash
# 1. check GitHub CLI auth (ask user to run `gh auth login` if this fails)
gh auth status

# 2. create project
mkdir trend-guard && cd trend-guard
git init -b main
python3.11 -m venv .venv && source .venv/bin/activate

# 3. create PRIVATE repo on GitHub and link it
gh repo create trend-guard --private --source=. --remote=origin \
  --description "Autonomous vol-targeted long/short trend bot for the Roostoo Quant Hackathon"

# 4. vendor the official API docs for reference (read-only)
git clone --depth 1 https://github.com/roostoo/Roostoo-API-Documents.git docs/vendor/roostoo-api
rm -rf docs/vendor/roostoo-api/.git
```

Create these files in Milestone 0:

**`.gitignore`**
```
.venv/
__pycache__/
*.pyc
.env
.env.*
!.env.example
data/
logs/
state/*.db
reports/
.pytest_cache/
.ruff_cache/
.DS_Store
.claude/settings.local.json
```

**`.env.example`**
```
ROOSTOO_API_KEY=
ROOSTOO_SECRET_KEY=
ROOSTOO_BASE_URL=https://mock-api.roostoo.com
ALLOW_LIVE_ORDERS=0          # must be 1 to send real orders
BOT_MODE=paper               # paper | live
ANTHROPIC_API_KEY=           # optional, only if news overlay enabled
```

**`requirements.txt`** (keep lean)
```
requests>=2.32
numpy>=1.26
pandas>=2.2
pyarrow>=15
pyyaml>=6
pydantic>=2.6
python-dotenv>=1.0
streamlit>=1.35
feedparser>=6.0
anthropic>=0.40
yfinance>=0.2.40
```

**`requirements-dev.txt`**
```
pytest>=8
pytest-cov>=5
ruff>=0.5
responses>=0.25
```

**`pyproject.toml`**: ruff config (line length 110, target py311), pytest config (`testpaths = ["tests"]`).

**`.github/workflows/ci.yml`**: on push and PR, set up Python 3.11, install both requirement files, run `ruff check .` and `pytest -q`.

**`LICENSE`**: MIT (needed because the repo becomes open source).

**`README.md`**: skeleton with the section headings from Section 13.

**`CLAUDE.md`** (Claude Code reads this at the start of every session):
```markdown
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
```

**`.claude/settings.json`** (project permissions, committed):
```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allow": [
      "Bash(pytest *)",
      "Bash(python -m pytest *)",
      "Bash(ruff *)",
      "Bash(pip install *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git log *)",
      "Bash(python backtest/*)",
      "Bash(python scripts/download_history.py *)",
      "Bash(python -m trend_guard.main --mode paper *)"
    ],
    "ask": [
      "Bash(git push *)",
      "Bash(gh *)",
      "Bash(ssh *)",
      "Bash(scp *)"
    ],
    "deny": [
      "Read(./.env)",
      "Bash(python -m trend_guard.main --mode live *)",
      "Bash(python scripts/probe_live.py *)"
    ]
  }
}
```
Add `.claude/settings.local.json` to `.gitignore` (personal overrides). The code-level `ALLOW_LIVE_ORDERS` gate is the real safety control; the deny rules are a second layer.

**`.claude/commands/checkpoint.md`** (becomes the `/checkpoint` command):
```markdown
Run the milestone checkpoint:
1. `ruff check .` and `pytest -q`; fix failures before continuing.
2. Summarise what was built this milestone, what was tested, and any deviations from BUILD_SPEC.md.
3. List open questions for the human.
4. Commit with a conventional message, then ask before pushing.
5. Append a line to the Progress log in CLAUDE.md.
Then stop and wait.
```

**`.claude/commands/backtest.md`** (`/backtest`):
```markdown
Run `python backtest/run_backtest.py --config config/strategy.yaml`, then summarise reports/<latest>/summary.md as a table:
p25/median composite (3 definitions), median return, p95 MDD, red-day share, trades, fees, and each baseline. Flag anything that fails the Section 11.3 objective.
```

**`.claude/commands/predeploy.md`** (`/predeploy`):
```markdown
Go through BUILD_SPEC.md Section 17 compliance checklist item by item against the current code. For each item report PASS/FAIL with file and line evidence. Also confirm CI is green and the working tree is clean.
```

Keep this spec in the repo root as `BUILD_SPEC.md`. Commit: `chore: bootstrap repo`. Ask before pushing.

**Checkpoint 0:** confirm repo is private (`gh repo view --json visibility`), CI green, CLAUDE.md and .claude/ committed. Run `/checkpoint`.

---

## 3. Repository layout

```
trend-guard/
  BUILD_SPEC.md
  CLAUDE.md
  .claude/settings.json
  .claude/commands/
  README.md
  LICENSE
  pyproject.toml
  requirements.txt
  requirements-dev.txt
  .env.example
  config/
    strategy.yaml          # all parameters
    events.yaml            # scheduled macro events (UTC)
    symbols.yaml           # Roostoo pair -> Binance / yfinance symbol overrides
  src/trend_guard/
    __init__.py
    config.py              # pydantic models, loads yaml + env
    logging_setup.py       # JSONL loggers
    clock.py               # Clock interface: RealClock, SimClock
    roostoo/
      client.py            # REST client, signing, v3 + v6
      models.py            # typed responses
      rules.py             # precision, MiniOrder, decimal formatting
      errors.py
    data/
      binance_vision.py    # bulk history download (klines, funding, metrics)
      binance_live.py      # recent klines, funding, OI via REST
      yf_stocks.py         # bStock underlying history
      ticker_recorder.py   # Roostoo ticker snapshots -> bars
      bars.py              # resampling, alignment, gap checks
      universe.py          # eligibility from exchangeInfo + liquidity + spread
    strategy/
      signals.py           # donchian, tsmom, ensemble
      regime.py            # market index, breadth, hysteresis state machine
      portfolio.py         # sizing, covariance, vol target, caps
    risk/
      overlays.py          # dd governor, daily brake, endgame, event, crowding, tournament
      protective.py        # 1-min loop: trailing stops, shock filter
      news_llm.py          # optional veto-only LLM overlay
    execution/
      broker.py            # Broker protocol
      sim_broker.py        # paper / backtest fills
      live_broker.py       # wraps RoostooClient
      order_manager.py     # target -> orders, maker-first, reprice, reconcile
    state/
      store.py             # SQLite
      schema.sql
    engine.py              # orchestrates one hourly cycle + minute loop
    main.py                # CLI entrypoint
    metrics.py             # Sharpe, Sortino, Calmar, composite (3 definitions)
  backtest/
    engine.py              # hourly event-driven backtest using same strategy code
    windows.py             # rolling 14-day evaluation
    pbo.py                 # CSCV probability of backtest overfitting
    report.py              # markdown + csv outputs
    run_backtest.py        # CLI
  scripts/
    download_history.py
    smoke_readonly.py      # serverTime, exchangeInfo, ticker, balance (no orders)
    probe_live.py          # prep-window probes via committed code, gated
  monitor/
    dashboard.py           # Streamlit, read-only SQLite
  deploy/
    setup_ec2.sh
    trend-guard.service
    recorder.service
    logrotate.conf
  tests/
  docs/
    vendor/roostoo-api/
    STRATEGY.md            # judge-facing explanation
    RUNBOOK.md
```

---

## 4. Roostoo API client (`src/trend_guard/roostoo/`)

### 4.1 Facts from the official docs (do not deviate)

- Auth levels: `RCL_NoVerification` (no params), `RCL_TSCheck` (`timestamp` param), `RCL_TopLevelCheck` (headers `RST-API-KEY` and `MSG-SIGNATURE` plus `timestamp`).
- Signature: sort params by key, join as `k1=v1&k2=v2`, HMAC-SHA256 with the secret key, lowercase hex. For GET this string is the query string; for POST it is the body. POST header `Content-Type: application/x-www-form-urlencoded`.
- Timestamp: 13-digit ms, must be within 60,000 ms of server time.
- **Only documented parameters are included in the server-side signature.** Never send extra params.
- Failures return HTTP 200 with `Success: false` and `ErrMsg`. Always check `Success`.
- Zero-valued fields are **omitted** from responses. Treat missing numeric fields as `0`.
- `/v3/pending_count` returns `Success: false, ErrMsg: "no pending order under this account"` when empty. That is not an error.
- `/v3/query_order` returns `Success: false, ErrMsg: "no order matched"` when empty. Not an error.
- `/v3/cancel_order` with neither `order_id` nor `pair` cancels **all** pending orders. Require an explicit `cancel_all=True` flag in our wrapper.

### 4.2 Endpoints to implement

| Method | Path | Auth | Params |
|---|---|---|---|
| GET | `/v3/serverTime` | none | none |
| GET | `/v3/exchangeInfo` | none | none |
| GET | `/v3/ticker` | TSCheck | `timestamp`, optional `pair` |
| GET | `/v3/balance` | signed | `timestamp` |
| GET | `/v3/pending_count` | signed | `timestamp` |
| POST | `/v3/place_order` | signed | `pair`, `side` (BUY/SELL), `type` (LIMIT/MARKET), `quantity`, `timestamp`, `price` if LIMIT |
| POST | `/v3/query_order` | signed | `timestamp` + either `order_id` alone, or `pair` (+ `offset`, `limit`, `pending_only` TRUE/FALSE) |
| POST | `/v3/cancel_order` | signed | `timestamp` + at most one of `order_id`, `pair` |
| POST | `/v6/short_open` | signed | `pair`, `collateral` (USD, min 1), `timestamp`, optional `order_type=LIMIT` + `price` |
| POST | `/v6/short_close` | signed | `pair`, `timestamp`, optional `close_qty` (precedence) or `close_pct` (0, 100] |
| GET | `/v6/short_positions` | signed | `timestamp` |

Short facts: market short opens at `MaxBid`; close always market at `MinAsk`; reduce-only; loss capped at collateral; `PositionValue = Collateral + UnrealizedPNL`; open fee is charged when the request is accepted (including pending LIMIT). Error `this competition does not allow short positions` means shorts are disabled: the client must raise `ShortsDisabledError` and the engine must permanently disable the short sleeve for the session and log it.

### 4.3 Client code (implement exactly this core, extend for all endpoints)

```python
# src/trend_guard/roostoo/client.py
import hashlib, hmac, time, logging
from decimal import Decimal
import requests

log = logging.getLogger("api")

class RoostooAPIError(Exception): ...
class OrderStateUnknownError(Exception): ...   # order may or may not exist server-side
class ShortsDisabledError(Exception): ...

class RoostooClient:
    def __init__(self, base_url: str, api_key: str, secret_key: str,
                 timeout: float = 10.0, rate_limiter=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.secret = secret_key.encode()
        self.timeout = timeout
        self.session = requests.Session()
        self.offset_ms = 0                      # server_time - local_time
        self.rl = rate_limiter                  # token bucket, may be None in tests

    # --- signing ---
    @staticmethod
    def build_total_params(params: dict) -> str:
        # sort by key and join k=v with '&' exactly as the docs require
        return "&".join(f"{k}={params[k]}" for k in sorted(params))

    def sign(self, total_params: str) -> str:
        # HMAC-SHA256 hex digest of the param string
        return hmac.new(self.secret, total_params.encode(), hashlib.sha256).hexdigest()

    def now_ms(self) -> str:
        # local clock corrected by measured server offset
        return str(int(time.time() * 1000) + self.offset_ms)

    def sync_clock(self) -> int:
        t0 = time.time() * 1000
        server = self._get("/v3/serverTime", {}, auth="none")["ServerTime"]
        t1 = time.time() * 1000
        self.offset_ms = int(server - (t0 + t1) / 2)   # midpoint estimate
        if abs(self.offset_ms) > 30_000:
            log.warning("clock offset large", extra={"offset_ms": self.offset_ms})
        return self.offset_ms

    # --- transport ---
    def _get(self, path: str, params: dict, auth: str, retries: int = 3) -> dict:
        # GETs are idempotent: safe to retry with backoff
        for attempt in range(retries):
            p = dict(params)
            headers = {}
            if auth in ("ts", "signed"):
                p["timestamp"] = self.now_ms()
            if auth == "signed":
                headers = {"RST-API-KEY": self.api_key,
                           "MSG-SIGNATURE": self.sign(self.build_total_params(p))}
            try:
                if self.rl: self.rl.acquire()
                r = self.session.get(self.base_url + path, params=p, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                log.info("GET", extra={"path": path, "ok": data.get("Success", True)})
                return data
            except (requests.RequestException, ValueError) as e:
                log.warning("GET failed", extra={"path": path, "attempt": attempt, "err": str(e)})
                time.sleep(2 ** attempt)
        raise RoostooAPIError(f"GET {path} failed after {retries} attempts")

    def _post(self, path: str, params: dict) -> dict:
        # POSTs that create orders are NEVER retried; ambiguous outcome -> OrderStateUnknownError
        p = dict(params)
        p["timestamp"] = self.now_ms()
        body = self.build_total_params(p)
        headers = {"RST-API-KEY": self.api_key, "MSG-SIGNATURE": self.sign(body),
                   "Content-Type": "application/x-www-form-urlencoded"}
        try:
            if self.rl: self.rl.acquire(order=True)
            r = self.session.post(self.base_url + path, data=body, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as e:
            log.error("POST ambiguous", extra={"path": path, "params": params, "err": str(e)})
            raise OrderStateUnknownError(path) from e
        log.info("POST", extra={"path": path, "params": params, "resp": data})
        if not data.get("Success", False) and "does not allow short" in data.get("ErrMsg", ""):
            raise ShortsDisabledError(data["ErrMsg"])
        return data
```

Note: `query_order` and `cancel_order` are POSTs but are safe to retry. Give `_post` a `retry_safe: bool` argument; when `True`, retry like `_get`.

### 4.4 Rules and formatting (`rules.py`)

```python
from decimal import Decimal, ROUND_DOWN

def truncate(x: Decimal, decimals: int) -> Decimal:
    # floor toward zero at the given precision, exact decimal math
    q = Decimal(1).scaleb(-decimals)
    return x.quantize(q, rounding=ROUND_DOWN)

def fmt(x: Decimal) -> str:
    # plain decimal string, never scientific notation
    s = format(x.normalize(), "f")
    return s if s not in ("-0", "") else "0"
```

`TradingRule(pair, price_precision, amount_precision, mini_order)` with `clamp_qty`, `clamp_price`, `meets_min(price, qty)` (`price * qty > mini_order`). Built from `/v3/exchangeInfo`.

### 4.5 Rate limiter

Token bucket in `roostoo/ratelimit.py`: default 30 requests/minute total, orders at least 1.0 s apart, configurable. On repeated failures (3 consecutive), halve the rate for 10 minutes.

### 4.6 Required tests (`tests/test_client.py`)

- Signature matches the docs vector: params `pair=BNB/USD, quantity=2000, side=BUY, timestamp=1580774512000, type=MARKET`, secret `S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep` produces `20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff`.
- Missing numeric fields parse as 0.
- `no pending order` and `no order matched` return empty results, not exceptions.
- POST timeout raises `OrderStateUnknownError` and is not retried.
- Shorts-disabled message raises `ShortsDisabledError`.
- `fmt(Decimal("0.00001")) == "0.00001"`; truncation of `0.29` at 2 decimals equals `0.29`.

---

## 5. Data layer (`src/trend_guard/data/`)

### 5.1 Symbol mapping

Roostoo pair `COIN/USD` maps to Binance spot `COINUSDT` by default. `config/symbols.yaml` holds overrides and marks special assets:

```yaml
overrides:        # roostoo_pair: binance_symbol
  "PEPE/USD": "PEPEUSDT"
asset_class:      # anything not listed = crypto
  "PAXG/USD": gold
bstocks:          # roostoo_pair: yfinance ticker of the underlying
  "NVDA/USD": NVDA
  "MSFT/USD": MSFT
  "GOOGL/USD": GOOGL
  "META/USD": META
crypto_beta_stocks: ["COIN/USD", "MSTR/USD", "CRCL/USD"]   # treated as crypto for regime purposes
stablecoins: ["USDT/USD", "USDC/USD", "FDUSD/USD", "DAI/USD"]
```

The exact bStock pair strings must be read from `/v3/exchangeInfo` at runtime; this file only maps them.

### 5.2 Historical data (`binance_vision.py`, `scripts/download_history.py`)

- Spot klines 1h: `https://data.binance.vision/data/spot/monthly/klines/{SYM}/1h/{SYM}-1h-{YYYY}-{MM}.zip` and daily files under `/data/spot/daily/klines/...` for the current month.
- Futures funding: `https://data.binance.vision/data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{YYYY}-{MM}.zip`.
- Futures metrics (open interest): `https://data.binance.vision/data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-{YYYY-MM-DD}.zip`.
- Kline CSV columns: `open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore`. **Newer spot files use microsecond timestamps.** Detect the unit by magnitude (> 1e14 means microseconds) and normalise to UTC `datetime64[ns]`.
- Some files have a header row, some do not. Handle both.
- Verify paths exist (HTTP 404 means no data for that month/symbol, not an error). Cache zips under `data/raw/`, store cleaned parquet at `data/bars/1h/{SYM}.parquet`.
- Default history: 2023-01-01 to now.

### 5.3 Live data (`binance_live.py`)

- Recent spot klines: `GET https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000` (market-data mirror that is not geo-blocked for US IPs). Fallback `https://api.binance.com/api/v3/klines`.
- Funding: `GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=100`. Open interest history: `GET https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=200`.
- `fapi.binance.com` rejects US IPs. At startup, probe it; if blocked, set the crowding overlay to use Bybit public endpoints (`https://api.bybit.com/v5/market/funding/history`, `/v5/market/open-interest`) or disable it and log a warning. Never crash the bot because of optional data.
- Only use **closed** bars (drop the forming bar). Check continuity: missing hours get forward-filled at most 2 bars, beyond that the asset is marked stale and excluded until healthy.

### 5.4 Roostoo ticker recorder (`ticker_recorder.py`)

Runs as its own systemd service from day 1 (`deploy/recorder.service`).
- Every 30 s: one `/v3/ticker` call without `pair`. Append raw snapshots to `data/roostoo_ticks/{YYYY-MM-DD}.parquet` (UTC day files).
- Build 1-minute and 1-hour OHLC from `LastPrice`, and record spread `(MinAsk - MaxBid) / mid`.
- Used for: bStocks bars, basis checks vs Binance, live protective loop, and the universe spread filter.

### 5.5 Price source rule

For signals: Binance 1h closes for crypto; Roostoo-recorded 1h bars for bStocks and any pair without a Binance symbol, with yfinance underlying history as backfill. For valuation, stops and execution: Roostoo ticker only. **Log the Roostoo vs Binance basis per asset hourly.** If an asset's median absolute basis over 24h exceeds 0.5%, switch its signal source to Roostoo bars and log it.

### 5.6 Universe (`universe.py`)

Recomputed at startup and daily at 00:05 UTC. Eligible if all hold:
- `CanTrade` true in `exchangeInfo` and a live ticker exists.
- Not a stablecoin.
- Crypto: Binance history of at least `min_history_days`, in the top `max_crypto` by Roostoo 24h `UnitTradeValue`, median spread over last 6h at most `max_spread_pct`.
- bStocks: listed in `symbols.yaml` and `bstocks.enabled` true in config.
- PAXG: if listed.
- Hysteresis: once eligible, an asset stays eligible unless it falls outside `max_crypto + 5` or fails the spread filter for 12 consecutive hours. Existing positions in a dropped asset are closed through normal rebalancing (target 0), not dumped.

---

## 6. Strategy (`src/trend_guard/strategy/`), pure functions only

All inputs are pandas objects indexed by UTC hour. All functions deterministic.

### 6.1 Volatility

```python
def ewma_vol(log_ret: pd.Series, halflife_h: int) -> pd.Series:
    # EWMA std of hourly log returns (not annualised)
    return log_ret.pow(2).ewm(halflife=halflife_h, adjust=False).mean().pow(0.5)

ANN_H = 24 * 365   # hours per year for annualisation
```

### 6.2 Ensemble signal (`signals.py`)

For each asset, at the latest closed bar `t`, for each `L` in `lookbacks_h`:

```
window      = close[t-L+1 : t]               # last L closes including t
upper, lower = max(window), min(window)
mid          = (upper + lower) / 2
d_L          = clip((close_t - mid) / (0.5*(upper - lower) + eps), -1, 1)   # Donchian position
m_L          = tanh( ln(close_t / close_{t-L}) / (sigma_h_t * sqrt(L) + eps) )  # vol-normalised momentum
component_L  = w_donchian * d_L + w_tsmom * m_L
signal_raw   = mean over L of component_L                 # in [-1, 1]
signal       = EMA(signal_raw, span = smooth_bars)        # light smoothing to cut churn
```

Return `NaN` if the asset has fewer than `max(lookbacks_h) + 1` bars. `NaN` means target weight 0.

### 6.3 Regime (`regime.py`)

- Market index: equal-weight mean of hourly log returns of the top `regime.index_size` eligible crypto (exclude PAXG, bStocks), cumulated into a synthetic price. `mkt = signal(index)`.
- `breadth = share of eligible crypto with signal > 0`.
- State machine with hysteresis:

```
RISK_ON   if mkt >  on.mkt  and breadth >  on.breadth
RISK_OFF  if mkt <  off.mkt and breadth <  off.breadth
exit RISK_ON  to NEUTRAL when mkt < on.mkt - hyst or breadth < on.breadth - hyst
exit RISK_OFF to NEUTRAL when mkt > off.mkt + hyst or breadth > off.breadth + hyst
else stay
```

- Budgets per regime (config):

| Regime | long_budget | short_budget | long_entry_min | short_entry_max |
|---|---|---|---|---|
| RISK_ON | 1.0 | 0.25 | 0.10 | -0.50 |
| NEUTRAL | 0.5 | 0.5 | 0.20 | -0.30 |
| RISK_OFF | 0.0 | 1.0 | n/a | -0.30 |

- Gate function per crypto asset:

```
if signal > long_entry_min:   s_eff = signal * long_budget
elif signal < short_entry_max and shorts_enabled: s_eff = signal * short_budget   # negative
else: s_eff = 0
```

- Diversifiers (PAXG, bStocks) ignore the crypto regime: `s_eff = signal if signal > diversifier_entry_min else 0`. Long-only by default (`diversifiers.allow_short: false`).
- Crypto-beta stocks (COIN, MSTR, CRCL) use the crypto regime gate.

### 6.4 Portfolio construction (`portfolio.py`)

```
sigma_ann_i = ewma_vol(i) * sqrt(ANN_H)
raw_w_i     = s_eff_i * sigma_asset_target / max(sigma_ann_i, vol_floor)
w           = apply_caps(raw_w)                               # per-asset caps (6.5)
Sigma       = EWMA covariance of hourly log returns, halflife cov_halflife_h, annualised
Sigma_s     = (1 - shrink) * Sigma + shrink * diag(Sigma)     # shrinkage toward diagonal
port_vol    = sqrt(w' Sigma_s w)
k           = min(sigma_portfolio_target / max(port_vol, 1e-6), k_max)
w           = apply_caps(k * w)
w           = enforce_gross(w, gross_max)                     # sum |w| <= gross_max, scale down pro rata
```

Weights are fractions of equity. Positive = long notional. Negative = short, and `|w| * equity` is the short collateral.

### 6.5 Caps

| Bucket | Cap on abs(weight) |
|---|---|
| BTC, ETH | 0.25 |
| Other majors (config list) | 0.12 |
| Other crypto | 0.06 |
| Single bStock | 0.08 |
| bStock sleeve total | 0.25 |
| PAXG | 0.20 |
| Total short collateral | 0.50 |
| Gross (long notional + short collateral) | 0.98 (keep 2% cash buffer for fees) |

---

## 7. Risk overlays (`src/trend_guard/risk/`)

All overlays return multipliers in `[0, 1]` (tournament may return up to its cap) and a reason string. The engine applies them in this order and logs every value each cycle:

```
w_final_i = w_i * dd_mult * day_mult * event_mult * endgame_mult * tournament_mult
                * crowd_mult_i(side) * news_mult_i * risk_scale
then enforce caps and gross again
```

`risk_scale` is a global config value (use 0.1 on 3 Oct for the live dry run, 1.0 from 4 Oct).

| Overlay | Rule | Default params |
|---|---|---|
| Drawdown governor | `DD = 1 - equity/peak`; `dd_mult = clip(1 - DD/dd_max, dd_floor, 1)` | `dd_max 0.05`, `dd_floor 0.25` |
| Daily brake | P&L since 00:00 UTC below `soft`: 0.5; below `hard`: 0 for new risk (only reductions allowed) until next UTC day | `soft -0.008`, `hard -0.015` |
| Event | 0.5 from `pre_min` before to `post_min` after each event in `config/events.yaml` | `pre 30`, `post 60` |
| End-game | from competition day `from_day`, if cumulative return >= `min_return`, multiply the vol target by `1 - sigma_cut` (implement as a multiplier on k) | `11`, `0.04`, `0.35` |
| Crowding (per asset) | long side: if `funding_z > z` and `oi_chg_24h > oi` and `signal > 0.7` then 0.6. Short side: if `funding_z < -z` and 72h return < -8% then 0.6 and block new shorts | `z 2.0`, `oi 0.10` |
| Tournament (optional, off) | if our return > 20th-place regional return + buffer: 0.8; if below: 1.15 | off until organizers approve |
| News LLM (optional, off) | see Section 9 | off by default |

`funding_z` uses the last 30 days of funding observations for that asset. If data is unavailable, the multiplier is 1.0 and a warning is logged.

### 7.1 Protective loop (`protective.py`), runs every 60 s on Roostoo tickers

- **Trailing stop per position:** exit at MARKET if (long) `price < max(best_price_since_entry - atr_mult * ATR, mid_channel_ref)` or (short) the mirror. `ATR` = mean true range of the last 24 hourly bars. `mid_channel_ref` = `mid_L` for the median lookback, ratcheted in the trade's favour only. Default `atr_mult 2.5`.
- **Shock filter:** `r15` = BTC 15-minute log return from recorded ticks; `z = r15 / (sigma_h * sqrt(0.25))`.
  - `z < -2.5`: cut all crypto longs by 50% at MARKET, freeze new crypto longs for 60 min.
  - `z > +2.5` and BTC 72h return < -8%: cover 50% of shorts at MARKET, freeze new shorts for 60 min (squeeze protection).
- **Daily brake hard trigger** checked here too.
- After a stop-out, the asset is blocked from re-entry in the same direction for `cooldown_h` (default 6).

---

## 8. Execution (`src/trend_guard/execution/`)

### 8.1 Broker protocol

```python
class Broker(Protocol):
    def get_equity(self) -> Decimal: ...                         # USD free+lock + longs at mid + short PositionValue
    def get_positions(self) -> dict[str, "Position"]: ...        # pair -> signed qty, avg price, side
    def get_tickers(self) -> dict[str, "Ticker"]: ...
    def place_limit(self, pair: str, side: str, qty: Decimal, price: Decimal) -> "OrderAck": ...
    def place_market(self, pair: str, side: str, qty: Decimal) -> "OrderAck": ...
    def cancel(self, order_id: str) -> None: ...
    def query(self, order_id: str) -> "OrderStatus": ...
    def short_open(self, pair: str, collateral: Decimal) -> "ShortAck": ...
    def short_close(self, pair: str, pct: Decimal) -> "ShortCloseAck": ...
```

`LiveBroker` wraps `RoostooClient`. `SimBroker` is used in paper mode (live Roostoo tickers, simulated fills, same fees) and backtests (bar-based fills, Section 11.2).

### 8.2 Order manager: target weights to orders

Each hourly cycle at minute `rebalance_minute` (default 3):

1. Reconcile: pull balance, short positions, pending orders. Exchange state is the truth. Cancel stale pending orders we created more than `limit_reprice_s * limit_max_tries` ago.
2. Compute `target_notional_i = w_final_i * equity`, `current_notional_i` (signed).
3. Skip if `|target - current| < deadband * equity` and signs match. Respect `min_hold_h` except for risk-reducing moves.
4. Sequence: **reductions first** (sells of longs, short closes), then **additions** (buys, short opens), so cash is available. If a target flips sign, close fully first, then open the other side next cycle or after the close confirms.
5. Longs: maker-first.
   - BUY LIMIT at `MaxBid` (or `MaxBid + tick` if spread > 2 ticks). SELL LIMIT at `MinAsk` (or `MinAsk - tick`).
   - Every `limit_reprice_s` (60): query; if filled, done; if partial, cancel the rest and re-place the residual at the new touch; after `limit_max_tries` (4), send the residual as MARKET.
   - Risk exits from overlays or the protective loop go straight to MARKET.
6. Shorts: `short_open(pair, collateral)` MARKET only (limit gives no fee saving). `short_close(pair, close_pct)` for reductions; compute pct from target vs current collateral.
7. Round every quantity and price with `TradingRule`; drop orders below `MiniOrder`.
8. Record `Role`, `CommissionPercent`, `CommissionChargeValue` from every response. Never assume fees.

### 8.3 Ambiguous order state

On `OrderStateUnknownError`: mark the intent `UNKNOWN` in the store; wait 3 s; `query_order(pair=..., pending_only=FALSE, limit=20)`; match by side, quantity and `CreateTimestamp` within 120 s of our send time. If matched, adopt it. If not, treat as not placed. Never resubmit before this check completes.

### 8.4 Activity guarantee

The strategy trades naturally, but log a warning at 18:00 UTC if no fill has happened that UTC day. Do **not** add fake trades. If this warning fires two days in a row, raise it at the checkpoint so we lower the deadband via a normal `strat:` commit.

---

## 9. Optional LLM news veto (`risk/news_llm.py`), off by default

Build last (Milestone 7), only if time allows. It can **only reduce** exposure.

- Sources polled every 30 min with `feedparser` (verify URLs work before relying on them): `https://www.coindesk.com/arc/outboundfeeds/rss/`, `https://cointelegraph.com/rss`, `https://www.theblock.co/rss.xml`. Deduplicate by URL hash; only headlines from the last 2 hours.
- Model: `claude-haiku-4-5-20251001` via the `anthropic` SDK, `temperature=0`, `max_tokens=400`. Key from `ANTHROPIC_API_KEY`.
- System prompt: classify crypto-market risk from headlines. Output strict JSON only:
  `{"scope":"market|coin|none","coins":["BTC"],"severity":0,"category":"hack|delisting|regulatory|macro|exchange_failure|other","confidence":0.0,"rationale":"..."}`
- Deterministic mapping in code:
  - market, severity 3, confidence >= 0.7: `news_mult = 0.5` on all crypto for 6 h.
  - coin, severity 3 (hack, exploit, delisting): target 0 for that coin and block entries for 24 h.
  - anything else: no action.
- Log prompt, raw response, parsed JSON and the resulting action to `logs/llm/{date}.jsonl`. On any API or parse failure: no-op.

---

## 10. State, logging, engine

### 10.1 SQLite schema (`state/schema.sql`)

```sql
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY, created_utc TEXT, pair TEXT, side TEXT, kind TEXT,  -- kind: LIMIT/MARKET/SHORT_OPEN/SHORT_CLOSE
  qty TEXT, price TEXT, collateral TEXT, exchange_order_id TEXT, status TEXT,  -- status: NEW/PENDING/FILLED/PARTIAL/CANCELED/UNKNOWN/REJECTED
  role TEXT, filled_qty TEXT, avg_price TEXT, fee TEXT, reason TEXT, cycle_id TEXT, raw_json TEXT);
CREATE TABLE IF NOT EXISTS equity (
  ts_utc TEXT PRIMARY KEY, equity TEXT, cash TEXT, long_notional TEXT, short_collateral TEXT, peak TEXT, dd TEXT);
CREATE TABLE IF NOT EXISTS decisions (
  cycle_id TEXT PRIMARY KEY, ts_utc TEXT, regime TEXT, json TEXT);            -- signals, weights, multipliers, targets
CREATE TABLE IF NOT EXISTS positions_meta (
  pair TEXT PRIMARY KEY, side TEXT, opened_utc TEXT, best_price TEXT, stop_ref TEXT, cooldown_until TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);                     -- start_equity, start_ts, flags
```

Store decimals as strings. Equity snapshot every minute.

### 10.2 Logs (JSONL, one file per UTC day, rotated)

- `logs/api/*.jsonl`: every request: ts, method, path, params (no secrets), success, errmsg, latency_ms.
- `logs/decisions/*.jsonl`: every hourly cycle: inputs summary, signal per asset, regime, raw and final weights, all multipliers with reasons, orders planned.
- `logs/fills/*.csv`: every fill with role and fee.
- On startup, log the git commit hash (`git rev-parse HEAD`) and a hash of `config/strategy.yaml`. This ties every trade to a commit for Screen 1.

### 10.3 Engine loop (`engine.py`, `main.py`)

```
startup:
  load config + env; sync clock; load exchangeInfo -> TradingRules; build universe
  reconcile from exchange; load or init kv(start_equity, start_ts, peak)
  warm up: load >= max(lookbacks)+50 hourly bars per asset (history + live)
loop (every 60 s):
  if HALT file exists: cancel our pending orders, do nothing else, log, continue
  refresh tickers; write equity snapshot; run protective loop
  every 5 min: reconcile; clock sync
  at minute == rebalance_minute: run hourly cycle (fetch latest closed bars -> signals -> regime
      -> portfolio -> overlays -> order manager); persist decision
  every 30 min (if enabled): news overlay refresh
  00:05 UTC: refresh universe; reset daily P&L anchor
```

CLI:
```
python -m trend_guard.main --mode paper|live --config config/strategy.yaml
python -m trend_guard.main --once          # single hourly cycle then exit (debugging)
```

`--mode live` refuses to start unless `ALLOW_LIVE_ORDERS=1` and both API keys are present.

HALT file path: `/etc/trend-guard/HALT` on EC2, `./HALT` locally.

---

## 11. Metrics and backtesting

### 11.1 Metrics (`metrics.py`)

Implement composite under three definitions and always report all three:

| Name | Returns | Sharpe | Sortino | Calmar |
|---|---|---|---|---|
| `daily_raw` | daily UTC-close returns | mean/std (population) | mean / sqrt(mean(min(r,0)^2)) | mean / max_dd |
| `daily_ann` | daily UTC-close returns | above x sqrt(365) | above x sqrt(365) | CAGR / max_dd |
| `hourly_raw` | hourly returns | mean/std | as above | mean / max_dd |

`composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar`. Use `eps = 1e-8` in denominators and flag windows where a denominator hit eps. Also report: total return, max drawdown, share of red days, trade count, fees paid, turnover.

### 11.2 Backtest engine (`backtest/engine.py`)

- Hourly event-driven. Uses the **same** `signals`, `regime`, `portfolio`, `overlays` and `order_manager` target logic with `SimBroker`.
- Fill model: decisions at bar `t` close. Maker limit for a buy is `close_t * (1 - half_spread)`; it fills at that price if bar `t+1` low <= limit, fee 5 bps. Otherwise the residual fills at bar `t+1` close plus `slippage_bps` at 10 bps fee. Sells mirror this. Risk exits fill at bar `t+1` open plus slippage at 10 bps. Shorts 10 bps each side.
- Protective stops evaluated on hourly high/low (conservative: if both stop and target touched in one bar, assume the stop).
- Defaults: `half_spread 0.0003`, `slippage_bps 2`.
- Warm-up period excluded from metrics.

### 11.3 Evaluation (`backtest/windows.py`, `report.py`)

- Run over 2024-01-01 to latest. Evaluate **every 14-day window** (step 1 day).
- Output per config: distribution table (p10, p25, median, p75, p90) of return, composite (3 definitions), MDD, red-day share; plus results in stress windows: 2024-08-01 to 2024-08-15, 2025-02-20 to 2025-03-06, 2025-10-05 to 2025-10-19, and the most recent 14 days.
- Baselines: BTC buy and hold, equal-weight top-10 hold, cash, long-only version of our strategy, no-overlay version.
- **Selection objective:** maximise p25 of `daily_raw` composite subject to median 14-day return >= 2%, median red-day share <= 30%, p95 MDD <= 5%.
- Parameter grid kept under 50 combinations. Sweep: `sigma_portfolio_target {0.20, 0.30, 0.40}`, `lookbacks {hourly set, daily set, blend}`, `dd_max {0.04, 0.05, 0.07}`, `deadband {0.015, 0.02, 0.03}`.
- `backtest/pbo.py`: CSCV with 16 partitions on the daily return matrix of all grid configs. Report PBO for the selected config.
- Report written to `reports/{timestamp}/summary.md` and CSVs.

Alternative lookback sets for the sweep:
- hourly set: `[24, 48, 96, 168, 336, 504]` hours
- daily set (paper-style, on daily bars): `[5, 10, 20, 30, 60, 90]` days
- blend: average of both signals

---

## 12. Full default config (`config/strategy.yaml`)

```yaml
competition:
  start_utc: "2026-10-03T16:00:00Z"    # 4 Oct 00:00 HKT; confirm with organizers
  end_utc:   "2026-10-17T15:59:59Z"
  risk_scale: 1.0                      # 0.1 for the 3 Oct live dry run
universe:
  max_crypto: 25
  max_spread_pct: 0.0015
  min_history_days: 180
  majors: ["SOL/USD", "BNB/USD", "XRP/USD", "DOGE/USD", "ADA/USD", "AVAX/USD", "LINK/USD", "SUI/USD"]
  bstocks_enabled: false               # flip after prep checks
signal:
  lookbacks_h: [24, 48, 96, 168, 336, 504]
  w_donchian: 0.5
  w_tsmom: 0.5
  smooth_bars: 3
vol:
  halflife_h: 48
  vol_floor_ann: 0.15
regime:
  index_size: 10
  on:  {mkt: 0.20, breadth: 0.55}
  off: {mkt: -0.20, breadth: 0.40}
  hysteresis: 0.10
  budgets:
    RISK_ON:  {long: 1.0, short: 0.25, long_entry_min: 0.10, short_entry_max: -0.50}
    NEUTRAL:  {long: 0.5, short: 0.50, long_entry_min: 0.20, short_entry_max: -0.30}
    RISK_OFF: {long: 0.0, short: 1.00, long_entry_min: 1.01, short_entry_max: -0.30}
diversifiers:
  entry_min: 0.10
  allow_short: false
portfolio:
  sigma_asset_target: 0.15
  sigma_portfolio_target: 0.30
  cov_halflife_h: 72
  shrink: 0.30
  k_max: 3.0
  gross_max: 0.98
  caps: {btc_eth: 0.25, major: 0.12, other: 0.06, bstock: 0.08, bstock_sleeve: 0.25, paxg: 0.20, short_total: 0.50}
risk:
  dd_max: 0.05
  dd_floor: 0.25
  daily_brake: {soft: -0.008, hard: -0.015}
  event: {pre_min: 30, post_min: 60, mult: 0.5}
  endgame: {from_day: 11, min_return: 0.04, sigma_cut: 0.35}
  crowding: {enabled: true, funding_z: 2.0, oi_chg_24h: 0.10, mult: 0.6}
  tournament: {enabled: false, buffer: 0.01, above_mult: 0.8, below_mult: 1.15}
  news_llm: {enabled: false}
protective:
  atr_mult: 2.5
  shock_z: 2.5
  squeeze_prior_drop: -0.08
  cooldown_h: 6
execution:
  rebalance_minute: 3
  deadband: 0.02
  min_hold_h: 3
  limit_reprice_s: 60
  limit_max_tries: 4
  ticker_poll_s: 30
  requests_per_min: 30
  min_order_gap_s: 1.0
backtest:
  start: "2024-01-01"
  half_spread: 0.0003
  slippage_bps: 2
  maker_fee: 0.0005
  taker_fee: 0.0010
  short_fee: 0.0010
  window_days: 14
```

`config/events.yaml`: list of `{name, utc}` for scheduled US macro releases between 4 and 17 Oct 2026. **Fill it from the official BLS and Federal Reserve calendars; do not invent dates.** Leave empty with a TODO if unverified.

---

## 13. Monitoring, deployment, docs

### 13.1 Dashboard (`monitor/dashboard.py`)

Streamlit, read-only on the SQLite file. Panels: equity curve and drawdown; live composite (3 definitions) since `start_ts`; exposure by sleeve (crypto long, crypto short, PAXG, bStocks); current regime and all multipliers with reasons; positions table; last 50 orders with role and fee; API error rate. Bind to `127.0.0.1:8501` only. Access via SSH tunnel: `ssh -L 8501:localhost:8501 ubuntu@<ec2>`. Never open the port publicly.

### 13.2 EC2 (`deploy/setup_ec2.sh`)

- Ubuntu, timezone UTC, `chrony` enabled for clock accuracy.
- Python 3.11 venv under `/opt/trend-guard`.
- Private repo access: create a read-only **deploy key** on EC2 (`ssh-keygen -t ed25519`), add it with `gh repo deploy-key add ~/.ssh/id_ed25519.pub --title ec2 -R <owner>/trend-guard` from the dev machine, then clone via SSH.
- Secrets in `/etc/trend-guard/env` (mode 600, owner = service user).
- `deploy/trend-guard.service`: `Restart=always`, `RestartSec=10`, `EnvironmentFile=/etc/trend-guard/env`, `ExecStart=/opt/trend-guard/.venv/bin/python -m trend_guard.main --mode live`.
- `deploy/recorder.service`: same pattern for the ticker recorder.
- `deploy/logrotate.conf`: daily rotation, keep 30 days, compress.

Deploy runbook (`docs/RUNBOOK.md`):
```bash
git tag -a v1.0.x -m "reason"; git push --tags        # on dev machine
ssh ec2 'cd /opt/trend-guard && git fetch --tags && git checkout v1.0.x && sudo systemctl restart trend-guard'
```

### 13.3 README sections (judges read this in Screen 4)

1. Overview and one-paragraph strategy summary
2. Why this design (link to `docs/STRATEGY.md` with the research citations)
3. Architecture diagram (Mermaid) and module map
4. Rule compliance table: each competition rule mapped to the file and function that enforces it
5. Risk management (every overlay, with defaults)
6. Execution (maker-first, reconciliation, no blind retries)
7. Backtest methodology and results table (from `reports/`)
8. How to run: install, paper mode, live mode, tests, backtest
9. Deployment on EC2
10. Logs and audit trail (how trades tie to commits)
11. Limitations and honest caveats

### 13.4 Going public before 14 Oct

1. `pip install detect-secrets` (or run `gitleaks detect`) over the full history; fix anything found by rotating the key and rewriting history if needed.
2. Confirm `.env`, `data/`, `logs/`, `state/*.db` were never committed.
3. `gh repo edit --visibility public --accept-visibility-change-consequences`.
4. Submit the repo link.

---

## 14. Milestones and checkpoints

| # | Target date (HKT) | Deliverable | Definition of done |
|---|---|---|---|
| M0 | 30 Sep | Bootstrap (Section 2) | Private repo, CI green |
| M1 | 30 Sep | Roostoo client, rules, rate limiter, tests, `scripts/smoke_readonly.py` | All client tests pass; smoke script prints server time, pair count, BTC ticker (balance once keys arrive) |
| M2 | 1 Oct | Data layer: history download, live klines, funding/OI, ticker recorder, universe | Parquet bars for all eligible assets since 2024; recorder running on EC2 |
| M3 | 1 to 2 Oct | Strategy + overlays (pure) + SimBroker + backtest engine + metrics + windows report | `run_backtest.py` produces the Section 11.3 table for default config and baselines |
| M4 | 2 Oct | LiveBroker, order manager, state store, engine loop, paper mode on EC2 | Paper mode runs 12 h with no unhandled exceptions; decisions logged hourly |
| M5 | 2 to 3 Oct | Prep probes (Section 15), dashboard, deploy scripts, README draft | Probe results recorded in `docs/PROBES.md` |
| M6 | 3 Oct | Parameter selection (grid + PBO), live dry run at `risk_scale 0.1` (human starts live mode) | Selected config committed with `strat:` message; 6 h live dry run clean |
| M7 | optional | News LLM overlay | Tests with mocked API; off by default |
| M8 | 3 Oct 23:00 | Freeze `v1.0.0`, set `risk_scale 1.0`, deploy | Bot live before trading starts |
| M9 | before 14 Oct | Secret scan, make public, submit | Link submitted |

---

## 15. Prep-window probes (1 to 3 Oct), via committed code only

`scripts/probe_live.py` runs a fixed, logged sequence and writes results to `docs/PROBES.md`. It requires `ALLOW_LIVE_ORDERS=1` and `--i-understand`, and **the human runs it, not Claude Code** (it is in the deny list). **Before running it, ask the organizers whether prep-period trades on the competition account are acceptable** (Screen 1 checks for manually called APIs). If not allowed, run the same probes through the bot itself in live mode at `risk_scale 0.1`.

| Probe | Action | Record |
|---|---|---|
| Maker behaviour | BTC/USD BUY LIMIT at `MaxBid` for ~$20, poll 5 min | Role, time to fill, fee |
| Inside-spread limit | Same at `MaxBid + tick` | Role, fill time |
| Market order | BUY then SELL ~$20 of ETH | Fill price vs ticker, fee |
| Shorts | `short_open` $10 collateral on BTC, then `short_close` 100% | Enabled or disabled, fees, PnL fields |
| Cancel | Place far-away limit, cancel by id | Works as documented |
| Basis | 24 h of recorder data vs Binance 1m | Median and max basis per asset, lag estimate |
| bStocks weekend | Recorder over Sat 3 Oct | Do prices move outside US hours? Spread? |
| Rate limit | Observe error responses at 30 s polling | Any throttling messages |

Decision rules: shorts disabled means set the short budgets to 0 (RISK_OFF = cash plus diversifiers). Limits not booked as maker means switch to MARKET and raise `deadband` to 0.03. bStocks static or spread above 0.3% means keep `bstocks_enabled: false`.

---

## 16. Known unknowns (ask the user or organizers; do not guess)

1. Exact scoring sampling and annualisation (organizers have not answered). Keep reporting all three definitions.
2. Whether shorts are enabled for this competition's accounts.
3. Exact trading start and end timestamps.
4. Numeric rate limits.
5. Whether prep-period test trades on the competition key are acceptable.
6. Whether reading the public leaderboard from the bot is acceptable (tournament overlay stays off until confirmed).
7. EC2 region (affects Binance futures access).

## 17. Compliance checklist (verify before each deploy)

- [ ] No order path bypasses `ALLOW_LIVE_ORDERS` gating
- [ ] Long notional + short collateral <= equity at all times (asserted in code)
- [ ] No quoting both sides of a pair; no sub-minute trading loops; signals only on closed hourly bars
- [ ] Every trade traceable to a logged decision and a commit hash
- [ ] No secrets in repo or history
- [ ] All tests and CI green
- [ ] README compliance table current
