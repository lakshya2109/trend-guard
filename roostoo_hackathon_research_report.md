# Roostoo HK vs AU vs IN Quant Trading Hackathon: Research Report and Build Spec

Prepared 29 Sep 2026 for a 4-person HKUST team. Written to be pasted into Claude Code (or any agentic coding assistant) and shared with teammates' AI assistants for the consensus round.

---

## 0. Executive summary (read this if nothing else)

**Recommendation:** a non-LLM, rules-based, long/short **trend-following ensemble with volatility targeting and a drawdown governor**, trading the most liquid Roostoo crypto pairs plus a small diversifying sleeve (PAXG gold token and a few bStocks). Add two thin overlays: a **derivatives crowding/crash filter** (Binance perp funding and open interest) and an **optional LLM news-risk veto** that can only *reduce* exposure. Execute maker-first with limit orders at the touch.

**Why this, and why it differentiates us:**

| Edge | Why it matters here | Evidence |
|---|---|---|
| Use the `/v6` short endpoints | Shorting was added to the API in Sep 2026. Last round (SG vs HK, Mar 2026) was long-only and a political-headline selloff knocked out teams that could not short. Most public repos for this round are still long-only. | API repo; Medium post-mortem; competitor repos |
| Symmetric trend ensemble (not a single indicator) | Donchian multi-lookback ensembles with vol sizing give Sharpe ~1.5 net of fees on top-20 crypto, survivorship-free, 2015 to 2025. Trend payoffs are positively skewed, which Sortino rewards. | Zarattini, Pagani, Barbon (2025) |
| Volatility management | Crypto momentum crashes; vol management is the documented fix. Vol scaling, not the signal, explains much of TSMOM's performance. | Grobys et al. (2025, FMPM); Kim, Tse, Wald (2016); Barroso & Santa-Clara (2015) |
| Optimise *the actual scoring function* | Over 14 days a single red day can wipe out the composite (Section 2). We design to minimise the count and size of down periods, not to maximise raw return. | Our own calculation below |
| Crowding / crash filter from perps | High futures carry predicts crashes. Almost no student bot uses derivatives data. It is free via Binance public data. | Schmeling, Schrimpf, Todorov (Management Science 2026) |
| Diversifiers (PAXG, bStocks) | Organisers confirmed bStocks are tradable. Lower correlation lowers portfolio vol and drawdown for the same return. | Organizer clarification (via Rustoo repo) |
| Maker-first execution | 0.05% vs 0.10% per side. A past participant got near-immediate fills at maker fee using aggressive limits at the touch. | Medium post-mortem |
| LLM only as a risk veto | LLM trading agents' reported edge degrades under rigorous testing; the provided MAS paper's own ablation shows the News Agent mostly dampens volatility rather than adding return. | FINSABER (KDD 2026); Luo et al. (2501.00826v3) Table II; Alpha Arena |

**Honest caveat:** over 14 days, outcome is dominated by market path. A strategy with an annual Sharpe of 1.5 at 30% vol has an expected 14-day return of about +1.7% with a standard deviation near 5.9%. Our job is to (1) avoid the left tail, (2) be positioned correctly if a trend appears, and (3) be engineered well enough to pass Screens 1 and 4. That is what this design does.

**Urgent admin:** team registration closes **29 Sep 23:59 HKT** via the Typeform (not Luma). Do this first.

---

## 1. Verified facts about the competition and platform

### 1.1 Timeline (HKT)

| Date | Event |
|---|---|
| 29 Sep 23:59 | Team registration deadline (Typeform) |
| 1 to 3 Oct | Preparation: build, deploy, test on Roostoo |
| 4 to 17 Oct | Live trading (14 full days). Redeploys allowed |
| Before 14 Oct | Submit open-source repo link with README |
| 21 Oct | Top 15 finalists announced (5 per region) |
| 27 Oct | Finalist deck / md deadline |
| 29 Oct to 4 Nov | Virtual finale |

### 1.2 Rules that constrain design

- $100,000 mock portfolio. Spot only, "1x long and short", no leverage.
- Fees: 0.10% taker (market), 0.05% maker (limit). Shorts: flat 0.10% on open and close, even for limit shorts.
- No HFT, market making or arbitrage. Excessive requests return failed responses.
- Must run autonomously on the provided AWS EC2. No manual API calls with the competition key. Commit history must be consistent and traceable.
- At least 8 active trading days. Organizers clarified: an active day = at least 1 trade, and trade history must match the stated strategy.
- Any data source allowed (you pay for non-Roostoo data and LLM calls).
- Open-source repo, strict anti-plagiarism. **Do not copy public competitor repos.**

### 1.3 Evaluation pipeline (order matters)

1. **Screen 1, compliance:** trade log integrity, commit history transparency.
2. **Screen 2, return:** top 20 per region by `(final - initial) / initial`.
3. **Screen 3, composite:** `0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar`.
4. **Screen 4, code review:** clear strategy logic, clean repo, runs continuously on Roostoo.
5. Judges pick top 3 per region from 5 finalists. Winning region = highest average return of its top 5.

So **return gets us through the gate, ratios rank us, code quality and presentation win it.**

### 1.4 API facts (from `roostoo/Roostoo-API-Documents`, cloned 29 Sep 2026)

- Base URL `https://mock-api.roostoo.com`. HMAC-SHA256 signed requests; `timestamp` must be within 60 s of server time (sync with `/v3/serverTime`).
- Market data: **only** `/v3/ticker` snapshots (`MaxBid`, `MinAsk`, `LastPrice`, 24h `Change`, 24h volumes). No OHLCV history. Call without `pair` to get all tickers in one request.
- Orders: `/v3/place_order` (`MARKET` or `LIMIT`), `/v3/query_order`, `/v3/cancel_order`, `/v3/pending_count`, `/v3/balance`.
- **Shorts (new, merged PR "feat/shortingapis"):** `/v6/short_open` (sized by USD `collateral`, market fills at `MaxBid`, optional LIMIT), `/v6/short_close` (always market at `MinAsk`, reduce-only, `close_pct` or `close_qty`), `/v6/short_positions`. Loss capped at collateral. Error string `this competition does not allow short positions` exists, so **verify shorts are enabled for our account during 1 to 3 Oct**.
- Quirks: failures still return HTTP 200 with `Success:false`; zero-valued fields are omitted (treat missing as 0); only documented params are included in the server-side signature; empty results (`no pending order`, `no order matched`) are not errors.
- Precision: round price to `PricePrecision`, quantity to `AmountPrecision`, and require `price x qty > MiniOrder`. Use decimal string formatting (never `str(float)`, which can emit `1e-05`).
- Maker vs taker is an outcome (`Role` in the response), not a flag. Always read realised fee from the response.

### 1.5 Organizer clarifications (received by another team 28 Sep, recorded in `Xanthen-w/Rustoo/docs/COMPETITION_RULES.md`)

- Active trading day = at least 1 trade.
- Teams may stop, restart, redeploy themselves (git pull on EC2, restart service).
- Shorts: losses capped at collateral. **No native stop-loss or take-profit orders.** Bots must monitor and close themselves.
- Long and short on the same pair simultaneously: allowed.
- bStocks: use symbols exactly as `/v3/exchangeInfo` returns; Binance or other sources fine for history.
- Exact Sharpe/Sortino/Calmar sampling and annualisation: **not answered.**
- After a timeout, verify via `/v3/query_order` (never blind-resubmit).

### 1.6 Universe

A competitor's live check on 22 Sep 2026 found **88 pairs** in `exchangeInfo`, 86 with ticker data. Past rounds included BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT, SUI, APT, NEAR, TRX, LTC, UNI, AAVE, PEPE, TAO, FET, WLD, ZEC, HBAR, ICP, FIL, TON, ENA, PAXG and more. The app now also lists **bStocks** (tokenized US equities: NVDA, MSFT, GOOGL, META, AMD, MU, INTC, TSLA, COIN, MSTR, CRCL, SNDK, SKHY, LITE, GLW, NBIS, CBRS, QCOM, WDC, PLTR, SPCX). bStocks are BNB Chain tokens that trade 24/7, so weekend prices move on thin liquidity and gap into the Monday US open.

**Always discover the universe dynamically from `exchangeInfo` at startup.** Never hardcode.

---

## 2. How we are scored, and what it implies

### 2.1 The formula's behaviour over 14 days

Organizers did not publish sampling or annualisation. A community rebuild of the March 2026 leaderboard (`eric15342335/roostoo-leaderboard/fetcher.py`) used **daily UTC closes, non-annualised**: Sortino = mean / downside deviation (zero-centred), Sharpe = mean / std, Calmar = mean daily return / max drawdown. We computed four illustrative 14-day paths under that definition and under an annualised definition:

| Path (14 daily returns) | Return | MDD | Composite (daily, raw) | Composite (annualised) |
|---|---|---|---|---|
| A: steady +0.3%/day, two -0.2% days | +3.25% | 0.40% | **1.77** | 128 |
| B: aggressive swings | +8.05% | 3.00% | 0.36 | 71 |
| C: same as A but one day is -2% | +1.38% | 2.20% | **0.14** | 8 |
| D: +0.1%/day, never red | +1.41% | 0.00% | degenerate (divide by ~0) | degenerate |

**Implications:**

1. **One bad day is catastrophic.** Path C differs from A by a single day and loses over 90% of its score. Sortino (40% weight) only sees negative periods, so size and number of red periods dominate.
2. **Calmar dominates if annualised** (annualised 14-day returns are huge relative to MDD). Either way, **MDD is the key lever.**
3. **Positively skewed strategies win.** Many small gains plus rare, capped losses. Trend following with fast risk cuts is naturally positively skewed; mean reversion is negatively skewed (small gains, occasional large losses) and is the wrong shape for this scoring.
4. **But we must first clear Screen 2 (top 20 by return).** Past evidence: in the March 2026 final, the top three returns were +20.5%, +8.1%, +5.7%, and a team that reached top 5 early fell back before the end. Many teams finish negative. A positive, controlled return (target +3 to +8%) likely clears the gate, but we should **monitor the public leaderboard** to calibrate.
5. **Do not game snapshot timing** (for example flattening just before 00:00 UTC). Sampling is unknown, it costs fees, and it would look manipulative in Screen 1. Instead make the equity curve smooth at every frequency.

### 2.2 Derived objective for backtests

Choose parameters to maximise the **25th percentile** of the composite across all rolling 14-day windows, subject to (a) median 14-day return above +2%, (b) fraction of red days below 30%, (c) 95th percentile of 14-day MDD below 5%. Report both daily-raw and annualised composite, and also an hourly-sampled version. Robustness across definitions beats peak performance under one.

---

## 3. Competitive landscape (what others are doing)

From public repos for this round and the March 2026 round:

| Approach seen | Typical outcome / weakness |
|---|---|
| Cross-sectional momentum top-N + "BTC below MA = cash" | Most common. Long-only, so flat in selloffs at best. Crowded idea. |
| RSI/EMA/MACD/Bollinger stacks, SMC "liquidity sweep" gates, Supertrend + ML filter | Many parameters, high turnover, fee drag. One repo's mean-reversion backtest showed -14% in-sample and -12% OOS with fees of ~9% of capital. |
| HMM / Hurst regime switchers on 1 to 4 coins | Concentrated, noisy on 14 days. |
| Ridge / ML on Binance features | Overfit risk; hard to explain. |
| LLM agents | Expensive, non-deterministic, hard to pass "consistent with declared strategy". |
| Cointegration / stat-arb with shorts (at least one team) | Closest to our differentiator; watch out for the "no arbitrage" rule. |

Lessons from the March 2026 post-mortems:
- A long-only bot "reacted too slowly to a sharp market selloff triggered by global political headlines" and could not recover without shorts.
- Churn between ranks 4 and 5 burned fees; fixes were a ranking buffer, minimum hold, and a fee dead-zone.
- Aggressive limit orders at the touch filled quickly at the 0.05% maker fee.
- A dashboard/observability layer was the most valuable debugging tool.

---

## 4. Literature review: what the evidence supports

### 4.1 Trend / time-series momentum in crypto

- **Liu & Tsyvinski (RFS 2021)** and **Liu, Tsyvinski & Wu (JF 2022):** strong time-series momentum at 1 to 4 week horizons in the crypto market; momentum is one of three priced crypto factors.
- **Zarattini, Pagani & Barbon (2025), "Catching Crypto Trends" (SSRN 5209907):** ensemble of Donchian channel breakouts over multiple lookbacks with volatility-based sizing and a trailing stop, rotating among the top 20 liquid coins, survivorship-bias-free since 2015. Net-of-fee Sharpe above 1.5; reported CAGR ~30%, Sortino ~2.0, alpha vs BTC 10.8 to 14%. **This is our core template.**
- **Borgards (2021):** momentum present at all data frequencies, marginally weaker at higher frequency.
- **Han et al. (2024):** time-series and cross-sectional momentum in crypto; time-series percentile ranks handle extreme returns better than raw regressions.
- **Kim, Tse & Wald (2016):** TSMOM's superiority is driven largely by volatility scaling. Implication: **never run an unscaled signal.**
- Recent: arXiv 2602.11708 (Feb 2026) on adaptive trend-following portfolio construction in crypto highlights non-stationary vol regimes, heavy left tails, and a fast-changing universe.

### 4.2 Volatility management and momentum crashes

- **Moreira & Muir (JF 2017):** volatility timing raises Sharpe because expected returns do not rise one-for-one with volatility.
- **Barroso & Santa-Clara (JFE 2015):** scaling momentum by realised variance nearly doubled Sharpe and removed crashes.
- **Daniel & Moskowitz (JFE 2016):** momentum crashes occur in panic states, after declines, when vol is high, and coincide with rebounds. Implication: **our short sleeve must de-risk after capitulation** (short squeezes).
- **"Cryptocurrency momentum has (not) its moments" (FMPM 2025):** crypto momentum is prone to severe crashes; a single coin can sink the portfolio; vol management mitigates it. Implication: **per-asset caps plus portfolio vol target.**
- **Harvey et al. (2018, Man Group):** vol targeting improves Sharpe for risk assets and reduces left tails.
- **Bongaerts, Kang & van Dijk (FAJ 2020), conditional volatility targeting:** adjust only in extreme vol states; lower turnover. Good fit for fee-sensitive, short windows.

### 4.3 Intraday structure

- **Shen, Urquhart & Wang (Financial Review 2022):** Bitcoin intraday time-series momentum; strongest after high-volume or high-volatility sessions; gains largest in downturns.
- **Wen, Bouri, Xu & Zhao (2022):** crypto shows both intraday momentum and reversal; patterns shift around large jumps and FOMC days. Implication: **cut exposure around scheduled macro releases.**
- **Padysak & Vojtko (2022):** BTC tends to trend when at its local max and revert at its local min (supports breakout entries, not dip-buying in downtrends); 21:00 to 23:00 UTC showed persistently strong returns historically.
- 2026 hourly-data study: apparent day-of-week effects collapse into Sunday 23:00 to 00:00 UTC. Activity, volatility and illiquidity peak around 16:00 to 17:00 UTC. Implication: seasonality is weak; use it at most as a small tilt, and avoid placing large rebalances in the thin Sunday late-UTC hour.

### 4.4 Cross-asset predictability

- **Guo et al. (JEDC 2024), "Cross-cryptocurrency return predictability":** lagged returns of other coins predict focal coins out of sample, via slow information diffusion; long-short portfolio profitable after costs.
- **"Seesaw effect" (JEF 2023):** intraday, large coins *negatively* predict small coins. Implication: an altcoin that has lagged a sharp BTC move is not automatically a buy. Test BTC-lead features as a secondary signal only.

### 4.5 Derivatives-based risk signals

- **Schmeling, Schrimpf & Todorov, "Crypto Carry" (Management Science 2026; BIS WP 1087):** futures basis/funding can reach 40 to 60% p.a., driven by retail trend-chasing leverage and scarce arbitrage capital, and **high carry predicts future crashes.**
- arXiv 2510.14435: carry profitability compressed sharply since 2024. Implication: do **not** try to earn carry (and we cannot trade perps anyway). Use funding and OI purely as a **crowding / crash-risk indicator.**

### 4.6 Stops and drawdown control

- **Kaminski & Lo (JFM 2014):** under a random walk stop-losses reduce expected return, but with momentum or regime switching they add value. Crypto trends, so a trend strategy with stops is consistent with theory.
- **Grossman & Zhou (1993):** optimal policy under a max-drawdown constraint scales risky exposure with the distance from the drawdown floor. This is our drawdown governor.

### 4.7 LLM trading agents: what the evidence says

- **FINSABER (Li et al., KDD 2026):** over 20 years and 100+ symbols, previously reported LLM advantages deteriorate significantly; LLM strategies are too conservative in bulls and too aggressive in bears.
- **Alpha Arena (Nof1, Season 1, Oct to Nov 2025):** six frontier LLMs with real money on crypto perps. Wide dispersion; several lost heavily; the organisers themselves flag prompt bias, small samples and short duration. Season 1.5 on tokenized stocks: most models negative.
- **The PDF you provided (Luo et al., arXiv 2501.00826v3):** best config (Hierarchical + Skill) made +133.5% in 2025, but: weekly rebalancing over 52 weeks (not 14 days); a single run at temperature 0 (the authors say multi-seed sweeps were cost-prohibitive); the winning "skill" variant is essentially classical TA indicators fed to the LLM; **every MAS configuration lost money in the bear regime**; and their ablation shows removing the News Agent barely changes return but raises volatility, meaning news acts as a risk dampener.
- Survey arXiv 2609.04917 (Sep 2026) stresses that many AI trading results do not survive realistic costs, leakage controls and execution modelling.

**Conclusion:** an LLM should not be the alpha engine for a 14-day scored window. It can add value as an **asymmetric news-risk veto** (it can only cut risk), which also gives judges a credible "creative hybrid" story.

### 4.8 Overfitting and statistics

- **Bailey & López de Prado, Deflated Sharpe Ratio (JPM 2014)** and **Probability of Backtest Overfitting (J. Comp. Finance 2017):** with four people testing many variants, multiple-testing bias is our biggest research risk.
- **Lo (FAJ 2002), "The Statistics of Sharpe Ratios":** Sharpe estimates on short samples are very noisy; serial correlation distorts annualisation.

---

## 5. Recommended strategy: full specification

Name for the deck: **"TREND-GUARD: vol-targeted long/short trend ensemble with a drawdown governor."**

### 5.1 Architecture of the decision

```
every hour (on closed 1h bars):
  universe   = eligible pairs (liquidity, spread, data checks)
  signal_i   = trend ensemble score in [-1, 1]            # 5.3
  regime     = market state from BTC + breadth            # 5.4
  raw_w_i    = gate(signal_i, regime) * risk_budget_i      # 5.5
  w          = portfolio vol target + caps                 # 5.5
  w          = w * crowding_mult * news_mult * dd_mult     # 5.6 overlays, each <= 1
  orders     = diff(w, current) with deadband + min hold   # 5.7
every 1 min:
  protective checks: trailing stops, shock filter, daily brake   # 5.6
```

### 5.2 Universe selection

- Start from `/v3/exchangeInfo`. Keep `CanTrade=true` pairs with a live ticker.
- Crypto: rank by Roostoo 24h `UnitTradeValue`; keep top ~25 with spread `(MinAsk - MaxBid)/mid <= 0.15%` and a Binance USDT spot history of at least 180 days. Exclude stablecoins.
- Diversifier sleeve: PAXG always eligible. bStocks eligible only after the prep-window checks in Section 9 pass; start with 4 to 6 large, liquid names (e.g. NVDA, MSFT, GOOGL, META). Treat COIN, MSTR, CRCL as crypto-beta, not diversifiers.
- Refresh eligibility once per day (00:05 UTC), with hysteresis so names do not flip in and out.

### 5.3 Core signal: trend ensemble

Compute on 1h closes from Binance (crypto) and on recorded Roostoo ticker bars (for anything without a clean external feed).

Two components averaged (ensemble across both method and lookback reduces parameter sensitivity):

1. **Donchian position score** for each lookback `L` in hours `{24, 48, 96, 168, 336, 504}`:
   `upper_L = max(close[t-L:t])`, `lower_L = min(close[t-L:t])`, `mid_L = (upper_L + lower_L) / 2`
   `d_L = clip((close_t - mid_L) / (0.5 * (upper_L - lower_L)), -1, 1)`
2. **Vol-normalised TSMOM** for the same `L`:
   `m_L = tanh( ret(t-L, t) / (sigma_1h * sqrt(L)) )`

`signal_i = mean over L of (0.5 * d_L + 0.5 * m_L)`.

Also test the paper's original daily lookbacks (5 to 360 days on daily bars) and a blend. Pick using the Section 7 protocol, not by eyeballing.

### 5.4 Market regime gate

- `mkt = signal` of an equal-weighted index of the top 10 crypto by liquidity, and `btc = signal_BTC`.
- `breadth = share of eligible crypto with signal_i > 0`.

| Regime | Condition | Crypto longs | Crypto shorts |
|---|---|---|---|
| RISK-ON | `mkt > 0.2` and `breadth > 0.55` | allowed, full budget | only if `signal_i < -0.5` (idiosyncratic losers), 25% budget |
| NEUTRAL | otherwise | 50% budget | 50% budget, `signal_i < -0.3` |
| RISK-OFF | `mkt < -0.2` and `breadth < 0.40` | none (except PAXG / bStocks by own signal) | full budget for `signal_i < -0.3` |

Use hysteresis (enter at thresholds above, exit at thresholds 0.1 closer to zero) to avoid flip-flopping.

### 5.5 Position sizing and portfolio construction

- `sigma_i`: EWMA of hourly returns, half-life 48h, annualised with `sqrt(24*365)`.
- Per-asset risk budget: `raw_w_i = signal_i * (sigma_asset_target / sigma_i)` with `sigma_asset_target` ~ 15% annualised.
- Portfolio vol target `sigma_P` using an EWMA covariance with shrinkage (Ledoit-Wolf or constant-correlation), half-life 72h. Scale all weights by `k = sigma_P / sqrt(w' Σ w)`, capped so that gross <= 1.0.
- Default `sigma_P = 30%` annualised (~1.6% daily). Test 20 to 40%.
- Caps: BTC/ETH `|w| <= 25%`; other majors `<= 12%`; small caps `<= 6%`; any single bStock `<= 8%`; bStock sleeve `<= 25%`; PAXG `<= 20%`; total short collateral `<= 50%`; long notional + short collateral `<= 100%` of equity.
- Shorts are sized by `collateral`: `collateral = |w_i| * equity`. Remember the 0.1% fee on both open and close.

### 5.6 Overlays (all multiplicative, all <= 1, all logged)

1. **Drawdown governor (Grossman-Zhou style):** `DD = 1 - equity / peak_equity`. `dd_mult = clip(1 - DD / DD_max, 0.25, 1.0)` with `DD_max = 5%`. Peak resets only upward.
2. **Daily brake:** if P&L since 00:00 UTC < -0.8%, `day_mult = 0.5` until next UTC day; < -1.5%, close risk-adding positions and keep only hedges.
3. **Shock filter (1-minute loop):** if BTC 15-min return < -2.5 sigma or > +2.5 sigma, freeze new entries 60 min. Downside shock with open longs: cut crypto longs 50% at market. Upside shock after a >8% 3-day decline with open shorts: cover 50% (momentum-crash / squeeze protection per Daniel & Moskowitz).
4. **Trailing stop per position:** exit if price crosses back through the `mid_L` of the median lookback (ratcheting, as in Zarattini et al.), or 2.5 x ATR(1h, 24) from the best price since entry, whichever is tighter.
5. **Crowding / crash filter (Binance USDT-M public data):** per coin and market-wide, 30-day z-scores of funding rate, 24h change in open interest, and perp-spot basis.
   - `funding_z > 2` and `OI_change_24h > +10%` and price extended (`signal > 0.7`): `crowd_mult = 0.6` for longs.
   - `funding_z < -2` after a decline: block new shorts, `crowd_mult = 0.6` for existing shorts.
   - Note: `fapi.binance.com` blocks US IPs. Check the EC2 region; fall back to Bybit or OKX public endpoints.
6. **Event calendar:** halve new risk from 30 min before to 60 min after scheduled US CPI and FOMC releases that fall in 4 to 17 Oct (check the BLS and Fed calendars). Static config file, committed to git.
7. **End-game lock-in:** from day 11 (14 Oct), if cumulative return >= +4%, lower `sigma_P` by 35%. Rebalancing still produces daily trades.
8. **Optional tournament-aware budget:** if the bot can read the public leaderboard (a `/v1/leader_board` endpoint was used publicly in March), estimate the 20th-place return in our region hourly. Above it by a buffer: `sigma_P x 0.8`; below it: `sigma_P x 1.15` (hard cap 40%). Grounded in fund-tournament behaviour (Brown, Harlow & Starks, JF 1996). **Ask organizers first** whether reading the leaderboard from the bot is acceptable; if not, skip.
9. **Optional LLM news-risk veto:** see 5.8.

### 5.7 Execution

- **Rebalance cadence:** hourly at a fixed offset (e.g. minute 3). Deadband: only trade if `|target - current| > 2%` of equity or the sign flips. Minimum hold 3 hours except for risk exits.
- **Maker-first spot orders:** BUY LIMIT at `MaxBid` (or one tick above if spread > 2 ticks), SELL LIMIT at `MinAsk` (or one tick below). Re-price every 60 s up to 4 times, then cross with MARKET if still needed. Risk-reducing exits from overlays 2 and 3 go straight to MARKET.
- **Verify** in prep that a limit at or inside the touch is booked as `MAKER` and at what fill speed. Log `Role` and `CommissionPercent` for every fill.
- **Shorts:** use MARKET for `/v6/short_open` (limit gives no fee benefit and charges the fee at placement). Close with `/v6/short_close` using `close_pct`.
- **Request budget:** one all-pairs ticker call every 30 to 60 s; orders paced at least 1 s apart; exponential backoff on failures. Never retry an order-creating call blindly: on timeout, reconcile via `query_order` and `balance`.
- **Reconciliation:** every 5 min, rebuild positions from `/v3/balance` and `/v6/short_positions`; the exchange is the source of truth.

### 5.8 Optional LLM overlay spec (veto-only)

- Inputs every 30 min: headlines from free sources (CoinDesk, The Block, Cointelegraph RSS; Binance announcements for delistings; optionally CryptoPanic API). Deduplicate by URL hash.
- Model: a small, fast model (e.g. Claude Haiku 4.5 class) at temperature 0. About 50 calls/day x 14 days of a few thousand tokens each; cost is small on any small model, but check current pricing.
- Output schema (strict JSON): `{"scope": "market"|"coin"|"none", "coins": [...], "severity": 0-3, "category": "hack|delisting|regulatory|macro|exchange_failure|other", "confidence": 0-1, "rationale": "..."}`.
- Mapping (deterministic, in code): market severity 3 with confidence > 0.7: `news_mult = 0.5` for 6h. Coin severity 3 (hack, exploit, delisting): exit that coin, block for 24h. Severity <= 2: no action. The LLM can **never** open or increase a position.
- Logging: store prompt, response, hash and resulting action in `logs/llm/` so judges can audit. If the API fails, `news_mult = 1` (no-op).

---

## 6. Default configuration (starting point, not tuned)

```yaml
universe:
  max_crypto: 25
  max_spread_pct: 0.15
  min_history_days: 180
  diversifiers: [PAXG]          # bStocks added after prep checks
signal:
  bar: 1h
  lookbacks_h: [24, 48, 96, 168, 336, 504]
  blend: {donchian: 0.5, tsmom: 0.5}
regime:
  on:  {mkt: 0.20, breadth: 0.55}
  off: {mkt: -0.20, breadth: 0.40}
  hysteresis: 0.10
risk:
  sigma_asset_target: 0.15
  sigma_portfolio_target: 0.30
  vol_halflife_h: 48
  cov_halflife_h: 72
  caps: {btc_eth: 0.25, major: 0.12, small: 0.06, bstock: 0.08, bstock_sleeve: 0.25, paxg: 0.20, short_total: 0.50}
  dd_max: 0.05
  dd_floor_mult: 0.25
  daily_brake: {soft: -0.008, hard: -0.015}
  endgame: {from_day: 11, min_return: 0.04, sigma_cut: 0.35}
execution:
  rebalance_minute: 3
  deadband: 0.02
  min_hold_h: 3
  limit_reprice_s: 60
  limit_max_tries: 4
  ticker_poll_s: 30
overlays:
  crowding: {enabled: true, funding_z: 2.0, oi_chg: 0.10}
  news_llm: {enabled: false}
  tournament: {enabled: false}
```

---

## 7. Backtest and validation protocol

### 7.1 Data

- Crypto: Binance Vision (`data.binance.vision`) spot 1h klines (and 1m for execution tests), 2023-01 to present, for all universe coins mapped `COIN/USD -> COINUSDT`.
- Derivatives: Binance Vision futures `fundingRate` and `metrics` (open interest) files.
- bStocks: underlying US stock hourly bars (yfinance limits 1h history to ~2 years) plus our own Roostoo ticker recordings. Model non-US hours as low-liquidity drift with a gap at the open.
- **Start recording Roostoo tickers now** (every 30 s, all pairs) to measure basis and lag vs Binance.

### 7.2 Simulation realism

- Signal at bar close, fill at next bar's open plus 1 to 2 bps slippage.
- Maker fills: assume 60% fill within the re-price window at 5 bps, remainder at taker 10 bps plus slippage. Shorts 10 bps each side.
- Include the deadband, min hold, caps and every overlay in the backtest, with the exact same code path as live (shared strategy module, simulated broker).

### 7.3 Evaluation

- **All rolling 14-day windows** (1-day step) over 2024 to 2026. For each: return, composite (daily-raw, annualised, hourly), MDD, share of red days, trade count, fee drag.
- Stress windows to inspect individually: Aug 2024 yen-carry unwind, Feb to Apr 2025 drawdown, the 10 to 11 Oct 2025 liquidation crash, and the most recent 60 days.
- Baselines: BTC hold, equal-weight top-10 hold, long-only version of our strategy (to prove the short sleeve adds value), cash.
- Objective from Section 2.2. Keep the parameter grid small (under ~50 combos). Compute PBO via combinatorially symmetric cross-validation and a Deflated Sharpe for the chosen config.
- **Team bake-off:** every teammate's strategy variant plugs into the same harness and is judged on the same table. This is also the cleanest way to reach consensus in the new chat.

---

## 8. Engineering spec for the coding assistant

### 8.1 Repo layout

```
trend-guard/
  README.md                 # strategy, rules compliance, how to run, risk controls, results
  config/strategy.yaml
  config/events.yaml        # macro calendar
  src/
    roostoo/client.py       # signing, clock sync, retries (GET only), typed responses, v3 + v6
    roostoo/rules.py        # precision, MiniOrder, decimal formatting
    data/binance.py         # klines, funding, OI; pagination, gap checks
    data/ticker_recorder.py # Roostoo ticker -> 1h bars + raw snapshots
    strategy/signals.py     # donchian, tsmom, ensemble
    strategy/regime.py
    strategy/portfolio.py   # vol targeting, covariance, caps
    risk/overlays.py        # dd governor, daily brake, shock, crowding, endgame
    risk/news_llm.py        # optional, veto-only
    execution/broker.py     # LiveBroker and SimBroker share one interface
    execution/orders.py     # maker-first logic, reprice, reconcile
    state/store.py          # SQLite: positions, orders, fills, equity, decisions
    monitor/dashboard.py    # read-only FastAPI or Streamlit
    main.py                 # scheduler loop
  backtest/engine.py, metrics.py, windows.py, pbo.py
  tests/                    # signing vector from docs, precision, reconcile, overlays
  deploy/trend-guard.service, setup_ec2.sh
  logs/                     # api.jsonl, decisions.jsonl, fills.csv, llm/
```

### 8.2 Non-negotiables

- One code path for backtest and live (strategy never touches I/O).
- Every API request logged with outcome (organizers recommend this). Every decision logged with inputs, signal values, multipliers and resulting orders.
- Persist state in SQLite; on restart, reconcile from exchange before trading.
- Unit test the HMAC signature against the documented example (`20b7fd55...7fff`).
- Clock sync every 5 min. Kill switch file (`/etc/trend-guard/HALT`) checked each loop.
- systemd service with `Restart=always`; health heartbeat every minute.
- **Commit hygiene:** conventional commits; every strategy change in a commit with a short rationale; no API keys in git; no manual order scripts that use the competition key.
- README must map each rule to where it is enforced in code (helps Screens 1 and 4).

### 8.3 Acceptance criteria before 3 Oct 23:00 HKT

1. Bot runs 24h on EC2 with zero unhandled exceptions and full reconciliation.
2. Verified: a maker limit fill, a market fill, a short open and close (or confirmation shorts are disabled), a cancel.
3. Backtest table from 7.3 produced for the final config and at least two teammate variants.
4. Dashboard shows equity, exposure by sleeve, current multipliers, live composite (daily and hourly).

---

## 9. Things to verify in the prep window (1 to 3 Oct)

| Question | How | If it fails |
|---|---|---|
| Are shorts enabled for our account? | Open and close a $10 collateral short | Drop short sleeve; RISK-OFF = cash + PAXG |
| Does a limit at the touch fill as MAKER, and how fast? | 10 small test orders on BTC/ETH | Use market orders, raise deadband |
| How far do Roostoo prices lag Binance? | Compare recorded tickers vs Binance 1m | If lag > 30 s, compute signals from Roostoo bars |
| Do bStock prices move on Saturday 3 Oct? | Record every 30 s through the weekend | Exclude bStocks, or trade them only in US hours |
| Numeric rate limit | Watch error responses while polling at 30 s | Slow to 60 s |
| Scoring sampling and annualisation | Email / WhatsApp organizers again | Keep robust objective |
| Is reading the leaderboard from the bot OK? | Ask organizers | Disable tournament overlay |

---

## 10. Rule-compliance guardrails (do not cross)

- No latency or cross-venue arbitrage (e.g. trading Roostoo because Binance moved seconds earlier). Our signals use closed 1h bars, which keeps us clearly outside "arbitrage".
- No market making (do not post two-sided quotes). Maker-first execution of directional trades is fine.
- No leverage: long notional plus short collateral never exceeds equity.
- No manual trades or ad-hoc scripts on the competition key. All changes via commits and redeploys.
- No copying of public competitor code. Write our own, cite papers.
- LLM output never directly creates orders; it only multiplies risk down, deterministically, and is logged.

---

## 11. Four-person plan

| Person | Owns | By 3 Oct |
|---|---|---|
| A | Roostoo client, execution, reconciliation, EC2 deploy, logging | Live-tested broker, systemd, kill switch |
| B | Data (Binance klines, funding/OI, ticker recorder), backtest engine, metrics, windows, PBO | Harness producing the 7.3 table |
| C | Signals, regime, portfolio construction, short sleeve | Tuned core with ablations (long-only vs long/short) |
| D | Overlays (DD governor, brake, shock, crowding, endgame), dashboard, optional LLM veto, README and deck | Overlays with tests, dashboard live |

Daily during 4 to 17 Oct: 15-minute review of dashboard and decision logs. Change parameters only for bugs or clear rule issues; no mid-competition curve fitting.

---

## 12. Kickoff prompt for Claude Code

> You are building `trend-guard`, an autonomous trading bot for the Roostoo mock exchange (HK vs AU vs IN Quant Trading Hackathon, live 4 to 17 Oct 2026). Read this entire report first; it is the spec. Start with `src/roostoo/client.py` implementing every endpoint in the official docs (v3 spot and v6 shorts), HMAC signing with a unit test against the documented example signature, server-time sync, GET-only retries, and order-state reconciliation instead of blind retries. Then build `SimBroker` and `LiveBroker` behind one interface, the SQLite state store, and the Binance Vision data loader. Then implement the strategy in Section 5 exactly, with every threshold read from `config/strategy.yaml`, and the backtest protocol in Section 7 reusing the same strategy code. Write tests as you go. Use conventional commits with a one-line rationale for each strategy change. Do not add features not in the spec without asking. Include short inline comments in all code.

---

## 13. Annotated bibliography

**Competition and platform**
- Roostoo API docs: https://github.com/roostoo/Roostoo-API-Documents
- Event page: https://luma.com/coghwiyt
- March 2026 community leaderboard and metric rebuild: https://github.com/eric15342335/roostoo-leaderboard
- Competitor notes with organizer clarifications: https://github.com/Xanthen-w/Rustoo (docs/API_NOTES.md, docs/COMPETITION_RULES.md)
- March 2026 top-5-then-fell-back bot: https://github.com/OccupiedPorcupine/HKvSG-TradingBot
- March 2026 post-mortem: https://medium.com/@akitek.mhh/i-built-a-trading-bot-for-a-crypto-hackathon-from-scratch-e0fd8858c754
- bStocks background: https://trustwallet.com/blog/company/access-bstocks-on-trust-wallet-tokenized-shares-on-chain

**Trend and momentum**
- Zarattini, Pagani, Barbon (2025). Catching Crypto Trends. SSRN 5209907. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907 (core template)
- Liu & Tsyvinski (2021). Risks and Returns of Cryptocurrency. RFS 34(6).
- Liu, Tsyvinski & Wu (2022). Common Risk Factors in Cryptocurrency. JF 77(2).
- Moskowitz, Ooi & Pedersen (2012). Time Series Momentum. JFE 104(2).
- Borgards (2021). Dynamic time series momentum of cryptocurrencies. https://www.sciencedirect.com/science/article/abs/pii/S1062940821000590
- Han et al. (2024). Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market. https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf
- Kim, Tse & Wald (2016). Time series momentum and volatility scaling. JFM.
- Systematic Trend-Following with Adaptive Portfolio Construction (2026). https://arxiv.org/html/2602.11708v1

**Volatility management and crashes**
- Moreira & Muir (2017). Volatility-Managed Portfolios. JF. https://www.stern.nyu.edu/sites/default/files/assets/documents/Volatility%20Managed%20Portfolios.pdf
- Barroso & Santa-Clara (2015). Momentum has its moments. JFE 116(1).
- Daniel & Moskowitz (2016). Momentum crashes. JFE 122(2).
- Cryptocurrency momentum has (not) its moments (2025). FMPM. https://link.springer.com/article/10.1007/s11408-025-00474-9
- Harvey et al. (2018). The Impact of Volatility Targeting. https://www.man.com/insights/the-impact-of-volatility-targeting
- Bongaerts, Kang & van Dijk (2020). Conditional Volatility Targeting. FAJ.

**Intraday and cross-asset**
- Shen, Urquhart & Wang (2022). Bitcoin intraday time series momentum. Financial Review. https://onlinelibrary.wiley.com/doi/abs/10.1111/fire.12290
- Wen, Bouri, Xu & Zhao (2022). Intraday return predictability in crypto: momentum, reversal, or both. https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833
- Padysak & Vojtko (2022). Seasonality, Trend-following, and Mean Reversion in Bitcoin. https://quantpedia.com/strategies/intraday-seasonality-in-bitcoin
- Intraday and daily dynamics of cryptocurrency (2024). https://www.sciencedirect.com/science/article/pii/S1059056024006506
- Guo et al. (2024). Cross-cryptocurrency return predictability. JEDC. https://www.sciencedirect.com/science/article/abs/pii/S0165188924000551
- A seesaw effect in the cryptocurrency market (2023). JEF. https://www.sciencedirect.com/science/article/abs/pii/S0927539823000956

**Derivatives and crash risk**
- Schmeling, Schrimpf & Todorov (2026). Crypto Carry. Management Science. https://pubsonline.informs.org/doi/10.1287/mnsc.2024.05069 ; BIS WP 1087: https://www.bis.org/publications/working-paper-1087-crypto-carry
- Cryptocurrency as an Investable Asset Class: Coming of Age (2025). https://arxiv.org/pdf/2510.14435

**Risk control**
- Kaminski & Lo (2014). When do stop-loss rules stop losses? JFM 18. https://dspace.mit.edu/handle/1721.1/114876
- Grossman & Zhou (1993). Optimal investment strategies for controlling drawdowns. Mathematical Finance.
- Brown, Harlow & Starks (1996). Of tournaments and temptations. JF.

**LLM agents**
- Li, Kim, Cucuringu & Ma (2026). FINSABER. KDD. https://arxiv.org/abs/2505.07078 ; code https://github.com/waylonli/FINSABER
- Nof1 Alpha Arena Season 1: https://nof1.ai/blog/TechPost1
- Luo et al. LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management. arXiv 2501.00826v3 (the PDF provided).
- AI in Equity and Crypto Markets: Progress, Profitability Evidence, and the Limits of Automated Investing (2026). https://arxiv.org/pdf/2609.04917

**Statistics**
- Bailey & López de Prado (2014). The Deflated Sharpe Ratio. JPM.
- Bailey, Borwein, López de Prado & Zhu (2017). The Probability of Backtest Overfitting. J. Computational Finance.
- Lo (2002). The Statistics of Sharpe Ratios. FAJ.
