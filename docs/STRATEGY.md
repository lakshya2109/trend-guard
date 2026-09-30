# STRATEGY.md

**trend-guard v2: a low-volatility trend book plus a long/short cross-sectional momentum book,
built for the contest's score rather than for raw return.**

Parameters live in [`config/strategy.yaml`](../config/strategy.yaml); the code is
[`src/trend_guard/strategy/book.py`](../src/trend_guard/strategy/book.py). Every number below is
reproducible with `python backtest/run_backtest.py` and `python backtest/sweep.py`.

## 1. What the contest rewards

After a top-20-by-return screen, teams are ranked on `0.4 Sortino + 0.3 Sharpe + 0.3 Calmar` over
about 14 daily returns. Three consequences drive the design:

1. **The ratios are almost scale-free.** Doubling the book doubles both the mean and the risk, so
   the score barely moves; what a smaller book does change is the chance of a qualifying
   fortnight (around -1% was the 20th-place cut-off in past rounds). So both sleeves run at modest
   volatility targets.
2. **One bad day dominates 14 daily returns.** Diversification between return streams that do not
   lose on the same days matters more than any single signal.
3. **Costs are a first-order effect.** At 0.10% fee plus slippage per trade, a book that turns
   over its capital three times a fortnight gives up about 0.45% per fortnight, roughly the whole
   median return of most books we tested.

## 2. The book

| Sleeve | Capital | What it does |
|---|---|---|
| **Trend** | 70% | Long-only time-series trend on every liquid Roostoo crypto pair (including PAXG, tokenized gold). Score = mean over lookbacks of the log return divided by the volatility expected over that horizon (a t-statistic), clipped. Weight ~ score / volatility for names above a threshold, the book scaled to 0.4%/day volatility with the 7-day covariance, per-name cap 50%, positions under 1% dropped because they cannot pay their round-trip cost. Rebalanced daily at 12:00 UTC. **Ensemble of 4 members**: lookbacks {7/14/30 days, 7/14/30/60 days} x entry thresholds {0.25, 0.40}. |
| **Momentum** | 30% | Cross-sectional long/short. Once a day (00:00 UTC) rank liquid coins (>= $5M/day, >= 30 days of history, PAXG excluded) by risk-adjusted momentum; long the best 6, short the worst 6 (a holding stays while ranked within 12), 60/40 long/short, equal weight per side, resized every hour to 1.5%/day volatility under a constant-correlation model, gross capped at 0.9. **Ensemble of 4 members**: lookbacks {3/7/14 days, 7/14/30 days} x skip {0h, 12h}. |

The account target is the capital-weighted sum of both sleeves, netted per pair. The two shares
sum to 1 and each sleeve caps its own gross, so long notional plus short collateral can never
exceed equity (asserted in code). A catastrophe breaker halves every target if the account falls
10% below its peak (a safety net for bugs or crashes, not a trading rule; it never triggered in the
backtest).

**Trade rule (identical in backtest and live, `execution/order_manager.plan_orders`):** act at the
decision hours (00:00 and 12:00 UTC) or when a target changes; skip a resize smaller than
max(1% of equity, 25% of the target) unless it opens or closes a position; sign flips close at
market and reopen on the other side in the same cycle. Between decisions quantities are held, so
winners are not trimmed every hour.

## 3. Results (every 14-day window, 12:00 UTC starts, $100k, 0.15% per side)

Same data (hourly Binance candles for all 65 tradable Roostoo crypto pairs), same windows, same
costs and the same score as the team's `roostoo-last-supper` harness; its published numbers are
shown for comparison.

| 2026 (258 windows) | Score | Median fortnight | Above -1% | Positive | Worst | Turnover |
|---|---|---|---|---|---|---|
| **trend-guard v2** | **0.159** | +0.72% | **85%** | 68% | **-2.2%** | **2.1x** |
| Team combined 70/30 (published) | 0.131 | +0.91% | 84% | 69% | -2.2% | 3.1x |

| All windows, Feb 2024 - Sep 2026 (944) | Score | Median fortnight | Above -1% | Positive | Worst |
|---|---|---|---|---|---|
| **trend-guard v2** | **0.076** | +0.45% | **76%** | **59%** | **-4.3%** |
| Team combined 70/30 (published) | 0.059 | +0.46% | 75% | 59% | -4.8% |

2024-25 only (686 windows): v2 0.055 vs 0.041 on score, and better on all five columns in our
replay of the team book. BTC buy-and-hold scored 0.008 over all windows with a -18% worst
fortnight.

**How strong is this?** Moderate, not decisive. v2 beats the team book in 56% of individual
windows. A block bootstrap over fortnights gives P(v2's median score > team's) = 77% over all
windows but only 57% on 2026 alone: 2026 contains about 18 independent fortnights, too few to
separate two good books with confidence. v2's 2026 mean fortnight return equals the team's
(+1.41% vs +1.40%); its median is lower because its distribution is more right-skewed.

## 4. Research log (what was tested, and why the book looks like this)

Method: all 944 contest windows plus a single continuous after-cost run (daily Sharpe) per variant,
which is far less noisy than a median over overlapping windows. Selection used 2024-25; 2026 was
kept for confirmation.

1. **trend-guard v1 (BUILD_SPEC design) failed after costs.** A Donchian + momentum ensemble with
   hourly rebalancing, regime gates and seven overlays: +22% gross over 2024-26 but -52% net,
   because turnover was 190x equity per six months. Its hourly-ATR trailing stop (about 1-2% from
   the peak) was hit by ordinary noise, producing constant stop-and-re-enter churn.
2. **The median window score is noisy.** Basket size k = 4/6/8/10 gave 0.020/0.078/0.035/0.079:
   not a pattern, just noise. Hence ensembles instead of single parameter picks.
3. **Trend sleeve: stop dropping small positions, use longer lookbacks.** A 0.4%/day book cut at a
   1% dust floor holds a concentrated few names; lookbacks out to 60 days and an ensemble over
   thresholds lifted the continuous 2024-25 Sharpe from 0.3 to 1.0-1.7 across the whole grid.
4. **Momentum: ensemble and hourly vol-targeting.** Averaging lookbacks x skips (Sharpe 2.0 / 2.1
   in 2024-25 / 2026) beat any single member and cut turnover. Including the unskipped member
   helped in every variant of the sweep.
5. **Rejected** (worse or not robust in both periods): residual (beta-adjusted) momentum (Sharpe
   2.1 in-sample, 0.1-0.4 in 2026), wider rank buffers and enter/exit bands (less turnover but
   lost alpha: the momentum signal decays quickly), market-trend-dependent short share (the
   momentum-crash hedge lowered the score in every variant), book-level volatility targeting on
   top of the sleeves, score-ensembles, dispersion scaling, trailing stops, tight drawdown brakes.
6. **Things that looked like edges and were not:** short-term (1-3 day) cross-sectional reversal
   looked like a Sharpe -5 anomaly; the mirror trade loses too once costs are counted and it
   flips sign with the decision hour. A "seasoned minus newly listed" factor (Sharpe 1.1 / 1.0)
   flipped sign when rebalanced every 3 instead of 7 days and came almost entirely from 2025.

7. **Second research round (literature ideas, none adopted).** Each was tested as a new sleeve or
   a change to v2, selected on 2024-25 and confirmed on 2026:

   | Idea (source) | Result on the Roostoo universe |
   |---|---|
   | Volume-weighted market time-series momentum (Huang, Sangiorgi & Urquhart 2024, SSRN 4825389; Sharpe 2.17 reported) | Sharpe -0.5 (2024-25) / -1.2 (2026), long-only version too. The paper's 3,000 mostly tiny coins behave differently from 65 liquid ones |
   | MAX / lottery effect (Li et al., Financial Innovation 2021) | Sharpe 0.0 / -1.3 |
   | Adaptive per-coin trend selection by trailing Sharpe (AdaptiveTrend, arXiv 2602.11708) | lower score in 2024-25 and overall |
   | Trailing stops on the momentum sleeve (AdaptiveTrend's largest ablation) | no gain at 2x, 3x or 4x daily volatility |
   | Sleeve momentum: tilt capital toward the sleeve with the better trailing Sharpe ("Cryptocurrency Factor Momentum") | only a 60-day window helped; 30, 45, 90 and 120 days did not: an isolated cell, treated as noise |
   | Volume-confirmed momentum (volume-based momentum literature) | better in 2024-25 in 23 of 24 variants, worse in 2026 in 24 of 24: a regime effect, not an edge |

   The last row is the clearest example of why 2026 was held out: tuned on all data at once it
   would have been adopted.

**Overfitting check.** `backtest/sweep.py` replays the 36-variant grid and computes the
probability of backtest overfitting (CSCV, 16 partitions): 74%. Among these close variants the
in-sample winner is not reliably the out-of-sample winner, so the exact pick carries no weight;
the structural choices (ensembles, both skips, trend lookbacks to 60 days, low turnover) hold
across the grid. The final pick was made among variants that beat the team book on all five
2024-25 metrics, then by 2026, so 2026 is not a perfectly clean out-of-sample test for it.

## 5. Honest caveats

- Over 14 days the market path dominates. In 2026 windows where BTC fell more than 5%, the book
  was positive only 44% of the time.
- Backtests fill at the hourly close plus 5 bps; live fills can be worse in fast markets. Live
  execution is maker-first (0.05% fee when the limit rests), which the backtest does not credit.
- The universe is today's Roostoo listings, so coins delisted before 2026 are missing
  (survivorship bias affects both books equally).
- Shorts use Roostoo `/v6`; if they are disabled for the account the bot drops the short leg
  automatically and runs long-only.

## References

- Zarattini, Pagani & Barbon (2025), *Catching Crypto Trends*, SSRN 5209907.
- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, JFE.
- Liu & Tsyvinski (2021), *Risks and Returns of Cryptocurrency*, RFS; Liu, Tsyvinski & Wu (2022), JF.
- Moreira & Muir (2017), *Volatility-Managed Portfolios*, JF; Barroso & Santa-Clara (2015), JFE.
- Daniel & Moskowitz (2016), *Momentum Crashes*, JFE.
- Bailey, Borwein, Lopez de Prado & Zhu (2017), *The Probability of Backtest Overfitting*.
- Roostoo API documentation: github.com/roostoo/Roostoo-API-Documents.
