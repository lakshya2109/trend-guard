# STRATEGY.md

**trend-guard: 70% trend-guard v2 (a low-volatility trend sleeve plus a long/short momentum sleeve)
blended with 30% of the teammates' combined book, built for the contest's score rather than for raw
return.**

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
| **Momentum** | 30% | Cross-sectional long/short. Once a day (00:00 UTC) rank liquid coins (>= $5M/day, >= 30 days of history, PAXG excluded) by risk-adjusted momentum; long the best 6, short the worst 6 (a holding stays while ranked within 12; no new short on a coin already down more than 40% over 60 days), 60/40 long/short, equal weight per side, resized every hour to 1.5%/day volatility under a constant-correlation model, gross capped at 0.9. **Ensemble of 4 members**: lookbacks {3/7/14 days, 7/14/30 days} x skip {0h, 12h}. |

The account target is the capital-weighted sum of both sleeves, netted per pair. The two shares
sum to 1 and each sleeve caps its own gross, so long notional plus short collateral can never
exceed equity (asserted in code). A catastrophe breaker halves every target if the account falls
20% below its peak (10% until 9 Oct, round 22) (a safety net for bugs or crashes, not a trading rule; it never triggered in the
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

## 3b. The blend with the teammates' book

The live book is 70% v2 + 30% of the teammates' combined book (Gunek-945's trend book and swyam-rd's
momentum book in their own 70/30 split), vendored unmodified in `src/trend_guard/strategy/team/`.
The two books' fortnightly returns are 91% correlated, so the gain is not hedging: the books pick
their names with different code, and averaging them cancels part of each one's selection noise.
Unlike the ideas rejected in the research log, the blend improves the score in both periods and at
both splits tested (50/50 and 70/30):

| Score | 2024-25 | 2026 | All |
|---|---|---|---|
| v2 alone | 0.055 | 0.159 | 0.076 |
| Team book alone | 0.041 | 0.131 | 0.060 |
| **70% v2 + 30% team** | **0.060** | **0.178** | **0.088** |
| 50% v2 + 50% team | 0.061 | 0.164 | 0.092 |

The 2026 median fortnight rises from +0.72% (v2) to +0.96%, the positive share from 68% to 69%, the
share above -1% stays at 85%, and the worst 2026 fortnight moves from -2.2% to -2.3%.

## 3c. Risk level (2x) and the ML model that was not adopted

**Risk dial.** The composite score is nearly scale-free, so the book's size sets the return, not the
ratios. At 1x the book was 63% cash (median gross 0.37). Scaling the whole book:

| Book size | Gross | Score (all) | Median fortnight (all / 2026) | P(> +5%) all | P(< -5%) all | Worst |
|---|---|---|---|---|---|---|
| 1x | 0.37 | 0.088 | +0.53% / +0.96% | 8% | 0% | -4.2% |
| 1.5x | 0.55 | 0.095 | +0.88% / +1.40% | 15% | 2% | -6.4% |
| **2x (live)** | 0.74 | 0.088 | **+1.08% / +2.14%** | **20%** | 3.5% | -8.5% |
| 3x (capped) | 1.00 | 0.068 | +1.06% / +2.42% | 27% | 7% | -11.4% |

Tilting risk toward either sleeve (trend x1-5 by momentum x1-3, 20 combinations) did not beat scaling
both evenly; a pre-declared rule (max 2024-25 median return subject to no loss of 2024-25 score) picked
1x, i.e. more return is not free: it is bought with deeper bad fortnights. 2x was chosen by the team
for its return profile, knowingly.

**ML model (walk-forward, not adopted).** A cross-sectional model (ridge and a regularised LightGBM,
17 rank features, monthly expanding-window retraining with a label embargo, every prediction
out-of-sample) predicts next-day relative returns with a genuine, stable rank IC of +0.07 to +0.13 in
each of 2024, 2025 and 2026. The signal is a low-risk effect: high-volatility, high-beta, recently
listed and "lottery" coins underperform the next day (survivorship bias would work against this
finding, not for it). It does not convert into return:

- a long/short ML book lost money after costs in 7 of 8 pre-declared variants (its short side is the
  riskiest coins, which squeeze hardest in rallies);
- used as a filter on the momentum sleeve it raised the score (all windows 0.088 -> 0.108) and the
  positive-fortnight share, but lowered the mean return in every variant, the opposite of the goal.

It is documented rather than deployed; it is the natural next step if the team ever wants a smoother
book instead of a higher-return one.

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

8. **Third round (pre-launch report ideas, none adopted).** Nine pre-declared variants on the live
   2x blend. Rule: median score up in both 2024-25 and 2026, share of fortnights above -1% down by at
   most 2 points, and the worst fortnight no worse.

   | Variant | Score 2024-25 / 2026 (base 0.057 / 0.191) | Worst | Why rejected |
   |---|---|---|---|
   | Trend: smooth entry ramp instead of the 0.25/0.40 thresholds (4 ramp widths) | 0.059-0.075 / 0.175-0.203 | -9.0% to -10.0% (base -8.5%) | 2026 up in only 2 of 4 widths; mean return unchanged; loss tail fatter in all 4 (P(< -5%) 3.5% -> 4.6-4.8%) |
   | Wider no-trade band (skip resizes < 40% of target or 1.5% of equity) | 0.070 / 0.161 | -7.7% | turnover -11%, but 2026 falls: the momentum signal decays too fast to trade late |
   | Shorts held at most 21 days, then a 7-day ban | 0.058 / 0.191 | -8.5% | no effect: shorts rarely last that long |
   | No shorts in the least liquid third | 0.065 / 0.185 | -8.5% | 2026 lower |
   | Momentum lookbacks 3/7/14 + 7/14/21 days (drop 30) | 0.052 / 0.210 | -8.1% | 2024-25 lower |
   | "CTREND-lite": rank on t-stat momentum + Bollinger, CCI, MACD | 0.056 / 0.126 | -9.1% | worse everywhere |

   A per-short cap does nothing here (shorts are already equal-weight, 1/6 of the short side each).
   Funding-rate filters were not tested: the bar cache has no futures funding history.

9. **Fourth round: the rest of the pre-launch report. Adopted: a crash filter on new shorts.**
   Same rule and base as round 8.

   | Variant | Score 2024-25 / 2026 | Worst | Result |
   |---|---|---|---|
   | No new shorts on coins down more than 20 / 25 / 30 / **40%** over 60 days | 0.060 / 0.209, 0.060 / 0.194, 0.063 / 0.199, **0.063 / 0.204** | -8.5%, -8.8%, -9.2%, **-8.1%** | every cut-off raises the score in both periods without lowering the above -1% share; **40% adopted** (passes the whole rule, blocks the fewest shorts). Consistent with crypto losers reversing after about a month (Dobrynskaya) |
   | Band on the hourly vol-target resize (skip if the scale moved < 20%) | 0.057 / 0.193 | -8.3% | passes by a trivial margin with unchanged turnover: noise. Freezing the whole book between the 00/12 decisions cuts turnover only 2%, so hourly resizing is not where the churn is |
   | Nearness to the 90-day high added to the momentum rank | 0.069 / 0.146 | -8.4% | 2026 much worse |

   Diagnostics (no change made): book return per unit of risk by forecast BTC-vol tercile is 0.15 /
   0.06 / 0.13 (low / mid / high), so no vol-day filter; weekends (0.14) are no worse than weekdays
   (0.11), so no weekend cut. 14 variants were tried across rounds 8-9, above the report's 8-10 cap,
   so the filter's +0.007 all-window gain (0.088 -> 0.095) should be read as modest.

10. **Sleeve volatility targets re-tested on the live book (2x, crash filter): kept at 0.4% / 1.5%.**
    Grid of trend targets 0.2-1.5%/day x momentum targets 1.0-2.5%/day (32 cells). No cell beats
    the live pair in both 2024-25 and 2026; the live pair has the highest 2026 score in the grid
    (0.204). Higher targets are worse everywhere (trend 1.0%: 0.040-0.050 / 0.158-0.163, worst
    fortnight -10.6% to -11.3%): more risk does not buy more score or median return. Lower targets
    (0.2% / 1.0%) raise the all-window score to 0.102 and soften the worst fortnight (-6.3%) but cut the
    median fortnight from +1.13% to +0.87%, the same trade as a smaller risk dial. Adjacent cells differ
    by about +-0.01 in 2024-25 with no pattern, so picking the grid maximum would be fitting noise.

11. **Faster signals and jump-chasing (not adopted).** Gross edge by horizon (2024-26, liquid coins):
    1-4 hour moves partly reverse (-1 to -2 bp per period, t = -5.9 at 1h) against a ~30 bp round trip;
    momentum appears only from ~12h. After a 1-hour jump above 5%, the next hour is flat relative to the
    market, so reacting within the hour gains nothing, but the coin beats the market by ~+1% over the
    next day on average. A sleeve buying at the close of the jump hour and holding 12-24h loses on its
    own after costs (median fortnight -0.4% to -2.9%, worst -19% to -27%), and 10-20% of it in the
    book lowers the score in both periods in all 4 pre-declared variants (best: 0.050 / 0.101 vs
    0.063 / 0.204). Jumps above 10% reverse (-4.6% vs the market over 3 days).

12. **Momentum re-rank frequency (kept daily).** Re-ranking every 1 / 3 / 6 / 12 / 24 / 48 hours
    scores 0.039 / 0.056 / 0.062 / 0.070 / 0.063 / 0.038 (2024-25) and 0.159 / 0.178 / 0.191 / 0.189 /
    0.204 / 0.152 (2026), turnover 6.9 / 5.8 / 5.3 / 4.8 / 4.8 / 4.4x. The daily re-rank at 00 / 06 / 12
    / 18 UTC alone spans 0.051-0.069 and 0.168-0.232, so 6-24h is a plateau inside that noise, while
    hourly and 3-hourly re-ranking are clearly worse: coins near the top-6 cut-off swap on noise and each
    swap pays a round trip. Hourly re-ranking without the rank buffer collapses (0.015 / 0.066, 10.6x).

13. **Basket size and market-dependent long/short split (both kept).** Momentum k = 3-15 per side
    (buffer 2k): 3-4 worse everywhere; 8-15 slightly better in 2024-25 (0.071-0.073) but 2026 falls
    steadily (0.188 -> 0.128), so k = 6 stays. The live book is already regime-adaptive: net exposure
    averages +15% of equity when BTC fell >10% over 14 days and +45% when it rose >10% (the trend
    sleeves only buy rising coins). Tilting the momentum short share with BTC's 14-day trend lowered
    both periods in all 3 variants (best 0.044 / 0.159 vs 0.063 / 0.204), matching the earlier
    rejection of a market-trend-dependent short share.

14. **Competition scoring, non-overlapping fortnights, and the 2.5x risk level (adopted).** The previous
    edition's leaderboard (52 teams) shows score = active days x (0.4 Sortino + 0.3 Sharpe + 0.3 Calmar),
    ratios normalised across the field, Calmar = total return / max drawdown (not mean daily return /
    max drawdown as in the score used above, so its values are about 14x larger). Evaluated on
    non-overlapping fortnights against that field: the 2x book would have ranked first in 62-66% of
    fortnights and traded on 14-15 days in every one. Variants on that basis (68 fortnights):

    | Variant | Capital deployed | Median fortnight | P(> +5%) | P(< -5%) | Calmar | P(rank 1) |
    |---|---|---|---|---|---|---|
    | 2x (previous) | 71% | +1.2% | 18% | 2.9% | 0.29 | 66% |
    | **2.5x** | **85%** | +1.1% | **24%** | 2.9% | 0.29 | 62% |
    | 3x | 93% | +1.3% | 26% | 4.4% | 0.24 | 65% |
    | 4x | 97% | +1.4% | 25% | 7.4% | 0.31 | 63% |
    | floor: always >= 80% invested | 82% | +1.3% | 22% | 8.8% | 0.32 | 66% |
    | momentum re-ranked every 12h | 70% | +1.2% | 19% | 1.5% | 0.33 | 62% |
    | momentum lookbacks 1/3/7 + 3/7/14 days | 70% | +1.1% | 22% | 2.9% | 0.27 | 59% |
    | v2 momentum 50% of v2 (from 30%) | 85% | +0.7% | 25% | 2.9% | 0.18 | 63% |
    | short side 50% (from 40%) | 73% | +1.1% | 18% | 1.5% | 0.32 | 68% |

    The rank against last year's field barely moves (59-68%): it is set by whether the fortnight is good,
    not by the variant. 2.5x was adopted at the team's request to keep more capital working: it raises
    P(> +5%) and keeps the ratios, at the cost of a fatter 2024-25 loss tail on all 944 windows (P(< -5%)
    4.4% -> 7.9%, worst -8.1% -> -9.8%). The 50% short side helped 2024-25 but hurt 2026 at both 2x and
    2.5x (also checked at 45% and 55%), so it was not adopted; faster re-ranking stays inside noise.
    In quiet, trendless markets the trend sleeves hold little and the book stays partly in cash by design
    (late September 2026: ~40% invested at 2x, ~50% at 2.5x).

15. **Aggressive settings for return (4x adopted).** The team asked for the highest-return book. All 944
    windows, competition scoring:

    | Book | Invested | Median | Mean | P(> +5%) | P(> +10%) | P(< -5%) | Worst | P(rank 1) |
    |---|---|---|---|---|---|---|---|---|
    | blend 2.5x | 85% | +1.1% | +2.1% | 26% | 10% | 6.0% | -9.8% | 63% |
    | **blend 4x** | **97%** | **+1.3%** | **+2.5%** | **30%** | **14%** | 9.6% | -12.2% | 63% |
    | v2 only 4x | 97% | +1.2% | +2.5% | 30% | 14% | 10.0% | -11.5% | 61% |
    | full-size momentum long/short, 6 per side | 98% | +0.7% | +2.6% | 31% | 21% | 18.6% | -16.1% | 57% |
    | full-size momentum long/short, 3 per side | 97% | +0.2% | +2.4% | 33% | 21% | 25.0% | -17.3% | 43% |
    | full-size momentum long-only | 75% | -0.3% | +3.1% | 36% | 25% | 37.5% | -24.8% | 25% |
    | full-size trend long-only | 68% | -0.7% | +2.3% | 27% | 18% | 24.6% | -14.8% | 7% |
    | 50/50 full-size trend + momentum | 84% | +0.3% | +2.9% | 31% | 18% | 16.4% | -12.7% | 54% |

    No design lifts the mean fortnight much above +2.5-3%: extra aggression mostly spreads outcomes.
    The blend at 4x has the highest median and keeps its rank against the previous field; the
    concentrated books buy more +10% fortnights with two to four times as many -5% fortnights.

16. **ML and new data sources (none adopted).** `research/` (data: `research/fetch_extra.py`). New data
    for all 66 coins, Jun 2023 - Sep 2026: spot taker-buy volume and trade counts, perp funding, and
    Binance futures open interest with top-trader / account / taker long-short ratios. Harness:
    `research/harness.py` (contest simulator, plus one continuous run from 2024 for annualised Sharpe).
    The live 4x book there: Sharpe 1.50 / 2.09 / 2.69 (2024 / 2025 / 2026), **3.0 before costs**.

    | Idea | Result |
    |---|---|
    | LightGBM / ridge, 30-48 cross-sectional features, monthly walk-forward, label = next-day return vs the market in units of vol (`alpha_model.py`) | rank IC +0.03-0.05 every year, but a vol-targeted market-neutral sleeve has Sharpe 0.8 before costs and -1.4 after (11x turnover a fortnight) |
    | Same plus order-flow features | IC unchanged (0.033 vs 0.037) |
    | Time-series LightGBM (each coin's own next 1-3 day return), long-only | Sharpe +1.3 in 2024, -1.1 / -2.0 in 2025 / 2026: learns the regime |
    | Low-risk effect (idiosyncratic vol, the strongest rank IC: -0.08, stable) | shows in ranks, not in mean returns (high-vol coins are right-skewed); quintile spread t = -0.3 |
    | Funding level / change as a signal | spread changes sign by year |
    | OI-confirmed momentum (price up with open interest up) | adds +23 bp/day over momentum (Fama-MacBeth t 2.4) but +37 / +29 / -6 bp by year; as a tilt in the momentum ranking it lifts 2024 (1.50 -> 1.85) and lowers 2026 at every strength (2.69 -> 2.61 / 2.29 / 2.02). The same pattern as volume-confirmed momentum (round 7) |
    | Patient maker execution: limit at the close, re-posted hourly up to 1-6 h, fills only if a bar trades 10 bp through it (`exec_sim.py`) | fees 14.6% -> 9.5% a year, but adverse selection (missed fills run away) cancels it: Sharpe 2.07 -> 2.04-2.10. Only a fill-on-touch assumption gains (2.33), which cannot be verified offline |
    | Partial rebalancing toward targets (lambda 0.1-0.5) | turnover -6 to -15% only (most trading is names entering and leaving); 2026 lower |
    | Inverse-vol weights within each momentum side | +0.01 to +0.04 Sharpe: noise |
    | Sleeve capital split (256-cell grid, chosen on 2024-25) | best 2024-25 cells (1.80 -> 2.18) all score lower in 2026 (2.69 -> 2.1-2.5) |

    Cost sensitivity, for reference: the same book at 10 / 7 / 5 bp per side has Sharpe 2.30 / 2.44 / 2.53
    and median fortnight +1.7 / +1.9 / +2.0% (vs +1.4% at 15 bp). Whatever maker share the live bot
    achieves within its ~4-minute limit window is upside the backtest does not count.

17. **Risk dampening, capacity and execution (none adopted).** Dampening only helps the score when it is
    selective (cuts exposure before bad days more than good ones); uniform dampening is the risk dial.

    | Idea | Result |
    |---|---|
    | Return per unit of gross before costs, by leg (`research/` inline) | momentum longs 1.37-1.55 a year, trend longs 0.9-1.0, momentum shorts 0.44-0.50 (Sharpe 0.5): the shorts are the hedge |
    | v2 momentum short share 0 / 0.2 / 0.3 / 0.5 (live 0.4; `short_share.py`) | less short: same return, P(fortnight < -5%) doubles (7% -> 15-18%). 0.5: Sharpe 2.07 -> 2.22 (2024 / 2025 up) but 2026 2.69 -> 2.58, as in round 14 |
    | Market-wide leverage as a risk gauge: total OI change, mean funding, account / top-trader / taker long-short ratios, breadth, vol ratio (`market_state.py`) | rank correlation with the book's next-day return below 0.05 for every gauge, sign changing by year |
    | Staggered sub-books (each v2 sleeve split over decisions at 00 / 06 / 12 / 18 UTC; `stagger.py`) | Sharpe 2.07 -> 2.12, turnover -13%, but 2026 2.69 -> 2.45 (the live hours were lucky in 2026) |
    | Maker patience on real 1-minute Binance bars (598 random 2026 decision times; `minute_exec.py`) | the live policy (touch, reprice each minute, market after 4 min) costs 10-12 bp per spot trade vs 12.5-15 bp taker; waiting 10-60 min saves < 1 bp (inside the standard error). The backtest's 15 bp is conservative |
    | Fading 1-hour jumps (round 11's lead; `jump_fade.py`) | > +10% jumps: -9.2% vs the market over 3 days in 2026 (t -3.0, 78% of events) but +2.1% in 2024: a regime (the 2025-26 listings), not a law |
    | No new longs for 24 / 72 h after a +7 / 10 / 15% hour (`jump_filter.py`) | +7% or 72 h hurt (they block momentum entries); +10% / 24 h lifts every year slightly (2.07 -> 2.14) but its neighbours do not: treated as noise |

    **Activity rule (adopted, not a strategy change):** if no fill lands by :15 inside an activity window
    (00-03 / 12-15 UTC), the engine sends one small market order toward the book's own target, retrying
    hourly, so every contest date is active (the rules need 8; last edition's score scaled with them).

18. **One test for every candidate; two adopted (`research/paired.py`, `research/combo.py`).** Rounds
    16-17 judged candidates by eye from yearly Sharpe, which is too noisy for books that are 98-99%
    correlated (a +0.1 difference is about one standard error) and led to inconsistent calls. Every
    candidate now goes through the same paired stationary block bootstrap (same resampled days for both
    books, mean block 10 days, 2,000 reps) on the continuous daily returns from 2024. Adoption needs a
    high P(dSharpe > 0) AND a gain in both 2024-25 and 2026. Base: trend-guard-final (Sharpe 2.12).

    | Candidate | dSharpe | P(> 0) | 2024-25 | 2026 | d return / yr |
    |---|---|---|---|---|---|
    | anant's trend covariance (14-day, shrink 0.1, 72h EWMA) vs the 4x one | +0.04 | 86% | +0.08 | -0.03 | +0.7% |
    | **jump guard: no new / larger longs 24h after a +10% hour** | **+0.06** | **97.5%** | **+0.02** | **+0.15** | **+2.0%** |
    | jump guard +15% / 24h (neighbour) | +0.03 | 96% | +0.01 | +0.07 | +0.9% |
    | **momentum: weight ~ 1/sqrt(vol) within each side** | **+0.07** | **86%** | **+0.06** | **+0.09** | -1.9% |
    | momentum short share 0.5 | +0.13 | 89% | +0.24 | -0.09 | -0.3% |
    | momentum OI tilt 0.25 (needs live futures data) | +0.20 | 96% | +0.30 | -0.02 | +6.3% |
    | staggered sub-books / partial rebalancing | 0.00 / -0.01 | 51% / 43% | | | |
    | **adopted: inverse-vol + jump guard together** | **+0.14** | **97.7%** | **+0.08** | **+0.27** | +0.2% |

    The adopted pair would also have been picked on 2024-25 alone and is confirmed on 2026. Continuous
    Sharpe by year 1.63 / 2.12 / 2.66 -> 1.66 / 2.25 / 2.93, max drawdown 28.6% -> 26.3%, P(fortnight
    < -5%) 7.4% -> 5.9%, return unchanged. Over all 944 contest windows it has the higher Sharpe in 57%,
    Sortino 56%, Calmar 56% and the smaller drawdown in 80%; across the 14 non-overlapping offsets its
    median Sharpe is higher in 10 (the official report's offset 0 is one of the 4 where it is lower).
    Nine candidates were tried, so a 97.5% result alone could be luck; the two-period requirement and the
    mechanism (2025-26 pumps revert: `jump_fade.py`) are what support it. The jump guard never sells:
    it only stops buying into a coin for a day after a pump.

19. **Mid-contest regime switch (adopted 9 Oct, `research/fast_adapt.py`).** Contest objective after a -7%
    start: maximise P(return over the ~230 hours left > the gap to the HK top-20 cut-off, about +4.5%); extra
    downside no longer costs anything. Bear mode = while the liquid-coin market is down > 8% over 5 days
    (leave above -2%), run v2 momentum alone, re-ranked every 6 h. Over 699 past starts right after such a
    crash: P(> +4.5%) 26.6% -> 29.0%, P(> +6%) 20.2% -> 24.3%, in both 2024-25 (20.1% -> 22.8%) and 2026
    (54.1% -> 55.6%); over all 4,006 starts 24.2% -> 26.1%. Rejected bear modes: cutting or halving longs
    (15-21%) and going net short (23-24%): the switch fires near the bottom and misses the rebound.

20. **Bear mode = oversold rebound (adopted 9 Oct, `research/rebound.py`).** Same regime switch; in bear mode the
    book holds the 10 liquid coins with the worst 3-day return, equal weight, gross 1, re-picked daily at 00 UTC.
    After past crashes (699 starts, 225 h left): P(> +4.5%) 27% (old) / 29% (fast momentum) -> 42%, P(> +6%)
    20% / 24% -> 35%, P(> +8%) 12% / 16% -> 25%; downside heavier (P(< -8%) 2% / 8% -> 15%), which no longer
    matters for the top-20 cut. Robust to k = 5 / 10, 3- or 5-day lookback and inverse-vol weights (39-42%).
    By crash episode (50 since 2024) it was best at clearing +4.5% in 34 (fast momentum 9, old bot 7), 6 of 11
    in 2026; its lower 2026 average comes from one long episode (29 Jan 2026, 42 overlapping starts).

21. **Bear books that adapt to the current crash: none adopted (`research/bear_shorts.py`,
    `research/adaptive_bear.py`).** Same regime, objective and 699 post-crash starts (220 h left). Shorting the
    same 10 worst coins (continuation): P(> +4.5%) 20%, median -2.9%. Designs fixed before the run that hold
    whatever works now: daily follow-the-leader on trailing 48 / 72 h return after costs over {rebound,
    short 10 worst} or over {those, fast momentum, short BTC+ETH, cash}: 15-28%; rebound only once the
    market's 24 h return is positive, else shorts / cash / momentum: 15-23%; 50% old book + 50% rebound: 33%.
    The fixed rebound book stays best (42%; episode-weighted 46%, best in 27 of 50 episodes against all nine).
    Mechanism: in a crash the trailing winner is the short book, and the selector switches into it near the
    bottom, just before the rebound it was waiting for. On 8 Oct 19:00 UTC (bear mode on, market -9.3% over
    5 days) the short book was +12.6% over 72 h and the rebound book -13.4%: the trap these designs fall into.

22. **Breaker level 0.10 -> 0.20 (adopted 9 Oct, `research/breaker_now.py`).** Rounds 19-21 ran with the breaker
    off and each window starting at its own peak; live, the account starts the window ~7% under its peak, so
    a further ~3% loss halved the book until it recovered inside 5%. Same 699 post-crash starts, 220 h left,
    simulated from 7% under the peak, P(> +4.5%) for the rebound switch / old bot: breaker 0.10 25.5% / 24.5%,
    0.15 35.9% / 26.3%, 0.20 41.6% / 26.3%, off 42.3% / 26.3% (5% and 9% under the peak: same order). At
    0.10 the breaker cancelled almost all of the switch's gain. 0.20 keeps a catastrophe net for bugs and
    outages at almost no cost. A trip saved under the old level is judged again under the new one on restart.

23. **Bear mode that follows the market's direction: not adopted (`research/market_timing.py`).** Inside the
    regime, every 6 h: short while the market's trailing 24 / 72 h return is down, long while it is up (short
    the 10 worst 3-day coins or an equal-weight liquid basket; long the rebound book or the basket). Live
    breaker, 7% under the peak, 699 starts: P(> +4.5%) 21-27% against 42% for the rebound book. By what the
    market did next: when it kept falling (30% of cases) the 72 h followers win (34-38% against 8%); when it
    bounced (35%) they lose (18-29% against 72%), and sideways they whipsaw (13-21% against 39%). The direction
    is only known after the move, and after a crash the next move was more often up.

24. **Bear mode switched off (13 Oct, `research/bear_mode_check.py`, `research/factor_regime.py`).**
    Contest so far (4-13 Oct): the average liquid coin -8.2%, BTC -2.6%, 16% of coins up. In the replay
    (`scripts/simulate.py`, -2.7%; live -4.7%) the bear-mode rebound book made about +11% (gross, 9-11 Oct)
    and every normal sleeve lost (v2 trend -3.9%, v2 momentum -4.4%, team book -3.9%): a short-term
    reversal regime, the one that hurts momentum and trend. It cannot be timed: the trailing 7 / 14 / 28-day
    return spread between reversal and momentum books predicts the next 5 days' spread with correlation
    -0.03 to +0.08 and a 49-55% sign hit rate.

    Rounds 19-20 judged bear mode on P(+4.5% within 225 h of a crash) and the median, both of which reward
    variance. On the continuous run from 2024 it is clearly harmful: Sharpe 0.68 / 1.25 (2025 / 2026) with it,
    2.25 / 2.65 without, P(better without) 99.7% in the paired bootstrap and in both periods, max drawdown
    56% vs 25%; on the 26% of days it was on it cost 0.69% a day (t -2.4). After past crashes the mean
    225 h return is +1.67% with it vs +1.96% without (the medians, +2.72% vs +1.04%, hid a 19% vs 2%
    chance of losing more than 8%). Every other crash override is also worse than letting the normal book
    trade through (dSharpe: fast momentum -0.58, normal book at half size -0.15, cash -0.56; all worse in
    both periods): its trend sleeve already goes to cash and its momentum sleeve already shorts the losers.

    With ~114 h left, from 6% under the peak, after the market fell > 5% in 9 days: P(> +2.5%) 39% with bear
    mode, 25% without, 13% at half size; P(< -3%) 25% / 12% / 1%. Bear mode buys a chance at a big rebound
    with a doubled bad tail and a lower mean. Switched off (the team asked for a steadier book). Also: the
    breaker now trips only after two hourly readings past its level (`confirm_cycles`), since a lone reading
    of $74,840 on 8 Oct (closed-short collateral briefly missing from the exchange balance) was enough to
    trip it at 0.20.

    Outside sources checked (none adopted): open-source crypto momentum / reversal backtests (e.g.
    github.com/prams2104/crypto-momentum-backtest: 1-day reversal alpha t 4.6 out of sample, destroyed by
    138% daily turnover; github.com/Owusuk/crypto-stat-arb: market-neutral alpha t 1.64 -> 0.52 out of
    sample) agree with rounds 5-6 and 16; "Cryptocurrency momentum has (not) its moments" (Financial
    Markets and Portfolio Management, 2025) agrees with round 5 that volatility scaling does not rescue
    crypto momentum the way it does equity momentum.

25. **Net hedge: short the 10 most liquid coins against half the book's net long (adopted 13 Oct,
    `research/steady_book.py`, `research/hedge_check.py`).** The live book swings with the altcoin market
    because it is ~100% invested and ~47% net long on average (60% in the contest). Steadier versions, same
    costs and trade rule, continuous run from 2024 and all 944 windows:

    | Book | Sharpe 24-25 / 2026 | Return / yr | Day vol | Max DD | Score (median) | Contest replay | P(+2.5% in 114 h) |
    |---|---|---|---|---|---|---|---|
    | live 4x | 1.95 / 2.65 | +95% | 1.6% | 24.6% | 0.090 | -6.3% | 25% |
    | 2x (risk dial) | 2.10 / 2.66 | +42% | 0.8% | 11.7% | 0.095 | -3.2% | 13% |
    | hedge 25% of net, all liquid coins | 2.16 / 2.77 | +84% | 1.4% | 17.3% | 0.097 | -5.5% | 23% |
    | **hedge 50% of net, 10 most liquid** | **2.03 / 2.96** | +77% | 1.3% | **16.1%** | 0.092 | **-4.0%** | 23% |
    | hedge 50% of net, all liquid coins | 2.03 / 2.82 | +74% | 1.3% | 16.6% | 0.096 | -4.5% | 23% |
    | hedge 75% / 100% of net | worse | | | | | | |
    | hedge with BTC + ETH | worse in both periods (this crash was in altcoins, not BTC) | | | | | | |
    | momentum sleeve alone | 2.04 / 1.28 | +84% | 1.8% | 30.4% | 0.080 | -4.4% | 25% |

    Paired bootstrap vs live (the round-18 test): 10 most liquid at 50% dSharpe +0.14, P 67% (2024-25 +0.08,
    2026 +0.31); all coins 25% +0.18, P 83%; 50% +0.10, P 62%; 75% -0.09. Better in both periods but far
    from the 97% of rounds 18's adoptions: the Sharpe gain is not proven. What it does reliably is cut the
    worst drawdown by a third and the daily swing by a fifth for a fifth of the return, with the same chance
    of catching up in the contest's last days. The 10-coin basket was chosen over the all-coin one for
    execution (about 14 short positions instead of 30, in the tightest spreads). Implemented as the
    `net_hedge` component (`strategy/book.py`), applied last on the whole book and then gross-capped at 1;
    it reproduces the research run exactly (2.029 / 2.959, contest -4.02%).

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
