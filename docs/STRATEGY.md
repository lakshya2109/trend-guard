# STRATEGY.md

**TREND-GUARD: vol-targeted long/short trend ensemble with a drawdown governor.**

This document is the judge-facing explanation of *why* the bot is built this way, with the
research it leans on. For exact parameter values see `config/strategy.yaml`; for the formulas
see BUILD_SPEC.md Sections 6-7 and the code in `src/trend_guard/strategy/` and
`src/trend_guard/risk/`.

## 1. The core bet

Crypto exhibits time-series momentum at hourly-to-weekly horizons (Liu & Tsyvinski, RFS 2021;
Liu, Tsyvinski & Wu, JF 2022; Moskowitz, Ooi & Pedersen, JFE 2012). A multi-lookback Donchian +
momentum ensemble, volatility-scaled and applied long/short across the most liquid pairs, has
shown net-of-fee Sharpe above 1.5 in a survivorship-bias-free backtest since 2015 (Zarattini,
Pagani & Barbon, "Catching Crypto Trends", SSRN 5209907) -- this is our core template. We
extended it long/short because the `/v6` short endpoints were only added to the Roostoo API in
Sep 2026: the March 2026 round was long-only, and a long-only bot in that round "reacted too
slowly to a sharp market selloff" and could not recover (public post-mortem).

## 2. Why an ensemble, not one indicator

Averaging a Donchian channel-position score and a volatility-normalized momentum score across six
lookbacks (24h to 504h) reduces sensitivity to any single lookback's whipsaw. Kim, Tse & Wald
(2016) show TSMOM's edge is driven largely by volatility scaling, not the raw signal -- so we
never run an unscaled signal; every asset's contribution is divided by its own realized
volatility before sizing.

## 3. Why volatility targeting, twice

Volatility is scaled at two levels: per-asset (raw weight divided by that asset's annualized
EWMA vol) and portfolio-level (all weights jointly scaled to hit a target portfolio volatility
using a shrunk covariance matrix). This follows a large literature: Barroso & Santa-Clara (JFE
2015) show variance scaling nearly doubles momentum's Sharpe and removes crashes; Moreira & Muir
(JF 2017) show vol timing raises Sharpe because expected returns don't rise one-for-one with
volatility; Daniel & Moskowitz (JFE 2016) document that momentum crashes cluster after declines,
in panic states, and during sharp rebounds -- which is exactly when an *unscaled* short sleeve
would get squeezed hardest. The shock filter and squeeze-cover overlay (Section 6 below) are a
direct response to that finding.

## 4. Why a regime gate

`mkt` (an equal-weight index momentum score) and `breadth` (share of eligible coins with a
positive signal) gate how much long vs. short budget is available, with hysteresis so the regime
doesn't flip-flop at the boundary. This keeps the short sleeve small in strong uptrends (where
shorting trend is adversarial) and the long sleeve small in confirmed downtrends, rather than
running a symmetric long/short book at all times.

## 5. Why these risk overlays

Every overlay is multiplicative on the target weight, capped at 1.0, and logged with a reason
string, in this order: drawdown governor -> daily brake -> event -> endgame -> crowding -> news
(optional) -> global risk_scale.

- **Drawdown governor** scales exposure down as equity approaches a max-drawdown budget, the
  Grossman & Zhou (Mathematical Finance 1993) result that optimal risky exposure under a
  drawdown constraint scales with distance from the floor.
- **Daily brake** halves (then zeroes new) risk after a bad UTC day, because the scoring formula
  (Section 6 below) punishes red days asymmetrically -- cutting risk early after a bad day is
  cheap insurance against a second one.
- **Event overlay** halves risk around scheduled US macro releases (BLS CPI, FOMC), since crypto
  shows measurable intraday reversal risk around large scheduled news (Wen, Bouri, Xu & Zhao,
  2022).
- **Endgame lock-in** shrinks the vol target late in the competition if cumulative return is
  already solidly positive -- protecting a lead is worth more than chasing a marginally higher
  one once Screen 2's return gate is comfortably cleared.
- **Crowding filter** uses Binance perp funding rate and open-interest changes as a free,
  almost-nobody-uses-it crash-risk signal: Schmeling, Schrimpf & Todorov ("Crypto Carry",
  Management Science 2026) show high futures carry predicts future crashes, driven by retail
  leverage rather than compensated risk -- so we treat it purely as a de-risking signal, never as
  something to earn carry from (we can't trade perps anyway, and 2510.14435 shows that carry
  profitability has compressed sharply since 2024).
- **Shock filter / squeeze protection** cuts longs on a sharp downside 15-minute BTC move, and
  covers shorts on a sharp upside move that follows a multi-day decline (a short squeeze), which
  is the momentum-crash pattern Daniel & Moskowitz describe.
- **News LLM veto** (off by default, Milestone 7) can only ever *reduce* exposure, never
  initiate or increase a position. This follows directly from the evidence on LLM trading agents:
  FINSABER (KDD 2026) finds reported LLM trading advantages deteriorate under rigorous
  out-of-sample testing; the MAS paper provided in our research brief (Luo et al., arXiv
  2501.00826v3) shows every configuration lost money in the bear regime, and its own ablation
  shows removing the News Agent barely changes return but raises volatility -- i.e. news acts as
  a risk dampener, not an alpha source. We use it exactly for that role and nothing more.

## 6. Why we optimize the scoring function directly, not raw return

Screen 3's composite (`0.4 Sortino + 0.3 Sharpe + 0.3 Calmar`) is dominated by drawdown and by
the size/count of red periods -- a single bad day can erase most of a strategy's score (see the
worked examples in `roostoo_hackathon_research_report.md` Section 2.1). The backtest selection
objective (`backtest/windows.py`, BUILD_SPEC.md Section 11.3) therefore maximizes the 25th
percentile of the rolling-14-day composite, not the mean or the peak, subject to a minimum median
return and a maximum p95 drawdown. This deliberately trades a bit of upside for a much smaller
left tail, which is the right trade for this scoring function.

## 7. Honest caveats

Over any single 14-day window, outcome is dominated by market path -- a strategy with an annual
Sharpe of 1.5 at 30% target vol still has a wide dispersion of possible 14-day outcomes. We
cannot eliminate that; we can only (a) avoid the left tail via the overlays above, (b) be
positioned correctly if a trend appears, and (c) be engineered cleanly enough to pass Screens 1
and 4. Parameter selection uses combinatorially symmetric cross-validation (`backtest/pbo.py`) to
report a probability-of-backtest-overfitting estimate for the chosen config, per Bailey & López
de Prado (JPM 2014 / J. Comp. Finance 2017), since four people iterating on the same 2024-2026
data is exactly the multiple-testing setting their work warns about.

See `roostoo_hackathon_research_report.md` Section 13 for the full annotated bibliography.
