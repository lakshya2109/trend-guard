# trend-guard

Autonomous spot and 1x-short trading bot for the Roostoo HK vs AU vs IN Quant Trading Hackathon (4–18 Oct 2026). It runs unattended on EC2 and trades the Roostoo mock exchange only through `python -m trend_guard.main --mode live`. `main` is the deployed `trend-guard-final-v2` history. The last change to strategy parameters is `c99b894`. Every number the bot uses is in [`config/strategy.yaml`](config/strategy.yaml). The research log is [`docs/STRATEGY.md`](docs/STRATEGY.md). How to match fills to those commits is [`docs/TRADE_LOG_NOTES.md`](docs/TRADE_LOG_NOTES.md).

## 1. How the portfolio is curated

One account, $100,000. Target weights are summed per pair and then capped so long notional plus short collateral never exceeds equity. There is no leverage.

In the normal regime the account is three books:

| Book | Share of account | What it holds | Code |
|---|---|---|---|
| trend-guard v2, trend sleeve | 49% | Long-only liquid crypto, including PAXG, that is in a 1–8 week uptrend. Ensemble of four parameter sets. Scaled to 0.4%/day volatility. Rebalanced daily at 12:00 UTC. | `strategy/book.py` |
| trend-guard v2, momentum sleeve | 21% | Long the 6 strongest and short the 6 weakest liquid coins on 3–30 day risk-adjusted momentum. Ensemble of four parameter sets. Resized hourly to 1.5%/day volatility. Weights inside each side are proportional to 1/sqrt(volatility). | `strategy/book.py` |
| Teammate trend book | 21% | Long-only trend on 7/14/30-day horizons, 0.4%/day vol target, volatility trailing stops. | `strategy/team/` (Gunek-945, vendored unmodified) |
| Teammate momentum book | 9% | Long/short cross-sectional momentum, 3/7/14-day lookbacks, 1.5%/day vol target. | `strategy/team/` (swyam-rd, vendored unmodified) |

The two teammate books are themselves a 70/30 trend/momentum split, and that pair is 30% of the account. v2 is the other 70%, split 70/30 between its own trend and momentum sleeves. The whole blend is then multiplied by `risk_multiplier: 4.0` and gross-capped at 1, so on a typical hour almost all equity is invested.

On top of that blend the live file also runs:

| Rule | What it does |
|---|---|
| Jump guard | No new or larger long for 24 hours after a coin rises more than 10% in one hour. |
| Momentum crash filter | No new short on a coin already down more than 40% over 60 days. |
| Net hedge | Short the 10 most liquid coins, equal weight, sized to half the book's net long, then gross-capped again. |
| Catch-up regime | While the average liquid coin is down more than 5% over 9 days, replace the whole book with the 10 coins that fell most over 3 days, equal weight, fully invested. Leave when that 9-day return is back to 0% or higher. While this is on, the net hedge is flat. |
| Profit lock | After two minute equity readings at or above $98,500, cancel resting orders, flatten, and stay in cash. |
| Breaker | Halve every target only after two hourly readings 20% below the peak. Release once the drawdown is back inside 5%. |

## 2. Why this strategy

After a top-20-by-return screen, teams are ranked on `0.4 Sortino + 0.3 Sharpe + 0.3 Calmar` over about 14 daily returns. Three facts drove the original book:

1. The ratios barely depend on position size, so size sets the return, not the score. Return is still the first screen, which is why the live book is run at 4x rather than at the low-volatility research setting.
2. One bad day dominates a two-week window, so the account averages several books that do not fail on the same names.
3. The taker fee is 0.10% per fill. A book that turns over a few times a fortnight can give the whole median return back to fees, so positions are resized only when the change is large enough to pay for itself, and each sleeve is an ensemble rather than a single fragile parameter set.

The normal book is that design. The catch-up regime is not. On 15 Oct the objective was changed from the long-run score to the chance of crossing the Hong Kong top-20 cutoff, with about four days left. Round 28 of [`docs/STRATEGY.md`](docs/STRATEGY.md) records the cost: a much fatter loss tail. The profit lock exists so a move through the cutoff is kept rather than given back. That terminal mode should be read as a qualification bet, not as a claim of a higher long-run Sharpe.

## 3. Implementation

Every hour at HH:03 UTC, after the Binance candle has closed, the engine:

1. Tops up the hourly Binance bar cache.
2. Calls `book_weights` in `src/trend_guard/strategy/book.py` for target weights.
3. Multiplies by the breaker and by `competition.risk_scale`.
4. Turns targets into orders in `execution/order_manager.plan_orders`.
5. Sends closes first, then limit orders at the touch. Limits are repriced every 60 seconds and sent as market orders after four tries.
6. Opens and closes shorts at market through Roostoo `/v6`.

`plan_orders` acts on the 00:00 and 12:00 UTC decision bars, or when a target has changed. A resize smaller than the larger of 1% of equity and 25% of the target is skipped, unless it opens or closes a position. Sign flips close at market and reopen on the other side in the same cycle. Between decisions, quantities are held.

The only process the systemd unit starts is `python -m trend_guard.main --mode live --config config/strategy.yaml` (`deploy/trend-guard.service`). `scripts/probe_live.py` and `overnight/flatten_account.py` can place orders, but a person has to run them, and neither is part of the service. No `docs/PROBES.md` was produced.

If no fill has landed since an activity window opened (01:00–03:59 and 13:00–15:59 UTC), the engine sends one small market order toward the book's own largest gap, at least 0.1% of equity. That is how the bot stays active on days the trade rule would otherwise hold. Until `4218eaf` those windows opened at 00:00 and 12:00 UTC, before the decision trades, so the rule fired early on 5–8 Oct.

## 4. How the code reflects the idea

The strategy is a pure function of the price panel. `book_weights` has no network, disk, or clock reads. The backtest and the live engine call that same function, then the same `plan_orders`. A parameter change is a commit that edits `config/strategy.yaml`; the engine does not have a second copy of the thresholds.

Sleeve shares are asserted to sum to at most 1, and gross is capped again after the 4x multiplier and after the net hedge. The catch-up path in `book_components` replaces every sleeve with the rebound book while the 9-day regime is on, which is why the hedge weight is zero in that regime. The profit lock is not a target weight. Once two minute snapshots clear $98,500, `engine._flatten_for_profit_lock` cancels resting orders and submits a flatten, then later cycles return before any new risk is taken.

## 5. Backtesting

The design was chosen on every 14-day window starting daily at 12:00 UTC from February 2024 to September 2026, each window starting flat in $100,000. The contest replay charges 0.10% fee plus 5 bps of slippage on every trade and does not assume a maker fill. Reproduce with `python scripts/download_history.py && python backtest/run_backtest.py`. A single fortnight with a full trade list is `python scripts/simulate.py --start ... --end ...`, which also checks that deleting the second half of the prices does not change the first half of the targets.

The tables below are the 4x blend before the 13–15 Oct contest changes (net hedge, then the terminal catch-up and profit lock). They are the evidence for the normal book. They are not a backtest of the catch-up rule.

| 2026 (258 windows) | Score | Median fortnight | P(fortnight > +5%) | Above -1% | Positive | Worst |
|---|---|---|---|---|---|---|
| trend-guard (blend, 4x risk) | 0.208 | +2.85% | 36% | 75% | 67% | -8.0% |
| trend-guard (blend, 2.5x risk) | 0.214 | +2.63% | 36% | 77% | 69% | -5.4% |
| trend-guard (blend, 2x risk) | 0.204 | +2.21% | 28% | 78% | 69% | -4.4% |
| trend-guard (blend, 1x risk, before the crash filter) | 0.178 | +0.96% | 10% | 85% | 69% | -2.3% |
| Team combined 70/30 alone | 0.131 | +1.04% | 10% | 84.5% | 69% | -2.3% |
| BTC buy and hold | -0.007 | -0.35% | | 55% | 48% | -18.3% |

| All 944 windows | Score | Median fortnight | P(fortnight > +5%) | P(fortnight < -5%) | Above -1% | Worst |
|---|---|---|---|---|---|---|
| trend-guard (blend, 4x risk) | 0.076 | +1.28% | 30% | 9.6% | 67% | -12.2% |
| trend-guard (blend, 2.5x risk) | 0.083 | +1.12% | 26% | 6.0% | 68% | -9.8% |
| trend-guard (blend, 2x risk) | 0.095 | +1.13% | 21% | 3.2% | 69% | -8.1% |
| Team combined 70/30 alone | 0.060 | +0.44% | | | 76% | -5.1% |

Round 25 gives the net hedge on the continuous run from 2024: Sharpe 2.03 / 2.96 (2024–25 / 2026), about 1.3% daily volatility, max drawdown 16.1%, against 1.95 / 2.65, 1.6% daily volatility, and 24.6% for the 4x blend without the hedge. The paired test put the chance the Sharpe gain is real at 67%, so the hedge was adopted for the smaller drawdown, not because the score was proven higher. Round 28 gives the catch-up: on 1,408 comparable historical starts it cleared a static +2.55% hurdle about 58% of the time, against about 20% for the hedged book, and the chance of losing more than 3% over the next 93 hours rose from about 9% to about 25%.

## 6. Maker and taker fees

The contest charges 0.10% on a taker (market) order and 0.05% on a maker (limit) order. The live bot tries to pay the maker rate. Spot orders are limits at the touch, repriced once a minute, and only become market orders after four minutes. Shorts have no limit endpoint on `/v6`, so short opens and closes pay the taker rate. The fee stored on each fill is the commission Roostoo returns (`CommissionChargeValue`, converted to USD when it is charged in the coin), together with the maker or taker role on that response. The bot does not assume a fee rate after the fact.

The backtest is harsher than that live policy. `backtest` in `config/strategy.yaml` charges 0.10% plus 5 bps of slippage on every trade, which is 0.15% per side, and does not credit a maker fill. A smaller trade is skipped when it cannot cover a round trip (`min_trade_weight: 0.01`, `min_trade_frac: 0.25`). Research round 17 found that waiting longer than the four-minute limit window saved less than 1 bp on real 1-minute bars, so the live escalation is left as it is.

## 7. Risk management

- Before the 4x dial, the trend sleeve targets 0.4%/day and the momentum sleeve 1.5%/day. With the dial and the net hedge, round 25 shows about 1.3% daily volatility.
- A single trend name is capped at 50% of its sleeve. Momentum gross is capped at 0.9 of its capital. Account gross is capped at 1 after the dial and again after the hedge.
- Names need $2M/day and 14 days of history to enter the trend sleeve, and $5M/day and 30 days to enter the momentum sleeve. Stablecoins, tokenized stocks, and pairs with no Binance feed are never traded.
- The breaker halves targets only after two hourly readings 20% below the peak. The level was 10% until 9 Oct, and one reading was enough until 13 Oct. A single bad equity print on 8 Oct was why the second confirmation was added. Tighter brakes were tested and lowered the score by selling dips.
- The profit lock flattens the account after two minute readings at or above $98,500 and then holds cash.
- A `HALT` file at `/etc/trend-guard/HALT` cancels resting orders and leaves the process idle.
- Order posts that time out are not retried. The client raises `OrderStateUnknownError` and the engine reconciles through `query_order`.

## 8. Rule compliance

| Competition rule | Where it is enforced |
|---|---|
| No leverage | Sleeve shares sum to at most 1. `plan_orders` spends only free cash plus immediate proceeds, minus a cash buffer. Gross is capped at 1. |
| Spot and 1x shorts only | Longs via `/v3/place_order`. Shorts only via `/v6/short_open`, sized by collateral. |
| No HFT, market making, or arbitrage | Decisions on closed hourly bars. One directional order per pair per cycle. No quotes on both sides. |
| No manual trading in the live service | Every order from the service comes from `engine.Engine`. |
| No accidental live orders | `main.py` refuses live mode unless `ALLOW_LIVE_ORDERS=1` and keys exist. |
| At least 8 active days | Activity rule in section 3. |
| Trade-log integrity | Each fill record stores the commit hash. Each hourly decision is stored in SQLite and `logs/decisions`. `docs/TRADE_LOG_NOTES.md` explains the paper-mode rows on 4 Oct and the later strategy commits. Those log files are on the contest machine; they are gitignored so keys and account history are not in the repository. |

## 9. How to run

```bash
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
cp .env.example .env                                        # keys are only needed for live mode

pytest -q && ruff check .
python scripts/download_history.py                          # hourly bars for every Roostoo crypto pair
python backtest/run_backtest.py                             # contest replay -> reports/<ts>/summary.md
python scripts/simulate.py --start "2026-09-01 12:00" --end "2026-09-15 12:00"
python -m trend_guard.main --mode paper --once              # one cycle, simulated fills
```

Deployment, the HALT switch, and log viewing are in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## 10. Limitations

- Over 14 days the market path dominates. In 2026 windows where BTC fell more than 5%, the book was positive less than half the time.
- The backtest fills at the hourly close plus 5 bps. Live limit fills can be better than that, and fast markets can be worse.
- The universe is today's Roostoo listings, so coins delisted before 2026 are missing.
- The catch-up regime raises the chance of a large gain and the chance of a large loss. It is not the book described by the tables in section 5.
- 2026 contains only about 18 independent fortnights, so small score gaps are weak evidence.

## Credits

The trend book in `src/trend_guard/strategy/team/trend_signal.py` and `trend_portfolio.py` was written by Gunek-945 ([roostoo-trend-book](https://github.com/Gunek-945/roostoo-trend-book)). The momentum book in `src/trend_guard/strategy/team/momentum_model.py` was written by swyam-rd. Both are included unmodified, with their authors' attribution headers.
