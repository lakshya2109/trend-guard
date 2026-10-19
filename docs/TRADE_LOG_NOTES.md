# Trade log notes

Facts a reviewer needs to read `logs/fills/*.csv` against the exchange's own history.

## 2026-10-04 13:19:59 UTC: ten simulated rows (paper mode), not exchange trades

`logs/fills/2026-10-04.csv` starts with ten `SHORT_OPEN` rows stamped 13:19:59 (ARB, ASTER, CAKE, POL,
TRUMP, TRX, WLFI, XPL, ZEC, ZEN). They were written by a paper-mode start (`python -m trend_guard.main`
without `--mode live`; paper is the default), which at the time shared the live fill log. No order reached
Roostoo. Evidence:

- the live bot opened the same ten shorts five minutes later (13:24:50 to 13:25:54) at almost identical sizes;
- Roostoo's `/v6/short_positions` on 4 Oct showed only the second set (for example WLFI 90,673.1,
  ZEC 2.859, CAKE 1,096.81), never the sum of both;
- the 13:19:59 quantities are unrounded (`11879.0125...`), as the simulated broker computes them; the
  exchange returns quantities at the pair's AmountPrecision (`11855.2`).

The rows are left in place (the log is not edited by hand). Since commit `4218eaf`, paper mode writes to
`logs/paper/fills/` and can no longer mix with the live log.

## Activity-rule fills (`reason` starts with `activity:`)

One small market order toward the book's own target when no fill landed since an activity window opened
(the contest needs at least 8 active days). Until `4218eaf` the windows opened at 00:00 / 12:00 UTC, before
the strategy's own decision trades at 01:03 / 13:03, so the rule fired once per window on 5-8 Oct (8 fills,
$12 in fees). From `4218eaf` the windows are 01:00-03:59 / 13:00-15:59 UTC.

## Strategy changes during the contest

Every change is a `strat:` commit with its reason; the bot logs its commit on every fill (`commit` column) and
at each startup. Match a fill's `commit` to this timeline (times are the commit timestamps, HKT):

| When | Commit | What the bot did |
|---|---|---|
| about 21:00 UTC on 8 Oct, through 12 Oct | rounds 19-20 | bear mode on: rebound book while the 5-day market return was below -8% |
| 14 Oct 02:32 | `a1aba05` | bear mode off. Breaker trips only after two hourly readings past 20% drawdown (round 24) |
| 14 Oct 02:55 | `8c1b191` | net hedge on: short the 10 most liquid coins against half the net long (round 25) |
| 15 Oct 00:55 | `6aed9ff` | terminal catch-up: bear mode on again, now a 9-day market return below -5%, rebound book until that return is back to 0%, and a profit lock at $98,000 (round 28) |
| 15 Oct 01:47 | `c99b894` | profit lock raised to $98,500, still after two minute readings (round 29). This is the submitted commit |

Fills after `6aed9ff` can be the rebound book even though round 24 had switched bear mode off. The
decision log names the regime (`bear_mode` in `logs/decisions/*.jsonl`). A `profit_lock` decision means
the bot flattened and then held cash.
