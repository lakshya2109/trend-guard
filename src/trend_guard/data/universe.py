"""Eligible-universe computation (BUILD_SPEC.md Section 5.6).

PURE function: no network, no disk, no clock reads. All inputs -- exchangeInfo, tickers, history
lengths, spread stats -- are passed in by the caller (the engine's daily universe-refresh step),
exactly like strategy/risk code elsewhere in this repo.

Classification note: `crypto_beta_stocks` (COIN, MSTR, CRCL) are real equities that only trade
during US market hours, so for *universe membership* they are grouped with the bStocks sleeve
and require `bstocks_enabled`, even though BUILD_SPEC Section 6.3 treats them as crypto for the
*regime gate* (a separate, later concern in strategy/regime.py).
"""

from __future__ import annotations

from trend_guard.config import SymbolsConfig, UniverseConfig
from trend_guard.roostoo.models import Ticker

SPREAD_FAIL_LIMIT_H = 12  # consecutive hours failing the spread filter before a hysteresis drop
RANK_SLACK = 5  # an already-eligible asset survives until it falls this far outside max_crypto


def compute_eligible_universe(
    exchange_info: dict,
    tickers: dict[str, Ticker],
    history_days: dict[str, int],
    median_spread_6h: dict[str, float],
    symbols_cfg: SymbolsConfig,
    universe_cfg: UniverseConfig,
    previously_eligible: set[str],
    spread_fail_streak: dict[str, int],
) -> tuple[set[str], dict[str, int]]:
    """Return (new_eligible_pairs, new_spread_fail_streak).

    `history_days`, `median_spread_6h` and `tickers` are keyed by Roostoo pair (e.g. "BTC/USD").
    `spread_fail_streak` carries forward each candidate's consecutive-failure count; entries for
    pairs no longer in contention (outside max_crypto + RANK_SLACK) are dropped from the result.
    """
    trade_pairs = exchange_info.get("TradePairs", {}) or {}
    tradeable = {
        pair for pair, info in trade_pairs.items() if info.get("CanTrade", False) and pair in tickers
    }
    tradeable -= set(symbols_cfg.stablecoins)

    bstock_pairs = set(symbols_cfg.bstocks) | set(symbols_cfg.crypto_beta_stocks)
    paxg_pairs = {p for p, cls in symbols_cfg.asset_class.items() if cls == "gold"}

    eligible: set[str] = set()
    eligible |= paxg_pairs & tradeable  # PAXG: always eligible if listed and tradeable
    if universe_cfg.bstocks_enabled:
        eligible |= bstock_pairs & tradeable

    crypto_pairs = tradeable - bstock_pairs - paxg_pairs
    ranked = sorted(crypto_pairs, key=lambda p: tickers[p].unit_trade_value, reverse=True)
    rank_of = {pair: i + 1 for i, pair in enumerate(ranked)}  # 1-based rank by 24h UnitTradeValue

    new_streak: dict[str, int] = {}
    for pair in ranked:
        rank = rank_of[pair]
        if rank > universe_cfg.max_crypto + RANK_SLACK:
            continue  # falls outside the hysteresis band entirely: not a candidate, drop tracking

        spread = median_spread_6h.get(pair, float("inf"))
        spread_ok = spread <= universe_cfg.max_spread_pct
        streak = 0 if spread_ok else spread_fail_streak.get(pair, 0) + 1
        new_streak[pair] = streak

        was_eligible = pair in previously_eligible
        if was_eligible:
            # hysteresis retention: rank-band already enforced above; only a 12h spread failure
            # streak can drop it now.
            if streak < SPREAD_FAIL_LIMIT_H:
                eligible.add(pair)
        else:
            history_ok = history_days.get(pair, 0) >= universe_cfg.min_history_days
            if rank <= universe_cfg.max_crypto and spread_ok and history_ok:
                eligible.add(pair)

    return eligible, new_streak
