"""Tests for trend_guard.data.universe.compute_eligible_universe: pure function, no I/O."""

from __future__ import annotations

from trend_guard.config import SymbolsConfig, UniverseConfig
from trend_guard.data.universe import compute_eligible_universe
from trend_guard.roostoo.models import Ticker


def _ticker(utv: float) -> Ticker:
    return Ticker(unit_trade_value=utv, max_bid=100, min_ask=100.1, last_price=100)


def _symbols_cfg(**overrides) -> SymbolsConfig:
    base = dict(
        overrides={},
        asset_class={"PAXG/USD": "gold"},
        bstocks={"NVDA/USD": "NVDA"},
        crypto_beta_stocks=["COIN/USD"],
        stablecoins=["USDT/USD"],
    )
    base.update(overrides)
    return SymbolsConfig(**base)


def _universe_cfg(**overrides) -> UniverseConfig:
    base = dict(max_crypto=3, max_spread_pct=0.002, min_history_days=180, majors=[], bstocks_enabled=False)
    base.update(overrides)
    return UniverseConfig(**base)


def _exchange_info(pairs: list[str]) -> dict:
    return {"TradePairs": {p: {"CanTrade": True} for p in pairs}}


def test_top_max_crypto_selected_by_unit_trade_value():
    pairs = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"]
    tickers = {
        "BTC/USD": _ticker(1000),
        "ETH/USD": _ticker(900),
        "SOL/USD": _ticker(800),
        "XRP/USD": _ticker(700),
        "DOGE/USD": _ticker(600),
    }
    history = dict.fromkeys(pairs, 400)
    spread = dict.fromkeys(pairs, 0.001)

    eligible, streak = compute_eligible_universe(
        _exchange_info(pairs), tickers, history, spread, _symbols_cfg(), _universe_cfg(), set(), {}
    )

    assert eligible == {"BTC/USD", "ETH/USD", "SOL/USD"}
    assert streak["BTC/USD"] == 0


def test_stablecoin_always_excluded():
    pairs = ["BTC/USD", "USDT/USD"]
    tickers = {"BTC/USD": _ticker(500), "USDT/USD": _ticker(1_000_000)}
    history = dict.fromkeys(pairs, 400)
    spread = dict.fromkeys(pairs, 0.001)

    eligible, _ = compute_eligible_universe(
        _exchange_info(pairs),
        tickers,
        history,
        spread,
        _symbols_cfg(),
        _universe_cfg(max_crypto=5),
        set(),
        {},
    )

    assert "USDT/USD" not in eligible
    assert "BTC/USD" in eligible


def test_paxg_always_eligible_if_listed():
    pairs = ["BTC/USD", "PAXG/USD"]
    tickers = {"BTC/USD": _ticker(500), "PAXG/USD": _ticker(1)}
    history = dict.fromkeys(pairs, 400)
    spread = dict.fromkeys(pairs, 0.001)

    eligible, _ = compute_eligible_universe(
        _exchange_info(pairs),
        tickers,
        history,
        spread,
        _symbols_cfg(),
        _universe_cfg(bstocks_enabled=False),
        set(),
        {},
    )

    assert "PAXG/USD" in eligible


def test_bstock_requires_bstocks_enabled():
    pairs = ["BTC/USD", "NVDA/USD"]
    tickers = {"BTC/USD": _ticker(500), "NVDA/USD": _ticker(1)}
    history = dict.fromkeys(pairs, 400)
    spread = dict.fromkeys(pairs, 0.001)

    disabled, _ = compute_eligible_universe(
        _exchange_info(pairs),
        tickers,
        history,
        spread,
        _symbols_cfg(),
        _universe_cfg(bstocks_enabled=False),
        set(),
        {},
    )
    enabled, _ = compute_eligible_universe(
        _exchange_info(pairs),
        tickers,
        history,
        spread,
        _symbols_cfg(),
        _universe_cfg(bstocks_enabled=True),
        set(),
        {},
    )

    assert "NVDA/USD" not in disabled
    assert "NVDA/USD" in enabled


def test_hysteresis_keeps_asset_just_outside_max_crypto():
    # max_crypto=3; E/USD ranks 5th (within the +5 slack band) but was already eligible.
    pairs = ["A/USD", "B/USD", "C/USD", "D/USD", "E/USD"]
    tickers = {
        "A/USD": _ticker(100),
        "B/USD": _ticker(90),
        "C/USD": _ticker(80),
        "D/USD": _ticker(70),
        "E/USD": _ticker(60),
    }
    history = dict.fromkeys(pairs, 400)
    spread = dict.fromkeys(pairs, 0.001)  # comfortably within max_spread_pct

    eligible, streak = compute_eligible_universe(
        _exchange_info(pairs),
        tickers,
        history,
        spread,
        _symbols_cfg(),
        _universe_cfg(max_crypto=3),
        previously_eligible={"E/USD"},
        spread_fail_streak={},
    )

    assert "E/USD" in eligible  # rank 5 <= max_crypto(3) + slack(5): retained by hysteresis
    assert streak["E/USD"] == 0


def test_asset_beyond_rank_slack_drops_immediately():
    # F/USD ranks 9th with max_crypto=3 -> 9 > 3 + 5, drops even though it was eligible and spread is fine.
    pairs = [f"P{i}/USD" for i in range(9)]
    tickers = {p: _ticker(100 - i) for i, p in enumerate(pairs)}  # P0 highest value ... P8 lowest
    history = dict.fromkeys(pairs, 400)
    spread = dict.fromkeys(pairs, 0.001)

    eligible, _ = compute_eligible_universe(
        _exchange_info(pairs),
        tickers,
        history,
        spread,
        _symbols_cfg(),
        _universe_cfg(max_crypto=3),
        previously_eligible={"P8/USD"},
        spread_fail_streak={},
    )

    assert "P8/USD" not in eligible


def test_spread_fail_streak_drops_after_12_consecutive_hours():
    pairs = ["A/USD", "B/USD"]
    tickers = {"A/USD": _ticker(100), "B/USD": _ticker(90)}
    history = dict.fromkeys(pairs, 400)
    bad_spread = {"A/USD": 0.001, "B/USD": 0.01}  # B always fails max_spread_pct=0.002
    cfg = _universe_cfg(max_crypto=3, max_spread_pct=0.002)

    eligible = {"A/USD", "B/USD"}
    streak: dict[str, int] = {}
    for hour in range(11):
        eligible, streak = compute_eligible_universe(
            _exchange_info(pairs), tickers, history, bad_spread, _symbols_cfg(), cfg, eligible, streak
        )
        assert "B/USD" in eligible, f"should still be eligible after {hour + 1} failing hours"

    # 12th consecutive failing hour: now drops
    eligible, streak = compute_eligible_universe(
        _exchange_info(pairs), tickers, history, bad_spread, _symbols_cfg(), cfg, eligible, streak
    )
    assert "B/USD" not in eligible
    assert streak["B/USD"] == 12


def test_new_crypto_needs_history_and_spread_ok():
    pairs = ["A/USD", "B/USD"]
    tickers = {"A/USD": _ticker(100), "B/USD": _ticker(90)}
    cfg = _universe_cfg(max_crypto=5, max_spread_pct=0.002, min_history_days=180)

    history = {"A/USD": 400, "B/USD": 10}  # B has too little history
    spread = {"A/USD": 0.001, "B/USD": 0.001}

    eligible, _ = compute_eligible_universe(
        _exchange_info(pairs), tickers, history, spread, _symbols_cfg(), cfg, set(), {}
    )

    assert "A/USD" in eligible
    assert "B/USD" not in eligible


def test_can_trade_false_or_missing_ticker_excludes():
    exchange_info = {"TradePairs": {"A/USD": {"CanTrade": False}, "B/USD": {"CanTrade": True}}}
    tickers = {"A/USD": _ticker(100), "B/USD": _ticker(90)}  # B has no ticker below
    tickers = {"B/USD": _ticker(90)}
    history = {"A/USD": 400, "B/USD": 400}
    spread = {"A/USD": 0.001, "B/USD": 0.001}

    eligible, _ = compute_eligible_universe(
        exchange_info, tickers, history, spread, _symbols_cfg(), _universe_cfg(max_crypto=5), set(), {}
    )

    assert eligible == {"B/USD"}
