"""Config loading: the committed config/*.yaml files must parse into the typed models."""

from trend_guard.config import config_hash, load_env, load_strategy_config, load_symbols


def test_strategy_config_matches_research_design():
    cfg = load_strategy_config("config/strategy.yaml")
    b = cfg.book
    assert b.trend_share + b.momentum_share <= 1.0  # no leverage by construction
    assert b.trend.lookback_sets == [[168, 336, 720], [168, 336, 720, 1440]]
    assert b.momentum.lookback_sets == [[72, 168, 336], [168, 336, 720]]
    assert b.momentum.skips == [0, 12]
    assert "PAXG/USD" in b.momentum.exclude
    assert cfg.execution.decision_hours_utc == [0, 12]
    assert cfg.backtest.fee + cfg.backtest.slippage == 0.0015


def test_config_hash_is_stable():
    assert config_hash("config/strategy.yaml") == config_hash("config/strategy.yaml")
    assert len(config_hash("config/strategy.yaml")) == 12


def test_symbols_exclude_stablecoins_and_delisted_feeds():
    s = load_symbols("config/symbols.yaml")
    assert "USDT/USD" in s.excluded and "TON/USD" in s.excluded


def test_load_env_defaults_to_paper_and_no_live_orders(monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE_ORDERS", raising=False)
    monkeypatch.delenv("BOT_MODE", raising=False)
    env = load_env(dotenv_path="does_not_exist.env")
    assert env.allow_live_orders is False
    assert env.bot_mode == "paper"


def test_universe_keeps_tradable_crypto_only():
    from trend_guard.universe import binance_symbol, tradable_pairs

    info = {
        "TradePairs": {
            "BTC/USD": {"CanTrade": True, "AssetType": "crypto"},
            "NVDAB/USD": {"CanTrade": True, "AssetType": "stock"},
            "USDT/USD": {"CanTrade": True, "AssetType": "crypto"},
            "OLD/USD": {"CanTrade": False, "AssetType": "crypto"},
            "PAXG/USD": {"CanTrade": True},
        }
    }
    s = load_symbols("config/symbols.yaml")
    assert tradable_pairs(info, s) == ["BTC/USD", "PAXG/USD"]
    assert binance_symbol("BTC/USD", s) == "BTCUSDT"
