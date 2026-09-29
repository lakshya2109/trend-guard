"""Config loading tests: the committed config/*.yaml files must parse into the typed models."""

from trend_guard.config import (
    config_hash,
    load_env,
    load_events,
    load_strategy_config,
    load_symbols,
)


def test_load_strategy_config_from_repo_default():
    cfg = load_strategy_config("config/strategy.yaml")
    assert cfg.signal.lookbacks_h == [24, 48, 96, 168, 336, 504]
    assert cfg.regime.budgets["RISK_ON"].long == 1.0
    assert cfg.portfolio.caps.btc_eth == 0.25
    assert cfg.risk.dd_max == 0.05


def test_config_hash_is_stable():
    h1 = config_hash("config/strategy.yaml")
    h2 = config_hash("config/strategy.yaml")
    assert h1 == h2
    assert len(h1) == 12


def test_load_events_from_repo_default():
    events = load_events("config/events.yaml")
    assert len(events) == 1
    assert events[0].name.startswith("CPI")


def test_load_events_missing_file_returns_empty():
    assert load_events("config/does_not_exist.yaml") == []


def test_load_symbols_from_repo_default():
    symbols = load_symbols("config/symbols.yaml")
    assert symbols.bstocks["NVDA/USD"] == "NVDA"
    assert "USDT/USD" in symbols.stablecoins


def test_load_env_defaults_paper_mode(monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE_ORDERS", raising=False)
    monkeypatch.delenv("BOT_MODE", raising=False)
    env = load_env(dotenv_path="does_not_exist.env")
    assert env.allow_live_orders is False
    assert env.bot_mode == "paper"
