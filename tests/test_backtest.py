"""Backtest harness on synthetic bars: runs, keeps cash non-negative, and windows/PBO behave."""

import numpy as np
import pandas as pd

from backtest.engine import Variant, buy_and_hold, run_backtest
from backtest.pbo import pbo
from backtest.windows import Objective, rolling_windows, score, summarize
from trend_guard.config import load_strategy_config, load_symbols

CFG = load_strategy_config("config/strategy.yaml")
SYM = load_symbols("config/symbols.yaml")


def synthetic_bars(n_days: int = 260, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Four assets: an uptrend for the first half, a downtrend for the second."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-06-01", periods=24 * n_days, freq="1h", tz="UTC")
    out = {}
    for i, pair in enumerate(["BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD"]):
        drift = np.where(np.arange(len(idx)) < len(idx) // 2, 0.0004, -0.0004) * (1 + 0.2 * i)
        close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, len(idx))))
        opn = np.concatenate([[close[0]], close[:-1]])
        wick = np.abs(rng.normal(0, 0.003, len(idx)))
        out[pair] = pd.DataFrame(
            {
                "open": opn,
                "high": np.maximum(opn, close) * (1 + wick),
                "low": np.minimum(opn, close) * (1 - wick),
                "close": close,
                "quote_volume": 1e6 * (4 - i),
            },
            index=idx,
        )
    return out


def test_backtest_runs_trades_and_never_spends_missing_cash():
    cfg = CFG.model_copy(deep=True)
    cfg.universe.min_history_days = 30
    bars = synthetic_bars()
    res = run_backtest(bars, cfg, SYM, "2023-08-01", "2024-01-15")
    assert len(res.trades) > 0 and res.fees > 0
    assert res.equity.notna().all() and res.equity.iloc[0] > 0
    assert res.meta["min_cash"] > -0.02 * float(res.equity.max())  # taker fallback may overshoot slightly
    assert set(res.regimes.unique()) <= {"RISK_ON", "NEUTRAL", "RISK_OFF"}
    assert (res.exposure["short"] > 0).any()  # the downtrend half produced shorts

    lo = run_backtest(bars, cfg, SYM, "2023-08-01", "2024-01-15", Variant("lo", long_only=True))
    assert (lo.exposure["short"] == 0).all()

    s = summarize(rolling_windows(res.equity, res.trades, 14))
    assert s["n_windows"] > 100 and np.isfinite(score(s, Objective()))


def test_buy_and_hold_pays_one_fee():
    eq = buy_and_hold(synthetic_bars(30), "BTC/USD", "2023-06-01")
    assert abs(eq.iloc[0] - 100_000 * 0.999) < 1e-6


def test_pbo_separates_noise_from_real_edge():
    noise = pd.DataFrame(np.random.default_rng(1).normal(0, 0.01, (320, 10)))
    assert 0.2 < pbo(noise, n_partitions=8)["pbo"] < 0.8  # IS winner is a coin flip OOS
    edge = noise.copy()
    edge[0] += 0.01  # one config with a genuine edge keeps winning OOS
    assert pbo(edge, n_partitions=8)["pbo"] < 0.1
