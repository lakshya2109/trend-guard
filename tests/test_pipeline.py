"""pipeline.decide and the shared helpers both engines call."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from trend_guard.config import EventItem, load_strategy_config, load_symbols
from trend_guard.strategy.pipeline import (
    OverlayInputs,
    StopState,
    asset_signal,
    bucket_for,
    classify,
    decide,
    ewma_cov_annualised,
    ratchet,
    stop_level,
    warmup_bars,
)

CFG = load_strategy_config("config/strategy.yaml")
SYM = load_symbols("config/symbols.yaml")
NOW = datetime(2026, 10, 6, 12, tzinfo=UTC)
PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "PAXG/USD"]


def _inputs(**kw) -> OverlayInputs:
    base = dict(
        now=NOW,
        equity=100_000.0,
        peak_equity=100_000.0,
        pnl_today_pct=0.0,
        competition_day=3,
        cumulative_return=0.01,
    )
    base.update(kw)
    return OverlayInputs(**base)


def _decide(signals, current=None, **kw):
    cls = {p: classify(p, SYM) for p in PAIRS}
    buckets = {p: bucket_for(p, cls[p], CFG.universe) for p in PAIRS}
    sig_ann = {p: 0.6 for p in PAIRS}
    cov = pd.DataFrame(np.eye(len(PAIRS)) * 0.36, index=PAIRS, columns=PAIRS)
    mkt, prev, shorts = kw.pop("mkt", 0.5), kw.pop("prev", "RISK_ON"), kw.pop("shorts", True)
    return decide(signals, mkt, prev, sig_ann, cov, cls, buckets, current or {}, shorts, _inputs(**kw), CFG)


def test_classification_and_buckets():
    assert classify("PAXG/USD", SYM) == "paxg" and bucket_for("PAXG/USD", "paxg", CFG.universe) == "paxg"
    assert bucket_for("BTC/USD", "crypto", CFG.universe) == "btc_eth"
    assert bucket_for("SOL/USD", "crypto", CFG.universe) == "major"
    assert classify("COIN/USD", SYM) == "crypto_beta"


def test_decide_respects_caps_and_gross():
    d = _decide({"BTC/USD": 1.0, "ETH/USD": 1.0, "SOL/USD": 1.0, "PAXG/USD": 1.0})
    assert d.regime == "RISK_ON"
    assert abs(d.weights_final["BTC/USD"]) <= CFG.portfolio.caps.btc_eth + 1e-12
    assert abs(d.weights_final["SOL/USD"]) <= CFG.portfolio.caps.major + 1e-12
    assert sum(abs(w) for w in d.weights_final.values()) <= CFG.portfolio.gross_max + 1e-12


def test_shorts_disabled_means_no_negative_weights():
    d = _decide({"BTC/USD": -0.9, "ETH/USD": -0.9, "SOL/USD": 0.5, "PAXG/USD": 0.0}, shorts=False)
    assert all(w >= 0 for w in d.weights_final.values())


def test_hard_daily_brake_allows_reductions_only():
    current = {"BTC/USD": 0.05}
    sig = {"BTC/USD": 1.0, "ETH/USD": 1.0, "SOL/USD": 0.0, "PAXG/USD": 0.0}
    d = _decide(sig, current, pnl_today_pct=-0.02)
    assert d.no_new_risk
    assert d.weights_final["BTC/USD"] <= 0.05 + 1e-12  # may hold, may not grow
    assert d.weights_final["ETH/USD"] == 0.0  # may not open


def test_drawdown_and_event_overlays_scale_weights_down():
    sig = {"BTC/USD": 1.0, "ETH/USD": 0.0, "SOL/USD": 0.0, "PAXG/USD": 0.0}
    base = _decide(sig).weights_final["BTC/USD"]
    dd = _decide(sig, equity=97_500.0).weights_final["BTC/USD"]  # 2.5% DD -> 0.5x
    ev = _decide(sig, events=[EventItem(name="CPI", utc="2026-10-06T12:30:00Z")]).weights_final["BTC/USD"]
    assert dd < base and ev < base


def test_cooldown_blocks_reentry_same_side():
    sig = {"BTC/USD": 1.0, "ETH/USD": 0.0, "SOL/USD": 0.0, "PAXG/USD": 0.0}
    assert _decide(sig, cooldown_long={"BTC/USD"}).weights_final["BTC/USD"] == 0.0


def test_ratchet_arms_mid_only_once_price_is_above_it_and_never_loosens():
    st = StopState("LONG", best_price=100.0)
    st = ratchet(st, 101, 99, 100, mid_now=105)  # price below mid: reference not armed
    assert st.stop_ref is None
    st = ratchet(st, 110, 104, 108, mid_now=103)
    st = ratchet(st, 109, 101, 102, mid_now=101)  # mid fell: reference must not loosen
    assert st.stop_ref == 103 and st.best_price == 110
    assert stop_level(st, atr_value=2.0, atr_mult=2.5) == max(110 - 5.0, 103)


def test_short_stop_mirrors_long():
    st = ratchet(StopState("SHORT", best_price=100.0), 101, 95, 96, mid_now=99)
    assert st.best_price == 95 and st.stop_ref == 99
    assert stop_level(st, 2.0, 2.5) == min(95 + 5.0, 99)


def test_asset_signal_modes_are_causal():
    idx = pd.date_range("2024-01-01", periods=24 * 120, freq="1h", tz="UTC")
    close = pd.Series(np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, len(idx)))), index=idx)
    for mode in ("hourly", "daily", "blend"):
        cfg = CFG.model_copy(deep=True)
        cfg.signal.mode = mode
        full = asset_signal(close, cfg)
        cut = asset_signal(close.iloc[: 24 * 100], cfg)
        # values computed on truncated history must equal the full-history values (no look-ahead);
        # the last day is excluded because a daily bar is still forming at the cut
        pd.testing.assert_series_equal(full.iloc[: 24 * 99], cut.iloc[: 24 * 99], check_names=False)
        assert warmup_bars(cfg) > 0


def test_ewma_cov_is_symmetric_psd_and_annualised():
    rets = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (500, 3)), columns=list("abc"))
    cov = ewma_cov_annualised(rets, 72)
    assert np.allclose(cov, cov.T) and np.all(np.linalg.eigvalsh(cov) >= -1e-12)
    assert 0.3 < cov.loc["a", "a"] ** 0.5 / (0.01 * (24 * 365) ** 0.5) < 3
