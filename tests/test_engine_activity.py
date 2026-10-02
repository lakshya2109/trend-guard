"""Engine activity rule end to end on the paper broker: no fill in a window -> one trade toward target."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path

from trend_guard.clock import SimClock
from trend_guard.config import EnvConfig, StrategyConfig, SymbolsConfig, load_strategy_config
from trend_guard.engine import Engine
from trend_guard.execution.sim_broker import SimBroker
from trend_guard.roostoo.models import Ticker
from trend_guard.state.store import Store

ROOT = Path(__file__).resolve().parents[1]
T = datetime(2026, 10, 5, 12, 15, tzinfo=UTC)


def engine(tmp_path: Path) -> tuple[Engine, SimBroker, SimClock]:
    cfg: StrategyConfig = load_strategy_config(ROOT / "config/strategy.yaml")
    clock = SimClock(T)
    broker = SimBroker(D("100000"), cfg.backtest, clock, mode="paper")  # paper: market orders fill now
    broker.mark_to_market({"BTC/USD": Ticker(MaxBid=49999, MinAsk=50001, LastPrice=50000)})
    eng = Engine(
        cfg,
        SymbolsConfig(),
        EnvConfig(),
        None,
        broker,
        Store(":memory:", clock),
        clock,  # type: ignore[arg-type]
        tmp_path / "HALT",
        fills_dir=tmp_path / "fills",
    )
    return eng, broker, clock


def test_no_targets_yet_means_no_activity_trade(tmp_path):
    eng, broker, _ = engine(tmp_path)
    assert eng.activity_check() is None  # never reads a missing target as "sell everything"


def test_activity_trade_fills_and_marks_the_window_active(tmp_path):
    eng, broker, clock = engine(tmp_path)
    eng.store.kv_set("prev_targets", '{"BTC/USD": 0.05}')
    order = eng.activity_check()
    assert order is not None and order.pair == "BTC/USD" and order.action == "BUY"
    assert broker.longs["BTC/USD"] > 0  # filled at market on the paper broker
    assert eng.store.kv_get("last_fill_ts") is not None
    clock.advance(timedelta(hours=1))
    assert eng.activity_check() is None  # 13:15: the window already has a fill
    clock.advance(timedelta(hours=12))
    assert eng.activity_check() is not None  # 01:15 next day: new window, no fill yet
