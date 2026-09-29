"""SQLite store round-trips. BUILD_SPEC.md Section 10.1."""

from datetime import UTC, datetime
from decimal import Decimal

from trend_guard.clock import SimClock
from trend_guard.state.store import Store

CLOCK = SimClock(datetime(2026, 10, 5, tzinfo=UTC))


def test_orders_round_trip_decimals_as_strings():
    s = Store(":memory:", CLOCK)
    oid = s.insert_order(
        {
            "pair": "BTC/USD",
            "side": "BUY",
            "kind": "LIMIT",
            "qty": Decimal("0.000123"),
            "price": Decimal("65000.5"),
            "status": "PENDING",
            "extra_field": 7,
        }
    )
    s.update_order(oid, status="FILLED", fee=Decimal("0.0325"))
    row = s.recent_orders(1)[0]
    assert row["qty"] == "0.000123" and row["price"] == "65000.5"
    assert row["status"] == "FILLED" and row["fee"] == "0.0325"
    assert '"extra_field"' in row["raw_json"]  # unknown keys are kept, not dropped


def test_equity_decisions_positions_kv():
    s = Store(":memory:", CLOCK)
    s.record_equity(
        "2026-10-05T00:00:00+00:00",
        Decimal("100000"),
        Decimal("50000"),
        Decimal("40000"),
        Decimal("10000"),
        Decimal("101000"),
        Decimal("0.0099"),
    )
    assert s.equity_series()[0][1] == Decimal("100000")
    s.record_decision("c1", "2026-10-05T00:03:00+00:00", "RISK_ON", {"signals": {"BTC/USD": 0.5}})
    assert s.last_decision()["json"]["signals"]["BTC/USD"] == 0.5
    s.upsert_positions_meta("BTC/USD", "LONG", "t0", Decimal("65000"), None, None)
    assert s.get_positions_meta("BTC/USD")["best_price"] == "65000"
    s.kv_set("start_equity", Decimal("100000"))
    assert s.kv_get("start_equity") == "100000" and s.kv_get("missing") is None


def test_schema_apply_is_idempotent(tmp_path):
    path = tmp_path / "db.sqlite"
    Store(path, CLOCK).kv_set("a", "1")
    assert Store(path, CLOCK).kv_get("a") == "1"  # reopening keeps data
