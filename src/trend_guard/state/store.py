"""SQLite state store. BUILD_SPEC.md Section 10.1.

Decimals are stored as strings so nothing is lost to binary floats. The schema is applied on every
connect with CREATE TABLE IF NOT EXISTS, so opening an existing DB is always safe.
"""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

from trend_guard.clock import Clock

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_ORDER_COLS = (
    "created_utc",
    "pair",
    "side",
    "kind",
    "qty",
    "price",
    "collateral",
    "exchange_order_id",
    "status",
    "role",
    "filled_qty",
    "avg_price",
    "fee",
    "reason",
    "cycle_id",
    "raw_json",
)


def _s(v) -> str | None:
    """Store Decimals/numbers as plain strings; keep None as NULL."""
    if v is None:
        return None
    if isinstance(v, Decimal):
        return format(v, "f")
    return str(v)


class Store:
    def __init__(self, db_path: str | Path, clock: Clock):
        self.clock = clock
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- orders -----------------------------------------------------------------
    def insert_order(self, record: dict) -> int:
        """Insert an order row; unknown keys go into raw_json so nothing is silently dropped."""
        row = {c: _s(record.get(c)) for c in _ORDER_COLS if c != "raw_json"}
        row["created_utc"] = row["created_utc"] or self.clock.now().isoformat()
        extra = {k: _s(v) for k, v in record.items() if k not in _ORDER_COLS}
        row["raw_json"] = json.dumps(extra) if extra else record.get("raw_json")
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        cur = self.conn.execute(f"INSERT INTO orders ({cols}) VALUES ({marks})", tuple(row.values()))
        self.conn.commit()
        return int(cur.lastrowid)

    def update_order(self, row_id: int, **fields) -> None:
        if not fields:
            return
        bad = set(fields) - set(_ORDER_COLS)
        if bad:
            raise ValueError(f"unknown order columns: {bad}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(f"UPDATE orders SET {sets} WHERE id = ?", (*map(_s, fields.values()), row_id))
        self.conn.commit()

    def recent_orders(self, limit: int = 50) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    # --- equity -----------------------------------------------------------------
    def record_equity(
        self,
        ts_utc: str,
        equity: Decimal,
        cash: Decimal,
        long_notional: Decimal,
        short_collateral: Decimal,
        peak: Decimal,
        dd: Decimal,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO equity VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts_utc, _s(equity), _s(cash), _s(long_notional), _s(short_collateral), _s(peak), _s(dd)),
        )
        self.conn.commit()

    def equity_series(self) -> list[tuple[str, Decimal]]:
        cur = self.conn.execute("SELECT ts_utc, equity FROM equity ORDER BY ts_utc")
        return [(r["ts_utc"], Decimal(r["equity"])) for r in cur.fetchall()]

    # --- decisions -----------------------------------------------------------------
    def record_decision(self, cycle_id: str, ts_utc: str, regime: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?)",
            (cycle_id, ts_utc, regime, json.dumps(payload, default=str)),
        )
        self.conn.commit()

    def last_decision(self) -> dict | None:
        cur = self.conn.execute("SELECT * FROM decisions ORDER BY ts_utc DESC LIMIT 1")
        r = cur.fetchone()
        if r is None:
            return None
        return {
            "cycle_id": r["cycle_id"],
            "ts_utc": r["ts_utc"],
            "regime": r["regime"],
            "json": json.loads(r["json"]),
        }

    # --- positions_meta -----------------------------------------------------------------
    def get_positions_meta(self, pair: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM positions_meta WHERE pair = ?", (pair,)).fetchone()
        return dict(r) if r else None

    def all_positions_meta(self) -> dict[str, dict]:
        cur = self.conn.execute("SELECT * FROM positions_meta")
        return {r["pair"]: dict(r) for r in cur.fetchall()}

    def upsert_positions_meta(
        self,
        pair: str,
        side: str | None = None,
        opened_utc: str | None = None,
        best_price: Decimal | None = None,
        stop_ref: Decimal | None = None,
        cooldown_until: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO positions_meta VALUES (?, ?, ?, ?, ?, ?)",
            (pair, side, opened_utc, _s(best_price), _s(stop_ref), cooldown_until),
        )
        self.conn.commit()

    def delete_positions_meta(self, pair: str) -> None:
        self.conn.execute("DELETE FROM positions_meta WHERE pair = ?", (pair,))
        self.conn.commit()

    # --- kv -----------------------------------------------------------------
    def kv_get(self, key: str) -> str | None:
        r = self.conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return r["v"] if r else None

    def kv_set(self, key: str, value) -> None:
        self.conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, _s(value)))
        self.conn.commit()
