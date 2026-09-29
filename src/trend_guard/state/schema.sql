CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY, created_utc TEXT, pair TEXT, side TEXT, kind TEXT,  -- kind: LIMIT/MARKET/SHORT_OPEN/SHORT_CLOSE
  qty TEXT, price TEXT, collateral TEXT, exchange_order_id TEXT, status TEXT,  -- status: NEW/PENDING/FILLED/PARTIAL/CANCELED/UNKNOWN/REJECTED
  role TEXT, filled_qty TEXT, avg_price TEXT, fee TEXT, reason TEXT, cycle_id TEXT, raw_json TEXT);
CREATE TABLE IF NOT EXISTS equity (
  ts_utc TEXT PRIMARY KEY, equity TEXT, cash TEXT, long_notional TEXT, short_collateral TEXT, peak TEXT, dd TEXT);
CREATE TABLE IF NOT EXISTS decisions (
  cycle_id TEXT PRIMARY KEY, ts_utc TEXT, regime TEXT, json TEXT);            -- signals, weights, multipliers, targets
CREATE TABLE IF NOT EXISTS positions_meta (
  pair TEXT PRIMARY KEY, side TEXT, opened_utc TEXT, best_price TEXT, stop_ref TEXT, cooldown_until TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);                     -- start_equity, start_ts, flags
