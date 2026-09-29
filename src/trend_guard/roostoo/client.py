"""Roostoo REST client: HMAC signing, clock sync, and every v3/v6 endpoint we use.

Facts baked in here (BUILD_SPEC.md Section 4.1, docs/vendor/roostoo-api/README.md):
  - Only documented parameters are included in the server-side signature. Never send extras.
  - Failures return HTTP 200 with Success: false and ErrMsg. Always check Success.
  - Zero-valued fields are omitted from responses; treat missing numeric fields as 0
    (handled by the pydantic defaults in models.py).
  - GETs are idempotent and safe to retry. Order-creating POSTs are never retried: on transport
    failure we raise OrderStateUnknownError and the caller must reconcile via query_order.
  - query_order/cancel_order are POSTs but are safe to retry (retry_safe=True).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from decimal import Decimal

import requests

from trend_guard.roostoo.errors import OrderStateUnknownError, RoostooAPIError, ShortsDisabledError
from trend_guard.roostoo.models import (
    OrderDetail,
    ShortCloseResult,
    ShortOpenResult,
    ShortPosition,
    Ticker,
    WalletEntry,
)
from trend_guard.roostoo.ratelimit import RateLimiter
from trend_guard.roostoo.rules import fmt

log = logging.getLogger("api")


def _fmt_param(v) -> str:
    """Format a value for inclusion in the signed param string. Decimals use fmt() to avoid
    scientific notation; everything else is str()'d as-is."""
    if isinstance(v, Decimal):
        return fmt(v)
    return str(v)


class RoostooClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        secret_key: str,
        timeout: float = 10.0,
        rate_limiter: RateLimiter | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.secret = secret_key.encode()
        self.timeout = timeout
        self.session = requests.Session()
        self.offset_ms = 0  # server_time - local_time, set by sync_clock()
        self.rl = rate_limiter  # token bucket; None is fine in tests

    # --- signing -----------------------------------------------------------------
    @staticmethod
    def build_total_params(params: dict) -> str:
        """Sort by key and join k=v with '&', exactly as the docs require."""
        return "&".join(f"{k}={_fmt_param(params[k])}" for k in sorted(params))

    def sign(self, total_params: str) -> str:
        """HMAC-SHA256 hex digest of the param string."""
        return hmac.new(self.secret, total_params.encode(), hashlib.sha256).hexdigest()

    def now_ms(self) -> str:
        """Local clock corrected by the measured server offset, as a 13-digit ms string."""
        return str(int(time.time() * 1000) + self.offset_ms)

    def sync_clock(self) -> int:
        t0 = time.time() * 1000
        server = self._get("/v3/serverTime", {}, auth="none")["ServerTime"]
        t1 = time.time() * 1000
        self.offset_ms = int(server - (t0 + t1) / 2)  # midpoint estimate
        if abs(self.offset_ms) > 30_000:
            log.warning("clock offset large", extra={"offset_ms": self.offset_ms})
        return self.offset_ms

    # --- transport -----------------------------------------------------------------
    def _get(self, path: str, params: dict, auth: str, retries: int = 3) -> dict:
        """GETs are idempotent: safe to retry with exponential backoff."""
        last_err: Exception | None = None
        for attempt in range(retries):
            p = dict(params)
            headers = {}
            if auth in ("ts", "signed"):
                p["timestamp"] = self.now_ms()
            if auth == "signed":
                headers = {
                    "RST-API-KEY": self.api_key,
                    "MSG-SIGNATURE": self.sign(self.build_total_params(p)),
                }
            try:
                if self.rl:
                    self.rl.acquire()
                r = self.session.get(self.base_url + path, params=p, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                log.info("GET", extra={"path": path, "ok": data.get("Success", True)})
                if self.rl:
                    self.rl.note_success()
                return data
            except (requests.RequestException, ValueError) as e:
                last_err = e
                if self.rl:
                    self.rl.note_failure()
                log.warning("GET failed", extra={"path": path, "attempt": attempt, "err": str(e)})
                if attempt < retries - 1:
                    time.sleep(2**attempt)
        raise RoostooAPIError(f"GET {path} failed after {retries} attempts") from last_err

    def _post(self, path: str, params: dict, retry_safe: bool = False, retries: int = 3) -> dict:
        """POSTs that create orders are NEVER retried; ambiguous outcome -> OrderStateUnknownError.

        query_order/cancel_order pass retry_safe=True since they cannot create state.
        """
        attempts = retries if retry_safe else 1
        last_err: Exception | None = None
        for attempt in range(attempts):
            p = dict(params)
            p["timestamp"] = self.now_ms()
            body = self.build_total_params(p)
            headers = {
                "RST-API-KEY": self.api_key,
                "MSG-SIGNATURE": self.sign(body),
                "Content-Type": "application/x-www-form-urlencoded",
            }
            try:
                if self.rl:
                    self.rl.acquire(order=not retry_safe)
                r = self.session.post(self.base_url + path, data=body, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as e:
                last_err = e
                if self.rl:
                    self.rl.note_failure()
                log.error("POST ambiguous", extra={"path": path, "params": params, "err": str(e)})
                if retry_safe and attempt < attempts - 1:
                    time.sleep(2**attempt)
                    continue
                raise OrderStateUnknownError(path) from e
            if self.rl:
                self.rl.note_success()
            log.info("POST", extra={"path": path, "params": params, "resp": data})
            if not data.get("Success", False) and "does not allow short" in data.get("ErrMsg", ""):
                raise ShortsDisabledError(data["ErrMsg"])
            return data
        raise OrderStateUnknownError(path) from last_err

    # --- v3: public / account -------------------------------------------------------
    def server_time(self) -> int:
        return int(self._get("/v3/serverTime", {}, auth="none")["ServerTime"])

    def exchange_info(self) -> dict:
        return self._get("/v3/exchangeInfo", {}, auth="none")

    def get_ticker(self, pair: str | None = None) -> dict[str, Ticker]:
        params = {"pair": pair} if pair else {}
        data = self._get("/v3/ticker", params, auth="ts")
        raw = data.get("Data", {}) or {}
        return {p: Ticker.model_validate(v) for p, v in raw.items()}

    def get_balance(self) -> dict[str, WalletEntry]:
        data = self._get("/v3/balance", {}, auth="signed")
        raw = data.get("Wallet", {}) or {}
        return {coin: WalletEntry.model_validate(v) for coin, v in raw.items()}

    def pending_count(self) -> tuple[int, dict[str, int]]:
        data = self._get("/v3/pending_count", {}, auth="signed")
        if not data.get("Success", False):
            # "no pending order under this account" is an empty result, not an error.
            return 0, {}
        return int(data.get("TotalPending", 0)), dict(data.get("OrderPairs", {}) or {})

    # --- v3: trading -----------------------------------------------------------------
    def place_order(
        self,
        pair: str,
        side: str,
        type_: str,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> OrderDetail:
        params: dict = {"pair": pair, "side": side, "type": type_, "quantity": quantity}
        if type_ == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders require a price")
            params["price"] = price
        data = self._post("/v3/place_order", params, retry_safe=False)
        if not data.get("Success", False):
            raise RoostooAPIError(data.get("ErrMsg", "place_order failed"))
        return OrderDetail.model_validate(data["OrderDetail"])

    def query_order(
        self,
        order_id: str | None = None,
        pair: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
        pending_only: bool | None = None,
    ) -> list[OrderDetail]:
        params: dict = {}
        if order_id is not None:
            params["order_id"] = order_id
        else:
            if pair is not None:
                params["pair"] = pair
            if offset is not None:
                params["offset"] = offset
            if limit is not None:
                params["limit"] = limit
            if pending_only is not None:
                params["pending_only"] = "TRUE" if pending_only else "FALSE"
        data = self._post("/v3/query_order", params, retry_safe=True)
        if not data.get("Success", False):
            # "no order matched" is an empty result, not an error.
            return []
        return [OrderDetail.model_validate(o) for o in data.get("OrderMatched", [])]

    def cancel_order(
        self, order_id: str | None = None, pair: str | None = None, cancel_all: bool = False
    ) -> list[int]:
        if order_id is None and pair is None and not cancel_all:
            raise ValueError(
                "cancel_order with neither order_id nor pair cancels ALL orders; pass cancel_all=True"
            )
        params: dict = {}
        if order_id is not None:
            params["order_id"] = order_id
        elif pair is not None:
            params["pair"] = pair
        data = self._post("/v3/cancel_order", params, retry_safe=True)
        if not data.get("Success", False):
            raise RoostooAPIError(data.get("ErrMsg", "cancel_order failed"))
        return list(data.get("CanceledList", []))

    # --- v6: shorts --------------------------------------------------------------------
    def short_open(
        self,
        pair: str,
        collateral: Decimal,
        order_type: str | None = None,
        price: Decimal | None = None,
    ) -> ShortOpenResult:
        params: dict = {"pair": pair, "collateral": collateral}
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("LIMIT short_open requires a price")
            params["order_type"] = "LIMIT"
            params["price"] = price
        data = self._post("/v6/short_open", params, retry_safe=False)
        if not data.get("Success", False):
            raise RoostooAPIError(data.get("ErrMsg", "short_open failed"))
        return ShortOpenResult.model_validate(data)

    def short_close(
        self, pair: str, close_qty: Decimal | None = None, close_pct: Decimal | None = None
    ) -> ShortCloseResult:
        params: dict = {"pair": pair}
        if close_qty is not None:
            params["close_qty"] = close_qty
        elif close_pct is not None:
            params["close_pct"] = close_pct
        data = self._post("/v6/short_close", params, retry_safe=False)
        if not data.get("Success", False):
            raise RoostooAPIError(data.get("ErrMsg", "short_close failed"))
        return ShortCloseResult.model_validate(data)

    def short_positions(self) -> list[ShortPosition]:
        data = self._get("/v6/short_positions", {}, auth="signed")
        if not data.get("Success", False):
            raise RoostooAPIError(data.get("ErrMsg", "short_positions failed"))
        return [ShortPosition.model_validate(p) for p in data.get("Positions", [])]
