"""Precision, MiniOrder and decimal formatting rules, built from /v3/exchangeInfo.

Never use str(float) for prices/quantities sent to the API: it can emit scientific notation
(e.g. "1e-05") which the exchange will reject or misparse. Always go through truncate()/fmt().
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from pydantic import BaseModel


def truncate(x: Decimal, decimals: int) -> Decimal:
    """Floor toward zero at the given precision, exact decimal math (no binary float rounding)."""
    q = Decimal(1).scaleb(-decimals)
    return x.quantize(q, rounding=ROUND_DOWN)


def fmt(x: Decimal) -> str:
    """Plain decimal string, never scientific notation, never a bare "-0"."""
    s = format(x.normalize(), "f")
    return s if s not in ("-0", "") else "0"


class TradingRule(BaseModel):
    """Per-pair trading rules parsed from exchangeInfo's TradePairs entry."""

    pair: str
    price_precision: int
    amount_precision: int
    mini_order: Decimal
    can_trade: bool = True

    model_config = {"arbitrary_types_allowed": True}

    def clamp_price(self, price: Decimal) -> Decimal:
        return truncate(price, self.price_precision)

    def clamp_qty(self, qty: Decimal) -> Decimal:
        return truncate(qty, self.amount_precision)

    def meets_min(self, price: Decimal, qty: Decimal) -> bool:
        """Order value must exceed MiniOrder: price * qty > mini_order."""
        return price * qty > self.mini_order

    @classmethod
    def from_exchange_info(cls, pair: str, entry: dict) -> TradingRule:
        return cls(
            pair=pair,
            price_precision=int(entry.get("PricePrecision", 0)),
            amount_precision=int(entry.get("AmountPrecision", 0)),
            mini_order=Decimal(str(entry.get("MiniOrder", 0))),
            can_trade=bool(entry.get("CanTrade", False)),
        )


def parse_trading_rules(exchange_info: dict) -> dict[str, TradingRule]:
    """Build {pair: TradingRule} from a raw /v3/exchangeInfo response."""
    pairs = exchange_info.get("TradePairs", {}) or {}
    return {pair: TradingRule.from_exchange_info(pair, entry) for pair, entry in pairs.items()}
