"""Precision/formatting tests. BUILD_SPEC.md Section 4.6."""

from decimal import Decimal

from trend_guard.roostoo.rules import TradingRule, fmt, parse_trading_rules, truncate


def test_fmt_no_scientific_notation():
    assert fmt(Decimal("0.00001")) == "0.00001"


def test_truncate_towards_zero():
    assert truncate(Decimal("0.29"), 2) == Decimal("0.29")
    assert truncate(Decimal("0.297"), 2) == Decimal("0.29")
    assert truncate(Decimal("-0.297"), 2) == Decimal("-0.29")


def test_fmt_never_bare_negative_zero():
    assert fmt(Decimal("-0.0000001").quantize(Decimal("1"))) == "0"


def test_trading_rule_meets_min():
    rule = TradingRule(pair="BTC/USD", price_precision=2, amount_precision=6, mini_order=Decimal("1.0"))
    assert rule.meets_min(Decimal("50000"), Decimal("0.001")) is True
    assert rule.meets_min(Decimal("0.5"), Decimal("1")) is False


def test_trading_rule_clamp():
    rule = TradingRule(pair="BTC/USD", price_precision=2, amount_precision=6, mini_order=Decimal("1.0"))
    assert rule.clamp_price(Decimal("50000.4567")) == Decimal("50000.45")
    assert rule.clamp_qty(Decimal("0.1234567")) == Decimal("0.123456")


def test_parse_trading_rules_from_exchange_info():
    info = {
        "TradePairs": {
            "BTC/USD": {
                "CanTrade": True,
                "PricePrecision": 2,
                "AmountPrecision": 6,
                "MiniOrder": 1.0,
            }
        }
    }
    rules = parse_trading_rules(info)
    assert rules["BTC/USD"].price_precision == 2
    assert rules["BTC/USD"].can_trade is True
