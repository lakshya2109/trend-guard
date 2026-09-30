"""Which Roostoo pairs the book may trade, and their Binance symbols.

Liquidity and history filters live in the strategy itself (strategy/book.py applies them per hour
from Binance volume), so the universe here is just: listed, tradable, crypto, not a stablecoin.
Tokenized stocks (bStocks) are left out on purpose: no clean history before mid-2026 and single-stock
weekend gap risk (docs/STRATEGY.md).
"""

from __future__ import annotations

from trend_guard.config import SymbolsConfig


def binance_symbol(pair: str, symbols: SymbolsConfig) -> str:
    """'BTC/USD' -> 'BTCUSDT' unless config/symbols.yaml overrides it."""
    return symbols.overrides.get(pair) or pair.split("/")[0] + "USDT"


def tradable_pairs(exchange_info: dict, symbols: SymbolsConfig) -> list[str]:
    """CanTrade crypto pairs from /v3/exchangeInfo, minus config/symbols.yaml `excluded`.

    exchangeInfo tags each pair with AssetType ("crypto" or "stock"; tokenized stocks are named like
    NVDAB/USD). A pair without the tag is treated as crypto.
    """
    out = []
    for pair, info in (exchange_info.get("TradePairs") or {}).items():
        if not info.get("CanTrade", False) or not pair.endswith("/USD"):
            continue
        if info.get("AssetType", "crypto") != "crypto" or pair in symbols.excluded:
            continue
        out.append(pair)
    return sorted(out)
