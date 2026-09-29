"""Typed Roostoo API responses.

Zero-valued fields are omitted by the server (BUILD_SPEC.md Section 4.1), so every numeric field
here defaults to 0 rather than being required.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Ticker(BaseModel):
    max_bid: float = Field(0, alias="MaxBid")
    min_ask: float = Field(0, alias="MinAsk")
    last_price: float = Field(0, alias="LastPrice")
    change: float = Field(0, alias="Change")
    coin_trade_value: float = Field(0, alias="CoinTradeValue")
    unit_trade_value: float = Field(0, alias="UnitTradeValue")

    model_config = {"populate_by_name": True}

    @property
    def mid(self) -> float:
        return (self.max_bid + self.min_ask) / 2 if (self.max_bid and self.min_ask) else self.last_price

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return (self.min_ask - self.max_bid) / m if m else 0.0


class WalletEntry(BaseModel):
    free: float = Field(0, alias="Free")
    lock: float = Field(0, alias="Lock")

    model_config = {"populate_by_name": True}


class OrderDetail(BaseModel):
    pair: str = Field("", alias="Pair")
    order_id: int = Field(0, alias="OrderID")
    status: str = Field("", alias="Status")
    role: str = Field("", alias="Role")
    create_ts: int = Field(0, alias="CreateTimestamp")
    finish_ts: int = Field(0, alias="FinishTimestamp")
    side: str = Field("", alias="Side")
    type: str = Field("", alias="Type")
    price: float = Field(0, alias="Price")
    quantity: float = Field(0, alias="Quantity")
    filled_quantity: float = Field(0, alias="FilledQuantity")
    filled_aver_price: float = Field(0, alias="FilledAverPrice")
    coin_change: float = Field(0, alias="CoinChange")
    unit_change: float = Field(0, alias="UnitChange")
    commission_coin: str = Field("", alias="CommissionCoin")
    commission_charge_value: float = Field(0, alias="CommissionChargeValue")
    commission_percent: float = Field(0, alias="CommissionPercent")

    model_config = {"populate_by_name": True}


class ShortOpenResult(BaseModel):
    id: int = Field(0, alias="ID")
    pair: str = Field("", alias="Pair")
    order_type: str = Field("", alias="OrderType")
    entry_price: float = Field(0, alias="EntryPrice")
    short_qty: float = Field(0, alias="ShortQty")
    collateral: float = Field(0, alias="Collateral")
    open_fee: float = Field(0, alias="OpenFee")
    status: str = Field("", alias="Status")
    create_ts: int = Field(0, alias="CreateTimestamp")

    model_config = {"populate_by_name": True}


class ShortCloseResult(BaseModel):
    close_price: float = Field(0, alias="ClosePrice")
    realized_pnl: float = Field(0, alias="RealizedPNL")
    close_fee: float = Field(0, alias="CloseFee")
    return_amount: float = Field(0, alias="ReturnAmount")
    closed_qty: float = Field(0, alias="ClosedQty")
    fully_closed: bool = Field(False, alias="FullyClosed")
    remaining_qty: float = Field(0, alias="RemainingQty")
    remaining_collateral: float = Field(0, alias="RemainingCollateral")

    model_config = {"populate_by_name": True}


class ShortPosition(BaseModel):
    id: int = Field(0, alias="ID")
    pair: str = Field("", alias="Pair")
    entry_price: float = Field(0, alias="EntryPrice")
    short_qty: float = Field(0, alias="ShortQty")
    collateral: float = Field(0, alias="Collateral")
    current_price: float = Field(0, alias="CurrentPrice")
    unrealized_pnl: float = Field(0, alias="UnrealizedPNL")
    unrealized_pnl_pct: float = Field(0, alias="UnrealizedPNLPct")
    position_value: float = Field(0, alias="PositionValue")
    create_ts: int = Field(0, alias="CreateTimestamp")
    status: str = Field("OPEN", alias="PositionStatus")

    model_config = {"populate_by_name": True}
