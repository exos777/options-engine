"""
Trade records for the wheel ledger.

One Trade row per fill or lifecycle event. A roll is logged as two rows:
a BUY_TO_CLOSE of the old contract and a SELL_TO_OPEN of the new one.

Prices follow broker conventions: option prices are per share (multiply
by 100 per contract), share prices are per share. Quantity is contracts
for option actions and shares for share actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

CONTRACT_MULTIPLIER = 100


class TradeAction(str, Enum):
    SELL_TO_OPEN = "STO"
    BUY_TO_CLOSE = "BTC"
    EXPIRED = "EXPIRED"
    ASSIGNED = "ASSIGNED"       # put assigned, or call called away
    BUY_SHARES = "BUY_SHARES"
    SELL_SHARES = "SELL_SHARES"

    @property
    def label(self) -> str:
        return _ACTION_LABELS[self]

    @property
    def is_option(self) -> bool:
        return self not in (TradeAction.BUY_SHARES, TradeAction.SELL_SHARES)


_ACTION_LABELS = {
    TradeAction.SELL_TO_OPEN: "Sell to Open",
    TradeAction.BUY_TO_CLOSE: "Buy to Close",
    TradeAction.EXPIRED: "Expired Worthless",
    TradeAction.ASSIGNED: "Assigned / Called Away",
    TradeAction.BUY_SHARES: "Buy Shares",
    TradeAction.SELL_SHARES: "Sell Shares",
}


@dataclass
class Trade:
    trade_date: date
    ticker: str
    action: TradeAction
    quantity: int
    price: float = 0.0
    option_type: Optional[str] = None   # "put" / "call" for option actions
    strike: Optional[float] = None
    expiration: Optional[date] = None
    fees: float = 0.0
    notes: str = ""
    id: Optional[int] = None

    @property
    def contract_key(self) -> tuple[str, float, date]:
        """Identifies the option contract this trade touches."""
        return (self.option_type or "", float(self.strike or 0.0), self.expiration)

    @property
    def cash_flow(self) -> float:
        """Signed cash impact in dollars (+ = cash received)."""
        q = self.quantity
        a = self.action
        if a == TradeAction.SELL_TO_OPEN:
            gross = self.price * CONTRACT_MULTIPLIER * q
        elif a == TradeAction.BUY_TO_CLOSE:
            gross = -self.price * CONTRACT_MULTIPLIER * q
        elif a == TradeAction.EXPIRED:
            gross = 0.0
        elif a == TradeAction.ASSIGNED:
            notional = (self.strike or 0.0) * CONTRACT_MULTIPLIER * q
            gross = -notional if self.option_type == "put" else notional
        elif a == TradeAction.BUY_SHARES:
            gross = -self.price * q
        else:  # SELL_SHARES
            gross = self.price * q
        return gross - self.fees

    def describe(self) -> str:
        """Short human label, e.g. 'STO 2× TSLA 240P 2026-10-24 @ 3.10'."""
        if self.action.is_option:
            kind = "P" if self.option_type == "put" else "C"
            text = (
                f"{self.action.value} {self.quantity}× {self.ticker} "
                f"{self.strike:g}{kind} {self.expiration}"
            )
            if self.action in (TradeAction.SELL_TO_OPEN, TradeAction.BUY_TO_CLOSE):
                text += f" @ {self.price:.2f}"
            return text
        verb = "Buy" if self.action == TradeAction.BUY_SHARES else "Sell"
        return f"{verb} {self.quantity} {self.ticker} @ {self.price:.2f}"
