"""
Wheel ledger — turns a flat list of trades into wheel cycles.

A *cycle* for a ticker starts with the first trade while flat (no shares,
no open short options) and ends the next time it is flat again — e.g.
CSP → assigned → CCs → called away. Everything inside the cycle counts
toward that cycle's premium and P&L, so the adjusted cost basis reflects
every put and call sold along the way.

Pure functions only — no Streamlit, no database.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from portfolio.models import CONTRACT_MULTIPLIER, Trade, TradeAction


@dataclass
class OpenShort:
    """An open short option (one contract/strike/expiration)."""
    ticker: str
    option_type: str
    strike: float
    expiration: date
    contracts: int
    avg_open_price: float      # per-share premium received
    opened_on: date

    @property
    def premium_collected(self) -> float:
        return self.avg_open_price * CONTRACT_MULTIPLIER * self.contracts

    @property
    def collateral(self) -> float:
        """Cash secured by a short put. Calls are covered by shares instead."""
        if self.option_type == "put":
            return self.strike * CONTRACT_MULTIPLIER * self.contracts
        return 0.0

    def dte(self, today: Optional[date] = None) -> int:
        return (self.expiration - (today or date.today())).days


@dataclass
class WheelPosition:
    """The open (in-progress) wheel cycle for one ticker."""
    ticker: str
    started: date
    shares: int = 0
    share_cost: float = 0.0           # total $ cost of shares held (avg-cost)
    net_premium: float = 0.0          # all option cash this cycle, after fees
    share_cash: float = 0.0           # net cash from share buys/sells this cycle
    max_capital: float = 0.0
    open_shorts: list[OpenShort] = field(default_factory=list)

    @property
    def avg_share_cost(self) -> Optional[float]:
        return self.share_cost / self.shares if self.shares > 0 else None

    @property
    def adjusted_basis(self) -> Optional[float]:
        """Per-share cost after subtracting every premium collected this cycle."""
        if self.shares <= 0:
            return None
        return (self.share_cost - self.net_premium) / self.shares

    @property
    def csp_collateral(self) -> float:
        return sum(s.collateral for s in self.open_shorts)

    @property
    def capital_at_risk(self) -> float:
        return self.csp_collateral + self.share_cost

    @property
    def phase(self) -> str:
        if self.shares > 0:
            return "CC"
        if self.open_shorts:
            return "CSP"
        return "—"


@dataclass
class ClosedCycle:
    """A completed wheel cycle (back to flat)."""
    ticker: str
    started: date
    ended: date
    net_premium: float
    share_pnl: float
    max_capital: float

    @property
    def realized_pnl(self) -> float:
        return self.net_premium + self.share_pnl

    @property
    def days(self) -> int:
        return max((self.ended - self.started).days, 1)

    @property
    def return_pct(self) -> float:
        return self.realized_pnl / self.max_capital if self.max_capital > 0 else 0.0

    @property
    def annualized_pct(self) -> float:
        return self.return_pct * 365.0 / self.days


@dataclass
class Ledger:
    positions: dict[str, WheelPosition] = field(default_factory=dict)
    closed: list[ClosedCycle] = field(default_factory=list)
    monthly_premium: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def open_shorts(self) -> list[OpenShort]:
        out = [s for p in self.positions.values() for s in p.open_shorts]
        return sorted(out, key=lambda s: (s.expiration, s.ticker, s.strike))

    @property
    def total_premium(self) -> float:
        return sum(self.monthly_premium.values())

    @property
    def realized_pnl(self) -> float:
        return sum(c.realized_pnl for c in self.closed)

    @property
    def capital_at_risk(self) -> float:
        return sum(p.capital_at_risk for p in self.positions.values())


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _close_short(
    pos: WheelPosition, trade: Trade, warnings: list[str],
) -> Optional[OpenShort]:
    """Reduce the matching open short by trade.quantity. Returns the match."""
    key = trade.contract_key
    for short in pos.open_shorts:
        if (short.option_type, short.strike, short.expiration) == key:
            if trade.quantity > short.contracts:
                warnings.append(
                    f"{trade.describe()}: closes {trade.quantity} but only "
                    f"{short.contracts} open — closed what was open."
                )
            short.contracts -= min(trade.quantity, short.contracts)
            if short.contracts <= 0:
                pos.open_shorts.remove(short)
            return short
    warnings.append(f"{trade.describe()}: no matching open contract.")
    return None


def _apply(pos: WheelPosition, trade: Trade, warnings: list[str]) -> None:
    a = trade.action
    cf = trade.cash_flow

    if a == TradeAction.SELL_TO_OPEN:
        pos.net_premium += cf
        key = trade.contract_key
        for short in pos.open_shorts:
            if (short.option_type, short.strike, short.expiration) == key:
                total = short.avg_open_price * short.contracts + trade.price * trade.quantity
                short.contracts += trade.quantity
                short.avg_open_price = total / short.contracts
                break
        else:
            pos.open_shorts.append(OpenShort(
                ticker=trade.ticker,
                option_type=trade.option_type or "put",
                strike=float(trade.strike or 0.0),
                expiration=trade.expiration,
                contracts=trade.quantity,
                avg_open_price=trade.price,
                opened_on=trade.trade_date,
            ))

    elif a in (TradeAction.BUY_TO_CLOSE, TradeAction.EXPIRED):
        pos.net_premium += cf
        _close_short(pos, trade, warnings)

    elif a == TradeAction.ASSIGNED:
        # Fees on assignment count against the option side.
        pos.net_premium -= trade.fees
        _close_short(pos, trade, warnings)
        shares = trade.quantity * CONTRACT_MULTIPLIER
        notional = float(trade.strike or 0.0) * shares
        if trade.option_type == "put":
            _buy_shares(pos, shares, notional)
        else:
            _sell_shares(pos, shares, notional, trade, warnings)

    elif a == TradeAction.BUY_SHARES:
        _buy_shares(pos, trade.quantity, trade.price * trade.quantity)
        pos.share_cash -= trade.fees

    elif a == TradeAction.SELL_SHARES:
        _sell_shares(pos, trade.quantity, trade.price * trade.quantity, trade, warnings)
        pos.share_cash -= trade.fees


def _buy_shares(pos: WheelPosition, shares: int, cost: float) -> None:
    pos.shares += shares
    pos.share_cost += cost
    pos.share_cash -= cost


def _sell_shares(
    pos: WheelPosition, shares: int, proceeds: float,
    trade: Trade, warnings: list[str],
) -> None:
    if shares > pos.shares:
        warnings.append(
            f"{trade.describe()}: sells {shares} shares but only "
            f"{pos.shares} held."
        )
    if pos.shares > 0:
        avg = pos.share_cost / pos.shares
        sold = min(shares, pos.shares)
        pos.share_cost -= avg * sold
        pos.shares -= sold
    pos.share_cash += proceeds
    if pos.shares <= 0:
        pos.shares = 0
        pos.share_cost = 0.0


def build_ledger(trades: list[Trade]) -> Ledger:
    """Replay *trades* in date order into open positions and closed cycles."""
    ledger = Ledger()
    monthly: dict[str, float] = defaultdict(float)
    ordered = sorted(
        trades, key=lambda t: (t.trade_date, t.id if t.id is not None else 0),
    )
    # A cycle may only close after a ticker's *last* trade of the day — a
    # roll (close + reopen on the same date) is momentarily flat in between
    # and must stay one cycle.
    last_of_day = {
        (t.ticker.upper(), t.trade_date): i for i, t in enumerate(ordered)
    }

    for i, trade in enumerate(ordered):
        ticker = trade.ticker.upper()
        pos = ledger.positions.get(ticker)
        if pos is None:
            pos = WheelPosition(ticker=ticker, started=trade.trade_date)
            ledger.positions[ticker] = pos

        _apply(pos, trade, ledger.warnings)
        if trade.action.is_option:
            month = trade.trade_date.strftime("%Y-%m")
            option_cash = (
                -trade.fees if trade.action == TradeAction.ASSIGNED
                else trade.cash_flow
            )
            monthly[month] += option_cash
        pos.max_capital = max(pos.max_capital, pos.capital_at_risk)

        if last_of_day[(ticker, trade.trade_date)] != i:
            continue
        if pos.shares == 0 and not pos.open_shorts:
            # A stray close with nothing open leaves an empty cycle — the
            # warning already reports it, so don't record a 0/0 cycle.
            if pos.max_capital == 0 and pos.net_premium == 0 and pos.share_cash == 0:
                del ledger.positions[ticker]
                continue
            ledger.closed.append(ClosedCycle(
                ticker=ticker,
                started=pos.started,
                ended=trade.trade_date,
                net_premium=pos.net_premium,
                share_pnl=pos.share_cash,
                max_capital=pos.max_capital,
            ))
            del ledger.positions[ticker]

    ledger.monthly_premium = dict(sorted(monthly.items()))
    return ledger
