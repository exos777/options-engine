"""
Exit alerts for open short options.

Rules (in priority order):
    expired       — past expiration but still open in the ledger
    strike_tested — ≤ 2 DTE and the stock is ITM or within 1 ATR of strike
    profit_take   — ≥ 50% of the premium captured: buy back, redeploy
    below_basis   — covered call strike under the wheel-adjusted basis
    itm           — in the money with time left
    earnings      — earnings date before expiration

Pure functions — callers supply spot / mark / ATR / earnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from portfolio.ledger import OpenShort

DEFAULT_PROFIT_TARGET = 0.50
TESTED_MAX_DTE = 2

SEVERITY_ORDER = {"action": 0, "warning": 1, "info": 2}


@dataclass
class Alert:
    ticker: str
    severity: str      # "action" | "warning" | "info"
    kind: str
    message: str
    contract: str = ""


def _is_itm(short: OpenShort, spot: float) -> bool:
    if short.option_type == "put":
        return spot < short.strike
    return spot > short.strike


def position_alerts(
    short: OpenShort,
    *,
    spot: Optional[float] = None,
    mark: Optional[float] = None,
    atr: Optional[float] = None,
    earnings_date: Optional[date] = None,
    adjusted_basis: Optional[float] = None,
    today: Optional[date] = None,
    profit_target: float = DEFAULT_PROFIT_TARGET,
) -> list[Alert]:
    """All alerts that currently apply to one open short."""
    today = today or date.today()
    kind_letter = "P" if short.option_type == "put" else "C"
    label = f"{short.strike:g}{kind_letter} {short.expiration:%b %d}"
    dte = short.dte(today)
    alerts: list[Alert] = []

    def add(severity: str, kind: str, message: str) -> None:
        alerts.append(Alert(short.ticker, severity, kind, message, label))

    if dte < 0:
        add("action", "expired",
            f"Expired {short.expiration:%b %d} — log it as Expired or Assigned.")
        return alerts

    itm = spot is not None and spot > 0 and _is_itm(short, spot)

    if spot is not None and spot > 0 and dte <= TESTED_MAX_DTE:
        buffer = atr if atr and atr > 0 else spot * 0.02
        if itm or abs(spot - short.strike) <= buffer:
            where = "in the money" if itm else f"within ${buffer:.2f} (1 ATR)"
            add("action", "strike_tested",
                f"Stock ${spot:.2f} is {where} of the ${short.strike:g} strike "
                f"with {dte}d left — decide: roll or accept assignment.")

    if mark is not None and short.avg_open_price > 0:
        captured = 1.0 - mark / short.avg_open_price
        if captured >= profit_target:
            add("action", "profit_take",
                f"{captured:.0%} of premium captured (sold {short.avg_open_price:.2f}, "
                f"now {mark:.2f}) — buy back and redeploy the capital.")

    if (
        short.option_type == "call"
        and adjusted_basis is not None
        and short.strike < adjusted_basis
    ):
        add("warning", "below_basis",
            f"Call strike ${short.strike:g} is below your adjusted basis "
            f"${adjusted_basis:.2f} — assignment would lock in a loss.")

    if itm and not any(a.kind == "strike_tested" for a in alerts):
        add("warning", "itm",
            f"In the money (stock ${spot:.2f} vs ${short.strike:g}) with {dte}d left.")

    if earnings_date is not None and today <= earnings_date <= short.expiration:
        add("warning", "earnings",
            f"Earnings {earnings_date:%b %d} lands before expiration.")

    return alerts


def sort_alerts(alerts: list[Alert]) -> list[Alert]:
    return sorted(alerts, key=lambda a: (SEVERITY_ORDER.get(a.severity, 9), a.ticker))
