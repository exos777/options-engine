"""
Portfolio exposure — how much capital is committed, and where.

Capital at risk per ticker = cash securing short puts + cost of shares held.
Compared against the account size to flag concentration (one ticker too
large) and over-commitment (too little cash left free).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from portfolio.ledger import WheelPosition

DEFAULT_MAX_TICKER_PCT = 0.20
DEFAULT_MAX_TOTAL_PCT = 0.80


@dataclass
class ExposureRow:
    ticker: str
    phase: str
    csp_collateral: float
    share_cost: float
    pct_of_account: float

    @property
    def total(self) -> float:
        return self.csp_collateral + self.share_cost


@dataclass
class ExposureReport:
    account_size: float
    rows: list[ExposureRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(r.total for r in self.rows)

    @property
    def pct_of_account(self) -> float:
        return self.total / self.account_size if self.account_size > 0 else 0.0

    @property
    def free_cash(self) -> float:
        return self.account_size - self.total


def compute_exposure(
    positions: list[WheelPosition],
    account_size: float,
    max_ticker_pct: float = DEFAULT_MAX_TICKER_PCT,
    max_total_pct: float = DEFAULT_MAX_TOTAL_PCT,
) -> ExposureReport:
    report = ExposureReport(account_size=account_size)
    for pos in positions:
        row = ExposureRow(
            ticker=pos.ticker,
            phase=pos.phase,
            csp_collateral=pos.csp_collateral,
            share_cost=pos.share_cost,
            pct_of_account=(
                pos.capital_at_risk / account_size if account_size > 0 else 0.0
            ),
        )
        if row.total > 0:
            report.rows.append(row)
    report.rows.sort(key=lambda r: r.total, reverse=True)

    if account_size <= 0:
        return report

    for row in report.rows:
        if row.pct_of_account > max_ticker_pct:
            report.warnings.append(
                f"{row.ticker} is {row.pct_of_account:.0%} of the account "
                f"(limit {max_ticker_pct:.0%})."
            )
    if report.pct_of_account > max_total_pct:
        report.warnings.append(
            f"{report.pct_of_account:.0%} of the account is committed "
            f"(limit {max_total_pct:.0%}) — only ${report.free_cash:,.0f} free."
        )
    return report


def max_new_csp_contracts(
    report: ExposureReport,
    ticker: str,
    strike: float,
    max_ticker_pct: float = DEFAULT_MAX_TICKER_PCT,
    max_total_pct: float = DEFAULT_MAX_TOTAL_PCT,
) -> int:
    """How many more puts at *strike* fit inside both limits (0 if none)."""
    if report.account_size <= 0 or strike <= 0:
        return 0
    per_contract = strike * 100
    current = next((r.total for r in report.rows if r.ticker == ticker), 0.0)
    room_ticker = report.account_size * max_ticker_pct - current
    room_total = report.account_size * max_total_pct - report.total
    room = min(room_ticker, room_total)
    return max(int(room // per_contract), 0)
