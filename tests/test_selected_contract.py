"""Tests for data.common.get_selected_contract and DTE auto-calculation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from data.common import (
    SelectedContract,
    days_to_expiration,
    get_selected_contract,
)
from strategies.models import OptionContract


# ── Fake data provider ───────────────────────────

@dataclass
class _Chain:
    calls: list[OptionContract]
    puts: list[OptionContract]


class _FakeDP:
    """Minimal data provider stub exposing get_option_chain."""

    def __init__(self, chains: dict[tuple[str, str], _Chain]):
        self._chains = chains

    def get_option_chain(
        self, ticker: str, expiration: str,
    ) -> tuple[list[OptionContract], list[OptionContract]]:
        key = (ticker, expiration)
        if key not in self._chains:
            raise ValueError(f"no chain for {ticker} {expiration}")
        ch = self._chains[key]
        return ch.calls, ch.puts


def _contract(
    strike: float, exp: str, side: str,
    bid: float = 1.00, ask: float = 1.10,
) -> OptionContract:
    return OptionContract(
        strike=strike,
        expiration=exp,
        option_type=side,
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        volume=100,
        open_interest=500,
        implied_volatility=0.30,
        delta=-0.25 if side == "put" else 0.30,
        gamma=0.01,
        theta=-0.05,
        vega=0.10,
    )


def _iso(days_out: int) -> str:
    return (date.today() + timedelta(days=days_out)).isoformat()


# ── DTE from expiration ──────────────────────────

def test_dte_from_expiration_seven_days():
    exp = _iso(7)
    assert days_to_expiration(exp) == 7


def test_dte_from_expiration_clamped_at_zero_for_past():
    exp = _iso(-3)
    assert days_to_expiration(exp) == 0


def test_get_selected_contract_dte_matches_expiration():
    exp = _iso(10)
    chain = _Chain(calls=[_contract(100, exp, "call")], puts=[_contract(100, exp, "put")])
    dp = _FakeDP({("XYZ", exp): chain})

    result = get_selected_contract(dp, "XYZ", exp, 100.0, "CSP")
    assert result.dte == 10
    assert result.found


# ── Strategy selects correct side ───────────────

def test_csp_returns_put_contract():
    exp = _iso(10)
    chain = _Chain(
        calls=[_contract(100, exp, "call", bid=5.0, ask=5.2)],
        puts=[_contract(100, exp, "put", bid=2.0, ask=2.2)],
    )
    dp = _FakeDP({("XYZ", exp): chain})

    result = get_selected_contract(dp, "XYZ", exp, 100.0, "CSP")
    assert result.found
    assert result.contract.option_type == "put"
    assert result.contract.bid == pytest.approx(2.0)


def test_cc_returns_call_contract():
    exp = _iso(10)
    chain = _Chain(
        calls=[_contract(100, exp, "call", bid=5.0, ask=5.2)],
        puts=[_contract(100, exp, "put", bid=2.0, ask=2.2)],
    )
    dp = _FakeDP({("XYZ", exp): chain})

    result = get_selected_contract(dp, "XYZ", exp, 100.0, "CC")
    assert result.found
    assert result.contract.option_type == "call"
    assert result.contract.bid == pytest.approx(5.0)


# ── Strike not found ────────────────────────────

def test_strike_not_found_returns_warning():
    exp = _iso(10)
    chain = _Chain(
        calls=[_contract(100, exp, "call")],
        puts=[_contract(100, exp, "put")],
    )
    dp = _FakeDP({("XYZ", exp): chain})

    result = get_selected_contract(dp, "XYZ", exp, 105.0, "CSP")
    assert not result.found
    assert result.contract is None
    assert result.error == "Strike not found for selected expiration."
    # DTE still present so UI can still show it
    assert result.dte == 10


# ── Expiration change yields new contract / DTE ──

def test_expiration_change_updates_contract_and_dte():
    exp_short = _iso(7)
    exp_long = _iso(14)
    chain_short = _Chain(
        calls=[_contract(100, exp_short, "call", bid=1.0, ask=1.1)],
        puts=[_contract(100, exp_short, "put", bid=0.80, ask=0.95)],
    )
    chain_long = _Chain(
        calls=[_contract(100, exp_long, "call", bid=2.0, ask=2.2)],
        puts=[_contract(100, exp_long, "put", bid=1.50, ask=1.70)],
    )
    dp = _FakeDP({
        ("XYZ", exp_short): chain_short,
        ("XYZ", exp_long): chain_long,
    })

    short_res = get_selected_contract(dp, "XYZ", exp_short, 100.0, "CSP")
    long_res = get_selected_contract(dp, "XYZ", exp_long, 100.0, "CSP")

    assert short_res.dte == 7
    assert long_res.dte == 14
    assert short_res.contract.bid == pytest.approx(0.80)
    assert long_res.contract.bid == pytest.approx(1.50)
    # The put for the longer expiration should carry more premium in a real
    # market; our fake enforces it to confirm the helper isn't reusing a cache.
    assert long_res.contract.bid > short_res.contract.bid


# ── Invalid expiration ──────────────────────────

def test_invalid_expiration_returns_error():
    dp = _FakeDP({})
    result = get_selected_contract(dp, "XYZ", "not-a-date", 100.0, "CSP")
    assert not result.found
    assert result.error is not None
