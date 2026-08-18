"""
Contract tests for data/tradier_provider.py.

Network is mocked — these tests verify request shape and response parsing
without hitting the real Tradier API.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from data import tradier_provider as tp
from strategies.models import OptionContract, Quote


# ---------------------------------------------------------------------------
# Token / availability
# ---------------------------------------------------------------------------

def test_tradier_available_false_without_token(monkeypatch):
    monkeypatch.delenv("TRADIER_TOKEN", raising=False)
    with patch.object(tp, "_get_token", return_value=""):
        assert tp.tradier_available() is False


def test_tradier_available_true_with_token():
    with patch.object(tp, "_get_token", return_value="abc123"):
        assert tp.tradier_available() is True


# ---------------------------------------------------------------------------
# Token normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "abc123",
    "  abc123  ",
    '"abc123"',
    "'abc123'",
    '  "abc123"  ',
])
def test_clean_token_strips_whitespace_and_quotes(raw):
    """
    TOML requires quotes in secrets.toml but env vars take them literally.
    Pasting the quoted form into Railway must not produce Bearer "abc123".
    """
    assert tp._clean_token(raw) == "abc123"


def test_clean_token_preserves_interior_quotes():
    assert tp._clean_token('ab"cd') == 'ab"cd'


def test_clean_token_handles_empty_and_none():
    assert tp._clean_token("") == ""
    assert tp._clean_token(None) == ""


def test_get_token_strips_quotes_from_env(monkeypatch):
    monkeypatch.setenv("TRADIER_TOKEN", '"quoted-token"')
    assert tp._get_token() == "quoted-token"


def test_token_fingerprint_is_non_secret(monkeypatch):
    monkeypatch.setenv("TRADIER_TOKEN", "supersecrettoken9999")
    fp = tp.tradier_token_fingerprint()
    assert fp == "20:9999"
    assert "supersecret" not in fp


def test_token_fingerprint_none_when_unset(monkeypatch):
    monkeypatch.delenv("TRADIER_TOKEN", raising=False)
    with patch.object(tp, "_get_token", return_value=""):
        assert tp.tradier_token_fingerprint() == "none"


# ---------------------------------------------------------------------------
# Live status probe
# ---------------------------------------------------------------------------

def test_status_false_without_token():
    with patch.object(tp, "_get_token", return_value=""):
        ok, msg = tp.tradier_status()
    assert ok is False
    assert "No TRADIER_TOKEN" in msg


def test_status_reports_401_as_rejected():
    resp = type("R", (), {"status_code": 401})()
    with patch.object(tp, "_get_token", return_value="badtokenabcd"), \
         patch.object(tp, "_request", return_value=resp):
        ok, msg = tp.tradier_status()
    assert ok is False
    assert "401" in msg and "abcd" in msg   # last-4 shown, full key not
    assert "badtokenabcd" not in msg


def test_status_ok_on_200():
    resp = type("R", (), {"status_code": 200})()
    with patch.object(tp, "_get_token", return_value="goodtoken123"), \
         patch.object(tp, "_request", return_value=resp):
        ok, msg = tp.tradier_status()
    assert ok is True
    assert "connected" in msg.lower()


def test_status_falls_back_to_market_data_probe_on_403():
    """Market-data-only keys 403 on /user/profile but still work."""
    profile = type("R", (), {"status_code": 403})()
    market = type("R", (), {"status_code": 200})()
    with patch.object(tp, "_get_token", return_value="marketonly99"), \
         patch.object(tp, "_request", side_effect=[profile, market]):
        ok, _ = tp.tradier_status()
    assert ok is True


def test_status_reports_network_failure():
    import requests
    with patch.object(tp, "_get_token", return_value="tok"), \
         patch.object(tp, "_request", side_effect=requests.ConnectionError("boom")):
        ok, msg = tp.tradier_status()
    assert ok is False
    assert "Could not reach" in msg


def test_request_retries_transport_errors_then_succeeds():
    import requests
    resp = type("R", (), {"status_code": 200})()
    with patch.object(tp, "_get_token", return_value="tok"), \
         patch.object(tp.time, "sleep"), \
         patch.object(tp.requests, "get",
                      side_effect=[requests.ConnectionError("reset"), resp]) as mock_get:
        out = tp._request("/markets/quotes")
    assert out is resp
    assert mock_get.call_count == 2


def test_request_does_not_retry_http_status_errors():
    """A 401 is a real answer — retrying it just wastes time."""
    resp = type("R", (), {"status_code": 401})()
    with patch.object(tp, "_get_token", return_value="tok"), \
         patch.object(tp.requests, "get", return_value=resp) as mock_get:
        out = tp._request("/markets/quotes")
    assert out is resp
    assert mock_get.call_count == 1


def test_get_calls_raise_without_token(monkeypatch):
    monkeypatch.delenv("TRADIER_TOKEN", raising=False)
    with patch.object(tp, "_get_token", return_value=""):
        with pytest.raises(RuntimeError, match="TRADIER_TOKEN not configured"):
            tp._get("/markets/quotes", {"symbols": "TSLA"})


# ---------------------------------------------------------------------------
# get_quote
# ---------------------------------------------------------------------------

def test_get_quote_parses_single_quote_dict():
    fake = {
        "quotes": {
            "quote": {
                "symbol": "TSLA",
                "last": 375.50,
                "prevclose": 370.00,
                "volume": 50_000_000,
                "average_volume": 80_000_000,
                "close": 375.40,
            }
        }
    }
    with patch.object(tp, "_get", return_value=fake), \
         patch.object(tp, "_get_token", return_value="t"):
        q = tp.get_quote("tsla")
    assert isinstance(q, Quote)
    assert q.ticker == "TSLA"
    assert q.price == 375.50
    assert q.prev_close == 370.00
    assert q.change == pytest.approx(5.50)
    assert q.change_pct == pytest.approx(5.50 / 370.00)
    assert q.volume == 50_000_000
    assert q.avg_volume == 80_000_000
    assert q.market_cap is None
    assert q.earnings_date is None


def test_get_quote_handles_list_response():
    fake = {"quotes": {"quote": [{"symbol": "TSLA", "last": 100.0, "prevclose": 99.0}]}}
    with patch.object(tp, "_get", return_value=fake), \
         patch.object(tp, "_get_token", return_value="t"):
        q = tp.get_quote("TSLA")
    assert q.price == 100.0


def test_get_quote_raises_on_empty_response():
    with patch.object(tp, "_get", return_value={"quotes": {"quote": None}}), \
         patch.object(tp, "_get_token", return_value="t"):
        with pytest.raises(ValueError, match="No quote data"):
            tp.get_quote("BADTKR")


# ---------------------------------------------------------------------------
# get_expirations
# ---------------------------------------------------------------------------

def test_get_expirations_returns_sorted_tuple():
    fake = {"expirations": {"date": ["2026-06-19", "2026-05-15", "2026-05-29"]}}
    with patch.object(tp, "_get", return_value=fake), \
         patch.object(tp, "_get_token", return_value="t"):
        exps = tp.get_expirations("TSLA")
    assert exps == ("2026-05-15", "2026-05-29", "2026-06-19")


def test_get_expirations_coerces_single_date_string():
    fake = {"expirations": {"date": "2026-05-15"}}
    with patch.object(tp, "_get", return_value=fake), \
         patch.object(tp, "_get_token", return_value="t"):
        exps = tp.get_expirations("TSLA")
    assert exps == ("2026-05-15",)


def test_get_expirations_raises_when_empty():
    with patch.object(tp, "_get", return_value={"expirations": {"date": []}}), \
         patch.object(tp, "_get_token", return_value="t"):
        with pytest.raises(ValueError, match="No option expirations"):
            tp.get_expirations("BADTKR")


# ---------------------------------------------------------------------------
# get_option_chain — Greeks parsing is the key bit
# ---------------------------------------------------------------------------

def test_get_option_chain_parses_greeks_from_nested_block():
    chain = {
        "options": {
            "option": [
                {
                    "symbol": "TSLA260515C00400000",
                    "strike": 400.0,
                    "option_type": "call",
                    "bid": 5.50,
                    "ask": 5.70,
                    "last": 5.60,
                    "volume": 1234,
                    "open_interest": 5678,
                    "greeks": {
                        "delta": 0.42,
                        "gamma": 0.012,
                        "theta": -0.18,
                        "vega": 0.35,
                        "mid_iv": 0.456,
                    },
                },
                {
                    "symbol": "TSLA260515P00370000",
                    "strike": 370.0,
                    "option_type": "put",
                    "bid": 4.80,
                    "ask": 5.00,
                    "last": 4.90,
                    "volume": 9876,
                    "open_interest": 4321,
                    "greeks": {
                        "delta": -0.38,
                        "gamma": 0.011,
                        "theta": -0.17,
                        "vega": 0.33,
                        "mid_iv": 0.512,
                    },
                },
            ]
        }
    }
    quote = Quote(
        ticker="TSLA", price=380.0, prev_close=378.0,
        change=2.0, change_pct=0.005, volume=0, avg_volume=0,
    )
    with patch.object(tp, "_get", return_value=chain), \
         patch.object(tp, "get_quote", return_value=quote), \
         patch.object(tp, "_get_token", return_value="t"):
        calls, puts = tp.get_option_chain("TSLA", "2026-05-15")

    assert len(calls) == 1
    assert len(puts) == 1
    c = calls[0]
    p = puts[0]
    assert isinstance(c, OptionContract)
    assert c.strike == 400.0
    assert c.option_type == "call"
    assert c.bid == 5.50
    assert c.ask == 5.70
    assert c.delta == pytest.approx(0.42)
    assert c.gamma == pytest.approx(0.012)
    assert c.theta == pytest.approx(-0.18)
    assert c.vega == pytest.approx(0.35)
    assert c.implied_volatility == pytest.approx(0.456)
    assert c.open_interest == 5678
    assert p.option_type == "put"
    assert p.delta == pytest.approx(-0.38)
    assert p.implied_volatility == pytest.approx(0.512)


def test_get_option_chain_coerces_single_option_dict():
    """When only one strike exists, Tradier returns a dict, not a list."""
    chain = {
        "options": {
            "option": {
                "strike": 100.0,
                "option_type": "call",
                "bid": 1.0, "ask": 1.1, "last": 1.05,
                "volume": 10, "open_interest": 20,
                "greeks": {"delta": 0.5, "gamma": 0.01, "theta": -0.1, "vega": 0.2, "mid_iv": 0.3},
            }
        }
    }
    quote = Quote(
        ticker="X", price=100.0, prev_close=99.0,
        change=1.0, change_pct=0.01, volume=0, avg_volume=0,
    )
    with patch.object(tp, "_get", return_value=chain), \
         patch.object(tp, "get_quote", return_value=quote), \
         patch.object(tp, "_get_token", return_value="t"):
        calls, puts = tp.get_option_chain("X", "2026-05-15")
    assert len(calls) == 1
    assert calls[0].strike == 100.0
    assert puts == []


# ---------------------------------------------------------------------------
# get_historical
# ---------------------------------------------------------------------------

def test_get_historical_builds_dataframe():
    days = [
        {"date": f"2026-04-{d:02d}", "open": 100 + d, "high": 102 + d,
         "low": 99 + d, "close": 101 + d, "volume": 1_000_000 + d * 1000}
        for d in range(1, 30)
    ]
    fake = {"history": {"day": days}}
    with patch.object(tp, "_get", return_value=fake), \
         patch.object(tp, "_get_token", return_value="t"):
        df = tp.get_historical("TSLA", months=2)

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 29
    # Sorted ascending by date.
    assert df.index.is_monotonic_increasing


def test_get_historical_raises_when_too_few_rows():
    fake = {"history": {"day": [{"date": "2026-04-01", "open": 1, "high": 1,
                                  "low": 1, "close": 1, "volume": 1}]}}
    with patch.object(tp, "_get", return_value=fake), \
         patch.object(tp, "_get_token", return_value="t"):
        with pytest.raises(ValueError, match="Insufficient historical"):
            tp.get_historical("TSLA")


# ---------------------------------------------------------------------------
# Re-exports from data.common
# ---------------------------------------------------------------------------

def test_module_re_exports_dte_and_earnings_warning():
    """app/main.py calls dp.days_to_expiration and dp.earnings_warning,
    so tradier_provider must expose them at module level."""
    assert callable(tp.days_to_expiration)
    assert callable(tp.earnings_warning)
