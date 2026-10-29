"""Tests for earnings-date lookup in data/provider.py (Auto mode)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from data import provider

TODAY = date(2026, 10, 29)


def test_yf_calendar_dict_format():
    t = SimpleNamespace(calendar={"Earnings Date": [date(2026, 10, 29), date(2026, 10, 30)]})
    assert provider._yf_next_earnings(t, today=TODAY) == "2026-10-29"


def test_yf_calendar_ignores_past_dates():
    t = SimpleNamespace(calendar={"Earnings Date": [date(2026, 7, 30)]})
    assert provider._yf_next_earnings(t, today=TODAY) is None


def test_yf_calendar_legacy_dataframe_format():
    cal = pd.DataFrame({"Earnings Date": [pd.Timestamp("2026-11-04")]})
    assert provider._yf_next_earnings(SimpleNamespace(calendar=cal), today=TODAY) == "2026-11-04"


def test_yf_calendar_missing_or_broken():
    assert provider._yf_next_earnings(SimpleNamespace(calendar=None), today=TODAY) is None
    assert provider._yf_next_earnings(SimpleNamespace(calendar={}), today=TODAY) is None

    class Boom:
        @property
        def calendar(self):
            raise RuntimeError("blocked")
    assert provider._yf_next_earnings(Boom(), today=TODAY) is None


def test_get_earnings_date_prefers_tradier_then_falls_back_to_yfinance():
    with patch.object(provider, "_try_tradier", return_value="2026-10-29"), \
         patch.object(provider, "_ticker") as yf:
        assert provider.get_earnings_date("AMZN") == "2026-10-29"
        yf.assert_not_called()
    future = date.today().replace(year=date.today().year + 1)
    with patch.object(provider, "_try_tradier", return_value=None), \
         patch.object(provider, "_ticker",
                      return_value=SimpleNamespace(calendar={"Earnings Date": [future]})):
        assert provider.get_earnings_date("AMZN") == future.isoformat()
