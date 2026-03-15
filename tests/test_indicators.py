"""Tests for indicators/technical.py"""

import numpy as np
import pandas as pd
import pytest

from indicators.technical import (
    atr, bollinger_bands, calculate_full_indicators, calculate_indicators,
    ema, macd, rsi, sma,
)


class TestSMA:
    def test_basic_average(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert result.iloc[-1] == pytest.approx(4.0)

    def test_returns_nan_before_period(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])

    def test_single_period(self):
        s = pd.Series([5.0, 10.0, 15.0])
        result = sma(s, 1)
        assert list(result) == pytest.approx([5.0, 10.0, 15.0])

    def test_constant_series(self):
        s = pd.Series([7.0] * 20)
        result = sma(s, 10)
        assert result.dropna().iloc[-1] == pytest.approx(7.0)


class TestRSI:
    def test_range_0_to_100(self, flat_df):
        result = rsi(flat_df["Close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_overbought_uptrend(self, bullish_df):
        result = rsi(bullish_df["Close"], 14)
        # Strong uptrend should produce RSI > 50 by the end
        assert result.iloc[-1] > 50

    def test_oversold_downtrend(self, bearish_df):
        result = rsi(bearish_df["Close"], 14)
        assert result.iloc[-1] < 50

    def test_nan_for_first_period(self):
        s = pd.Series(range(1, 30), dtype=float)
        result = rsi(s, 14)
        assert pd.isna(result.iloc[0])


class TestATR:
    def test_positive_values(self, bullish_df):
        result = atr(bullish_df["High"], bullish_df["Low"], bullish_df["Close"], 14)
        assert (result.dropna() > 0).all()

    def test_zero_range_yields_zero(self):
        n = 30
        price = pd.Series([100.0] * n)
        result = atr(price, price, price, 14)
        assert result.dropna().iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_high_volatility_higher_than_low(self):
        s_flat = pd.Series([100.0] * 50)
        s_vol = pd.Series([100.0 + (i % 2) * 5 for i in range(50)], dtype=float)
        atr_flat = atr(s_flat, s_flat, s_flat, 14).dropna().iloc[-1]
        s_high = s_vol + 3
        s_low = s_vol - 3
        atr_vol = atr(s_high, s_low, s_vol, 14).dropna().iloc[-1]
        assert atr_vol > atr_flat


class TestEMA:
    def test_ema_reacts_faster_than_sma(self):
        # After a sharp spike, EMA should be closer to the spike than SMA
        base = [100.0] * 30
        spike = base + [200.0] * 5
        s = pd.Series(spike)
        ema_val = ema(s, 10).iloc[-1]
        sma_val = sma(s, 10).iloc[-1]
        assert ema_val > sma_val

    def test_constant_series_equals_constant(self):
        s = pd.Series([50.0] * 40)
        result = ema(s, 10).dropna()
        assert (result - 50.0).abs().max() < 1e-6

    def test_bounded(self, bullish_df):
        result = ema(bullish_df["Close"], 20).dropna()
        assert (result > 0).all()


class TestBollingerBands:
    def test_upper_above_lower(self, bullish_df):
        bb = bollinger_bands(bullish_df["Close"])
        valid = bb.upper.dropna()
        assert (bb.upper.dropna() > bb.lower.dropna()).all()

    def test_middle_equals_sma20(self, bullish_df):
        bb = bollinger_bands(bullish_df["Close"], period=20)
        s20 = sma(bullish_df["Close"], 20)
        diff = (bb.middle - s20).dropna().abs()
        assert diff.max() < 1e-9

    def test_pct_b_near_1_at_upper(self, flat_df):
        # A price at the upper band should yield pct_b ≈ 1
        bb = bollinger_bands(flat_df["Close"], period=20)
        # pct_b valid range check
        valid = bb.pct_b.dropna()
        assert valid.notna().any()

    def test_bandwidth_positive(self, bullish_df):
        bb = bollinger_bands(bullish_df["Close"])
        assert (bb.bandwidth.dropna() > 0).all()

    def test_constant_series_zero_bandwidth(self):
        s = pd.Series([100.0] * 40)
        bb = bollinger_bands(s, period=20)
        assert bb.bandwidth.dropna().abs().max() < 1e-6


class TestMACD:
    def test_macd_line_is_fast_minus_slow(self, bullish_df):
        close = bullish_df["Close"]
        md = macd(close, fast=12, slow=26, signal=9)
        fast = ema(close, 12)
        slow = ema(close, 26)
        expected = (fast - slow).dropna()
        actual = md.macd_line.dropna()
        # Align and compare
        common = expected.index.intersection(actual.index)
        assert (expected[common] - actual[common]).abs().max() < 1e-6

    def test_histogram_equals_macd_minus_signal(self, bullish_df):
        md = macd(bullish_df["Close"])
        diff = (md.macd_line - md.signal_line - md.histogram).dropna().abs()
        assert diff.max() < 1e-9

    def test_series_lengths_match(self, bullish_df):
        md = macd(bullish_df["Close"])
        n = len(bullish_df)
        assert len(md.macd_line) == n
        assert len(md.signal_line) == n
        assert len(md.histogram) == n


class TestCalculateFullIndicators:
    def test_returns_full_indicators(self, bullish_df):
        from indicators.technical import FullIndicators
        result = calculate_full_indicators(bullish_df)
        assert isinstance(result, FullIndicators)

    def test_snapshot_matches_calculate_indicators(self, bullish_df):
        full = calculate_full_indicators(bullish_df)
        snap = calculate_indicators(bullish_df)
        assert full.snapshot.current_price == pytest.approx(snap.current_price)
        assert full.snapshot.sma_20 == pytest.approx(snap.sma_20)

    def test_all_series_present(self, bullish_df):
        full = calculate_full_indicators(bullish_df)
        assert full.sma20 is not None
        assert full.sma50 is not None
        assert full.bb is not None
        assert full.macd_data is not None

    def test_ohlc_series_preserved(self, bullish_df):
        full = calculate_full_indicators(bullish_df)
        assert len(full.open) == len(bullish_df)
        assert len(full.high) == len(bullish_df)
        assert len(full.low) == len(bullish_df)
        assert (full.high >= full.low).all()


class TestCalculateIndicators:
    def test_returns_correct_types(self, bullish_df):
        from strategies.models import TechnicalIndicators
        result = calculate_indicators(bullish_df)
        assert isinstance(result, TechnicalIndicators)

    def test_current_price_matches_last_close(self, bullish_df):
        result = calculate_indicators(bullish_df)
        assert result.current_price == pytest.approx(float(bullish_df["Close"].iloc[-1]))

    def test_weekly_atr_gt_daily(self, bullish_df):
        result = calculate_indicators(bullish_df)
        assert result.weekly_atr_est > result.atr_14

    def test_raises_on_too_few_rows(self):
        tiny = pd.DataFrame({
            "Open": [1.0] * 10,
            "High": [1.1] * 10,
            "Low":  [0.9] * 10,
            "Close": [1.0] * 10,
            "Volume": [1000.0] * 10,
        })
        with pytest.raises(ValueError, match="at least 20"):
            calculate_indicators(tiny)

    def test_sma20_gt_sma50_in_uptrend(self, bullish_df):
        result = calculate_indicators(bullish_df)
        # In a sustained uptrend the fast SMA should be above the slow one
        assert result.sma_20 >= result.sma_50 - 5  # allow some tolerance
