"""
Technical indicators module.

All functions accept pandas Series / DataFrames with numeric data and return
numeric results. No I/O or side effects.

Indicators available:
  - SMA (any period)
  - EMA (any period)
  - RSI (Wilder's smoothing)
  - ATR (Wilder's smoothing)
  - Bollinger Bands (SMA ± N * stddev)
  - MACD (fast EMA - slow EMA, signal EMA, histogram)
  - Average Volume
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.models import TechnicalIndicators


# ---------------------------------------------------------------------------
# Individual indicator calculations
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (EMA-style).
    Returns a Series of values in [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def average_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    return volume.rolling(window=period, min_periods=1).mean()


@dataclass
class BollingerBands:
    middle: pd.Series
    upper: pd.Series
    lower: pd.Series
    bandwidth: pd.Series
    pct_b: pd.Series


def bollinger_bands(series: pd.Series, period: int = 20, n_std: float = 2.0) -> BollingerBands:
    """Bollinger Bands: middle ± n_std * rolling stddev."""
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    bandwidth = (upper - lower) / mid
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return BollingerBands(middle=mid, upper=upper, lower=lower,
                          bandwidth=bandwidth, pct_b=pct_b)


@dataclass
class MACDData:
    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> MACDData:
    """MACD: fast EMA - slow EMA, signal EMA, histogram."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return MACDData(macd_line=macd_line, signal_line=signal_line, histogram=histogram)


# ---------------------------------------------------------------------------
# Full indicators dataclass — carries series for charting
# ---------------------------------------------------------------------------

@dataclass
class FullIndicators:
    """
    Full indicator series for charting, plus scalar TechnicalIndicators
    snapshot used by the scoring engine (unchanged interface).
    """
    snapshot: TechnicalIndicators
    dates: pd.DatetimeIndex
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    sma20: pd.Series
    sma50: pd.Series
    rsi14: pd.Series
    atr14: pd.Series
    bb: BollingerBands
    macd_data: MACDData
    volume: pd.Series
    avg_volume: pd.Series


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> TechnicalIndicators:
    """
    Backward-compatible entry point used by the scoring engine.
    Returns scalar TechnicalIndicators snapshot only.
    """
    return _calculate_full(df).snapshot


def calculate_full_indicators(df: pd.DataFrame) -> FullIndicators:
    """
    Extended entry point for chart.py.
    Returns FullIndicators with all series for plotting.
    """
    return _calculate_full(df)


def _calculate_full(df: pd.DataFrame) -> FullIndicators:
    if len(df) < 20:
        raise ValueError("Need at least 20 rows of historical data.")

    open_  = df["Open"]
    high   = df["High"]
    low    = df["Low"]
    close  = df["Close"]
    volume = df["Volume"]

    sma20_s  = sma(close, 20)
    sma50_s  = sma(close, 50)
    rsi14_s  = rsi(close, 14)
    atr14_s  = atr(high, low, close, 14)
    avgvol_s = average_volume(volume, 20)
    bb_data  = bollinger_bands(close, period=20, n_std=2.0)
    macd_d   = macd(close, fast=12, slow=26, signal=9)

    current_price = float(close.iloc[-1])

    def _last(s: pd.Series, default: float) -> float:
        v = s.iloc[-1]
        return float(v) if not pd.isna(v) else default

    snapshot = TechnicalIndicators(
        sma_20=_last(sma20_s, current_price),
        sma_50=_last(sma50_s, current_price),
        rsi_14=_last(rsi14_s, 50.0),
        atr_14=_last(atr14_s, 0.0),
        avg_volume_20=_last(avgvol_s, 0.0),
        current_price=current_price,
        weekly_atr_est=_last(atr14_s, 0.0) * math.sqrt(5),
    )

    return FullIndicators(
        snapshot=snapshot,
        dates=df.index,
        open=open_,
        high=high,
        low=low,
        close=close,
        sma20=sma20_s,
        sma50=sma50_s,
        rsi14=rsi14_s,
        atr14=atr14_s,
        bb=bb_data,
        macd_data=macd_d,
        volume=volume,
        avg_volume=avgvol_s,
    )
