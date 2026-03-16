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


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average Directional Index (Wilder's smoothing).
    Returns values 0–100: >25 = trending, <20 = ranging.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    smooth_tr   = tr.ewm(com=period - 1, min_periods=period).mean()
    smooth_plus = plus_dm.ewm(com=period - 1, min_periods=period).mean()
    smooth_minus = minus_dm.ewm(com=period - 1, min_periods=period).mean()

    plus_di  = 100 * smooth_plus  / smooth_tr.replace(0, np.nan)
    minus_di = 100 * smooth_minus / smooth_tr.replace(0, np.nan)

    di_sum  = (plus_di + minus_di).replace(0, np.nan)
    dx      = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(com=period - 1, min_periods=period).mean()


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Cumulative VWAP anchored to the start of the data window.
    Uses typical price = (H + L + C) / 3.
    """
    typical = (high + low + close) / 3
    cum_vol = volume.cumsum().replace(0, np.nan)
    return (typical * volume).cumsum() / cum_vol


def ttm_squeeze(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> pd.Series:
    """
    TTM Squeeze: True when Bollinger Bands are inside Keltner Channels.
    Squeeze ON  (True)  = low-volatility compression — avoid selling yet.
    Squeeze OFF (False) = expansion starting — better time to sell premium.
    """
    # Bollinger Bands
    bb_mid   = sma(close, bb_period)
    bb_std   = close.rolling(bb_period, min_periods=bb_period).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    # Keltner Channels (ATR-based)
    kc_mid   = sma(close, kc_period)
    kc_atr   = atr(high, low, close, kc_period)
    kc_upper = kc_mid + kc_mult * kc_atr
    kc_lower = kc_mid - kc_mult * kc_atr

    # Squeeze = BB entirely inside KC
    return (bb_upper < kc_upper) & (bb_lower > kc_lower)


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
    adx14: pd.Series
    vwap_series: pd.Series
    squeeze: pd.Series      # bool Series: True = squeeze on


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

    sma20_s   = sma(close, 20)
    sma50_s   = sma(close, 50)
    rsi14_s   = rsi(close, 14)
    atr14_s   = atr(high, low, close, 14)
    avgvol_s  = average_volume(volume, 20)
    bb_data   = bollinger_bands(close, period=20, n_std=2.0)
    macd_d    = macd(close, fast=12, slow=26, signal=9)
    adx14_s   = adx(high, low, close, period=14)
    vwap_s    = vwap(high, low, close, volume)
    squeeze_s = ttm_squeeze(high, low, close)

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
        adx_14=_last(adx14_s, 25.0),
        squeeze_on=bool(squeeze_s.iloc[-1]) if not pd.isna(squeeze_s.iloc[-1]) else False,
        vwap=_last(vwap_s, current_price),
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
        adx14=adx14_s,
        vwap_series=vwap_s,
        squeeze=squeeze_s,
    )
