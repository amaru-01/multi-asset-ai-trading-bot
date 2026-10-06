"""Shared normalization and feature engineering for training and inference."""

import numpy as np
import pandas as pd
import ta

from .contract import (
    LEGACY_CRYPTO_FEATURE_COLUMNS,
    LEGACY_STOCK_FEATURE_COLUMNS,
    STOCK_FEATURE_COLUMNS,
)


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def normalize_market_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance's optional multi-index response into sorted OHLCV rows."""
    required = set(OHLCV_COLUMNS)
    if isinstance(data.columns, pd.MultiIndex):
        for level in range(data.columns.nlevels):
            names = data.columns.get_level_values(level)
            if required.issubset(set(names)):
                data = data.copy()
                data.columns = names
                break
        else:
            raise ValueError("Could not identify OHLCV columns in market data")

    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Market data is missing columns: {sorted(missing)}")
    data = data.loc[:, OHLCV_COLUMNS].copy()
    index = pd.DatetimeIndex(data.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    data.index = index.normalize()
    data.sort_index(inplace=True)
    data = data.groupby(level=0).last()
    for column in OHLCV_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=OHLCV_COLUMNS, inplace=True)
    return data


def add_regression_features(
    data: pd.DataFrame,
    spy_close: pd.Series,
) -> pd.DataFrame:
    """Calculate the five shared inputs for the three-day return regressor."""
    data = normalize_market_columns(data)
    close = data["Close"]
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    atr_14 = ta.volatility.average_true_range(
        data["High"], data["Low"], close, window=14
    )
    benchmark_close = pd.to_numeric(spy_close, errors="coerce")
    benchmark_roc_5 = benchmark_close.pct_change(5).reindex(data.index)

    data["SMA_20_50_Ratio"] = sma_20 / sma_50
    data["ATR_Normalized"] = atr_14 / close
    data["RSI_14"] = ta.momentum.rsi(close, window=14)
    data["ROC_5"] = close.pct_change(5)
    data["SPY_ROC_5"] = benchmark_roc_5.fillna(0.0)
    data["ATR_14"] = atr_14
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


def build_regression_training_frame(
    data: pd.DataFrame,
    spy_close: pd.Series,
    horizon_days: int = 3,
    return_threshold: float = 0.005,
) -> pd.DataFrame:
    """Add complete forward-return regression targets and drop rolling/trailing NaNs."""
    frame = add_regression_features(data, spy_close)
    close = frame["Close"]
    frame["Target_Return"] = (close.shift(-horizon_days) - close) / close
    frame.dropna(subset=STOCK_FEATURE_COLUMNS + ["Target_Return"], inplace=True)
    frame["Target_Direction"] = (
        frame["Target_Return"] > return_threshold
    ).astype("int8")
    return frame


def add_legacy_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add historical stock and crypto feature names needed by legacy JSON models."""
    frame = normalize_market_columns(data)
    close = frame["Close"]
    frame["RSI"] = ta.momentum.rsi(close, window=14)
    frame["MACD"] = ta.trend.macd_diff(close)
    frame["ATR"] = ta.volatility.average_true_range(
        frame["High"], frame["Low"], close, window=14
    )
    frame["Return_1D"] = close.pct_change(1)
    frame["Vol_Change"] = frame["Volume"].pct_change(1)
    frame["RSI_lag1"] = frame["RSI"].shift(1)
    frame["Return_lag1"] = frame["Return_1D"].shift(1)
    frame["SMA_10"] = close.rolling(window=10).mean()
    frame["SMA_50"] = close.rolling(window=50).mean()
    frame["MACD_signal"] = ta.trend.macd_signal(close)
    frame["BB_high"] = ta.volatility.bollinger_hband(close)
    frame["BB_low"] = ta.volatility.bollinger_lband(close)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def build_inference_features(
    data: pd.DataFrame,
    feature_columns: list[str],
    spy_close: pd.Series,
) -> pd.DataFrame:
    """Build both current and legacy columns, returning a finite aligned feature frame."""
    frame = add_regression_features(data, spy_close)
    legacy_columns = set(LEGACY_STOCK_FEATURE_COLUMNS + LEGACY_CRYPTO_FEATURE_COLUMNS)
    if set(feature_columns).issubset(legacy_columns):
        legacy = add_legacy_features(data)
        for column in feature_columns:
            frame[column] = legacy[column]
    missing = set(feature_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Unsupported model features: {sorted(missing)}")
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.dropna(subset=feature_columns + ["Close", "ATR_14"], inplace=True)
    return frame


def create_legacy_spy_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Build the historical one-day SPY CSV using the shared legacy indicators."""
    frame = add_legacy_features(data)
    frame["Tomorrow_Return"] = frame["Close"].pct_change(1).shift(-1)
    frame.dropna(
        subset=LEGACY_STOCK_FEATURE_COLUMNS + ["Tomorrow_Return"],
        inplace=True,
    )
    frame["Target"] = (frame.pop("Tomorrow_Return") > 0).astype("int8")
    return frame.loc[:, LEGACY_STOCK_FEATURE_COLUMNS + ["Target"]]
