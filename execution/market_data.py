"""yfinance downloads and model inputs for live signals."""

import logging
import math
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from features.pipeline import build_inference_features, normalize_market_columns

logger = logging.getLogger("multi_asset_trading_bot")

# The models are trained on finished daily candles, so build signals from
# completed bars only (today's still-forming bar is dropped before features).
USE_COMPLETED_BARS = True
_SPY_CLOSE_CACHE: pd.Series | None = None


def drop_partial_bar(data: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop today's still-forming daily bar so signals use completed candles only.

    Falls back to returning the data unchanged (with a warning) if the index is
    not a DatetimeIndex, so a format surprise never breaks trading.
    """
    if not USE_COMPLETED_BARS or len(data) < 2:
        return data
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning("Cannot detect partial bar, index is not dates label=%s", label)
        return data
    last = data.index[-1]
    if last.tzinfo is not None:
        last = last.tz_localize(None)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if last.normalize() >= today:
        logger.info("Dropped partial daily bar label=%s bar_date=%s", label, last.date())
        return data.iloc[:-1]
    return data


def get_spy_close_history() -> pd.Series:
    """Fetch one year of SPY closes for the shared benchmark feature."""
    global _SPY_CLOSE_CACHE
    if _SPY_CLOSE_CACHE is None:
        response = yf.download(
            "SPY",
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if response is None or response.empty:
            raise ValueError("yfinance returned no SPY data for benchmark feature")
        normalized = drop_partial_bar(
            normalize_market_columns(response), "SPY-benchmark"
        )
        _SPY_CLOSE_CACHE = pd.to_numeric(normalized["Close"], errors="coerce")
    return _SPY_CLOSE_CACHE


def get_latest_features(
    ticker: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, float, float]:
    """Fetch latest prices and calculate model inputs with the shared pipeline."""
    try:
        response = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if response is None or response.empty:
            raise ValueError(f"yfinance returned no data for {ticker}")
        market_data = normalize_market_columns(response)
        # Latest traded price (including today's forming bar) is used for sizing
        # and the bracket levels; the signal itself uses completed bars only.
        latest_price = float(
            pd.to_numeric(market_data["Close"], errors="coerce").dropna().iloc[-1]
        )
        signal_data = drop_partial_bar(market_data, ticker)
        spy_close = (
            pd.to_numeric(signal_data["Close"], errors="coerce")
            if ticker == "SPY"
            else get_spy_close_history()
        )
        frame = build_inference_features(signal_data, feature_columns, spy_close)
        if frame.empty:
            raise ValueError(f"Insufficient valid feature history for {ticker}")
        latest_features = frame.loc[:, feature_columns].tail(1).astype(float)
        reference_price = latest_price
        current_atr = float(frame["ATR_14"].iloc[-1])
        if not math.isfinite(reference_price) or reference_price <= 0:
            raise ValueError(f"Invalid reference price for {ticker}: {reference_price}")
        if not math.isfinite(current_atr) or current_atr <= 0:
            raise ValueError(f"Invalid ATR-14 for {ticker}: {current_atr}")
        return latest_features, reference_price, current_atr
    except Exception as exc:
        logger.exception("Failed to fetch or prepare market data ticker=%s", ticker)
        raise RuntimeError(f"Unable to prepare market features for {ticker}") from exc
