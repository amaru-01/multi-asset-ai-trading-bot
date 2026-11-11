"""Model paths, trading costs and market-history download for retraining."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import MIN_TRAINING_ROWS, MODEL_DIR
from features.pipeline import normalize_market_columns

from .settings import ROUND_TRIP_COST


def resolve_model_path(asset: dict) -> Path:
    path = Path(asset["model_path"])
    return path if path.is_absolute() else MODEL_DIR / path


def trading_cost(asset: dict) -> float:
    return ROUND_TRIP_COST.get(asset["type"], ROUND_TRIP_COST["crypto"])


def download_history(ticker: str) -> pd.DataFrame:
    """Download and normalize five years of daily market history."""
    response = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if response is None or response.empty:
        raise ValueError(f"yfinance returned no history for {ticker}")
    history = normalize_market_columns(response)
    # Today's daily bar is still forming (crypto trades 24/7), so its close keeps moving
    # during the day. Using it would make targets, back-tests and marked-to-market open
    # trades change from run to run. Keep completed bars only.
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if len(history) and pd.DatetimeIndex(history.index)[-1] >= today:
        history = history[pd.DatetimeIndex(history.index) < today]
    if len(history) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Insufficient valid history for {ticker}: {len(history)} rows; "
            f"need at least {MIN_TRAINING_ROWS}"
        )
    return history
