"""Shared feature contracts and market-data transformations."""

from .contract import (
    CRYPTO_FEATURE_COLUMNS,
    ETH_FEATURE_COLUMNS,
    FEATURE_COLUMNS_BY_TICKER,
    LEGACY_CRYPTO_FEATURE_COLUMNS,
    LEGACY_STOCK_FEATURE_COLUMNS,
    STOCK_FEATURE_COLUMNS,
)
from .pipeline import (
    add_legacy_features,
    build_inference_features,
    build_regression_training_frame,
    create_legacy_spy_dataset,
    normalize_market_columns,
)

__all__ = [
    "CRYPTO_FEATURE_COLUMNS",
    "ETH_FEATURE_COLUMNS",
    "FEATURE_COLUMNS_BY_TICKER",
    "LEGACY_CRYPTO_FEATURE_COLUMNS",
    "LEGACY_STOCK_FEATURE_COLUMNS",
    "STOCK_FEATURE_COLUMNS",
    "add_legacy_features",
    "build_inference_features",
    "build_regression_training_frame",
    "create_legacy_spy_dataset",
    "normalize_market_columns",
]
