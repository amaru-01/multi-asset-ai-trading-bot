"""Model wrappers with backward-compatible native XGBoost artifact loading."""

from pathlib import Path

import xgboost as xgb

from features.contract import LEGACY_CRYPTO_FEATURE_COLUMNS, LEGACY_STOCK_FEATURE_COLUMNS

from .base_model import BaseTradingModel
from .classifier import XGBClassifierModel
from .regressor import XGBRegressorModel


def load_trading_model(
    path: Path,
    symbol: str,
    expected_features: list[str],
) -> BaseTradingModel:
    """Choose the classifier wrapper only for recognized legacy feature contracts."""
    booster = xgb.Booster()
    booster.load_model(str(path))
    trained_features = booster.feature_names or []
    legacy_contracts = (
        LEGACY_STOCK_FEATURE_COLUMNS,
        LEGACY_CRYPTO_FEATURE_COLUMNS,
    )
    if trained_features and trained_features not in legacy_contracts:
        if trained_features != expected_features:
            raise ValueError(
                "Model feature contract mismatch: "
                f"model={trained_features}, execution={expected_features}"
            )
    model: BaseTradingModel
    if trained_features in legacy_contracts:
        model = XGBClassifierModel(
            feature_columns=trained_features,
        )
    else:
        model = XGBRegressorModel(symbol=symbol, feature_columns=expected_features)
    model.load_artifact(path)
    return model


__all__ = [
    "BaseTradingModel",
    "XGBClassifierModel",
    "XGBRegressorModel",
    "load_trading_model",
]
