"""Wrapper for current three-day forward-return XGBoost regressors."""

from pathlib import Path

import pandas as pd
import xgboost as xgb

from config import ASSETS, REGRESSION_RETURN_THRESHOLD
from .base_model import BaseTradingModel


class XGBRegressorModel(BaseTradingModel):
    """Fit, signal, and persist a configured three-day return regressor."""

    def __init__(
        self,
        symbol: str,
        feature_columns: list[str] | None = None,
        decision_threshold: float = REGRESSION_RETURN_THRESHOLD,
        estimator: xgb.XGBRegressor | None = None,
    ) -> None:
        self.symbol = symbol
        self._feature_columns = feature_columns
        self.decision_threshold = decision_threshold
        self.estimator = estimator or self._new_estimator()

    def _new_estimator(self) -> xgb.XGBRegressor:
        asset = next((item for item in ASSETS if item["symbol"] == self.symbol), None)
        parameters = asset["model_parameters"] if asset else {}
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
            **parameters,
        )

    def fit(self, features: pd.DataFrame, target) -> "XGBRegressorModel":
        self.estimator.fit(features, target)
        self._feature_columns = list(features.columns)
        return self

    def predict_signal(self, features: pd.DataFrame) -> dict[str, str | float]:
        predicted_return = float(self.predict_values(features)[0])
        return {
            "signal": "BUY" if predicted_return > self.decision_threshold else "HOLD",
            "raw_prediction": predicted_return,
            "threshold": float(self.decision_threshold),
        }

    def predict_values(self, features: pd.DataFrame):
        """Return predicted continuous forward returns for validation batches."""
        return self.estimator.predict(features)

    def save_artifact(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.estimator.get_booster().set_attr(
            prediction_type="three_day_forward_return",
            decision_threshold=str(self.decision_threshold),
        )
        self.estimator.save_model(str(path))

    def load_artifact(self, path: Path) -> "XGBRegressorModel":
        self.estimator.load_model(str(path))
        self._feature_columns = self.estimator.get_booster().feature_names
        attributes = self.estimator.get_booster().attributes()
        saved_threshold = attributes.get("decision_threshold")
        if saved_threshold is not None:
            self.decision_threshold = float(saved_threshold)
        return self

    @property
    def feature_names(self) -> list[str] | None:
        return self._feature_columns or self.estimator.get_booster().feature_names

    @property
    def model_type(self) -> str:
        return "3-Day Regressor"
