"""Wrapper for legacy XGBoost classifier artifacts."""

from pathlib import Path

import pandas as pd
import xgboost as xgb

from config import LEGACY_CLASSIFIER_THRESHOLD
from .base_model import BaseTradingModel


class XGBClassifierModel(BaseTradingModel):
    """Adapt a native legacy XGBClassifier to the shared model interface."""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        decision_threshold: float = LEGACY_CLASSIFIER_THRESHOLD,
        estimator: xgb.XGBClassifier | None = None,
    ) -> None:
        self._feature_columns = feature_columns
        self.decision_threshold = decision_threshold
        self.estimator = estimator or xgb.XGBClassifier()

    def fit(self, features: pd.DataFrame, target) -> "XGBClassifierModel":
        self.estimator.fit(features, target)
        return self

    def predict_signal(self, features: pd.DataFrame) -> dict[str, str | float]:
        probability = float(self.estimator.predict_proba(features)[0, 1])
        return {
            "signal": "BUY" if probability >= self.decision_threshold else "HOLD",
            "raw_prediction": probability,
            "threshold": float(self.decision_threshold),
        }

    def save_artifact(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.estimator.get_booster().set_attr(
            decision_threshold=str(self.decision_threshold)
        )
        self.estimator.save_model(str(path))

    def load_artifact(self, path: Path) -> "XGBClassifierModel":
        self.estimator.load_model(str(path))
        self._feature_columns = self.estimator.get_booster().feature_names
        saved_threshold = self.estimator.get_booster().attributes().get(
            "decision_threshold"
        )
        if saved_threshold is not None:
            self.decision_threshold = float(saved_threshold)
        return self

    @property
    def feature_names(self) -> list[str] | None:
        return self._feature_columns or self.estimator.get_booster().feature_names

    @property
    def model_type(self) -> str:
        return "Legacy Classifier"
