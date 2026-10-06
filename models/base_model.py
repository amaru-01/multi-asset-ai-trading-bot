"""Abstract interface implemented by current and legacy trading models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseTradingModel(ABC):
    """Common fitting, inference, and persistence contract for bot models."""

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: Any) -> "BaseTradingModel":
        """Fit the underlying estimator and return this wrapper."""

    @abstractmethod
    def predict_signal(self, features: pd.DataFrame) -> dict[str, str | float]:
        """Return a BUY/HOLD signal plus the raw estimator output and cutoff."""

    @abstractmethod
    def save_artifact(self, path: Path) -> None:
        """Save the estimator in native XGBoost JSON format."""

    @abstractmethod
    def load_artifact(self, path: Path) -> "BaseTradingModel":
        """Load the estimator and any supported metadata from native JSON."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str] | None:
        """Return training feature names when recorded in the model artifact."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Human-readable model family for notification and diagnostics."""
