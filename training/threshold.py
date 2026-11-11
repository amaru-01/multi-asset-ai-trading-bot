"""Prediction-threshold selection on a trailing validation slice of the training data."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config import FORWARD_HORIZON_DAYS
from models import XGBRegressorModel

from .settings import (
    MIN_THRESHOLD_PRECISION,
    MIN_THRESHOLD_RECALL,
    THRESHOLD_PERCENTILES,
)

logger = logging.getLogger("multi_asset_trading_bot")


def select_prediction_threshold(
    training_rows: pd.DataFrame,
    feature_columns: list[str],
    asset: dict,
) -> tuple[float, dict]:
    """Select a percentile cutoff on a trailing validation slice of training only."""
    validation_start = int(len(training_rows) * 0.75)
    fit_end = max(0, validation_start - FORWARD_HORIZON_DAYS)
    fit_rows = training_rows.iloc[:fit_end]
    validation_rows = training_rows.iloc[validation_start:]
    if fit_rows.empty or validation_rows.empty:
        raise ValueError(f"Insufficient rows to calibrate threshold for {asset['ticker']}")

    validation_model = XGBRegressorModel(
        asset["symbol"], feature_columns=feature_columns
    ).fit(fit_rows[feature_columns], fit_rows["Target_Return"])
    validation_predictions = validation_model.predict_values(
        validation_rows[feature_columns]
    )
    validation_targets = validation_rows["Target_Direction"].to_numpy(dtype="int8")
    baseline = float(max(np.mean(validation_targets), 1 - np.mean(validation_targets)))

    candidates = []
    for percentile in THRESHOLD_PERCENTILES:
        threshold = float(np.percentile(validation_predictions, percentile))
        predicted_signal = (validation_predictions > threshold).astype("int8")
        accuracy = float(accuracy_score(validation_targets, predicted_signal))
        precision = float(
            precision_score(validation_targets, predicted_signal, zero_division=0)
        )
        recall = float(recall_score(validation_targets, predicted_signal, zero_division=0))
        candidates.append(
            {
                "threshold": threshold,
                "percentile": percentile,
                "accuracy": accuracy,
                "margin": accuracy - baseline,
                "precision": precision,
                "recall": recall,
                "balanced": (
                    precision > MIN_THRESHOLD_PRECISION
                    and recall > MIN_THRESHOLD_RECALL
                ),
            }
        )

    eligible = [candidate for candidate in candidates if candidate["balanced"]]
    pool = eligible or candidates
    selected = max(
        pool,
        key=lambda candidate: (
            candidate["margin"],
            min(candidate["precision"], candidate["recall"]),
        ),
    )
    logger.info(
        "Threshold calibration ticker=%s percentile=%d threshold=%.6f "
        "validation_accuracy=%.4f baseline=%.4f margin=%+.4f precision=%.4f "
        "recall=%.4f balanced=%s",
        asset["ticker"],
        selected["percentile"],
        selected["threshold"],
        selected["accuracy"],
        baseline,
        selected["margin"],
        selected["precision"],
        selected["recall"],
        bool(eligible),
    )
    return selected["threshold"], selected
