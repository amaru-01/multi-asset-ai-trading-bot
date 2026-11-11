"""Train a candidate model and run it through every gate."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from config import (
    FORWARD_HORIZON_DAYS,
    MIN_TRAINING_ROWS,
    REGRESSION_RETURN_THRESHOLD,
    TEST_FRACTION,
)
from features.pipeline import build_regression_training_frame
from models import XGBRegressorModel

from .data import trading_cost
from .gates import (
    log_prediction_calibration,
    quality_gate,
    simulate_strategy,
    walk_forward_check,
)
from .settings import ENFORCE_EVIDENCE_GATES, REGRESSOR_ENTRY_MIN
from .threshold import select_prediction_threshold

logger = logging.getLogger("multi_asset_trading_bot")


def _prepare_frame(asset: dict, history: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """Build the labelled feature frame and require enough complete rows."""
    feature_columns = asset["feature_columns"]
    frame = build_regression_training_frame(
        history,
        spy_close,
        horizon_days=FORWARD_HORIZON_DAYS,
        return_threshold=REGRESSION_RETURN_THRESHOLD,
    )
    frame.dropna(
        subset=feature_columns + ["Target_Return", "Target_Direction"], inplace=True
    )
    if len(frame) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Insufficient complete feature rows for {asset['ticker']}: {len(frame)}"
        )
    return frame


def _split_holdout(asset: dict, frame: pd.DataFrame, feature_columns: list[str]):
    """Purged chronological split. Returns (training_rows, holdout_rows, y_true)."""
    split_at = int(len(frame) * (1 - TEST_FRACTION))
    training_rows = frame.iloc[: max(0, split_at - FORWARD_HORIZON_DAYS)]
    holdout_rows = frame.iloc[split_at:]
    y_true = holdout_rows["Target_Direction"].to_numpy(dtype="int8")
    if training_rows.empty or len(np.unique(y_true)) < 2:
        raise ValueError(f"Training or holdout labels are insufficient for {asset['ticker']}")

    zero_variance = [
        name for name in feature_columns if training_rows[name].nunique(dropna=True) <= 1
    ]
    if zero_variance:
        raise ValueError(f"Zero-variance features for {asset['ticker']}: {zero_variance}")
    return training_rows, holdout_rows, y_true


def _holdout_metrics(
    asset: dict,
    holdout_rows: pd.DataFrame,
    y_true: np.ndarray,
    predicted_returns: np.ndarray,
    threshold: float,
    threshold_metrics: dict,
) -> dict:
    """Score the holdout predictions."""
    y_pred = (predicted_returns > threshold).astype("int8")

    accuracy = float(accuracy_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, predicted_returns))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    baseline = float(max(np.mean(y_true), 1 - np.mean(y_true)))
    return {
        "ticker": asset["ticker"],
        "rows": len(holdout_rows),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "majority_baseline": baseline,
        "accuracy_margin": accuracy - baseline,
        "decision_threshold": threshold,
        "threshold_percentile": threshold_metrics["percentile"],
        "threshold_validation_precision": threshold_metrics["precision"],
        "threshold_validation_recall": threshold_metrics["recall"],
        "threshold_validation_balanced": threshold_metrics["balanced"],
    }


def _log_holdout(asset: dict, metrics: dict, threshold_metrics: dict, basic_path: str) -> None:
    logger.info(
        "Holdout ticker=%s rows=%d accuracy=%.4f baseline=%.4f margin=%+.4f "
        "roc_auc=%.4f precision=%.4f recall=%.4f threshold=%.6f "
        "percentile=%d gate=%s",
        asset["ticker"],
        metrics["rows"],
        metrics["accuracy"],
        metrics["majority_baseline"],
        metrics["accuracy_margin"],
        metrics["roc_auc"],
        metrics["precision"],
        metrics["recall"],
        metrics["decision_threshold"],
        threshold_metrics["percentile"],
        basic_path,
    )


def _evidence_checks(
    asset: dict,
    frame: pd.DataFrame,
    feature_columns: list[str],
    holdout_rows: pd.DataFrame,
    predicted_returns: np.ndarray,
    threshold: float,
) -> dict:
    """Evidence gates: does the live rule make money after costs, and does the
    ranking signal hold up across sequential folds, not just one split?"""
    cost = trading_cost(asset)
    entry_bar = max(threshold, REGRESSOR_ENTRY_MIN)
    backtest = simulate_strategy(
        holdout_rows["Close"].to_numpy(dtype=float),
        predicted_returns,
        entry_bar,
        threshold,
        cost,
    )
    quantiles = log_prediction_calibration(
        asset["ticker"],
        predicted_returns,
        holdout_rows["Target_Return"].to_numpy(dtype=float),
        cost,
    )
    try:
        walk_forward = walk_forward_check(frame, feature_columns, asset)
    except Exception:
        logger.exception("Walk-forward check failed ticker=%s", asset["ticker"])
        walk_forward = {
            "folds_used": 0,
            "positive_folds": 0,
            "mean_auc": float("nan"),
            "fold_aucs": [],
            "passes": False,
        }
    return {
        "cost": cost,
        "entry_bar": entry_bar,
        "backtest": backtest,
        "quantiles": quantiles,
        "walk_forward": walk_forward,
    }


def _verdict(passed_basic: bool, basic_path: str, evidence: dict) -> tuple[bool, str]:
    """Combine the basic gate with the evidence gates. Returns (passed, gate_path)."""
    evidence_ok = evidence["backtest"]["passes"] and evidence["walk_forward"]["passes"]
    passed = passed_basic and (evidence_ok or not ENFORCE_EVIDENCE_GATES)
    if not passed_basic:
        gate_path = "FAIL"
    elif passed:
        gate_path = basic_path
    else:
        gate_path = "FAIL-EVIDENCE"
    return passed, gate_path


def _log_evidence(asset: dict, evidence: dict, gate_path: str) -> None:
    backtest = evidence["backtest"]
    walk_forward = evidence["walk_forward"]
    logger.info(
        "Evidence ticker=%s cost=%.2f%% entry_bar=%+.4f backtest_trades=%d "
        "backtest_mean_net=%+.3f%% backtest_win_rate=%.2f wf_folds=%d "
        "wf_positive=%d wf_mean_auc=%.4f verdict=%s",
        asset["ticker"],
        evidence["cost"] * 100,
        evidence["entry_bar"],
        backtest["trades"],
        backtest["mean_net"] * 100,
        backtest["win_rate"],
        walk_forward["folds_used"],
        walk_forward["positive_folds"],
        walk_forward["mean_auc"],
        gate_path,
    )


def _evidence_metrics(
    evidence: dict,
    passed: bool,
    gate_path: str,
    holdout_rows: pd.DataFrame,
) -> dict:
    backtest = evidence["backtest"]
    walk_forward = evidence["walk_forward"]
    return {
        "gate_path": gate_path,
        "passed": passed,
        "trading_cost": evidence["cost"],
        "entry_bar": evidence["entry_bar"],
        "backtest_trades": backtest["trades"],
        "backtest_mean_net": backtest["mean_net"],
        "backtest_win_rate": backtest["win_rate"],
        "backtest_passes": backtest["passes"],
        "wf_folds": walk_forward["folds_used"],
        "wf_positive_folds": walk_forward["positive_folds"],
        "wf_mean_auc": walk_forward["mean_auc"],
        "wf_passes": walk_forward["passes"],
        "prediction_quantiles": evidence["quantiles"],
        "_holdout_index": holdout_rows.index,
    }


def train_and_validate(
    asset: dict,
    history: pd.DataFrame,
    spy_close: pd.Series,
) -> tuple[XGBRegressorModel | None, dict]:
    """Fit an asset regressor and evaluate it on a purged chronological holdout."""
    if spy_close is None:
        raise ValueError("SPY benchmark history is unavailable")
    feature_columns = asset["feature_columns"]
    frame = _prepare_frame(asset, history, spy_close)
    training_rows, holdout_rows, y_true = _split_holdout(asset, frame, feature_columns)

    threshold, threshold_metrics = select_prediction_threshold(
        training_rows, feature_columns, asset
    )
    holdout_model = XGBRegressorModel(
        asset["symbol"], feature_columns=feature_columns
    ).fit(training_rows[feature_columns], training_rows["Target_Return"])
    predicted_returns = np.asarray(
        holdout_model.predict_values(holdout_rows[feature_columns]), dtype=float
    )

    metrics = _holdout_metrics(
        asset, holdout_rows, y_true, predicted_returns, threshold, threshold_metrics
    )
    passed_basic, basic_path = quality_gate(metrics)
    _log_holdout(asset, metrics, threshold_metrics, basic_path)

    evidence = _evidence_checks(
        asset, frame, feature_columns, holdout_rows, predicted_returns, threshold
    )
    passed, gate_path = _verdict(passed_basic, basic_path, evidence)
    _log_evidence(asset, evidence, gate_path)
    metrics.update(_evidence_metrics(evidence, passed, gate_path, holdout_rows))
    if not passed:
        return None, metrics

    final_model = XGBRegressorModel(
        asset["symbol"],
        feature_columns=feature_columns,
        decision_threshold=threshold,
    ).fit(frame[feature_columns], frame["Target_Return"])
    return final_model, metrics
