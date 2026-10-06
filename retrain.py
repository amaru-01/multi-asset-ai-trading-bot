"""Retrain per-asset three-day XGBoost return models and enforce holdout gates."""

import logging
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from config import (
    ASSETS,
    FORWARD_HORIZON_DAYS,
    MAX_BASELINE_ACCURACY_DEFICIT,
    MIN_ACCURACY_OVER_BASELINE,
    MIN_OUT_OF_SAMPLE_ACCURACY,
    MIN_OUT_OF_SAMPLE_ROC_AUC,
    MIN_TRAINING_ROWS,
    MODEL_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REGRESSION_RETURN_THRESHOLD,
    STRONG_RANKING_ROC_AUC,
    TEST_FRACTION,
)
from features.pipeline import build_regression_training_frame, normalize_market_columns
from models import XGBRegressorModel
from telegram_notifier import send_telegram_message


logger = logging.getLogger("multi_asset_trading_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
load_dotenv(PROJECT_ROOT / ".env")

THRESHOLD_PERCENTILES = (55, 60, 65, 70, 75)
MIN_THRESHOLD_PRECISION = 0.35
MIN_THRESHOLD_RECALL = 0.35


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
    if len(history) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Insufficient valid history for {ticker}: {len(history)} rows; "
            f"need at least {MIN_TRAINING_ROWS}"
        )
    return history


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


def quality_gate(metrics: dict) -> tuple[bool, str]:
    """Accept either strict baseline improvement or a strong ranking signal."""
    path_a = (
        metrics["accuracy"]
        > metrics["majority_baseline"] + MIN_ACCURACY_OVER_BASELINE
        and metrics["roc_auc"] > MIN_OUT_OF_SAMPLE_ROC_AUC
    )
    path_b = (
        metrics["roc_auc"] >= STRONG_RANKING_ROC_AUC
        and metrics["accuracy"]
        >= metrics["majority_baseline"] - MAX_BASELINE_ACCURACY_DEFICIT
    )
    if path_a:
        return True, "A"
    if path_b:
        return True, "B"
    return False, "FAIL"


def train_and_validate(
    asset: dict,
    history: pd.DataFrame,
    spy_close: pd.Series,
) -> tuple[XGBRegressorModel | None, dict]:
    """Fit an asset regressor and evaluate it on a purged chronological holdout."""
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

    threshold, threshold_metrics = select_prediction_threshold(
        training_rows, feature_columns, asset
    )
    holdout_model = XGBRegressorModel(
        asset["symbol"], feature_columns=feature_columns
    ).fit(training_rows[feature_columns], training_rows["Target_Return"])
    predicted_returns = holdout_model.predict_values(holdout_rows[feature_columns])
    y_pred = (predicted_returns > threshold).astype("int8")

    accuracy = float(accuracy_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, predicted_returns))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    baseline = float(max(np.mean(y_true), 1 - np.mean(y_true)))
    metrics = {
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
    passed, path = quality_gate(metrics)
    metrics["gate_path"] = path
    logger.info(
        "Holdout ticker=%s rows=%d accuracy=%.4f baseline=%.4f margin=%+.4f "
        "roc_auc=%.4f precision=%.4f recall=%.4f threshold=%.6f "
        "percentile=%d gate=%s",
        asset["ticker"],
        metrics["rows"],
        accuracy,
        baseline,
        metrics["accuracy_margin"],
        roc_auc,
        precision,
        recall,
        threshold,
        threshold_metrics["percentile"],
        path,
    )
    if not passed:
        return None, metrics

    final_model = XGBRegressorModel(
        asset["symbol"],
        feature_columns=feature_columns,
        decision_threshold=threshold,
    ).fit(frame[feature_columns], frame["Target_Return"])
    return final_model, metrics


def _send_retraining_report(
    results: list[dict],
    updated_assets: list[str],
) -> None:
    """Send per-asset validation metrics and artifact outcomes through Telegram."""
    latest_by_ticker = {result["ticker"]: result for result in results}
    table_lines = [
        f"{'Ticker':<7} {'Base%':>5} {'Acc%':>5} {'Margin':>7} "
        f"{'AUC':>6} {'Gate':<6} Artifact"
    ]
    for asset in ASSETS:
        ticker = asset["ticker"]
        result = latest_by_ticker.get(ticker)
        export = "🟢 Updated" if ticker in updated_assets else "🔴 Retained"
        if result is None or "error" in result:
            table_lines.append(
                f"{ticker:<7} {'--':>5} {'--':>5} {'--':>7} "
                f"{'--':>6} {'🔴 Error':<10} {export}"
            )
            continue
        passed, path = quality_gate(result)
        gate = f"🟢 Path {path}" if passed else "🔴 Fail"
        table_lines.append(
            f"{escape(ticker, quote=False):<7} "
            f"{result['majority_baseline'] * 100:>5.1f} "
            f"{result['accuracy'] * 100:>5.1f} "
            f"{result['accuracy_margin'] * 100:>+6.2f} "
            f"{result['roc_auc']:>6.4f} {gate:<10} {export}"
        )
    message = (
        "<b>MODEL RETRAINING REPORT</b>\n"
        "<pre>" + "\n".join(table_lines) + "</pre>"
    )
    send_telegram_message(message, parse_mode="HTML")


def main() -> int:
    """Download shared benchmark history and update each passing asset immediately."""
    failed_assets = []
    updated_assets = []
    retained_assets = []
    results = []

    histories = {}
    for asset in ASSETS:
        try:
            histories[asset["ticker"]] = download_history(asset["ticker"])
        except Exception:
            failed_assets.append(asset["ticker"])
            retained_assets.append(asset["ticker"])
            results.append({"ticker": asset["ticker"], "error": "download error"})
            logger.exception("History download failed ticker=%s", asset["ticker"])

    spy_close = histories.get("SPY", {}).get("Close")
    for asset in ASSETS:
        ticker = asset["ticker"]
        history = histories.get(ticker)
        if history is None:
            continue
        try:
            model, metrics = train_and_validate(asset, history, spy_close)
            results.append(metrics)
            if model is None:
                failed_assets.append(ticker)
                retained_assets.append(ticker)
                logger.warning("Model retained ticker=%s gate=%s", ticker, metrics["gate_path"])
                continue
            model.save_artifact(asset["model_path"])
            updated_assets.append(ticker)
            logger.info(
                "Model updated ticker=%s path=%s gate=Path %s threshold=%.6f",
                ticker,
                asset["model_path"],
                metrics["gate_path"],
                metrics["decision_threshold"],
            )
        except Exception:
            failed_assets.append(ticker)
            retained_assets.append(ticker)
            results.append({"ticker": ticker, "error": "training or save error"})
            logger.exception("Training failed; model retained ticker=%s", ticker)

    logger.info("Updated model artifacts=%s", updated_assets or "none")
    logger.info("Retained model artifacts=%s", retained_assets or "none")
    _send_retraining_report(results, updated_assets)
    return 1 if failed_assets else 0


if __name__ == "__main__":
    raise SystemExit(main())
