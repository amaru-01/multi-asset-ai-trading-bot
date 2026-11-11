"""Retrain per-asset three-day XGBoost return models and enforce holdout gates.

On top of the original accuracy / ROC-AUC holdout gate, this script now:
  * walk-forward validates every candidate on sequential folds,
  * back-tests the live entry/exit rule on the holdout, net of trading costs,
  * scores the model already in Model/ when a candidate is rejected,
  * writes Model/model_status.json so run_execution.py can stop *buying* an asset
    whose live model has no demonstrated edge (existing positions are still managed),
  * logs prediction quantiles and realized returns by predicted bucket,
  * returns a non-zero exit code only for real errors, not for "model retained".

This file is the command-line entry point (``python retrain.py``). The logic lives in
the ``training`` package; the names below are re-exported so that ``import retrain``
keeps exposing what it always did.
"""

import logging
from datetime import datetime, timezone

from dotenv import load_dotenv

from config import ASSETS, PROJECT_ROOT
from training.candidate import train_and_validate  # noqa: F401  (re-exported)
from training.data import (  # noqa: F401  (re-exported)
    download_history,
    resolve_model_path,
    trading_cost,
)
from training.gates import (  # noqa: F401  (re-exported)
    fit_classifier as _fit_classifier,
    log_prediction_calibration,
    quality_gate,
    simulate_strategy,
    walk_forward_check,
)
from training.incumbent import (  # noqa: F401  (re-exported)
    evaluate_classifier_challenger,
    evaluate_incumbent,
)
from training.report import send_retraining_report
from training.settings import (  # noqa: F401  (re-exported)
    BLOCK_BUYING_ON_FAILURE,
    CLASSIFIER_ENTRY_MIN,
    CLASSIFIER_EXIT_BAR,
    DEPLOY_CLASSIFIER_CHALLENGER,
    ENFORCE_EVIDENCE_GATES,
    MIN_BACKTEST_TRADES,
    MIN_FOLD_TRAIN_ROWS,
    MIN_FORWARD_TRADES,
    MIN_THRESHOLD_PRECISION,
    MIN_THRESHOLD_RECALL,
    REGRESSOR_ENTRY_MIN,
    REQUIRED_PASS_STREAK,
    ROUND_TRIP_COST,
    STATUS_FILE_NAME,
    THRESHOLD_PERCENTILES,
    WALK_FORWARD_FOLDS,
    WALK_FORWARD_MIN_MEAN_AUC,
    WALK_FORWARD_MIN_POSITIVE_FOLDS,
)
from training.status import (  # noqa: F401  (re-exported)
    decide_status,
    json_safe as _json_safe,
    read_previous_assets,
    tracking_start,
    write_model_status,
)
from training.threshold import select_prediction_threshold  # noqa: F401  (re-exported)

_send_retraining_report = send_retraining_report  # old private name, kept for compatibility

logger = logging.getLogger("multi_asset_trading_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
load_dotenv(PROJECT_ROOT / ".env")


def _download_histories(
    errors: list[str],
    retained_assets: list[str],
    results: list[dict],
) -> dict:
    """Download every asset's history; a failed download retains that asset's model."""
    histories = {}
    for asset in ASSETS:
        try:
            histories[asset["ticker"]] = download_history(asset["ticker"])
        except Exception:
            errors.append(asset["ticker"])
            retained_assets.append(asset["ticker"])
            results.append({"ticker": asset["ticker"], "error": "download error"})
            logger.exception("History download failed ticker=%s", asset["ticker"])
    return histories


def _status_for_rejected(
    asset: dict,
    history,
    spy_close,
    holdout_index,
    metrics: dict,
    previous_assets: dict,
) -> dict:
    """The candidate was rejected: score the live model and decide whether to keep buying."""
    ticker = asset["ticker"]
    previous = previous_assets.get(ticker, {})
    since = tracking_start(previous)
    cost = trading_cost(asset)
    incumbent = evaluate_incumbent(asset, history, spy_close, holdout_index, cost)
    forward = evaluate_incumbent(asset, history, spy_close, None, cost, since=since)
    challenger = None
    if "classifier" in str(incumbent.get("model_type", "")).lower():
        challenger = evaluate_classifier_challenger(asset, history, spy_close)
    return decide_status(
        False,
        metrics,
        incumbent,
        previous=previous,
        forward=forward,
        challenger=challenger,
        tracking_since=since,
    )


def main() -> int:
    """Retrain every asset, score incumbents of rejected candidates, record status.

    Exit code is 1 only for real errors (downloads, training, saving, status file),
    not when a candidate is rejected and the existing model is kept.
    """
    errors: list[str] = []
    updated_assets: list[str] = []
    retained_assets: list[str] = []
    results: list[dict] = []
    statuses: dict = {}
    previous_assets = read_previous_assets()

    histories = _download_histories(errors, retained_assets, results)

    spy_close = histories.get("SPY", {}).get("Close")
    for asset in ASSETS:
        ticker = asset["ticker"]
        history = histories.get(ticker)
        if history is None:
            continue
        try:
            model, metrics = train_and_validate(asset, history, spy_close)
            holdout_index = metrics.pop("_holdout_index", None)
            results.append(metrics)
            if model is None:
                retained_assets.append(ticker)
                logger.warning("Model retained ticker=%s gate=%s", ticker, metrics["gate_path"])
                statuses[ticker] = _status_for_rejected(
                    asset, history, spy_close, holdout_index, metrics, previous_assets
                )
                logger.warning(
                    "Asset status ticker=%s buying=%s reason=%s",
                    ticker,
                    statuses[ticker]["buying"],
                    statuses[ticker]["reason"],
                )
                continue
            model.save_artifact(asset["model_path"])
            updated_assets.append(ticker)
            statuses[ticker] = decide_status(
                True,
                metrics,
                None,
                previous=previous_assets.get(ticker, {}),
                tracking_since=datetime.now(timezone.utc).date().isoformat(),
            )
            logger.info(
                "Model updated ticker=%s path=%s gate=Path %s threshold=%.6f",
                ticker,
                asset["model_path"],
                metrics["gate_path"],
                metrics["decision_threshold"],
            )
        except Exception:
            errors.append(ticker)
            retained_assets.append(ticker)
            results.append({"ticker": ticker, "error": "training or save error"})
            logger.exception("Training failed; model retained ticker=%s", ticker)

    if not write_model_status(statuses):
        errors.append("model_status")

    logger.info("Updated model artifacts=%s", updated_assets or "none")
    logger.info("Retained model artifacts=%s", retained_assets or "none")
    send_retraining_report(results, updated_assets, statuses)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
