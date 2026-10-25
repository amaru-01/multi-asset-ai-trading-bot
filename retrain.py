"""Retrain per-asset three-day XGBoost return models and enforce holdout gates.

On top of the original accuracy / ROC-AUC holdout gate, this script now:
  * walk-forward validates every candidate on sequential folds,
  * back-tests the live entry/exit rule on the holdout, net of trading costs,
  * scores the model already in Model/ when a candidate is rejected,
  * writes Model/model_status.json so run_execution.py can stop *buying* an asset
    whose live model has no demonstrated edge (existing positions are still managed),
  * logs prediction quantiles and realized returns by predicted bucket,
  * returns a non-zero exit code only for real errors, not for "model retained".
"""

import json
import logging
from datetime import datetime, timezone
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
from features.pipeline import (
    build_inference_features,
    build_regression_training_frame,
    normalize_market_columns,
)
from models import XGBRegressorModel, load_trading_model
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

# --- Evidence gates: judge models on money, not only on accuracy ---------------
# Estimated cost of one round trip as a fraction of the position. Alpaca crypto
# charges about 0.25% per side; US stocks are commission-free, so only a small
# allowance for spread and slippage is assumed. Adjust if your costs differ.
ROUND_TRIP_COST = {"crypto": 0.005, "stock": 0.0005}
# Mirrors of the live entry bars in run_execution.py (REGRESSOR_ENTRY_MIN and
# CLASSIFIER_ENTRY_MIN). Keep these in sync with that file.
REGRESSOR_ENTRY_MIN = 0.006
CLASSIFIER_ENTRY_MIN = 0.60
# The holdout back-test of the live rule needs at least this many trades and a
# positive average net return per trade.
MIN_BACKTEST_TRADES = 10
# Walk-forward: sequential expanding-window folds over the whole history.
WALK_FORWARD_FOLDS = 4
WALK_FORWARD_MIN_POSITIVE_FOLDS = 3  # folds whose ROC-AUC must beat 0.5
WALK_FORWARD_MIN_MEAN_AUC = 0.52
MIN_FOLD_TRAIN_ROWS = 150
# When True, a candidate must also pass the evidence gates to replace the live model.
ENFORCE_EVIDENCE_GATES = True
# When True, an asset whose live model shows no demonstrated edge is marked "blocked"
# in Model/model_status.json and run_execution.py stops opening new positions in it.
BLOCK_BUYING_ON_FAILURE = True
STATUS_FILE_NAME = "model_status.json"
# A blocked asset only reopens after this many consecutive weekly evidence passes.
REQUIRED_PASS_STREAK = 2
# Forward test: the live model is replayed on bars that arrived after tracking began
# (after the model was last trained), so the result is genuinely out of sample. Once it
# has this many trades it overrides the in-sample checks: a positive mean net return
# allows buying, a non-positive one blocks it.
MIN_FORWARD_TRADES = 10
# Classifier challenger for assets whose live model is a legacy classifier. It is trained
# and scored like the regressors, but only reported: set True only if models.py can save
# an XGBClassifier in the format load_trading_model() reads.
DEPLOY_CLASSIFIER_CHALLENGER = False
CLASSIFIER_EXIT_BAR = 0.55  # mirrors LEGACY_CLASSIFIER_THRESHOLD


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


def simulate_strategy(
    closes: np.ndarray,
    predictions: np.ndarray,
    enter_bar: float,
    exit_bar: float,
    cost: float,
) -> dict:
    """Back-test the live rule on completed daily bars, net of trading costs.

    A prediction made from day i's completed bar is acted on at day i+1's close
    (a conservative one-day delay). A position opens when the prediction reaches
    `enter_bar` and closes when it falls below `exit_bar` (the model's own
    threshold). `cost` is charged once per round trip. A position still open at
    the end is marked to the last close. Stop-loss, cooldown and minimum-hold
    rules are not simulated.
    """
    entry_price = None
    trade_returns: list[float] = []
    for i in range(len(predictions) - 1):
        fill_price = closes[i + 1]
        if entry_price is None:
            if predictions[i] >= enter_bar:
                entry_price = fill_price
        elif predictions[i] < exit_bar:
            trade_returns.append(fill_price / entry_price - 1.0 - cost)
            entry_price = None
    if entry_price is not None and len(closes) > 0:
        trade_returns.append(closes[-1] / entry_price - 1.0 - cost)

    trades = len(trade_returns)
    mean_net = float(np.mean(trade_returns)) if trades else float("nan")
    return {
        "trades": trades,
        "mean_net": mean_net,
        "total_net": float(np.sum(trade_returns)) if trades else 0.0,
        "win_rate": (
            float(np.mean(np.array(trade_returns) > 0)) if trades else float("nan")
        ),
        "passes": bool(trades >= MIN_BACKTEST_TRADES and mean_net > 0),
    }


def _fit_classifier(train: pd.DataFrame, feature_columns: list[str]):
    """Fit an XGBoost classifier for the 'return above the target threshold' label."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1,
    ).fit(train[feature_columns], train["Target_Direction"].astype(int))


def walk_forward_check(
    frame: pd.DataFrame,
    feature_columns: list[str],
    asset: dict,
    classifier: bool = False,
) -> dict:
    """Train on an expanding window and score each later fold out of sample.

    One holdout split is a noisy verdict, so require the ranking signal (ROC-AUC)
    to beat chance in most sequential folds and to be better than chance on average.
    """
    rows = len(frame)
    fold_size = rows // (WALK_FORWARD_FOLDS + 1)
    fold_aucs: list[float] = []
    for fold in range(1, WALK_FORWARD_FOLDS + 1):
        test_start = fold * fold_size
        test_end = rows if fold == WALK_FORWARD_FOLDS else (fold + 1) * fold_size
        train = frame.iloc[: max(0, test_start - FORWARD_HORIZON_DAYS)]
        test = frame.iloc[test_start:test_end]
        labels = test["Target_Direction"].to_numpy(dtype="int8")
        if len(train) < MIN_FOLD_TRAIN_ROWS or len(np.unique(labels)) < 2:
            continue
        if classifier:
            fitted = _fit_classifier(train, feature_columns)
            predictions = fitted.predict_proba(test[feature_columns])[:, 1]
        else:
            model = XGBRegressorModel(
                asset["symbol"], feature_columns=feature_columns
            ).fit(train[feature_columns], train["Target_Return"])
            predictions = np.asarray(
                model.predict_values(test[feature_columns]), dtype=float
            )
        fold_aucs.append(float(roc_auc_score(labels, predictions)))

    folds_used = len(fold_aucs)
    positive_folds = int(sum(auc > 0.5 for auc in fold_aucs))
    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    passes = (
        folds_used >= WALK_FORWARD_MIN_POSITIVE_FOLDS
        and positive_folds >= WALK_FORWARD_MIN_POSITIVE_FOLDS
        and mean_auc >= WALK_FORWARD_MIN_MEAN_AUC
    )
    return {
        "folds_used": folds_used,
        "positive_folds": positive_folds,
        "mean_auc": mean_auc,
        "fold_aucs": fold_aucs,
        "passes": bool(passes),
    }


def log_prediction_calibration(
    ticker: str,
    predictions: np.ndarray,
    realized: np.ndarray,
    cost: float,
) -> dict:
    """Log prediction quantiles and realized 3-day returns by predicted quintile.

    Use this to see what size of prediction the model actually produces and whether
    larger predictions really earn more than the round-trip cost.
    """
    predictions = np.asarray(predictions, dtype=float)
    realized = np.asarray(realized, dtype=float)
    quantiles = {
        f"p{q}": float(np.percentile(predictions, q)) for q in (50, 75, 90, 95, 99)
    }
    logger.info(
        "Prediction quantiles ticker=%s %s",
        ticker,
        " ".join(f"{name}={value * 100:+.3f}%" for name, value in quantiles.items()),
    )
    try:
        buckets = pd.qcut(
            pd.Series(predictions).rank(method="first"), 5, labels=False
        ).to_numpy()
    except ValueError:
        return quantiles
    for bucket in range(5):
        mask = buckets == bucket
        if not mask.any():
            continue
        mean_prediction = float(np.mean(predictions[mask]))
        mean_realized = float(np.mean(realized[mask]))
        logger.info(
            "Calibration ticker=%s quintile=%d/5 rows=%d mean_prediction=%+.3f%% "
            "mean_realized_3d=%+.3f%% net_of_cost=%+.3f%%",
            ticker,
            bucket + 1,
            int(mask.sum()),
            mean_prediction * 100,
            mean_realized * 100,
            (mean_realized - cost) * 100,
        )
    return quantiles


def train_and_validate(
    asset: dict,
    history: pd.DataFrame,
    spy_close: pd.Series,
) -> tuple[XGBRegressorModel | None, dict]:
    """Fit an asset regressor and evaluate it on a purged chronological holdout."""
    if spy_close is None:
        raise ValueError("SPY benchmark history is unavailable")
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
    predicted_returns = np.asarray(
        holdout_model.predict_values(holdout_rows[feature_columns]), dtype=float
    )
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
    passed_basic, basic_path = quality_gate(metrics)
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
        basic_path,
    )

    # Evidence gates: does the live rule make money after costs, and does the
    # ranking signal hold up across sequential folds, not just one split?
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
    evidence_ok = backtest["passes"] and walk_forward["passes"]
    passed = passed_basic and (evidence_ok or not ENFORCE_EVIDENCE_GATES)
    if not passed_basic:
        gate_path = "FAIL"
    elif passed:
        gate_path = basic_path
    else:
        gate_path = "FAIL-EVIDENCE"

    logger.info(
        "Evidence ticker=%s cost=%.2f%% entry_bar=%+.4f backtest_trades=%d "
        "backtest_mean_net=%+.3f%% backtest_win_rate=%.2f wf_folds=%d "
        "wf_positive=%d wf_mean_auc=%.4f verdict=%s",
        asset["ticker"],
        cost * 100,
        entry_bar,
        backtest["trades"],
        backtest["mean_net"] * 100,
        backtest["win_rate"],
        walk_forward["folds_used"],
        walk_forward["positive_folds"],
        walk_forward["mean_auc"],
        gate_path,
    )
    metrics.update(
        {
            "gate_path": gate_path,
            "passed": passed,
            "trading_cost": cost,
            "entry_bar": entry_bar,
            "backtest_trades": backtest["trades"],
            "backtest_mean_net": backtest["mean_net"],
            "backtest_win_rate": backtest["win_rate"],
            "backtest_passes": backtest["passes"],
            "wf_folds": walk_forward["folds_used"],
            "wf_positive_folds": walk_forward["positive_folds"],
            "wf_mean_auc": walk_forward["mean_auc"],
            "wf_passes": walk_forward["passes"],
            "prediction_quantiles": quantiles,
            "_holdout_index": holdout_rows.index,
        }
    )
    if not passed:
        return None, metrics

    final_model = XGBRegressorModel(
        asset["symbol"],
        feature_columns=feature_columns,
        decision_threshold=threshold,
    ).fit(frame[feature_columns], frame["Target_Return"])
    return final_model, metrics


def evaluate_incumbent(
    asset: dict,
    history: pd.DataFrame,
    spy_close: pd.Series,
    holdout_index,
    cost: float,
    since=None,
) -> dict:
    """Score the model currently in Model/ with the live rule.

    Default: the holdout dates. The incumbent may already have been trained on part of
    that window, so a good result is an optimistic upper bound ("unverified").

    With `since` (a date), score only completed bars after that date instead. These
    bars arrived after tracking began, so the result is a genuine forward test.
    """
    forward = since is not None
    path = resolve_model_path(asset)
    if not path.is_file():
        return {"evaluated": False, "reason": "no model file"}
    if holdout_index is None and not forward:
        return {"evaluated": False, "reason": "no holdout window"}
    try:
        model = load_trading_model(path, asset["symbol"], asset["feature_columns"])
        columns = list(model.feature_names or asset["feature_columns"])
        features = build_inference_features(history, columns, spy_close)
        if forward:
            today = pd.Timestamp(datetime.now(timezone.utc).date())
            dates = pd.DatetimeIndex(features.index)
            rows = features.loc[(dates > pd.Timestamp(since)) & (dates < today)]
            rows = rows.dropna(subset=columns + ["Close"])
            if len(rows) < 2:
                return {"evaluated": False, "reason": "no new completed bars yet"}
        else:
            rows = features.loc[features.index.intersection(holdout_index)]
            rows = rows.dropna(subset=columns + ["Close"])
            if len(rows) < 60:
                return {"evaluated": False, "reason": f"only {len(rows)} comparable rows"}
        inputs = rows[columns].astype(float)

        if hasattr(model, "predict_values"):
            predictions = np.asarray(model.predict_values(inputs), dtype=float)
        else:
            predictions = np.array(
                [
                    float(model.predict_signal(inputs.iloc[[i]])["raw_prediction"])
                    for i in range(len(inputs))
                ]
            )
        threshold = float(model.predict_signal(inputs.iloc[[-1]])["threshold"])
        floor = (
            REGRESSOR_ENTRY_MIN
            if isinstance(model, XGBRegressorModel)
            else CLASSIFIER_ENTRY_MIN
        )
        enter_bar = max(threshold, floor)
        backtest = simulate_strategy(
            rows["Close"].to_numpy(dtype=float),
            predictions,
            enter_bar,
            threshold,
            cost,
        )
        logger.info(
            "%s ticker=%s model=%s rows=%d threshold=%.4f "
            "entry_bar=%.4f trades=%d mean_net=%+.3f%% win_rate=%.2f passes=%s",
            "Forward test" if forward else "Incumbent back-test",
            asset["ticker"],
            getattr(model, "model_type", "unknown"),
            len(rows),
            threshold,
            enter_bar,
            backtest["trades"],
            backtest["mean_net"] * 100,
            backtest["win_rate"],
            backtest["passes"],
        )
        return {
            "evaluated": True,
            "model_type": str(getattr(model, "model_type", "unknown")),
            "threshold": threshold,
            "entry_bar": enter_bar,
            "rows": len(rows),
            "backtest": backtest,
        }
    except Exception:
        logger.exception("Incumbent evaluation failed ticker=%s", asset["ticker"])
        return {"evaluated": False, "reason": "evaluation error"}


def evaluate_classifier_challenger(
    asset: dict,
    history: pd.DataFrame,
    spy_close: pd.Series,
) -> dict:
    """Train a fresh XGBoost classifier and score it like the regressor candidates.

    Assets whose live model is a legacy classifier were never validated by the regressor
    candidate, so this gives an honest out-of-sample read on the classifier approach:
    train on the early part of the history, score the purged holdout, back-test the live
    classifier rule (enter at CLASSIFIER_ENTRY_MIN, exit below CLASSIFIER_EXIT_BAR) net of
    costs, and run the same walk-forward check. Report-only unless
    DEPLOY_CLASSIFIER_CHALLENGER is enabled.
    """
    try:
        feature_columns = asset["feature_columns"]
        frame = build_regression_training_frame(
            history,
            spy_close,
            horizon_days=FORWARD_HORIZON_DAYS,
            return_threshold=REGRESSION_RETURN_THRESHOLD,
        )
        frame = frame.dropna(
            subset=feature_columns + ["Target_Return", "Target_Direction"]
        )
        if len(frame) < MIN_TRAINING_ROWS:
            return {"evaluated": False, "reason": "not enough rows"}
        split_at = int(len(frame) * (1 - TEST_FRACTION))
        training_rows = frame.iloc[: max(0, split_at - FORWARD_HORIZON_DAYS)]
        holdout_rows = frame.iloc[split_at:]
        y_true = holdout_rows["Target_Direction"].to_numpy(dtype="int8")
        if training_rows.empty or len(np.unique(y_true)) < 2:
            return {"evaluated": False, "reason": "labels insufficient"}

        model = _fit_classifier(training_rows, feature_columns)
        probabilities = model.predict_proba(holdout_rows[feature_columns])[:, 1]
        auc = float(roc_auc_score(y_true, probabilities))
        cost = trading_cost(asset)
        backtest = simulate_strategy(
            holdout_rows["Close"].to_numpy(dtype=float),
            probabilities,
            CLASSIFIER_ENTRY_MIN,
            CLASSIFIER_EXIT_BAR,
            cost,
        )
        walk_forward = walk_forward_check(
            frame, feature_columns, asset, classifier=True
        )
        passed = bool(
            auc > MIN_OUT_OF_SAMPLE_ROC_AUC
            and backtest["passes"]
            and walk_forward["passes"]
        )
        logger.info(
            "Classifier challenger ticker=%s holdout_auc=%.4f backtest_trades=%d "
            "backtest_mean_net=%+.3f%% wf_positive=%d/%d wf_mean_auc=%.4f passed=%s "
            "(report-only=%s)",
            asset["ticker"],
            auc,
            backtest["trades"],
            backtest["mean_net"] * 100,
            walk_forward["positive_folds"],
            walk_forward["folds_used"],
            walk_forward["mean_auc"],
            passed,
            not DEPLOY_CLASSIFIER_CHALLENGER,
        )
        return {
            "evaluated": True,
            "roc_auc": auc,
            "backtest_trades": backtest["trades"],
            "backtest_mean_net": backtest["mean_net"],
            "wf_positive_folds": walk_forward["positive_folds"],
            "wf_folds": walk_forward["folds_used"],
            "wf_mean_auc": walk_forward["mean_auc"],
            "passed": passed,
        }
    except Exception:
        logger.exception("Classifier challenger failed ticker=%s", asset["ticker"])
        return {"evaluated": False, "reason": "error"}


def decide_status(
    deployed: bool,
    metrics: dict,
    incumbent: dict | None,
    previous: dict | None = None,
    forward: dict | None = None,
    challenger: dict | None = None,
    tracking_since: str | None = None,
) -> dict:
    """Summarise which model is live for an asset and whether buying is allowed.

    Order of evidence: (1) a forward test with enough trades overrides everything;
    otherwise (2) a candidate that passed, or a live model that passes the in-sample
    back-test; (3) a previously blocked asset needs REQUIRED_PASS_STREAK consecutive
    passes from (2) before buying reopens.
    """
    previous = previous or {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    candidate = {
        key: metrics.get(key)
        for key in (
            "gate_path",
            "roc_auc",
            "accuracy",
            "majority_baseline",
            "backtest_trades",
            "backtest_mean_net",
            "wf_folds",
            "wf_positive_folds",
            "wf_mean_auc",
            "decision_threshold",
            "entry_bar",
            "prediction_quantiles",
        )
    }
    status = {
        "checked_at": now,
        "candidate": candidate,
        "tracking_since": tracking_since or now[:10],
        "deployed_candidate": bool(deployed),
    }
    if challenger:
        status["classifier_challenger"] = challenger
    if forward and forward.get("evaluated"):
        status["forward"] = {
            "since": status["tracking_since"],
            "rows": forward["rows"],
            "trades": forward["backtest"]["trades"],
            "mean_net": forward["backtest"]["mean_net"],
            "win_rate": forward["backtest"]["win_rate"],
        }

    detail = ""
    evidence_pass = False
    fail_open = False
    if deployed:
        status.update(
            live_model="3-Day Regressor (retrained)",
            buying="allowed",
            reason="candidate passed the holdout, walk-forward and net-of-cost gates",
        )
        evidence_pass = True
    elif incumbent is None or not incumbent.get("evaluated"):
        why = (incumbent or {}).get("reason", "not evaluated")
        fail_open = True
        status.update(
            live_model="existing model file (not scored)",
            buying="allowed",
            reason=f"candidate rejected; existing model could not be scored ({why})",
        )
    else:
        backtest = incumbent["backtest"]
        model_type = incumbent["model_type"]
        if backtest["trades"] and backtest["mean_net"] == backtest["mean_net"]:
            detail = (
                f"{backtest['trades']} trades, mean net {backtest['mean_net']:+.2%} per trade"
            )
            if backtest["trades"] < MIN_BACKTEST_TRADES:
                detail += f"; fewer than the {MIN_BACKTEST_TRADES} needed to trust it"
        else:
            detail = f"only {backtest['trades']} trades (too few to judge)"
        status["incumbent"] = {
            "model_type": model_type,
            "threshold": incumbent["threshold"],
            "entry_bar": incumbent["entry_bar"],
            "rows": incumbent["rows"],
            "backtest": backtest,
        }
        if backtest["passes"]:
            status.update(
                live_model=f"{model_type} (existing, unverified)",
                buying="allowed",
                reason=(
                    f"candidate rejected; existing {model_type} is profitable on the holdout "
                    f"({detail}) but may have trained on that data"
                ),
            )
            # A model deployed last week was trained on this very window, so its
            # in-sample pass proves nothing new.
            evidence_pass = not previous.get("deployed_candidate", False)
        else:
            status.update(
                live_model=f"{model_type} (existing)",
                buying="blocked" if BLOCK_BUYING_ON_FAILURE else "allowed",
                reason=(
                    f"no model passed: candidate rejected and existing {model_type} is not "
                    f"proven by the net-of-cost back-test ({detail})"
                ),
            )

    # (1) Forward test with enough trades is real out-of-sample evidence.
    fwd = status.get("forward")
    forward_decided = False
    if fwd and fwd["trades"] >= MIN_FORWARD_TRADES and fwd["mean_net"] == fwd["mean_net"]:
        text = (
            f"forward test since {fwd['since']}: {fwd['trades']} trades, "
            f"mean net {fwd['mean_net']:+.2%} per trade"
        )
        forward_decided = True
        if fwd["mean_net"] > 0:
            status.update(buying="allowed", reason=f"{text} (out of sample, positive)")
            evidence_pass = True
        else:
            status.update(
                buying="blocked" if BLOCK_BUYING_ON_FAILURE else "allowed",
                reason=f"{text} (out of sample, losing)",
            )
            evidence_pass = False

    # (3) Streak rule for assets that were blocked before this run.
    streak = int(previous.get("pass_streak", 0)) + 1 if evidence_pass else 0
    status["pass_streak"] = streak
    was_blocked = previous.get("buying") == "blocked"
    if (
        was_blocked
        and status["buying"] == "allowed"
        and not fail_open
        and not forward_decided
        and streak < REQUIRED_PASS_STREAK
    ):
        status.update(
            buying="blocked",
            reason=(
                f"pass streak {streak}/{REQUIRED_PASS_STREAK}; buying reopens after "
                f"{REQUIRED_PASS_STREAK} consecutive weekly passes"
            ),
        )
    status["evidence_pass"] = bool(evidence_pass)
    return status


def _json_safe(value):
    """Convert numpy types and NaN/inf to plain JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def read_previous_assets() -> dict:
    """Last run's per-asset status (empty when the file is missing or unreadable)."""
    try:
        data = json.loads((MODEL_DIR / STATUS_FILE_NAME).read_text(encoding="utf-8"))
        return data.get("assets", {}) or {}
    except Exception:
        return {}


def tracking_start(previous: dict) -> str:
    """Date from which forward results count: the last run, or today if there is none.

    The live model was last saved before the previous run, so every completed bar after
    that date is unseen by it.
    """
    for key in ("tracking_since", "checked_at"):
        value = str(previous.get(key) or "")[:10]
        if len(value) == 10:
            return value
    return datetime.now(timezone.utc).date().isoformat()


def write_model_status(statuses: dict) -> bool:
    """Merge this run's per-asset status into Model/model_status.json."""
    path = MODEL_DIR / STATUS_FILE_NAME
    try:
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("assets", {})
        except Exception:
            existing = {}
        existing.update(statuses)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "assets": existing,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("Model status written path=%s assets=%s", path, sorted(statuses))
        return True
    except Exception:
        logger.exception("Could not write model status file")
        return False


def _send_retraining_report(
    results: list[dict],
    updated_assets: list[str],
    statuses: dict,
) -> None:
    """Send per-asset validation metrics, live-model status and buying state to Telegram."""
    latest_by_ticker = {result["ticker"]: result for result in results}
    table_lines = [
        f"{'Ticker':<7} {'AUC':>5} {'WF':>3} {'Trd':>3} {'Net%':>5} Buy"
    ]
    detail_lines = []
    for asset in ASSETS:
        ticker = asset["ticker"]
        label = escape(ticker, quote=False)
        result = latest_by_ticker.get(ticker)
        status = statuses.get(ticker)
        if status is None:
            buying = "--"
        else:
            buying = "🔴" if status["buying"] == "blocked" else "🟢"
        if result is None or "error" in result:
            table_lines.append(
                f"{label:<7} {'--':>5} {'--':>3} {'--':>3} {'--':>5} ERR"
            )
            continue
        net = result["backtest_mean_net"]
        net_text = f"{net * 100:+.2f}" if np.isfinite(net) else "--"
        if result["passed"]:
            gate = f"Path {result['gate_path']}"
        elif result["gate_path"] == "FAIL-EVIDENCE":
            gate = "No edge"
        else:
            gate = "Fail"
        table_lines.append(
            f"{label:<7} {result['roc_auc']:>5.3f} "
            f"{result['wf_positive_folds']}/{result['wf_folds']:<1} "
            f"{result['backtest_trades']:>3} {net_text:>5} {buying}"
        )
        if status is not None:
            marker = "updated" if ticker in updated_assets else "kept"
            detail_lines.append(
                f"{label}: {escape(status['live_model'], quote=False)} ({marker}). "
                f"{escape(status['reason'][:600], quote=False)} [{gate}]"
            )
            fwd = status.get("forward")
            if fwd:
                mean = fwd["mean_net"]
                mean_text = f"{mean:+.2%}" if mean == mean else "n/a"
                detail_lines.append(
                    f"  Forward test since {fwd['since']}: {fwd['trades']} trades, "
                    f"mean net {mean_text} (counts at {MIN_FORWARD_TRADES}+ trades)"
                )
            chal = status.get("classifier_challenger")
            if chal and chal.get("evaluated"):
                detail_lines.append(
                    f"  Classifier challenger: AUC {chal['roc_auc']:.3f}, "
                    f"WF {chal['wf_positive_folds']}/{chal['wf_folds']}, "
                    f"{chal['backtest_trades']} trades, "
                    f"{'passed' if chal['passed'] else 'did not pass'} (report-only)"
                )
    message = (
        "<b>MODEL RETRAINING REPORT</b>\n"
        "<pre>" + "\n".join(table_lines) + "</pre>\n"
        "<i>WF = walk-forward folds with AUC&gt;0.5, Trd = back-test trades, "
        "Net% = mean net per trade, 🟢 buying allowed / 🔴 blocked</i>"
    )
    if detail_lines:
        message += "\n" + "\n".join(detail_lines)
    send_telegram_message(message, parse_mode="HTML")


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

    histories = {}
    for asset in ASSETS:
        try:
            histories[asset["ticker"]] = download_history(asset["ticker"])
        except Exception:
            errors.append(asset["ticker"])
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
            holdout_index = metrics.pop("_holdout_index", None)
            results.append(metrics)
            if model is None:
                retained_assets.append(ticker)
                logger.warning("Model retained ticker=%s gate=%s", ticker, metrics["gate_path"])
                previous = previous_assets.get(ticker, {})
                since = tracking_start(previous)
                cost = trading_cost(asset)
                incumbent = evaluate_incumbent(
                    asset, history, spy_close, holdout_index, cost
                )
                forward = evaluate_incumbent(
                    asset, history, spy_close, None, cost, since=since
                )
                challenger = None
                if "classifier" in str(incumbent.get("model_type", "")).lower():
                    challenger = evaluate_classifier_challenger(
                        asset, history, spy_close
                    )
                statuses[ticker] = decide_status(
                    False,
                    metrics,
                    incumbent,
                    previous=previous,
                    forward=forward,
                    challenger=challenger,
                    tracking_since=since,
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
    _send_retraining_report(results, updated_assets, statuses)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
