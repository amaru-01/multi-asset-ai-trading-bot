"""Score the model already in Model/ (and a classifier challenger) with the live rule."""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import (
    FORWARD_HORIZON_DAYS,
    MIN_OUT_OF_SAMPLE_ROC_AUC,
    MIN_TRAINING_ROWS,
    REGRESSION_RETURN_THRESHOLD,
    TEST_FRACTION,
)
from features.pipeline import build_inference_features, build_regression_training_frame
from models import XGBRegressorModel, load_trading_model

from .data import resolve_model_path, trading_cost
from .gates import fit_classifier, simulate_strategy, walk_forward_check
from .settings import (
    CLASSIFIER_ENTRY_MIN,
    CLASSIFIER_EXIT_BAR,
    DEPLOY_CLASSIFIER_CHALLENGER,
    REGRESSOR_ENTRY_MIN,
)

logger = logging.getLogger("multi_asset_trading_bot")


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

        model = fit_classifier(training_rows, feature_columns)
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
