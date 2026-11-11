"""Quality gate, strategy simulation, walk-forward check and calibration logging."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import (
    FORWARD_HORIZON_DAYS,
    MAX_BASELINE_ACCURACY_DEFICIT,
    MIN_ACCURACY_OVER_BASELINE,
    MIN_OUT_OF_SAMPLE_ROC_AUC,
    RANDOM_STATE,
    STRONG_RANKING_ROC_AUC,
)
from models import XGBRegressorModel

from .settings import (
    MIN_BACKTEST_TRADES,
    MIN_FOLD_TRAIN_ROWS,
    WALK_FORWARD_FOLDS,
    WALK_FORWARD_MIN_MEAN_AUC,
    WALK_FORWARD_MIN_POSITIVE_FOLDS,
)

logger = logging.getLogger("multi_asset_trading_bot")


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


def fit_classifier(train: pd.DataFrame, feature_columns: list[str]):
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
            fitted = fit_classifier(train, feature_columns)
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
