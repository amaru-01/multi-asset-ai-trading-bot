"""Characterization tests for retrain.py (model gates, buy/block decisions, status, report).

They describe how retraining behaves today so the code can be restructured safely.
Market data is synthetic and seeded; nothing touches the network or the real Model/ folder.
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import retrain  # noqa: E402
from features.pipeline import build_regression_training_frame  # noqa: E402
from helpers import RETRAIN_MODULES, patch_name, patch_value  # noqa: E402
from models import XGBRegressorModel  # noqa: E402

TODAY = datetime.now(timezone.utc).date().isoformat()


def make_history(n=1200, seed=1):
    """Seeded random-walk daily OHLCV that ends yesterday (every bar is completed)."""
    end = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    index = pd.date_range(end=end, periods=n, freq="D")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def make_frame(history, spy_close):
    asset = config.ASSETS[1]
    frame = build_regression_training_frame(
        history, spy_close,
        horizon_days=config.FORWARD_HORIZON_DAYS,
        return_threshold=config.REGRESSION_RETURN_THRESHOLD,
    )
    return frame.dropna(
        subset=asset["feature_columns"] + ["Target_Return", "Target_Direction"]
    )


def btc(**overrides):
    asset = dict(next(a for a in config.ASSETS if a["ticker"] == "BTC-USD"))
    asset.update(overrides)
    return asset


def sim(trades, mean_net, passes=None, win_rate=0.5):
    """A simulate_strategy-style result."""
    return {
        "trades": trades, "mean_net": mean_net, "total_net": trades * mean_net,
        "win_rate": win_rate,
        "passes": (trades >= 10 and mean_net > 0) if passes is None else passes,
    }


HISTORY = make_history()
SPY = make_history(seed=2)["Close"]
FRAME = make_frame(HISTORY, SPY)
COLUMNS = config.ASSETS[1]["feature_columns"]


class QualityGateTests(unittest.TestCase):
    def test_path_a_needs_accuracy_above_baseline_and_some_auc(self):
        m = {"accuracy": 0.60, "majority_baseline": 0.55, "roc_auc": 0.53}
        self.assertEqual(retrain.quality_gate(m), (True, "A"))

    def test_path_b_accepts_a_strong_ranking_signal(self):
        m = {"accuracy": 0.54, "majority_baseline": 0.55, "roc_auc": 0.55}
        self.assertEqual(retrain.quality_gate(m), (True, "B"))

    def test_weak_signal_fails(self):
        m = {"accuracy": 0.60, "majority_baseline": 0.55, "roc_auc": 0.50}
        self.assertEqual(retrain.quality_gate(m), (False, "FAIL"))

    def test_strong_auc_with_accuracy_far_below_baseline_fails(self):
        m = {"accuracy": 0.40, "majority_baseline": 0.55, "roc_auc": 0.60}
        self.assertEqual(retrain.quality_gate(m), (False, "FAIL"))

    def test_path_a_wins_when_both_apply(self):
        m = {"accuracy": 0.70, "majority_baseline": 0.55, "roc_auc": 0.60}
        self.assertEqual(retrain.quality_gate(m), (True, "A"))


class SimulateStrategyTests(unittest.TestCase):
    def test_one_round_trip_is_filled_one_bar_late_and_pays_cost(self):
        closes = np.array([100, 100, 110, 121, 121], dtype=float)
        preds = np.array([0, 0.05, 0, 0, 0], dtype=float)
        out = retrain.simulate_strategy(closes, preds, 0.01, 0.01, 0.002)
        self.assertEqual(out["trades"], 1)
        self.assertAlmostEqual(out["mean_net"], 121 / 110 - 1 - 0.002)
        self.assertEqual(out["win_rate"], 1.0)
        self.assertFalse(out["passes"])  # fewer than 10 trades

    def test_open_position_is_marked_to_the_last_close(self):
        closes = np.array([100, 100, 100, 130], dtype=float)
        preds = np.array([0, 0.05, 0.05, 0.05], dtype=float)
        out = retrain.simulate_strategy(closes, preds, 0.01, 0.01, 0.0)
        self.assertEqual(out["trades"], 1)
        self.assertAlmostEqual(out["mean_net"], 0.3)

    def test_no_signal_means_no_trades(self):
        out = retrain.simulate_strategy(np.ones(6), np.zeros(6), 0.01, 0.01, 0.0)
        self.assertEqual(out["trades"], 0)
        self.assertTrue(np.isnan(out["mean_net"]))
        self.assertEqual(out["total_net"], 0.0)
        self.assertFalse(out["passes"])

    def test_enough_profitable_trades_pass(self):
        closes, preds = [100.0], []
        for _ in range(12):  # enter, then exit one bar later at +10%
            closes += [closes[-1], closes[-1] * 1.10]
            preds += [0.05, 0.0]
        closes.append(closes[-1])
        preds.append(0.0)
        out = retrain.simulate_strategy(
            np.array(closes[:len(preds) + 1]), np.array(preds), 0.01, 0.01, 0.005
        )
        self.assertGreaterEqual(out["trades"], 10)
        self.assertGreater(out["mean_net"], 0)
        self.assertTrue(out["passes"])

    def test_losing_trades_do_not_pass_even_when_numerous(self):
        closes, preds = [100.0], []
        for _ in range(12):
            closes += [closes[-1], closes[-1] * 0.95]
            preds += [0.05, 0.0]
        closes.append(closes[-1])
        preds.append(0.0)
        out = retrain.simulate_strategy(
            np.array(closes[:len(preds) + 1]), np.array(preds), 0.01, 0.01, 0.0
        )
        self.assertGreaterEqual(out["trades"], 10)
        self.assertFalse(out["passes"])


class WalkForwardTests(unittest.TestCase):
    def test_result_shape_and_determinism(self):
        first = retrain.walk_forward_check(FRAME, COLUMNS, btc())
        again = retrain.walk_forward_check(FRAME, COLUMNS, btc())
        self.assertEqual(set(first), {"folds_used", "positive_folds", "mean_auc", "fold_aucs", "passes"})
        self.assertEqual(first["folds_used"], len(first["fold_aucs"]))
        self.assertLessEqual(first["folds_used"], retrain.WALK_FORWARD_FOLDS)
        self.assertIsInstance(first["passes"], bool)
        self.assertEqual(first, again)

    def test_classifier_mode_runs_and_differs_from_regressor(self):
        clf = retrain.walk_forward_check(FRAME, COLUMNS, btc(), classifier=True)
        reg = retrain.walk_forward_check(FRAME, COLUMNS, btc())
        self.assertEqual(clf["folds_used"], len(clf["fold_aucs"]))
        self.assertNotEqual(clf["fold_aucs"], reg["fold_aucs"])

    def test_too_little_data_uses_no_folds_and_fails(self):
        out = retrain.walk_forward_check(FRAME.iloc[:120], COLUMNS, btc())
        self.assertEqual(out["folds_used"], 0)
        self.assertTrue(np.isnan(out["mean_auc"]))
        self.assertFalse(out["passes"])

    def test_pass_rule_counts_positive_folds_and_mean_auc(self):
        patch_name(self, "roc_auc_score", modules=RETRAIN_MODULES, side_effect=[0.6, 0.6, 0.6, 0.4])
        out = retrain.walk_forward_check(FRAME, COLUMNS, btc())
        self.assertEqual((out["folds_used"], out["positive_folds"]), (4, 3))
        self.assertTrue(out["passes"])
        patch_name(self, "roc_auc_score", modules=RETRAIN_MODULES, side_effect=[0.6, 0.6, 0.4, 0.4])
        out = retrain.walk_forward_check(FRAME, COLUMNS, btc())
        self.assertFalse(out["passes"])


class ThresholdAndCalibrationTests(unittest.TestCase):
    def test_threshold_is_one_of_the_candidate_percentiles(self):
        rows = FRAME.iloc[: int(len(FRAME) * 0.8)]
        threshold, selected = retrain.select_prediction_threshold(rows, COLUMNS, btc())
        self.assertIn(selected["percentile"], retrain.THRESHOLD_PERCENTILES)
        self.assertEqual(threshold, selected["threshold"])
        self.assertEqual(
            set(selected),
            {"threshold", "percentile", "accuracy", "margin", "precision", "recall", "balanced"},
        )
        again, _ = retrain.select_prediction_threshold(rows, COLUMNS, btc())
        self.assertEqual(threshold, again)

    def test_too_few_rows_raises(self):
        with self.assertRaises(ValueError):
            retrain.select_prediction_threshold(FRAME.iloc[:3], COLUMNS, btc())

    def test_calibration_returns_quantiles(self):
        rng = np.random.default_rng(0)
        out = retrain.log_prediction_calibration(
            "BTC-USD", rng.normal(0, 0.01, 200), rng.normal(0, 0.02, 200), 0.005
        )
        self.assertEqual(list(out), ["p50", "p75", "p90", "p95", "p99"])
        self.assertEqual(sorted(out.values()), list(out.values()))

    def test_calibration_with_constant_predictions_still_returns(self):
        out = retrain.log_prediction_calibration("X", np.zeros(50), np.zeros(50), 0.0)
        self.assertEqual(set(out), {"p50", "p75", "p90", "p95", "p99"})


class TrainAndValidateTests(unittest.TestCase):
    def run_validate(self):
        return retrain.train_and_validate(btc(), HISTORY, SPY)

    def test_metrics_contain_everything_downstream_code_reads(self):
        _, metrics = self.run_validate()
        expected = {
            "ticker", "rows", "accuracy", "roc_auc", "precision", "recall",
            "majority_baseline", "accuracy_margin", "decision_threshold",
            "threshold_percentile", "threshold_validation_precision",
            "threshold_validation_recall", "threshold_validation_balanced",
            "gate_path", "passed", "trading_cost", "entry_bar", "backtest_trades",
            "backtest_mean_net", "backtest_win_rate", "backtest_passes", "wf_folds",
            "wf_positive_folds", "wf_mean_auc", "wf_passes", "prediction_quantiles",
            "_holdout_index",
        }
        self.assertEqual(set(metrics), expected)
        self.assertEqual(metrics["ticker"], "BTC-USD")
        self.assertEqual(metrics["trading_cost"], retrain.ROUND_TRIP_COST["crypto"])
        self.assertEqual(
            metrics["entry_bar"], max(metrics["decision_threshold"], retrain.REGRESSOR_ENTRY_MIN)
        )
        self.assertEqual(len(metrics["_holdout_index"]), metrics["rows"])

    def test_model_is_returned_exactly_when_the_candidate_passed(self):
        model, metrics = self.run_validate()
        self.assertEqual(model is not None, metrics["passed"])
        if model is not None:
            self.assertEqual(model.decision_threshold, metrics["decision_threshold"])

    def test_deterministic(self):
        _, a = self.run_validate()
        _, b = self.run_validate()
        for key in ("accuracy", "roc_auc", "decision_threshold", "backtest_trades", "wf_mean_auc"):
            self.assertEqual(a[key], b[key], key)

    def test_missing_benchmark_raises(self):
        with self.assertRaises(ValueError):
            retrain.train_and_validate(btc(), HISTORY, None)

    def test_too_little_history_raises(self):
        with self.assertRaises(ValueError):
            retrain.train_and_validate(btc(), HISTORY.iloc[-200:], SPY)

    # ---- verdict logic, with the gates forced ---------------------------------
    def force(self, basic, backtest_passes, wf_passes, enforce=True):
        patch_name(self, "quality_gate", modules=RETRAIN_MODULES, return_value=basic)
        patch_name(
            self, "simulate_strategy", modules=RETRAIN_MODULES,
            return_value=sim(20, 0.01 if backtest_passes else -0.01, backtest_passes),
        )
        patch_name(
            self, "walk_forward_check", modules=RETRAIN_MODULES,
            return_value={
                "folds_used": 4, "positive_folds": 4 if wf_passes else 1, "mean_auc": 0.6,
                "fold_aucs": [0.6] * 4, "passes": wf_passes,
            },
        )
        patch_value(self, "ENFORCE_EVIDENCE_GATES", enforce, modules=RETRAIN_MODULES)
        return self.run_validate()

    def test_all_gates_pass_returns_a_model(self):
        model, metrics = self.force((True, "A"), True, True)
        self.assertIsNotNone(model)
        self.assertEqual((metrics["passed"], metrics["gate_path"]), (True, "A"))

    def test_basic_gate_failure_is_a_plain_fail(self):
        model, metrics = self.force((False, "FAIL"), True, True)
        self.assertIsNone(model)
        self.assertEqual((metrics["passed"], metrics["gate_path"]), (False, "FAIL"))

    def test_missing_evidence_is_fail_evidence(self):
        for backtest_ok, wf_ok in ((False, True), (True, False), (False, False)):
            model, metrics = self.force((True, "B"), backtest_ok, wf_ok)
            self.assertIsNone(model)
            self.assertEqual(metrics["gate_path"], "FAIL-EVIDENCE")

    def test_evidence_can_be_switched_off(self):
        model, metrics = self.force((True, "B"), False, False, enforce=False)
        self.assertIsNotNone(model)
        self.assertEqual((metrics["passed"], metrics["gate_path"]), (True, "B"))

    def test_walk_forward_crash_counts_as_failed_evidence(self):
        patch_name(self, "quality_gate", modules=RETRAIN_MODULES, return_value=(True, "A"))
        patch_name(
            self, "simulate_strategy", modules=RETRAIN_MODULES,
            return_value=sim(20, 0.01, True),
        )
        patch_name(
            self, "walk_forward_check", modules=RETRAIN_MODULES, side_effect=RuntimeError("boom")
        )
        model, metrics = self.run_validate()
        self.assertIsNone(model)
        self.assertEqual(metrics["gate_path"], "FAIL-EVIDENCE")
        self.assertEqual((metrics["wf_folds"], metrics["wf_passes"]), (0, False))
        self.assertTrue(np.isnan(metrics["wf_mean_auc"]))


class IncumbentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "model.json"
        XGBRegressorModel("BTC", feature_columns=COLUMNS).fit(
            FRAME[COLUMNS], FRAME["Target_Return"]
        ).save_artifact(cls.path)
        cls.holdout_index = FRAME.index[-250:]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def asset(self):
        return btc(model_path=self.path)

    def test_no_model_file(self):
        out = retrain.evaluate_incumbent(
            btc(model_path=Path("does/not/exist.json")), HISTORY, SPY, self.holdout_index, 0.005
        )
        self.assertEqual(out, {"evaluated": False, "reason": "no model file"})

    def test_no_window_to_score(self):
        out = retrain.evaluate_incumbent(self.asset(), HISTORY, SPY, None, 0.005)
        self.assertEqual(out, {"evaluated": False, "reason": "no holdout window"})

    def test_scores_the_holdout_window(self):
        out = retrain.evaluate_incumbent(self.asset(), HISTORY, SPY, self.holdout_index, 0.005)
        self.assertTrue(out["evaluated"])
        self.assertEqual(out["model_type"], "3-Day Regressor")
        self.assertEqual(set(out), {"evaluated", "model_type", "threshold", "entry_bar", "rows", "backtest"})
        self.assertEqual(out["rows"], len(self.holdout_index))
        self.assertEqual(out["entry_bar"], max(out["threshold"], retrain.REGRESSOR_ENTRY_MIN))

    def test_too_few_comparable_rows(self):
        out = retrain.evaluate_incumbent(self.asset(), HISTORY, SPY, self.holdout_index[:10], 0.005)
        self.assertEqual(out, {"evaluated": False, "reason": "only 10 comparable rows"})

    def test_forward_test_scores_only_bars_after_the_date(self):
        since = HISTORY.index[-40]
        out = retrain.evaluate_incumbent(self.asset(), HISTORY, SPY, None, 0.005, since=since)
        self.assertTrue(out["evaluated"])
        self.assertEqual(out["rows"], 39)

    def test_forward_test_with_no_new_bars(self):
        since = HISTORY.index[-1]
        out = retrain.evaluate_incumbent(self.asset(), HISTORY, SPY, None, 0.005, since=since)
        self.assertEqual(out, {"evaluated": False, "reason": "no new completed bars yet"})

    def test_unreadable_model_is_reported_not_raised(self):
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text("not a model", encoding="utf-8")
        out = retrain.evaluate_incumbent(
            btc(model_path=bad), HISTORY, SPY, self.holdout_index, 0.005
        )
        self.assertEqual(out, {"evaluated": False, "reason": "evaluation error"})


class ClassifierChallengerTests(unittest.TestCase):
    def test_scores_a_fresh_classifier(self):
        out = retrain.evaluate_classifier_challenger(btc(), HISTORY, SPY)
        self.assertTrue(out["evaluated"])
        self.assertEqual(
            set(out),
            {"evaluated", "roc_auc", "backtest_trades", "backtest_mean_net",
             "wf_positive_folds", "wf_folds", "wf_mean_auc", "passed"},
        )
        self.assertIsInstance(out["passed"], bool)

    def test_not_enough_rows(self):
        out = retrain.evaluate_classifier_challenger(btc(), HISTORY.iloc[-200:], SPY)
        self.assertEqual(out, {"evaluated": False, "reason": "not enough rows"})

    def test_errors_are_reported_not_raised(self):
        out = retrain.evaluate_classifier_challenger(btc(), HISTORY, None)
        self.assertEqual(out, {"evaluated": False, "reason": "error"})


class DecideStatusTests(unittest.TestCase):
    METRICS = {"ticker": "BTC-USD", "gate_path": "FAIL", "roc_auc": 0.5, "decision_threshold": 0.01}

    def incumbent(self, trades=20, mean_net=0.01, passes=None, model_type="3-Day Regressor"):
        return {
            "evaluated": True, "model_type": model_type, "threshold": 0.01,
            "entry_bar": 0.01, "rows": 300, "backtest": sim(trades, mean_net, passes),
        }

    def forward(self, trades, mean_net):
        return {"evaluated": True, "rows": 50, "backtest": sim(trades, mean_net)}

    def test_deployed_candidate_allows_buying_and_counts_toward_the_streak(self):
        s = retrain.decide_status(True, self.METRICS, None, previous={"pass_streak": 3})
        self.assertEqual(s["buying"], "allowed")
        self.assertEqual(s["live_model"], "3-Day Regressor (retrained)")
        self.assertEqual(s["reason"], "candidate passed the holdout, walk-forward and net-of-cost gates")
        self.assertEqual((s["evidence_pass"], s["pass_streak"], s["deployed_candidate"]), (True, 4, True))
        self.assertEqual(s["tracking_since"], TODAY)
        self.assertEqual(s["candidate"]["gate_path"], "FAIL")
        self.assertNotIn("incumbent", s)

    def test_unscored_incumbent_fails_open(self):
        for incumbent, why in ((None, "not evaluated"), ({"evaluated": False, "reason": "no model file"}, "no model file")):
            s = retrain.decide_status(False, self.METRICS, incumbent)
            self.assertEqual(s["buying"], "allowed")
            self.assertEqual(s["live_model"], "existing model file (not scored)")
            self.assertIn(f"could not be scored ({why})", s["reason"])
            self.assertEqual((s["evidence_pass"], s["pass_streak"]), (False, 0))

    def test_profitable_incumbent_is_allowed_but_unverified(self):
        s = retrain.decide_status(False, self.METRICS, self.incumbent())
        self.assertEqual(s["buying"], "allowed")
        self.assertEqual(s["live_model"], "3-Day Regressor (existing, unverified)")
        self.assertIn("20 trades, mean net +1.00% per trade", s["reason"])
        self.assertIn("may have trained on that data", s["reason"])
        self.assertTrue(s["evidence_pass"])
        self.assertEqual(s["incumbent"]["rows"], 300)

    def test_an_in_sample_pass_does_not_count_right_after_a_deployment(self):
        s = retrain.decide_status(
            False, self.METRICS, self.incumbent(), previous={"deployed_candidate": True}
        )
        self.assertEqual(s["buying"], "allowed")
        self.assertFalse(s["evidence_pass"])
        self.assertEqual(s["pass_streak"], 0)

    def test_losing_incumbent_blocks_buying(self):
        s = retrain.decide_status(False, self.METRICS, self.incumbent(mean_net=-0.01))
        self.assertEqual(s["buying"], "blocked")
        self.assertEqual(s["live_model"], "3-Day Regressor (existing)")
        self.assertIn("no model passed", s["reason"])
        self.assertIn("mean net -1.00% per trade", s["reason"])
        self.assertFalse(s["evidence_pass"])

    def test_blocking_can_be_switched_off(self):
        patch_value(self, "BLOCK_BUYING_ON_FAILURE", False, modules=RETRAIN_MODULES)
        s = retrain.decide_status(False, self.METRICS, self.incumbent(mean_net=-0.01))
        self.assertEqual(s["buying"], "allowed")

    def test_few_trades_are_flagged_as_untrustworthy(self):
        s = retrain.decide_status(False, self.METRICS, self.incumbent(trades=4, mean_net=0.02, passes=False))
        self.assertEqual(s["buying"], "blocked")
        self.assertIn("fewer than the 10 needed to trust it", s["reason"])

    def test_no_trades_means_too_few_to_judge(self):
        s = retrain.decide_status(
            False, self.METRICS, self.incumbent(trades=0, mean_net=float("nan"), passes=False)
        )
        self.assertIn("only 0 trades (too few to judge)", s["reason"])

    def test_forward_test_with_enough_trades_overrides_a_block(self):
        s = retrain.decide_status(
            False, self.METRICS, self.incumbent(mean_net=-0.01), forward=self.forward(12, 0.02),
            tracking_since="2026-10-01",
        )
        self.assertEqual(s["buying"], "allowed")
        self.assertIn("forward test since 2026-10-01: 12 trades, mean net +2.00% per trade", s["reason"])
        self.assertIn("(out of sample, positive)", s["reason"])
        self.assertTrue(s["evidence_pass"])
        self.assertEqual(s["forward"]["trades"], 12)

    def test_losing_forward_test_blocks_even_a_profitable_incumbent(self):
        s = retrain.decide_status(
            False, self.METRICS, self.incumbent(), forward=self.forward(12, -0.02)
        )
        self.assertEqual(s["buying"], "blocked")
        self.assertIn("(out of sample, losing)", s["reason"])
        self.assertFalse(s["evidence_pass"])

    def test_forward_test_with_too_few_trades_is_only_recorded(self):
        s = retrain.decide_status(
            False, self.METRICS, self.incumbent(mean_net=-0.01), forward=self.forward(3, 0.05)
        )
        self.assertEqual(s["buying"], "blocked")
        self.assertEqual(s["forward"]["trades"], 3)

    def test_unevaluated_forward_test_is_ignored(self):
        s = retrain.decide_status(
            False, self.METRICS, self.incumbent(), forward={"evaluated": False, "reason": "x"}
        )
        self.assertNotIn("forward", s)

    def test_blocked_asset_needs_consecutive_passes_to_reopen(self):
        first = retrain.decide_status(
            False, self.METRICS, self.incumbent(), previous={"buying": "blocked", "pass_streak": 0}
        )
        self.assertEqual(first["buying"], "blocked")
        self.assertEqual(first["pass_streak"], 1)
        self.assertIn("pass streak 1/2", first["reason"])
        second = retrain.decide_status(
            False, self.METRICS, self.incumbent(), previous={"buying": "blocked", "pass_streak": 1}
        )
        self.assertEqual(second["buying"], "allowed")
        self.assertEqual(second["pass_streak"], 2)

    def test_streak_rule_does_not_apply_to_fail_open_or_forward_decisions(self):
        open_ = retrain.decide_status(False, self.METRICS, None, previous={"buying": "blocked"})
        self.assertEqual(open_["buying"], "allowed")
        fwd = retrain.decide_status(
            False, self.METRICS, self.incumbent(mean_net=-0.01),
            forward=self.forward(12, 0.02), previous={"buying": "blocked"},
        )
        self.assertEqual(fwd["buying"], "allowed")

    def test_a_failing_week_resets_the_streak(self):
        s = retrain.decide_status(
            False, self.METRICS, self.incumbent(mean_net=-0.01), previous={"pass_streak": 5}
        )
        self.assertEqual(s["pass_streak"], 0)

    def test_challenger_and_tracking_date_are_recorded(self):
        s = retrain.decide_status(
            False, self.METRICS, self.incumbent(), challenger={"evaluated": True, "passed": False},
            tracking_since="2026-09-01",
        )
        self.assertEqual(s["classifier_challenger"], {"evaluated": True, "passed": False})
        self.assertEqual(s["tracking_since"], "2026-09-01")


class StatusFileTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        patch_value(self, "MODEL_DIR", self.dir, modules=RETRAIN_MODULES)
        self.file = self.dir / retrain.STATUS_FILE_NAME

    def test_json_safe_converts_numpy_and_non_finite_values(self):
        out = retrain._json_safe(
            {"a": np.float64(1.5), "b": np.int64(3), "c": float("nan"), "d": float("inf"),
             "e": np.bool_(True), "f": [np.float32(2.0), (1, 2)], 5: "x"}
        )
        self.assertEqual(out, {"a": 1.5, "b": 3, "c": None, "d": None, "e": True, "f": [2.0, [1, 2]], "5": "x"})
        json.dumps(out)

    def test_missing_or_corrupt_file_reads_as_empty(self):
        self.assertEqual(retrain.read_previous_assets(), {})
        self.file.write_text("{broken", encoding="utf-8")
        self.assertEqual(retrain.read_previous_assets(), {})

    def test_write_merges_with_existing_assets(self):
        self.file.write_text(json.dumps({"assets": {"SPY": {"buying": "allowed"}}}), encoding="utf-8")
        self.assertTrue(retrain.write_model_status({"BTC-USD": {"buying": "blocked", "x": float("nan")}}))
        data = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertEqual(sorted(data["assets"]), ["BTC-USD", "SPY"])
        self.assertIsNone(data["assets"]["BTC-USD"]["x"])
        self.assertIn("updated_at", data)
        self.assertEqual(retrain.read_previous_assets()["SPY"], {"buying": "allowed"})

    def test_write_replaces_an_asset_and_starts_fresh_if_the_file_is_corrupt(self):
        self.file.write_text("{broken", encoding="utf-8")
        self.assertTrue(retrain.write_model_status({"SPY": {"buying": "allowed"}}))
        self.assertEqual(sorted(json.loads(self.file.read_text(encoding="utf-8"))["assets"]), ["SPY"])

    def test_write_failure_returns_false(self):
        blocker = self.dir / "blocker"
        blocker.write_text("a file, not a folder", encoding="utf-8")
        patch_value(self, "MODEL_DIR", blocker, modules=RETRAIN_MODULES)
        self.assertFalse(retrain.write_model_status({"SPY": {}}))

    def test_tracking_start_prefers_tracking_since_then_checked_at_then_today(self):
        self.assertEqual(retrain.tracking_start({"tracking_since": "2026-10-01", "checked_at": "2026-09-05T10:00:00"}), "2026-10-01")
        self.assertEqual(retrain.tracking_start({"checked_at": "2026-09-05T10:00:00+00:00"}), "2026-09-05")
        self.assertEqual(retrain.tracking_start({}), TODAY)
        self.assertEqual(retrain.tracking_start({"tracking_since": "bad"}), TODAY)

    def test_resolve_model_path_and_trading_cost(self):
        self.assertEqual(retrain.resolve_model_path({"model_path": "m.json"}), self.dir / "m.json")
        absolute = (self.dir / "x.json").resolve()
        self.assertEqual(retrain.resolve_model_path({"model_path": absolute}), absolute)
        self.assertEqual(retrain.trading_cost({"type": "crypto"}), 0.005)
        self.assertEqual(retrain.trading_cost({"type": "stock"}), 0.0005)
        self.assertEqual(retrain.trading_cost({"type": "other"}), 0.005)


class DownloadHistoryTests(unittest.TestCase):
    def raw(self, history):
        return history.copy()

    def test_empty_download_raises(self):
        yf = patch_name(self, "yf", modules=RETRAIN_MODULES)
        yf.download.return_value = pd.DataFrame()
        with self.assertRaises(ValueError):
            retrain.download_history("BTC-USD")

    def test_forming_bar_is_dropped(self):
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        history = HISTORY.copy()
        history.loc[today] = history.iloc[-1]
        yf = patch_name(self, "yf", modules=RETRAIN_MODULES)
        yf.download.return_value = history
        out = retrain.download_history("BTC-USD")
        self.assertLess(out.index[-1], today)
        self.assertEqual(len(out), len(HISTORY))

    def test_short_history_raises(self):
        yf = patch_name(self, "yf", modules=RETRAIN_MODULES)
        yf.download.return_value = HISTORY.iloc[-100:]
        with self.assertRaises(ValueError):
            retrain.download_history("BTC-USD")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.send = patch_name(self, "send_telegram_message", modules=RETRAIN_MODULES)

    def result(self, ticker, **kw):
        base = {
            "ticker": ticker, "roc_auc": 0.5312, "wf_positive_folds": 3, "wf_folds": 4,
            "backtest_trades": 12, "backtest_mean_net": 0.0123, "passed": True,
            "gate_path": "A",
        }
        base.update(kw)
        return base

    def status(self, **kw):
        base = {"buying": "allowed", "live_model": "3-Day Regressor (retrained)", "reason": "all good"}
        base.update(kw)
        return base

    def test_full_report_layout(self):
        results = [
            self.result("SPY"),
            self.result("BTC-USD", passed=False, gate_path="FAIL-EVIDENCE", backtest_mean_net=-0.004),
            {"ticker": "ETH-USD", "error": "download error"},
        ]
        statuses = {
            "SPY": self.status(),
            "BTC-USD": self.status(
                buying="blocked", live_model="Legacy Classifier (existing)", reason="no edge",
                forward={"since": "2026-10-01", "trades": 4, "mean_net": 0.01},
                classifier_challenger={"evaluated": True, "roc_auc": 0.51, "wf_positive_folds": 2,
                                       "wf_folds": 4, "backtest_trades": 7, "passed": False},
            ),
        }
        retrain._send_retraining_report(results, ["SPY"], statuses)
        self.send.assert_called_once()
        message = self.send.call_args.args[0]
        self.assertEqual(self.send.call_args.kwargs, {"parse_mode": "HTML"})
        self.assertTrue(message.startswith("<b>MODEL RETRAINING REPORT</b>\n<pre>Ticker"))
        self.assertIn("SPY     0.531 3/4  12 +1.23 \U0001F7E2\n", message)
        self.assertIn("BTC-USD 0.531 3/4  12 -0.40 \U0001F534\n", message)
        self.assertIn("ETH-USD    --  --  --    -- ERR</pre>", message)
        self.assertIn("SPY: 3-Day Regressor (retrained) (updated). all good [Path A]", message)
        self.assertIn("BTC-USD: Legacy Classifier (existing) (kept). no edge [No edge]", message)
        self.assertIn("Forward test since 2026-10-01: 4 trades, mean net +1.00% (counts at 10+ trades)", message)
        self.assertIn("Classifier challenger: AUC 0.510, WF 2/4, 7 trades, did not pass (report-only)", message)
        self.assertTrue(message.rstrip().endswith("not pass (report-only)"))

    def test_html_in_names_is_escaped(self):
        retrain._send_retraining_report(
            [self.result("SPY")], [], {"SPY": self.status(reason="a < b & c")}
        )
        self.assertIn("a &lt; b &amp; c", self.send.call_args.args[0])

    def test_long_reasons_are_cut(self):
        retrain._send_retraining_report(
            [self.result("SPY")], [], {"SPY": self.status(reason="x" * 1000)}
        )
        message = self.send.call_args.args[0]
        self.assertIn("x" * 600, message)
        self.assertNotIn("x" * 601, message)


class MainTests(unittest.TestCase):
    def setUp(self):
        mods = RETRAIN_MODULES
        self.download = patch_name(self, "download_history", modules=mods, side_effect=lambda t: HISTORY)
        self.validate = patch_name(self, "train_and_validate", modules=mods)
        self.incumbent = patch_name(
            self, "evaluate_incumbent", modules=mods,
            return_value={"evaluated": False, "reason": "no model file"},
        )
        self.challenger = patch_name(
            self, "evaluate_classifier_challenger", modules=mods, return_value={"evaluated": False}
        )
        self.write = patch_name(self, "write_model_status", modules=mods, return_value=True)
        self.previous = patch_name(self, "read_previous_assets", modules=mods, return_value={})
        self.report = patch_name(self, "send_retraining_report", modules=mods)
        self.metrics = {
            "ticker": "X", "gate_path": "A", "roc_auc": 0.6, "decision_threshold": 0.01,
            "_holdout_index": FRAME.index[-50:],
        }

    def accepted(self):
        model = mock.Mock()
        self.validate.side_effect = lambda asset, history, spy: (model, dict(self.metrics))
        return model

    def rejected(self):
        self.validate.side_effect = lambda asset, history, spy: (
            None, dict(self.metrics, gate_path="FAIL", passed=False)
        )

    def test_all_accepted_saves_every_model_and_returns_zero(self):
        model = self.accepted()
        self.assertEqual(retrain.main(), 0)
        self.assertEqual(model.save_artifact.call_count, len(config.ASSETS))
        statuses = self.write.call_args.args[0]
        self.assertEqual(sorted(statuses), sorted(a["ticker"] for a in config.ASSETS))
        for status in statuses.values():
            self.assertEqual(status["buying"], "allowed")
            self.assertTrue(status["deployed_candidate"])
        updated = self.report.call_args.args[1]
        self.assertEqual(sorted(updated), sorted(a["ticker"] for a in config.ASSETS))
        self.incumbent.assert_not_called()

    def test_rejection_scores_the_incumbent_and_a_forward_test(self):
        self.rejected()
        self.assertEqual(retrain.main(), 0)  # a retained model is not an error
        self.assertEqual(self.incumbent.call_count, 2 * len(config.ASSETS))
        forward_calls = [c for c in self.incumbent.call_args_list if "since" in c.kwargs]
        self.assertEqual(len(forward_calls), len(config.ASSETS))
        self.assertIsNone(forward_calls[0].args[3])
        statuses = self.write.call_args.args[0]
        self.assertEqual(statuses["BTC-USD"]["live_model"], "existing model file (not scored)")
        self.assertEqual(self.report.call_args.args[1], [])
        self.challenger.assert_not_called()

    def test_classifier_incumbent_triggers_the_challenger(self):
        self.rejected()
        self.incumbent.return_value = {
            "evaluated": True, "model_type": "Legacy Classifier", "threshold": 0.55,
            "entry_bar": 0.6, "rows": 300, "backtest": sim(20, 0.01),
        }
        retrain.main()
        self.assertEqual(self.challenger.call_count, len(config.ASSETS))

    def test_previous_status_feeds_the_decision(self):
        self.rejected()
        self.previous.return_value = {"BTC-USD": {"buying": "blocked", "pass_streak": 0, "tracking_since": "2026-10-01"}}
        scored = {
            "evaluated": True, "model_type": "3-Day Regressor", "threshold": 0.01,
            "entry_bar": 0.01, "rows": 300, "backtest": sim(20, 0.01),
        }
        # The holdout score passes, but the forward test has no new bars yet.
        self.incumbent.side_effect = lambda *a, **k: (
            {"evaluated": False, "reason": "no new completed bars yet"} if "since" in k else scored
        )
        retrain.main()
        status = self.write.call_args.args[0]["BTC-USD"]
        self.assertEqual(status["buying"], "blocked")
        self.assertEqual(status["tracking_since"], "2026-10-01")
        self.assertEqual(status["pass_streak"], 1)

    def test_a_positive_forward_test_reopens_buying_despite_a_previous_block(self):
        self.rejected()
        self.previous.return_value = {"BTC-USD": {"buying": "blocked", "pass_streak": 0}}
        self.incumbent.return_value = {
            "evaluated": True, "model_type": "3-Day Regressor", "threshold": 0.01,
            "entry_bar": 0.01, "rows": 300, "backtest": sim(20, 0.01),
        }
        retrain.main()
        status = self.write.call_args.args[0]["BTC-USD"]
        self.assertEqual(status["buying"], "allowed")
        self.assertIn("out of sample, positive", status["reason"])

    def test_training_error_returns_one_and_keeps_going(self):
        model = mock.Mock()

        def validate(asset, history, spy):
            if asset["ticker"] == "SPY":
                raise RuntimeError("boom")
            return model, dict(self.metrics)

        self.validate.side_effect = validate
        self.assertEqual(retrain.main(), 1)
        self.assertEqual(model.save_artifact.call_count, len(config.ASSETS) - 1)
        results = self.report.call_args.args[0]
        self.assertIn({"ticker": "SPY", "error": "training or save error"}, results)

    def test_download_error_returns_one_and_skips_that_asset(self):
        def download(ticker):
            if ticker == "ETH-USD":
                raise RuntimeError("no data")
            return HISTORY

        self.download.side_effect = download
        self.accepted()
        self.assertEqual(retrain.main(), 1)
        self.assertEqual(self.validate.call_count, len(config.ASSETS) - 1)
        self.assertIn({"ticker": "ETH-USD", "error": "download error"}, self.report.call_args.args[0])

    def test_status_file_failure_returns_one(self):
        self.accepted()
        self.write.return_value = False
        self.assertEqual(retrain.main(), 1)

    def test_benchmark_is_the_spy_close(self):
        self.accepted()
        retrain.main()
        spy_arg = self.validate.call_args.args[2]
        self.assertTrue(spy_arg.equals(HISTORY["Close"]))

    def test_missing_spy_download_passes_none_benchmark(self):
        def download(ticker):
            if ticker == "SPY":
                raise RuntimeError("no data")
            return HISTORY

        self.download.side_effect = download
        self.accepted()
        retrain.main()
        self.assertIsNone(self.validate.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
