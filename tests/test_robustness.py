"""Tests for the pure helpers in robustness.py."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import robustness as rb  # noqa: E402


class HoldingPeriodTests(unittest.TestCase):
    def test_runs_are_extracted(self):
        self.assertEqual(rb.holding_periods(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])), [2, 1, 3])

    def test_trailing_run_is_kept(self):
        self.assertEqual(rb.holding_periods(np.array([1, 1])), [2])

    def test_no_positions(self):
        self.assertEqual(rb.holding_periods(np.zeros(5)), [])


class RandomTimingTests(unittest.TestCase):
    def test_full_window_hold_is_deterministic(self):
        returns = np.array([0.01, -0.02, 0.03, 0.0])
        out = rb.random_timing_returns(returns, [4], 0.0, 20, np.random.default_rng(0))
        self.assertTrue(np.allclose(out, np.prod(1 + returns) - 1))

    def test_costs_are_charged_per_trade(self):
        returns = np.zeros(10)
        out = rb.random_timing_returns(returns, [2, 3], 0.01, 50, np.random.default_rng(1))
        # Two trades, each pays one_side on entry and on exit unless the run reaches the end.
        self.assertTrue(np.all(out <= 0.0))
        self.assertTrue(np.all(out >= (1 - 0.01) ** 4 - 1 - 1e-12))

    def test_never_overlaps_or_overflows(self):
        returns = np.ones(12) * 0.01
        out = rb.random_timing_returns(returns, [5, 4], 0.0, 200, np.random.default_rng(2))
        # Exactly 9 invested bars every time, so the return is identical in every trial.
        self.assertTrue(np.allclose(out, 1.01 ** 9 - 1))

    def test_impossible_placement_returns_nan(self):
        out = rb.random_timing_returns(np.zeros(5), [4, 4], 0.0, 3, np.random.default_rng(3))
        self.assertTrue(np.all(np.isnan(out)))

    def test_no_trades_returns_nan(self):
        out = rb.random_timing_returns(np.zeros(5), [], 0.0, 3, np.random.default_rng(3))
        self.assertTrue(np.all(np.isnan(out)))


if __name__ == "__main__":
    unittest.main()
