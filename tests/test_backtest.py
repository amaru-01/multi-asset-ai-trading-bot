"""Tests for the pure simulation and metrics code in backtest.py."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research import backtest as bt  # noqa: E402


def run(opens, preds, thresholds, first=0, cost=0.0, final_close=None, entry_min=0.0):
    opens = np.asarray(opens, dtype=float)
    return bt.simulate(
        opens,
        float(opens[-1] if final_close is None else final_close),
        np.asarray(preds, dtype=float),
        np.asarray(thresholds, dtype=float),
        first,
        cost,
        entry_min,
    )


class SimulateTests(unittest.TestCase):
    def test_never_signalling_earns_nothing(self):
        out = run([100, 101, 102, 103, 104], [0.0] * 5, [0.01] * 5)
        self.assertTrue(np.allclose(out["strategy_returns"], 0.0))
        self.assertEqual(out["trades"], [])

    def test_always_long_matches_buy_and_hold_when_free(self):
        # Prediction is always above the threshold, so the bot enters at the first open.
        opens = [100, 102, 101, 105, 107, 110]
        out = run(opens, [0.05] * 6, [0.01] * 6)
        # Entered at open[1]; returns from open[1] onward equal buy-and-hold over j>=2.
        self.assertAlmostEqual(
            np.prod(1 + out["strategy_returns"]), opens[-1] / opens[1], places=9
        )

    def test_signal_acts_on_next_open_not_same_bar(self):
        # Signal fires on bar 1 only; entry must be at open[2], exit at open[3].
        opens = [100, 100, 110, 121, 121]
        preds = [0, 0.05, 0, 0, 0]
        out = run(opens, preds, [0.01] * 5)
        self.assertEqual(len(out["trades"]), 1)
        self.assertAlmostEqual(out["trades"][0], 121 / 110 - 1.0, places=9)

    def test_costs_are_charged_on_entry_and_exit(self):
        opens = [100, 100, 100, 100, 100]
        preds = [0, 0.05, 0, 0, 0]
        free = run(opens, preds, [0.01] * 5, cost=0.0)
        paid = run(opens, preds, [0.01] * 5, cost=0.01)
        self.assertAlmostEqual(free["trades"][0], 0.0)
        self.assertAlmostEqual(paid["trades"][0], -0.02)
        self.assertAlmostEqual(float(np.sum(paid["strategy_returns"])), -0.02, places=9)

    def test_entry_minimum_blocks_marginal_signals(self):
        opens = [100, 100, 110, 121, 121]
        preds = [0, 0.004, 0, 0, 0]
        out = run(opens, preds, [0.001] * 5, entry_min=0.006)
        self.assertEqual(out["trades"], [])

    def test_open_position_is_marked_to_final_close(self):
        opens = [100, 100, 100, 100]
        out = run(opens, [0, 0.05, 0.05, 0.05], [0.01] * 4, final_close=120.0)
        self.assertAlmostEqual(out["trades"][0], 0.2, places=9)

    def test_first_bar_offset_ignores_earlier_signals(self):
        opens = [100, 110, 121, 121]
        out = run(opens, [0.05, 0.05, 0.0, 0.0], [0.01] * 4, first=2)
        self.assertEqual(out["trades"], [])


class MetricsTests(unittest.TestCase):
    def test_total_return_and_drawdown(self):
        m = bt.metrics(np.array([0.10, -0.10, 0.10]), 252)
        self.assertAlmostEqual(m["total_return"], 1.1 * 0.9 * 1.1 - 1.0)
        self.assertAlmostEqual(m["max_drawdown"], -0.10)

    def test_zero_volatility_sharpe_is_nan(self):
        self.assertTrue(np.isnan(bt.metrics(np.zeros(10), 252)["sharpe"]))


if __name__ == "__main__":
    unittest.main()
