"""Tests for the year-by-year arithmetic in research/yearly.py."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research import yearly as yr  # noqa: E402


def make_index(*year_lengths):
    """Consecutive daily index with the given (year, days) blocks."""
    parts = [pd.date_range(f"{year}-01-01", periods=days, freq="D") for year, days in year_lengths]
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def rows(index, strategy, buy_hold, position, split=2022, cost=0.0, trials=200):
    return yr.yearly_rows(
        index, np.asarray(strategy, float), np.asarray(buy_hold, float),
        np.asarray(position, float), cost, split, trials, np.random.default_rng(0),
    )


def by_period(result):
    return {row["period"]: row for row in result}


class YearlyRowsTests(unittest.TestCase):
    def setUp(self):
        self.index = make_index((2021, 10), (2022, 10), (2023, 10))
        self.bh = np.full(30, 0.01)

    def test_one_row_per_year_plus_the_two_split_periods(self):
        out = rows(self.index, np.zeros(30), self.bh, np.zeros(30), split=2023)
        self.assertEqual(
            [r["period"] for r in out], ["2021", "2022", "2023", "before 2023", "2023 onward"]
        )
        self.assertEqual([r["days"] for r in out], [10, 10, 10, 20, 10])

    def test_a_split_year_outside_the_data_gives_one_period_only(self):
        out = rows(self.index, np.zeros(30), self.bh, np.zeros(30), split=2030)
        self.assertEqual([r["period"] for r in out][-1], "before 2030")
        self.assertNotIn("2030 onward", [r["period"] for r in out])

    def test_never_invested_earns_nothing_and_matches_nothing(self):
        out = by_period(rows(self.index, np.zeros(30), self.bh, np.zeros(30)))
        for row in out.values():
            self.assertEqual((row["exposure"], row["trades"], row["strategy"]), (0.0, 0, 0.0))
            self.assertEqual(row["matched_bh"], 0.0)
            self.assertTrue(np.isnan(row["random_p"]))
        self.assertAlmostEqual(out["2021"]["buy_hold"], 1.01 ** 10 - 1)

    def test_always_invested_equals_buy_and_hold_with_one_trade(self):
        out = by_period(rows(self.index, self.bh, self.bh, np.ones(30)))
        self.assertEqual(out["2021"]["trades"], 1)  # the entry is counted once, in year one
        self.assertEqual(out["2022"]["trades"], 0)
        for year in ("2021", "2022", "2023"):
            self.assertAlmostEqual(out[year]["exposure"], 1.0)
            self.assertAlmostEqual(out[year]["strategy"], out[year]["buy_hold"])
            self.assertAlmostEqual(out[year]["excess_vs_matched"], 0.0)

    def test_matched_is_buy_and_hold_scaled_by_exposure(self):
        position = np.zeros(30)
        position[:5] = 1.0  # invested half of 2021 only
        strategy = position * self.bh
        out = by_period(rows(self.index, strategy, self.bh, position))
        self.assertAlmostEqual(out["2021"]["exposure"], 0.5)
        self.assertAlmostEqual(out["2021"]["matched_bh"], 1.005 ** 10 - 1)
        self.assertAlmostEqual(out["2021"]["strategy"], 1.01 ** 5 - 1)
        # constant market: half the time invested earns about half the market, no edge
        self.assertAlmostEqual(out["2021"]["excess_vs_matched"], 0.0, places=3)
        self.assertEqual(out["2022"]["exposure"], 0.0)

    def test_trades_are_counted_in_the_year_they_start(self):
        position = np.zeros(30)
        position[8:12] = 1.0   # starts 2021, runs into 2022
        position[25:28] = 1.0  # starts 2023
        out = by_period(rows(self.index, position * self.bh, self.bh, position, split=2023))
        self.assertEqual((out["2021"]["trades"], out["2022"]["trades"], out["2023"]["trades"]), (1, 0, 1))
        self.assertEqual(out["before 2023"]["trades"], 1)
        self.assertEqual(out["2023 onward"]["trades"], 1)

    def test_random_p_is_high_when_the_trades_are_unremarkable(self):
        position = np.zeros(30)
        position[2:5] = 1.0
        out = by_period(rows(self.index, position * self.bh, self.bh, position, trials=300))
        # constant market: any placement of the same trade earns the same, so p = 1
        self.assertEqual(out["2021"]["random_p"], 1.0)

    def test_random_p_is_small_when_the_strategy_catches_the_big_move(self):
        market = np.zeros(30)
        market[3:6] = 0.10  # the only big move of 2021
        position = np.zeros(30)
        position[3:6] = 1.0
        out = by_period(rows(self.index, position * market, market, position, trials=500))
        self.assertLess(out["2021"]["random_p"], 0.2)

    def test_costs_enter_the_random_comparison(self):
        position = np.zeros(30)
        position[2:5] = 1.0
        free = by_period(rows(self.index, position * self.bh, self.bh, position, cost=0.0))
        self.assertEqual(free["2021"]["random_p"], 1.0)


class PeriodRowTests(unittest.TestCase):
    def test_exposure_matched_comparison_uses_the_selected_days_only(self):
        mask = np.array([True, True, False, False])
        out = yr.period_row(
            "x", mask, np.array([0.1, 0.0, 9.0, 9.0]), np.array([0.1, 0.1, 9.0, 9.0]),
            np.array([1.0, 0.0, 1.0, 1.0]), np.array([True, False, False, False]),
            0.0, 50, np.random.default_rng(0),
        )
        self.assertEqual(out["days"], 2)
        self.assertAlmostEqual(out["exposure"], 0.5)
        self.assertAlmostEqual(out["strategy"], 0.1)
        self.assertAlmostEqual(out["buy_hold"], 1.1 * 1.1 - 1)
        self.assertEqual(out["trades"], 1)


if __name__ == "__main__":
    unittest.main()
