"""Tests for the live-trading safeguards in run_execution.py.

Run from the project root:  python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_execution as rx  # noqa: E402
from execution import safeguards  # noqa: E402


def env(**values):
    """Patch the safeguard env vars to exactly `values` (everything else unset)."""
    keys = (
        "ALPACA_PAPER",
        rx.LIVE_CONFIRM_ENV,
        rx.LIVE_MAX_CAPITAL_ENV,
        rx.HALT_ENV,
        rx.MAX_DAILY_LOSS_ENV,
    )
    base = {k: v for k, v in os.environ.items() if k not in keys}
    base.update({k: str(v) for k, v in values.items()})
    return mock.patch.dict(os.environ, base, clear=True)


class LiveCapitalCapTests(unittest.TestCase):
    def test_paper_mode_has_no_cap(self):
        with env():
            self.assertTrue(rx.is_paper_mode())
            self.assertIsNone(rx.live_capital_cap())

    def test_live_without_confirmation_is_refused(self):
        with env(ALPACA_PAPER="false", ALPACA_LIVE_MAX_CAPITAL="100"):
            with self.assertRaises(ValueError):
                rx.live_capital_cap()

    def test_live_with_wrong_phrase_is_refused(self):
        with env(
            ALPACA_PAPER="false",
            ALPACA_LIVE_CONFIRM="yes",
            ALPACA_LIVE_MAX_CAPITAL="100",
        ):
            with self.assertRaises(ValueError):
                rx.live_capital_cap()

    def test_live_without_capital_cap_is_refused(self):
        with env(ALPACA_PAPER="false", ALPACA_LIVE_CONFIRM=rx.LIVE_CONFIRM_PHRASE):
            with self.assertRaises(ValueError):
                rx.live_capital_cap()

    def test_live_with_bad_capital_cap_is_refused(self):
        for bad in ("0", "-5", "abc", "nan", "inf"):
            with env(
                ALPACA_PAPER="false",
                ALPACA_LIVE_CONFIRM=rx.LIVE_CONFIRM_PHRASE,
                ALPACA_LIVE_MAX_CAPITAL=bad,
            ):
                with self.assertRaises(ValueError, msg=bad):
                    rx.live_capital_cap()

    def test_live_fully_configured_returns_cap(self):
        with env(
            ALPACA_PAPER="false",
            ALPACA_LIVE_CONFIRM=rx.LIVE_CONFIRM_PHRASE,
            ALPACA_LIVE_MAX_CAPITAL="250",
        ):
            self.assertEqual(rx.live_capital_cap(), 250.0)


class HaltTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(safeguards, "HALT_FILE", Path(self._tmp.name) / "HALT_TRADING")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_halt_when_flat_day(self):
        with env():
            self.assertIsNone(rx.entries_halt_reason(1000.0, 1000.0))

    def test_env_kill_switch(self):
        with env(TRADING_HALTED="true"):
            self.assertIn("kill switch", rx.entries_halt_reason(1000.0, 1000.0))

    def test_file_kill_switch(self):
        safeguards.HALT_FILE.write_text("stop", encoding="utf-8")
        with env():
            self.assertIn("kill switch", rx.entries_halt_reason(1000.0, 1000.0))

    def test_daily_loss_breaker_trips_at_default_limit(self):
        with env():
            self.assertIn("circuit breaker", rx.entries_halt_reason(969.0, 1000.0))

    def test_small_loss_does_not_trip(self):
        with env():
            self.assertIsNone(rx.entries_halt_reason(985.0, 1000.0))

    def test_custom_loss_limit(self):
        with env(MAX_DAILY_LOSS_PCT="0.01"):
            self.assertIn("circuit breaker", rx.entries_halt_reason(985.0, 1000.0))

    def test_invalid_loss_limit_falls_back_to_default(self):
        with env(MAX_DAILY_LOSS_PCT="banana"):
            self.assertEqual(rx.max_daily_loss_pct(), rx.DEFAULT_MAX_DAILY_LOSS_PCT)

    def test_missing_last_equity_does_not_trip(self):
        with env():
            self.assertIsNone(rx.entries_halt_reason(500.0, None))


class ApplyCapitalCapTests(unittest.TestCase):
    def test_paper_mode_is_unchanged(self):
        self.assertEqual(
            rx.apply_capital_cap(mock.Mock(), None, 100000.0, 200000.0),
            (100000.0, 200000.0),
        )

    def test_live_cap_limits_equity_and_buying_power(self):
        with mock.patch.object(safeguards, "get_position_value", return_value=0.0):
            self.assertEqual(
                rx.apply_capital_cap(mock.Mock(), 100.0, 5000.0, 5000.0), (100.0, 100.0)
            )

    def test_live_cap_subtracts_existing_positions(self):
        with mock.patch.object(safeguards, "get_position_value", return_value=30.0):
            equity, buying_power = rx.apply_capital_cap(mock.Mock(), 100.0, 5000.0, 5000.0)
        # 3 assets x $30 invested = $90 of the $100 cap used
        self.assertEqual(equity, 100.0)
        self.assertAlmostEqual(buying_power, 10.0)

    def test_overinvested_never_negative(self):
        with mock.patch.object(safeguards, "get_position_value", return_value=500.0):
            _, buying_power = rx.apply_capital_cap(mock.Mock(), 100.0, 5000.0, 5000.0)
        self.assertEqual(buying_power, 0.0)


if __name__ == "__main__":
    unittest.main()
