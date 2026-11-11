"""Tests for the risk-only mode and the kill-switch source message."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_execution as rx  # noqa: E402

BTC = {
    "ticker": "BTC-USD", "symbol": "BTC", "type": "crypto",
    "alpaca_symbol": "BTC/USD", "feature_columns": ["a"],
}
ETH = dict(BTC, ticker="ETH-USD", symbol="ETH", alpaca_symbol="ETH/USD")
SPY = {
    "ticker": "SPY", "symbol": "SPY", "type": "stock",
    "alpaca_symbol": "SPY", "feature_columns": ["a"],
}


def position(entry, current):
    return mock.Mock(avg_entry_price=str(entry), current_price=str(current), symbol="BTCUSD")


class RiskOnlyTests(unittest.TestCase):
    def setUp(self):
        patches = {
            "get_open_position": mock.patch.object(rx, "get_open_position"),
            "get_latest_features": mock.patch.object(rx, "get_latest_features"),
            "cancel": mock.patch.object(rx, "cancel_open_orders"),
            "close": mock.patch.object(rx, "close_position_if_open"),
            "telegram": mock.patch.object(rx, "send_telegram_message"),
        }
        self.m = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_no_position_skips_market_data(self):
        self.m["get_open_position"].return_value = None
        self.assertEqual(rx.run_risk_only([BTC, ETH], mock.Mock()), 0)
        self.m["get_latest_features"].assert_not_called()
        self.m["close"].assert_not_called()

    def test_stock_assets_are_ignored(self):
        self.assertEqual(rx.run_risk_only([SPY], mock.Mock()), 0)
        self.m["get_open_position"].assert_not_called()

    def test_price_within_stop_does_nothing(self):
        # entry 100, ATR 2 -> stop at 97; price 99 is above it
        self.m["get_open_position"].return_value = position(100, 99)
        self.m["get_latest_features"].return_value = (None, 99.0, 2.0)
        self.assertEqual(rx.run_risk_only([BTC], mock.Mock()), 0)
        self.m["close"].assert_not_called()
        self.m["telegram"].assert_not_called()

    def test_price_through_stop_closes_and_alerts(self):
        self.m["get_open_position"].return_value = position(100, 96)
        self.m["get_latest_features"].return_value = (None, 96.0, 2.0)
        self.assertEqual(rx.run_risk_only([BTC], mock.Mock()), 0)
        self.m["cancel"].assert_called_once()
        self.m["close"].assert_called_once()
        self.m["telegram"].assert_called_once()
        self.assertIn("STOP-LOSS", self.m["telegram"].call_args.args[0])

    def test_failure_on_one_asset_still_checks_the_next_and_returns_1(self):
        self.m["get_open_position"].side_effect = [RuntimeError("boom"), position(100, 90)]
        self.m["get_latest_features"].return_value = (None, 90.0, 2.0)
        self.assertEqual(rx.run_risk_only([BTC, ETH], mock.Mock()), 1)
        self.m["close"].assert_called_once()

    def test_never_places_buy_orders(self):
        client = mock.Mock()
        self.m["get_open_position"].return_value = position(100, 99)
        self.m["get_latest_features"].return_value = (None, 99.0, 2.0)
        rx.run_risk_only([BTC, ETH], client)
        client.submit_order.assert_not_called()


class MainRiskOnlyTests(unittest.TestCase):
    def test_risk_only_does_not_require_live_confirmation(self):
        env = {"ALPACA_PAPER": "false"}  # live, with no confirm phrase or capital cap
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(rx, "create_trading_client", return_value=mock.Mock()), \
                mock.patch.object(rx, "run_risk_only", return_value=0) as risk:
            self.assertEqual(rx.main(["--risk-only"]), 0)
        risk.assert_called_once()

    def test_normal_live_run_still_requires_confirmation(self):
        with mock.patch.dict(os.environ, {"ALPACA_PAPER": "false"}), \
                mock.patch.object(rx, "create_trading_client") as create:
            self.assertEqual(rx.main([]), 1)
        create.assert_not_called()

    def test_risk_only_returns_1_if_client_cannot_start(self):
        with mock.patch.object(rx, "create_trading_client", side_effect=ValueError("no keys")):
            self.assertEqual(rx.main(["--risk-only"]), 1)


class KillSwitchSourceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(rx, "HALT_FILE", Path(tmp.name) / "HALT_TRADING")
        patcher.start()
        self.addCleanup(patcher.stop)

    def clean_env(self, **values):
        base = {k: v for k, v in os.environ.items() if k != rx.HALT_ENV}
        base.update(values)
        return mock.patch.dict(os.environ, base, clear=True)

    def test_off(self):
        with self.clean_env():
            self.assertIsNone(rx.kill_switch_source())
            self.assertFalse(rx.kill_switch_active())

    def test_variable_is_named(self):
        with self.clean_env(TRADING_HALTED="true"):
            self.assertEqual(rx.kill_switch_source(), "TRADING_HALTED environment variable")
            self.assertIn("TRADING_HALTED environment variable", rx.entries_halt_reason(1.0, 1.0))

    def test_file_is_named(self):
        rx.HALT_FILE.write_text("x", encoding="utf-8")
        with self.clean_env():
            self.assertEqual(rx.kill_switch_source(), "HALT_TRADING file")
            self.assertIn("HALT_TRADING file", rx.entries_halt_reason(1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
