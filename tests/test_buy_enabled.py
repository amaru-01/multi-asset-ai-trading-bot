"""Tests for the per-asset buying switch."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import run_execution as rx  # noqa: E402
from helpers import patch_name  # noqa: E402


def asset(ticker):
    return next(a for a in config.ASSETS if a["ticker"] == ticker)


def env(**values):
    base = {k: v for k, v in os.environ.items() if k != rx.BUY_ASSETS_ENV}
    base.update(values)
    return mock.patch.dict(os.environ, base, clear=True)


class ConfigDefaultsTests(unittest.TestCase):
    def test_only_btc_is_enabled_by_default(self):
        with env():
            self.assertTrue(rx.buy_enabled(asset("BTC-USD")))
            self.assertFalse(rx.buy_enabled(asset("ETH-USD")))
            self.assertFalse(rx.buy_enabled(asset("SPY")))

    def test_every_asset_declares_the_flag(self):
        for a in config.ASSETS:
            self.assertIn("buying_enabled", a, a["ticker"])

    def test_missing_flag_defaults_to_enabled(self):
        bare = {"ticker": "X-USD", "symbol": "X", "alpaca_symbol": "X/USD"}
        with env():
            self.assertTrue(rx.buy_enabled(bare))


class OverrideTests(unittest.TestCase):
    def test_variable_overrides_config(self):
        with env(BUY_ASSETS="ETH-USD"):
            self.assertTrue(rx.buy_enabled(asset("ETH-USD")))
            self.assertFalse(rx.buy_enabled(asset("BTC-USD")))

    def test_list_with_spaces_and_case(self):
        with env(BUY_ASSETS=" btc-usd , spy "):
            self.assertTrue(rx.buy_enabled(asset("BTC-USD")))
            self.assertTrue(rx.buy_enabled(asset("SPY")))
            self.assertFalse(rx.buy_enabled(asset("ETH-USD")))

    def test_short_symbol_and_alpaca_symbol_also_match(self):
        with env(BUY_ASSETS="ETH"):
            self.assertTrue(rx.buy_enabled(asset("ETH-USD")))
        with env(BUY_ASSETS="BTC/USD"):
            self.assertTrue(rx.buy_enabled(asset("BTC-USD")))

    def test_blank_variable_falls_back_to_config(self):
        # An unset GitHub variable arrives as an empty string.
        for blank in ("", "   "):
            with env(BUY_ASSETS=blank):
                self.assertTrue(rx.buy_enabled(asset("BTC-USD")))
                self.assertFalse(rx.buy_enabled(asset("ETH-USD")))

    def test_stray_separator_enables_nothing_rather_than_everything(self):
        with env(BUY_ASSETS=","):
            for ticker in ("SPY", "BTC-USD", "ETH-USD"):
                self.assertFalse(rx.buy_enabled(asset(ticker)))

    def test_unknown_names_are_reported(self):
        with env(BUY_ASSETS="BTC-USD,DOGE-USD"):
            with self.assertLogs("multi_asset_trading_bot", level="WARNING") as logs:
                rx.warn_unknown_buy_assets()
        self.assertTrue(any("DOGE-USD" in line for line in logs.output))

    def test_known_names_produce_no_warning(self):
        with env(BUY_ASSETS="BTC-USD,ETH"):
            with self.assertNoLogs("multi_asset_trading_bot", level="WARNING"):
                rx.warn_unknown_buy_assets()


class ExecuteAssetTests(unittest.TestCase):
    """Run the real execute_asset with the broker and market data mocked out."""

    def setUp(self):
        names = {
            "features": ("get_latest_features", (None, 100.0, 2.0)),
            "stop": ("close_if_stop_hit", False),
            "blocked": ("buying_blocked", (True, "stop here")),
            "cancel": ("cancel_open_orders", True),
            "close": ("close_position_if_open", True),
            "held": ("get_open_position", None),
            "recent": ("get_recent_filled_order", None),
            "telegram": ("send_telegram_message", None),
        }
        self.m = {
            key: patch_name(self, attr, return_value=value)
            for key, (attr, value) in names.items()
        }
        self.client = mock.Mock()

    def model(self, signal):
        model = mock.Mock(feature_names=["a"], model_type="test")
        model.predict_signal.return_value = {
            "signal": signal, "raw_prediction": 0.02, "threshold": 0.01,
        }
        return model

    def test_disabled_asset_never_reaches_buy_logic(self):
        with env():
            rx.execute_asset(asset("ETH-USD"), self.client, self.model("BUY"), 1000.0, 1000.0)
        self.m["blocked"].assert_not_called()
        self.client.submit_order.assert_not_called()

    def test_enabled_asset_continues_past_the_switch(self):
        with env():
            rx.execute_asset(asset("BTC-USD"), self.client, self.model("BUY"), 1000.0, 1000.0)
        self.m["blocked"].assert_called_once()

    def test_disabled_asset_still_exits_on_hold(self):
        self.m["held"].return_value = mock.Mock()
        with env():
            rx.execute_asset(asset("ETH-USD"), self.client, self.model("HOLD"), 1000.0, 1000.0)
        self.m["close"].assert_called_once()

    def test_disabled_asset_still_checked_for_stop_loss(self):
        self.m["stop"].return_value = True
        with env():
            rx.execute_asset(asset("ETH-USD"), self.client, self.model("BUY"), 1000.0, 1000.0)
        self.m["stop"].assert_called_once()

    def test_override_enables_a_disabled_asset(self):
        with env(BUY_ASSETS="ETH-USD"):
            rx.execute_asset(asset("ETH-USD"), self.client, self.model("BUY"), 1000.0, 1000.0)
        self.m["blocked"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
