"""Characterization tests for execute_asset (sizing, order building, gates) and main.

These pin the current behaviour so the code can be restructured safely.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import run_execution as rx  # noqa: E402
from alpaca.trading.enums import OrderClass, OrderSide  # noqa: E402
from helpers import patch_name  # noqa: E402
from models import XGBRegressorModel  # noqa: E402


def asset(ticker):
    return next(a for a in config.ASSETS if a["ticker"] == ticker)


def regressor(raw, threshold=0.005):
    model = XGBRegressorModel("BTC", feature_columns=["a"])
    model.predict_signal = mock.Mock(
        return_value={"signal": "BUY", "raw_prediction": raw, "threshold": threshold}
    )
    return model


def classifier(raw, threshold=0.55):
    model = mock.Mock(feature_names=["a"], model_type="Legacy Classifier")
    model.predict_signal.return_value = {
        "signal": "BUY", "raw_prediction": raw, "threshold": threshold,
    }
    return model


class ExecuteAssetTests(unittest.TestCase):
    def setUp(self):
        self.recent = {OrderSide.BUY: None, OrderSide.SELL: None}
        defaults = {
            "get_latest_features": (None, 100.0, 2.0),
            "close_if_stop_hit": False,
            "get_open_position": None,
            "get_open_orders": [],
            "get_position_value": 0.0,
            "us_market_open": True,
            "buying_blocked": (False, ""),
            "cancel_open_orders": False,
            "close_position_if_open": False,
            "send_telegram_message": None,
        }
        self.m = {
            name: patch_name(self, name, return_value=value)
            for name, value in defaults.items()
        }
        self.m["get_recent_filled_order"] = patch_name(
            self,
            "get_recent_filled_order",
            side_effect=lambda client, symbol, side, hours: self.recent[side],
        )
        env = {k: v for k, v in os.environ.items() if k != "BUY_ASSETS"}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.client = mock.Mock()
        self.client.submit_order.return_value = mock.Mock(id="ord-1", status="accepted")
        self.client.get_account.return_value = mock.Mock(buying_power="9000")

    def run_asset(self, ticker, model, equity=10000.0, buying_power=10000.0, halt=None):
        rx.execute_asset(asset(ticker), self.client, model, equity, buying_power, halt)

    def request(self):
        self.client.submit_order.assert_called_once()
        return self.client.submit_order.call_args.args[0]

    def message(self):
        return self.m["send_telegram_message"].call_args.args[0]

    # ---- crypto buys -------------------------------------------------------
    def test_crypto_buy_is_sized_by_exposure_cap(self):
        # risk-sized 5000, but one asset may hold at most 25% of equity (2500)
        self.run_asset("BTC-USD", regressor(0.02))
        req = self.request()
        self.assertEqual(req.symbol, "BTC/USD")
        self.assertEqual(req.side, OrderSide.BUY)
        self.assertEqual(float(req.notional), 2500.0)
        self.assertIsNone(req.order_class)
        msg = self.message()
        self.assertIn("BUY SIGNAL EXECUTED", msg)
        self.assertIn("$2,500.00", msg)
        self.assertIn("ord-1", msg)

    def test_weak_regression_signal_halves_the_size(self):
        # atr 5% -> risk-sized 2000; prediction below 1% halves it to 1000
        self.m["get_latest_features"].return_value = (None, 100.0, 5.0)
        self.run_asset("BTC-USD", regressor(0.008))
        self.assertEqual(float(self.request().notional), 1000.0)

    def test_strong_regression_signal_keeps_full_size(self):
        self.m["get_latest_features"].return_value = (None, 100.0, 5.0)
        self.run_asset("BTC-USD", regressor(0.02))
        self.assertEqual(float(self.request().notional), 2000.0)

    def test_legacy_classifier_low_confidence_halves_the_size(self):
        self.m["get_latest_features"].return_value = (None, 100.0, 5.0)
        self.run_asset("BTC-USD", classifier(0.61))
        self.assertEqual(float(self.request().notional), 1000.0)

    def test_legacy_classifier_high_confidence_keeps_full_size(self):
        self.m["get_latest_features"].return_value = (None, 100.0, 5.0)
        self.run_asset("BTC-USD", classifier(0.70))
        self.assertEqual(float(self.request().notional), 2000.0)

    def test_buying_power_caps_the_order(self):
        self.run_asset("BTC-USD", regressor(0.02), buying_power=300.0)
        self.assertEqual(float(self.request().notional), 300.0)

    def test_existing_exposure_reduces_the_room(self):
        self.m["get_position_value"].return_value = 1000.0
        self.run_asset("BTC-USD", regressor(0.02))
        self.assertEqual(float(self.request().notional), 1500.0)

    def test_buying_power_refresh_failure_falls_back_to_estimate(self):
        self.client.get_account.side_effect = RuntimeError("down")
        self.run_asset("BTC-USD", regressor(0.02))
        self.assertIn("$7,500.00", self.message())  # 10000 - 2500 order value

    # ---- stock buys ----------------------------------------------------------
    def test_stock_buy_is_a_whole_share_bracket_order(self):
        os.environ["BUY_ASSETS"] = "SPY"
        self.m["get_latest_features"].return_value = (None, 400.0, 4.0)
        self.run_asset("SPY", regressor(0.02))
        req = self.request()
        self.assertEqual(req.symbol, "SPY")
        self.assertEqual(int(req.qty), 6)  # floor(2500 / 400)
        self.assertEqual(req.order_class, OrderClass.BRACKET)
        self.assertEqual(float(req.take_profit.limit_price), 410.0)  # +2.5 ATR
        self.assertEqual(float(req.stop_loss.stop_price), 394.0)  # -1.5 ATR
        self.assertIn("6 shares", self.message())

    def test_stock_allocation_below_one_share_raises(self):
        os.environ["BUY_ASSETS"] = "SPY"
        self.m["get_latest_features"].return_value = (None, 5000.0, 50.0)
        with self.assertRaises(ValueError):
            self.run_asset("SPY", regressor(0.02))
        self.client.submit_order.assert_not_called()

    # ---- entry gates ---------------------------------------------------------
    def assert_no_order(self):
        self.client.submit_order.assert_not_called()

    def test_halt_blocks_buying(self):
        self.run_asset("BTC-USD", regressor(0.02), halt="kill switch active")
        self.assert_no_order()

    def test_disabled_asset_blocks_buying(self):
        self.run_asset("ETH-USD", regressor(0.02))
        self.assert_no_order()

    def test_model_status_block_stops_buying(self):
        self.m["buying_blocked"].return_value = (True, "no edge")
        self.run_asset("BTC-USD", regressor(0.02))
        self.assert_no_order()

    def test_closed_us_market_blocks_stock_buy_only(self):
        os.environ["BUY_ASSETS"] = "SPY,BTC-USD"
        self.m["us_market_open"].return_value = False
        self.run_asset("SPY", regressor(0.02))
        self.assert_no_order()
        self.run_asset("BTC-USD", regressor(0.02))  # crypto ignores the clock
        self.client.submit_order.assert_called_once()

    def test_open_orders_block_a_duplicate_buy(self):
        self.m["get_open_orders"].return_value = [mock.Mock()]
        self.run_asset("BTC-USD", regressor(0.02))
        self.assert_no_order()

    def test_open_position_blocks_a_duplicate_buy(self):
        self.m["get_open_position"].return_value = mock.Mock()
        self.run_asset("BTC-USD", regressor(0.02))
        self.assert_no_order()

    def test_prediction_below_entry_margin_is_skipped(self):
        self.run_asset("BTC-USD", regressor(0.003, threshold=0.002))
        self.assert_no_order()

    def test_cooldown_after_a_recent_exit_blocks_buying(self):
        self.recent[OrderSide.SELL] = mock.Mock(id="s1", filled_at="now")
        self.run_asset("BTC-USD", regressor(0.02))
        self.assert_no_order()

    def test_full_exposure_cap_blocks_buying(self):
        self.m["get_position_value"].return_value = 2500.0
        self.run_asset("BTC-USD", regressor(0.02))
        self.assert_no_order()

    # ---- stop-loss and HOLD handling ----------------------------------------
    def test_stop_loss_hit_ends_processing_before_any_buy(self):
        self.m["close_if_stop_hit"].return_value = True
        self.run_asset("BTC-USD", regressor(0.02))
        self.assert_no_order()

    def test_stop_loss_is_not_checked_for_stocks(self):
        os.environ["BUY_ASSETS"] = "SPY"
        self.run_asset("SPY", regressor(0.02))
        self.m["close_if_stop_hit"].assert_not_called()

    def hold_model(self):
        model = regressor(0.0)
        model.predict_signal.return_value = {
            "signal": "HOLD", "raw_prediction": -0.01, "threshold": 0.005,
        }
        return model

    def test_hold_closes_the_position_and_reports(self):
        self.m["get_open_position"].return_value = mock.Mock()
        self.m["cancel_open_orders"].return_value = True
        self.m["close_position_if_open"].return_value = True
        self.run_asset("BTC-USD", self.hold_model())
        self.m["close_position_if_open"].assert_called_once()
        msg = self.message()
        self.assertIn("POSITION CLEANUP", msg)
        self.assertIn("Cancelled", msg)
        self.assertIn("Close submitted", msg)
        self.assert_no_order()

    def test_hold_with_nothing_to_clean_sends_no_message(self):
        self.run_asset("BTC-USD", self.hold_model())
        self.m["send_telegram_message"].assert_not_called()

    def test_minimum_hold_keeps_a_just_opened_position(self):
        self.m["get_open_position"].return_value = mock.Mock()
        self.recent[OrderSide.BUY] = mock.Mock(id="b1", submitted_at="now")
        self.run_asset("BTC-USD", self.hold_model())
        self.m["close_position_if_open"].assert_not_called()
        self.m["cancel_open_orders"].assert_not_called()


class MainTests(unittest.TestCase):
    """main() wiring: account checks, halt alert, per-asset loop, exit codes."""

    def setUp(self):
        env = {
            k: v for k, v in os.environ.items()
            if k not in ("ALPACA_PAPER", "TRADING_HALTED", "BUY_ASSETS", "MAX_DAILY_LOSS_PCT")
        }
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.client = mock.Mock()
        self.account = mock.Mock(
            status="ACTIVE", equity="10000", buying_power="10000", last_equity="10000"
        )
        self.client.get_account.return_value = self.account
        names = {
            "create_trading_client": self.client,
            "load_trading_model": mock.Mock(),
            "execute_asset": None,
            "send_telegram_message": None,
            "apply_capital_cap": (10000.0, 10000.0),
        }
        self.m = {}
        for name, value in names.items():
            patcher = mock.patch.object(rx, name, return_value=value)
            self.m[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def test_paper_run_processes_every_asset_and_succeeds(self):
        self.assertEqual(rx.main([]), 0)
        self.assertEqual(self.m["execute_asset"].call_count, len(config.ASSETS))
        self.assertIsNone(self.m["execute_asset"].call_args.args[5])  # not halted

    def test_symbol_filter_limits_the_run(self):
        self.assertEqual(rx.main(["--symbol", "BTC-USD"]), 0)
        self.assertEqual(self.m["execute_asset"].call_count, 1)

    def test_unknown_symbol_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            rx.main(["--symbol", "NOPE"])

    def test_halt_is_announced_and_passed_to_every_asset(self):
        os.environ["TRADING_HALTED"] = "true"
        self.assertEqual(rx.main([]), 0)
        self.assertIn("ENTRIES HALTED", self.m["send_telegram_message"].call_args.args[0])
        for call in self.m["execute_asset"].call_args_list:
            self.assertIn("kill switch", call.args[5])

    def test_daily_loss_breaker_halts_entries(self):
        self.account.equity = "9000"  # -10% versus last_equity
        self.assertEqual(rx.main([]), 0)
        self.assertIn("circuit breaker", self.m["execute_asset"].call_args.args[5])

    def test_one_failing_asset_does_not_stop_the_others_but_returns_1(self):
        self.m["execute_asset"].side_effect = [RuntimeError("boom"), None, None]
        self.assertEqual(rx.main([]), 1)
        self.assertEqual(self.m["execute_asset"].call_count, len(config.ASSETS))

    def test_account_failure_returns_1(self):
        self.client.get_account.side_effect = RuntimeError("down")
        self.assertEqual(rx.main([]), 1)
        self.m["execute_asset"].assert_not_called()

    def test_live_without_confirmation_never_creates_a_client(self):
        os.environ["ALPACA_PAPER"] = "false"
        self.assertEqual(rx.main([]), 1)
        self.m["create_trading_client"].assert_not_called()

    def test_capital_cap_is_applied_to_sizing_inputs(self):
        self.m["apply_capital_cap"].return_value = (111.0, 222.0)
        rx.main(["--symbol", "BTC-USD"])
        args = self.m["execute_asset"].call_args.args
        self.assertEqual((args[3], args[4]), (111.0, 222.0))


if __name__ == "__main__":
    unittest.main()
