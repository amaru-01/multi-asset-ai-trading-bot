"""run_execution.py was split into the execution package; its old public names stay importable."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_execution  # noqa: E402

# Every public name run_execution.py exposed before it was split.
LEGACY_NAMES = [
    # constants
    "RISK_FACTOR", "MAX_POSITION_EQUITY_FRACTION", "CANCEL_TIMEOUT_SECONDS",
    "CANCEL_POLL_INTERVAL_SECONDS", "STOP_LOSS_ATR_MULTIPLE", "COOLDOWN_HOURS",
    "CLASSIFIER_ENTRY_MIN", "REGRESSOR_ENTRY_MIN", "MIN_HOLD_HOURS", "USE_COMPLETED_BARS",
    "LIVE_CONFIRM_ENV", "LIVE_CONFIRM_PHRASE", "LIVE_MAX_CAPITAL_ENV", "HALT_ENV",
    "HALT_FILE", "DEFAULT_MAX_DAILY_LOSS_PCT", "MAX_DAILY_LOSS_ENV", "BUY_ASSETS_ENV",
    # functions
    "is_paper_mode", "mode_label", "live_capital_cap", "buy_enabled",
    "warn_unknown_buy_assets", "kill_switch_source", "kill_switch_active",
    "max_daily_loss_pct", "entries_halt_reason", "apply_capital_cap",
    "create_trading_client", "drop_partial_bar", "get_spy_close_history",
    "get_latest_features", "round_price", "get_open_orders", "cancel_open_orders",
    "get_open_position", "close_position_if_open", "get_position_value",
    "us_market_open", "buying_blocked", "entry_margin_required",
    "get_recent_filled_order", "stop_loss_triggered", "close_if_stop_hit",
    "run_risk_only", "execute_asset", "main",
]


class PublicNameTests(unittest.TestCase):
    def test_legacy_names_still_importable(self):
        missing = [name for name in LEGACY_NAMES if not hasattr(run_execution, name)]
        self.assertEqual(missing, [])

    def test_script_entry_point_is_unchanged(self):
        self.assertTrue(callable(run_execution.main))


if __name__ == "__main__":
    unittest.main()
