"""Execution orchestration for regression and legacy classifier models.

Runs against the Alpaca paper account by default. Live trading requires
ALPACA_PAPER=false plus ALPACA_LIVE_CONFIRM and ALPACA_LIVE_MAX_CAPITAL (see
execution/safeguards.py). A kill switch and a daily-loss circuit breaker stop new
entries without blocking exits.

This file is the command-line entry point (``python run_execution.py``). The logic
lives in the ``execution`` package; the names below are re-exported so that
``import run_execution`` keeps exposing what it always did.
"""

import argparse
import logging
import sys
from html import escape
from pathlib import Path

from dotenv import load_dotenv

from config import ASSETS, MODEL_DIR, PROJECT_ROOT
from execution.broker import (  # noqa: F401  (re-exported)
    CANCEL_POLL_INTERVAL_SECONDS,
    CANCEL_TIMEOUT_SECONDS,
    cancel_open_orders,
    close_position_if_open,
    create_trading_client,
    get_open_orders,
    get_open_position,
    get_position_value,
    get_recent_filled_order,
    us_market_open,
)
from execution.market_data import (  # noqa: F401  (re-exported)
    USE_COMPLETED_BARS,
    drop_partial_bar,
    get_latest_features,
    get_spy_close_history,
)
from execution.mode import is_paper_mode, mode_label
from execution.risk import (  # noqa: F401  (re-exported)
    STOP_LOSS_ATR_MULTIPLE,
    close_if_stop_hit,
    run_risk_only,
    stop_loss_triggered,
)
from execution.safeguards import (  # noqa: F401  (re-exported)
    BUY_ASSETS_ENV,
    DEFAULT_MAX_DAILY_LOSS_PCT,
    HALT_ENV,
    HALT_FILE,
    LIVE_CONFIRM_ENV,
    LIVE_CONFIRM_PHRASE,
    LIVE_MAX_CAPITAL_ENV,
    MAX_DAILY_LOSS_ENV,
    apply_capital_cap,
    buy_enabled,
    entries_halt_reason,
    kill_switch_active,
    kill_switch_source,
    live_capital_cap,
    max_daily_loss_pct,
    warn_unknown_buy_assets,
)
from execution.gates import (  # noqa: F401  (re-exported)
    CLASSIFIER_ENTRY_MIN,
    COOLDOWN_HOURS,
    REGRESSOR_ENTRY_MIN,
    buying_blocked,
    entry_margin_required,
)
from execution.orders import round_price  # noqa: F401  (re-exported)
from execution.sizing import MAX_POSITION_EQUITY_FRACTION, RISK_FACTOR  # noqa: F401
from execution.trading import MIN_HOLD_HOURS, execute_asset  # noqa: F401
from models import load_trading_model
from notifications.telegram import send_telegram_message

logger = logging.getLogger("multi_asset_trading_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
load_dotenv(PROJECT_ROOT / ".env")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute trades from model signals")
    parser.add_argument(
        "--symbol",
        help="Optional ticker to process (e.g. BTC-USD, SPY, or ETH-USD)",
    )
    parser.add_argument(
        "--risk-only",
        action="store_true",
        help="Only check crypto stop-losses (no signals, never opens positions)",
    )
    arguments = parser.parse_args(argv)
    if arguments.symbol:
        matches = [
            asset
            for asset in ASSETS
            if arguments.symbol.upper()
            in {
                asset["ticker"].upper(),
                asset["symbol"].upper(),
                asset["alpaca_symbol"].upper(),
            }
        ]
        if not matches:
            parser.error(f"Unknown symbol {arguments.symbol!r}")
        arguments.assets = matches
    else:
        arguments.assets = ASSETS
    return arguments


def _run_risk_only_mode(arguments: argparse.Namespace) -> int:
    # Closing positions is always allowed, so the live-mode confirmation and capital
    # cap are deliberately not required here: a misconfigured live setup must never
    # leave an open position without its stop-loss check.
    try:
        return run_risk_only(arguments.assets, create_trading_client())
    except Exception:
        logger.exception("Risk-only run could not start (%s)", mode_label())
        return 1


def _prepare_entry_controls(account, assets: list[dict]) -> str | None:
    """Log which assets may buy and return why entries are halted, if they are."""
    try:
        last_equity = float(getattr(account, "last_equity", None) or 0) or None
    except (TypeError, ValueError):
        last_equity = None
    warn_unknown_buy_assets()
    logger.info(
        "Buying enabled for: %s",
        [a["ticker"] for a in assets if buy_enabled(a)] or "none",
    )
    halt_reason = entries_halt_reason(float(account.equity), last_equity)
    if halt_reason:
        logger.error("Entries halted: %s", halt_reason)
        send_telegram_message(
            f"<b>ENTRIES HALTED</b> [{mode_label()}]\n{escape(halt_reason)}\n"
            "Exits and stop-losses still run."
        )
    return halt_reason


def _process_assets(
    assets: list[dict],
    trading_client,
    capital_cap: float | None,
    halt_reason: str | None,
) -> bool:
    """Evaluate each asset in turn. Returns True if any asset failed."""
    had_asset_failure = False
    for asset in assets:
        model_path = Path(asset["model_path"])
        if not model_path.is_absolute():
            model_path = MODEL_DIR / model_path
        if not model_path.is_file() and asset["type"] == "crypto":
            logger.warning("Model file missing ticker=%s path=%s", asset["ticker"], model_path)
            continue
        try:
            model = load_trading_model(
                model_path,
                asset["symbol"],
                asset["feature_columns"],
            )
            current_account = trading_client.get_account()
            sizing_equity, sizing_buying_power = apply_capital_cap(
                trading_client,
                capital_cap,
                float(current_account.equity),
                float(current_account.buying_power),
            )
            execute_asset(
                asset,
                trading_client,
                model,
                sizing_equity,
                sizing_buying_power,
                halt_reason,
            )
        except Exception:
            had_asset_failure = True
            logger.exception("Asset execution failed ticker=%s", asset["ticker"])
    return had_asset_failure


def main(argv: list[str] | None = None) -> int:
    """Execute requested assets serially using the configured Alpaca account."""
    arguments = _parse_args(argv)
    if arguments.risk_only:
        return _run_risk_only_mode(arguments)
    try:
        capital_cap = live_capital_cap()
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    try:
        trading_client = create_trading_client()
        account = trading_client.get_account()
    except Exception:
        logger.exception("Unable to initialize or query Alpaca %s account", mode_label())
        return 1

    logger.info(
        "Alpaca %s account status=%s equity=%s buying_power=%s capital_cap=%s",
        mode_label(),
        account.status,
        account.equity,
        account.buying_power,
        f"${capital_cap:,.2f}" if capital_cap is not None else "none (paper)",
    )
    halt_reason = _prepare_entry_controls(account, arguments.assets)
    if _process_assets(arguments.assets, trading_client, capital_cap, halt_reason):
        logger.error("One or more requested assets failed execution")
        return 1
    logger.info("%s execution completed assets=%s", mode_label(), [item["ticker"] for item in arguments.assets])
    return 0


if __name__ == "__main__":
    sys.exit(main())
