"""Per-asset signal handling: stop-loss, HOLD cleanup, then gates, sizing and the order."""

import logging
from html import escape

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide

from models import XGBRegressorModel
from telegram_notifier import send_telegram_message

from .broker import (
    cancel_open_orders,
    close_position_if_open,
    get_open_position,
    get_recent_filled_order,
)
from .gates import entry_allowed
from .market_data import get_latest_features
from .mode import mode_label
from .orders import build_order, submit_and_notify
from .risk import close_if_stop_hit
from .sizing import size_position

logger = logging.getLogger("multi_asset_trading_bot")

# A HOLD signal may not close a position opened less than this many hours ago.
# The crypto stop-loss is exempt. Set to 0 to disable.
MIN_HOLD_HOURS = 8


def _signal_details(model, signal: dict) -> tuple[str, str]:
    """Return display model type and predicted value/cutoff text."""
    if isinstance(model, XGBRegressorModel):
        return (
            model.model_type,
            f"Predicted return: {signal['raw_prediction']:+.2%} "
            f"(threshold: {signal['threshold']:+.2%})",
        )
    return (
        model.model_type,
        f"Up probability: {signal['raw_prediction']:.2%} "
        f"(threshold: {signal['threshold']:.2%})",
    )


def _handle_hold(asset: dict, trading_client: TradingClient, signal_text: str) -> None:
    """A non-BUY signal: keep a just-opened position, otherwise clean up and report."""
    alpaca_symbol = asset["alpaca_symbol"]
    # Don't let a marginal signal flip close a position that was only just
    # opened. This also leaves SPY's bracket stop/take-profit orders in place.
    # The crypto stop-loss is unaffected.
    if (
        MIN_HOLD_HOURS > 0
        and get_open_position(trading_client, alpaca_symbol) is not None
    ):
        recent_entry = get_recent_filled_order(
            trading_client, alpaca_symbol, OrderSide.BUY, MIN_HOLD_HOURS
        )
        if recent_entry is not None:
            logger.info(
                "Holding through HOLD signal (minimum hold) ticker=%s "
                "entry_order=%s submitted_at=%s min_hold_hours=%s",
                asset["ticker"],
                recent_entry.id,
                recent_entry.submitted_at,
                MIN_HOLD_HOURS,
            )
            return
    cancelled = cancel_open_orders(trading_client, alpaca_symbol)
    closed = close_position_if_open(trading_client, alpaca_symbol)
    if cancelled or closed:
        send_telegram_message(
            f"<b>POSITION CLEANUP</b> [{mode_label()}]\n"
            f"<b>Symbol:</b> {escape(asset['ticker'])}\n"
            f"<b>Signal:</b> HOLD ({escape(signal_text)})\n"
            f"<b>Pending orders:</b> {'Cancelled' if cancelled else 'None pending'}\n"
            f"<b>Position:</b> {'Close submitted' if closed else 'No open position'}"
        )
    else:
        logger.info("HOLD signal, nothing to clean up ticker=%s", asset["ticker"])


def execute_asset(
    asset: dict,
    trading_client: TradingClient,
    model,
    equity: float,
    buying_power: float,
    halt_reason: str | None = None,
) -> None:
    """Evaluate a signal, manage the position, and notify Telegram.

    `halt_reason`, when set, blocks opening new positions only; exits, stop-losses
    and HOLD cleanups still run.
    """
    feature_columns = model.feature_names or asset["feature_columns"]
    latest_features, reference_price, current_atr = get_latest_features(
        asset["ticker"], feature_columns
    )
    result = model.predict_signal(latest_features)
    model_type, signal_text = _signal_details(model, result)

    logger.info(
        "Asset signal ticker=%s model=%s signal=%s raw_prediction=%.6f "
        "threshold=%.6f",
        asset["ticker"],
        model_type,
        result["signal"],
        result["raw_prediction"],
        result["threshold"],
    )

    # Bot-side stop-loss for crypto (SPY already has an exchange-side bracket stop).
    if asset["type"] == "crypto" and close_if_stop_hit(
        asset, trading_client, reference_price, current_atr
    ):
        return

    if result["signal"] != "BUY":
        _handle_hold(asset, trading_client, signal_text)
        return

    if not entry_allowed(asset, trading_client, model, result, halt_reason):
        return

    allocation = size_position(
        asset, trading_client, model, result,
        equity, buying_power, reference_price, current_atr,
    )
    if allocation is None:
        return

    request, order_value, order_size = build_order(
        asset, allocation, reference_price, current_atr
    )
    submit_and_notify(
        asset, trading_client, request, order_value, order_size,
        model_type, signal_text, reference_price, equity, buying_power,
    )
