"""Crypto stop-loss and the --risk-only mode."""

import logging
from html import escape

from alpaca.trading.client import TradingClient

from notifications.telegram import send_telegram_message

from .broker import cancel_open_orders, close_position_if_open, get_open_position
from .market_data import get_latest_features
from .mode import mode_label

logger = logging.getLogger("multi_asset_trading_bot")

# Crypto has no bracket orders on Alpaca, so exits are checked by the bot itself:
# close a crypto position when price falls this many ATRs below its entry price.
STOP_LOSS_ATR_MULTIPLE = 1.5


def stop_loss_triggered(
    position, reference_price: float, current_atr: float
) -> tuple[bool, float, float, float]:
    """Check the ATR stop. Returns (hit, entry_price, current_price, stop_price)."""
    entry_price = float(position.avg_entry_price)
    current_price = float(position.current_price or reference_price)
    stop_price = entry_price - STOP_LOSS_ATR_MULTIPLE * current_atr
    return current_price <= stop_price, entry_price, current_price, stop_price


def close_if_stop_hit(
    asset: dict,
    trading_client: TradingClient,
    reference_price: float,
    current_atr: float,
    held=None,
) -> bool:
    """Close a held crypto position if it fell through its ATR stop. True if it did.

    Pass `held` when the caller already looked the position up, to save an API call.
    """
    alpaca_symbol = asset["alpaca_symbol"]
    if held is None:
        held = get_open_position(trading_client, alpaca_symbol)
    if held is None:
        return False
    hit, entry_price, current_price, stop_price = stop_loss_triggered(
        held, reference_price, current_atr
    )
    if not hit:
        return False
    cancel_open_orders(trading_client, alpaca_symbol)
    close_position_if_open(trading_client, alpaca_symbol)
    logger.warning(
        "Stop-loss triggered ticker=%s entry=%.2f price=%.2f stop=%.2f",
        asset["ticker"],
        entry_price,
        current_price,
        stop_price,
    )
    send_telegram_message(
        f"<b>STOP-LOSS TRIGGERED</b> [{mode_label()}]\n"
        f"<b>Symbol:</b> {escape(asset['ticker'])}\n"
        f"<b>Entry price:</b> ${entry_price:,.2f}\n"
        f"<b>Current price:</b> ${current_price:,.2f}\n"
        f"<b>Stop level:</b> ${stop_price:,.2f} "
        f"({STOP_LOSS_ATR_MULTIPLE} ATR below entry)\n"
        "<b>Position:</b> Close submitted"
    )
    return True


def run_risk_only(assets: list[dict], trading_client: TradingClient) -> int:
    """Check crypto stop-losses only: no model, no signals, never opens a position.

    Meant to run often (every 10-15 minutes). Assets with no open position are skipped
    without fetching any market data, so most runs are cheap. SPY is skipped because its
    stop is an exchange-side bracket order. Returns 1 if any stop check failed.
    """
    failed = False
    for asset in assets:
        if asset["type"] != "crypto":
            continue
        try:
            held = get_open_position(trading_client, asset["alpaca_symbol"])
            if held is None:
                logger.info("Risk check: no open position ticker=%s", asset["ticker"])
                continue
            _, reference_price, current_atr = get_latest_features(
                asset["ticker"], asset["feature_columns"]
            )
            stopped = close_if_stop_hit(
                asset, trading_client, reference_price, current_atr, held
            )
            logger.info(
                "Risk check ticker=%s result=%s",
                asset["ticker"],
                "stop-loss closed position" if stopped else "within stop",
            )
        except Exception:
            failed = True
            logger.exception("Risk check failed ticker=%s", asset["ticker"])
    if failed:
        logger.error("One or more risk checks failed")
        return 1
    logger.info("%s risk check completed", mode_label())
    return 0
