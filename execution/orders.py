"""Build and submit market orders, then report them."""

import logging
import math
from decimal import Decimal, ROUND_HALF_UP
from html import escape

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from telegram_notifier import send_telegram_message

from .mode import mode_label

logger = logging.getLogger("multi_asset_trading_bot")


def round_price(price: Decimal) -> float:
    """Round to cents using conventional half-up rounding."""
    return float(price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def build_order(
    asset: dict,
    allocation: float,
    reference_price: float,
    current_atr: float,
) -> tuple[MarketOrderRequest, float, str]:
    """Build the market order: notional for crypto, whole-share bracket for stocks.

    Returns (request, order_value, order_size_text).
    """
    alpaca_symbol = asset["alpaca_symbol"]
    if asset["type"] == "crypto":
        order_value = round(allocation, 2)
        estimated_quantity = order_value / reference_price
        order_size = (
            f"${order_value:,.2f} (~{estimated_quantity:.8f} "
            f"{escape(asset['symbol'])})"
        )
        request = MarketOrderRequest(
            symbol=alpaca_symbol,
            notional=order_value,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
    else:
        quantity = math.floor(allocation / reference_price)
        if quantity <= 0:
            raise ValueError(
                f"Allocation ${allocation:.2f} is below one {alpaca_symbol} share"
            )
        order_value = quantity * reference_price
        order_size = f"{quantity} shares (~${order_value:,.2f})"
        entry = Decimal(str(reference_price))
        atr = Decimal(str(current_atr))
        request = MarketOrderRequest(
            symbol=alpaca_symbol,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(
                limit_price=round_price(entry + Decimal("2.5") * atr)
            ),
            stop_loss=StopLossRequest(
                stop_price=round_price(entry - Decimal("1.5") * atr)
            ),
        )
    return request, order_value, order_size


def submit_and_notify(
    asset: dict,
    trading_client: TradingClient,
    request: MarketOrderRequest,
    order_value: float,
    order_size: str,
    model_type: str,
    signal_text: str,
    reference_price: float,
    equity: float,
    buying_power: float,
) -> None:
    """Submit the order, refresh buying power, then log and send the Telegram report."""
    order = trading_client.submit_order(request)
    try:
        buying_power_remaining = float(trading_client.get_account().buying_power)
    except Exception:
        logger.exception("Could not refresh buying power after order ticker=%s", asset["ticker"])
        buying_power_remaining = max(0.0, buying_power - order_value)

    logger.info(
        "Order submitted ticker=%s order_id=%s status=%s value=%.2f",
        asset["ticker"],
        order.id,
        order.status,
        order_value,
    )
    send_telegram_message(
        f"<b>BUY SIGNAL EXECUTED</b> [{mode_label()}]\n"
        f"<b>Symbol:</b> {escape(asset['ticker'])}\n"
        "<b>Direction:</b> BUY\n"
        f"<b>Model type:</b> {escape(model_type)}\n"
        f"<b>Signal:</b> {escape(signal_text)}\n"
        f"<b>Reference price:</b> ${reference_price:,.2f}\n"
        f"<b>Order size (requested):</b> {order_size}\n"
        f"<b>Account equity:</b> ${equity:,.2f}\n"
        f"<b>Buying power remaining:</b> ${buying_power_remaining:,.2f}\n"
        f"<b>Order:</b> {escape(str(order.id))} ({escape(str(order.status))})"
    )
