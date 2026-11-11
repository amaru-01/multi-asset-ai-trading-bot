"""Alpaca client, orders and positions."""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from config import ALPACA_API_KEY_ENV, ALPACA_SECRET_KEY_ENV

from .mode import is_paper_mode

logger = logging.getLogger("multi_asset_trading_bot")

CANCEL_TIMEOUT_SECONDS = 10
CANCEL_POLL_INTERVAL_SECONDS = 0.5


def create_trading_client() -> TradingClient:
    """Create an Alpaca client (paper by default) using environment credentials."""
    api_key = os.getenv(ALPACA_API_KEY_ENV)
    secret_key = os.getenv(ALPACA_SECRET_KEY_ENV)
    if not api_key or not secret_key:
        raise ValueError(
            f"{ALPACA_API_KEY_ENV} and {ALPACA_SECRET_KEY_ENV} must be set"
        )
    return TradingClient(api_key, secret_key, paper=is_paper_mode())


def _norm_symbol(symbol: str) -> str:
    """Normalize 'ETH/USD' (orders) and 'ETHUSD' (positions) to one form."""
    return symbol.replace("/", "").upper()


def get_open_orders(trading_client: TradingClient, symbol: str) -> list:
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    return trading_client.get_orders(filter=request)


def cancel_open_orders(trading_client: TradingClient, symbol: str) -> bool:
    """Cancel all open orders for a symbol and report if any were cancelled."""
    deadline = time.monotonic() + CANCEL_TIMEOUT_SECONDS
    cancelled_any = False
    while time.monotonic() < deadline:
        open_orders = get_open_orders(trading_client, symbol)
        if not open_orders:
            return cancelled_any
        order = open_orders[0]
        try:
            trading_client.cancel_order_by_id(order.id)
        except Exception:
            remaining = {str(item.id) for item in get_open_orders(trading_client, symbol)}
            if str(order.id) in remaining:
                raise
        cancelled_any = True
        time.sleep(CANCEL_POLL_INTERVAL_SECONDS)
    remaining = [str(order.id) for order in get_open_orders(trading_client, symbol)]
    raise TimeoutError(f"Timed out cancelling orders for {symbol}: {remaining}")


def get_open_position(trading_client: TradingClient, symbol: str):
    """Find an open position, tolerating Alpaca's crypto symbol formats.

    Orders use 'ETH/USD' but the positions list reports 'ETHUSD', so compare
    normalized symbols instead of raw strings.
    """
    target = _norm_symbol(symbol)
    return next(
        (
            position
            for position in trading_client.get_all_positions()
            if _norm_symbol(position.symbol) == target
        ),
        None,
    )


def close_position_if_open(trading_client: TradingClient, symbol: str) -> bool:
    position = get_open_position(trading_client, symbol)
    if position is None:
        return False
    # Use the symbol exactly as Alpaca reports it for the position.
    order = trading_client.close_position(position.symbol)
    logger.info(
        "Position close submitted symbol=%s order_id=%s status=%s",
        position.symbol,
        order.id,
        order.status,
    )
    return True


def get_position_value(trading_client: TradingClient, symbol: str) -> float:
    """Total market value held in a symbol (0.0 if none), tolerating symbol formats."""
    target = _norm_symbol(symbol)
    return sum(
        abs(float(position.market_value or 0))
        for position in trading_client.get_all_positions()
        if _norm_symbol(position.symbol) == target
    )


def us_market_open(trading_client: TradingClient) -> bool:
    """True when the US stock market is open. Fails open if the clock call errors."""
    try:
        return bool(trading_client.get_clock().is_open)
    except Exception:
        logger.exception("Could not read market clock; assuming open")
        return True


def get_recent_filled_order(
    trading_client: TradingClient, symbol: str, side: OrderSide, hours: float
):
    """Return the most recent filled order on `side` submitted within `hours`, if any."""
    request = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        symbols=[symbol],
        side=side,
        after=datetime.now(timezone.utc) - timedelta(hours=hours),
        limit=50,
    )
    return next(
        (
            order
            for order in trading_client.get_orders(filter=request)
            if float(order.filled_qty or 0) > 0
        ),
        None,
    )
