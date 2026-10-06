"""Paper execution orchestration for regression and legacy classifier models."""

import argparse
import logging
import math
import os
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from pathlib import Path

import pandas as pd
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from dotenv import load_dotenv

from config import (
    ALPACA_API_KEY_ENV,
    ALPACA_SECRET_KEY_ENV,
    ASSETS,
    LEGACY_CLASSIFIER_THRESHOLD,
    MODEL_DIR,
    PROJECT_ROOT,
    REGRESSION_RETURN_THRESHOLD,
)
from features.pipeline import build_inference_features, normalize_market_columns
from models import XGBRegressorModel, load_trading_model
from telegram_notifier import send_telegram_message


logger = logging.getLogger("multi_asset_trading_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
load_dotenv(PROJECT_ROOT / ".env")

RISK_FACTOR = 0.01
MAX_POSITION_EQUITY_FRACTION = 0.25
CANCEL_TIMEOUT_SECONDS = 10
CANCEL_POLL_INTERVAL_SECONDS = 0.5
_SPY_CLOSE_CACHE: pd.Series | None = None


def create_trading_client() -> TradingClient:
    """Create an Alpaca paper client using standard environment credentials."""
    api_key = os.getenv(ALPACA_API_KEY_ENV)
    secret_key = os.getenv(ALPACA_SECRET_KEY_ENV)
    if not api_key or not secret_key:
        raise ValueError(
            f"{ALPACA_API_KEY_ENV} and {ALPACA_SECRET_KEY_ENV} must be set"
        )
    return TradingClient(api_key, secret_key, paper=True)


def get_spy_close_history() -> pd.Series:
    """Fetch one year of SPY closes for the shared benchmark feature."""
    global _SPY_CLOSE_CACHE
    if _SPY_CLOSE_CACHE is None:
        response = yf.download(
            "SPY",
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if response is None or response.empty:
            raise ValueError("yfinance returned no SPY data for benchmark feature")
        normalized = normalize_market_columns(response)
        _SPY_CLOSE_CACHE = pd.to_numeric(normalized["Close"], errors="coerce")
    return _SPY_CLOSE_CACHE


def get_latest_features(
    ticker: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, float, float]:
    """Fetch latest prices and calculate model inputs with the shared pipeline."""
    try:
        response = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if response is None or response.empty:
            raise ValueError(f"yfinance returned no data for {ticker}")
        market_data = normalize_market_columns(response)
        spy_close = (
            pd.to_numeric(market_data["Close"], errors="coerce")
            if ticker == "SPY"
            else get_spy_close_history()
        )
        frame = build_inference_features(market_data, feature_columns, spy_close)
        if frame.empty:
            raise ValueError(f"Insufficient valid feature history for {ticker}")
        latest_features = frame.loc[:, feature_columns].tail(1).astype(float)
        reference_price = float(frame["Close"].iloc[-1])
        current_atr = float(frame["ATR_14"].iloc[-1])
        if not math.isfinite(reference_price) or reference_price <= 0:
            raise ValueError(f"Invalid reference price for {ticker}: {reference_price}")
        if not math.isfinite(current_atr) or current_atr <= 0:
            raise ValueError(f"Invalid ATR-14 for {ticker}: {current_atr}")
        return latest_features, reference_price, current_atr
    except Exception as exc:
        logger.exception("Failed to fetch or prepare market data ticker=%s", ticker)
        raise RuntimeError(f"Unable to prepare market features for {ticker}") from exc


def round_price(price: Decimal) -> float:
    """Round to cents using conventional half-up rounding."""
    return float(price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


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
    return next(
        (position for position in trading_client.get_all_positions() if position.symbol == symbol),
        None,
    )


def close_position_if_open(trading_client: TradingClient, symbol: str) -> bool:
    position = get_open_position(trading_client, symbol)
    if position is None:
        return False
    order = trading_client.close_position(symbol)
    logger.info(
        "Position close submitted symbol=%s order_id=%s status=%s",
        symbol,
        order.id,
        order.status,
    )
    return True


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


def execute_asset(
    asset: dict,
    trading_client: TradingClient,
    model,
    equity: float,
    buying_power: float,
) -> None:
    """Evaluate a signal, manage the paper position, and notify Telegram."""
    feature_columns = model.feature_names or asset["feature_columns"]
    latest_features, reference_price, current_atr = get_latest_features(
        asset["ticker"], feature_columns
    )
    result = model.predict_signal(latest_features)
    model_type, signal_text = _signal_details(model, result)
    alpaca_symbol = asset["alpaca_symbol"]

    logger.info(
        "Asset signal ticker=%s model=%s signal=%s raw_prediction=%.6f "
        "threshold=%.6f",
        asset["ticker"],
        model_type,
        result["signal"],
        result["raw_prediction"],
        result["threshold"],
    )

    if result["signal"] != "BUY":
        cancelled = cancel_open_orders(trading_client, alpaca_symbol)
        closed = close_position_if_open(trading_client, alpaca_symbol)
        send_telegram_message(
            "<b>POSITION CLEANUP</b>\n"
            f"<b>Symbol:</b> {escape(asset['ticker'])}\n"
            f"<b>Signal:</b> HOLD ({escape(signal_text)})\n"
            f"<b>Pending orders:</b> {'Cancelled' if cancelled else 'None pending'}\n"
            f"<b>Position:</b> {'Close submitted' if closed else 'No open position'}"
        )
        return

    open_orders = get_open_orders(trading_client, alpaca_symbol)
    position = get_open_position(trading_client, alpaca_symbol)
    if open_orders or position is not None:
        logger.warning(
            "Skipping duplicate buy ticker=%s open_orders=%d has_position=%s",
            asset["ticker"],
            len(open_orders),
            position is not None,
        )
        return

    normalized_atr = current_atr / reference_price
    if not math.isfinite(normalized_atr) or normalized_atr <= 0:
        raise ValueError(f"Invalid normalized ATR for {asset['ticker']}")
    confidence_scale = (
        0.5
        if isinstance(model, XGBRegressorModel)
        and result["raw_prediction"] < 0.01
        else 1.0
    )
    if model.model_type == "Legacy Classifier":
        confidence_scale = 0.5 if result["raw_prediction"] <= 0.62 else 1.0

    risk_sized_value = equity * RISK_FACTOR / normalized_atr
    max_position_value = min(equity * MAX_POSITION_EQUITY_FRACTION, buying_power)
    allocation = min(risk_sized_value * confidence_scale, max_position_value)
    if not math.isfinite(allocation) or allocation <= 0:
        raise ValueError(f"Calculated position value is invalid for {asset['ticker']}")

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
        "<b>BUY SIGNAL EXECUTED</b>\n"
        f"<b>Symbol:</b> {escape(asset['ticker'])}\n"
        "<b>Direction:</b> BUY\n"
        f"<b>Model type:</b> {escape(model_type)}\n"
        f"<b>Signal:</b> {escape(signal_text)}\n"
        f"<b>Reference price:</b> ${reference_price:,.2f}\n"
        f"<b>Position size:</b> {order_size}\n"
        f"<b>Account equity:</b> ${equity:,.2f}\n"
        f"<b>Buying power remaining:</b> ${buying_power_remaining:,.2f}\n"
        f"<b>Order:</b> {escape(str(order.id))} ({escape(str(order.status))})"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute paper trades from model signals")
    parser.add_argument(
        "--symbol",
        help="Optional ticker to process (e.g. BTC-USD, SPY, or ETH-USD)",
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


def main(argv: list[str] | None = None) -> int:
    """Execute requested assets serially using Alpaca paper trading."""
    arguments = _parse_args(argv)
    try:
        trading_client = create_trading_client()
        account = trading_client.get_account()
    except Exception:
        logger.exception("Unable to initialize or query Alpaca paper account")
        return 1

    logger.info(
        "Alpaca paper account status=%s equity=%s buying_power=%s",
        account.status,
        account.equity,
        account.buying_power,
    )
    had_asset_failure = False
    for asset in arguments.assets:
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
            execute_asset(
                asset,
                trading_client,
                model,
                float(current_account.equity),
                float(current_account.buying_power),
            )
        except Exception:
            had_asset_failure = True
            logger.exception("Asset execution failed ticker=%s", asset["ticker"])
    if had_asset_failure:
        logger.error("One or more requested assets failed execution")
        return 1
    logger.info("Paper execution completed assets=%s", [item["ticker"] for item in arguments.assets])
    return 0


if __name__ == "__main__":
    sys.exit(main())
