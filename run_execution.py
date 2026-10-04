import logging
import math
import os
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
import ta
import xgboost as xgb
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


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "Model" / "xgb_trading_model.json"
FEATURE_COLUMNS = [
    "RSI",
    "MACD",
    "ATR",
    "Return_1D",
    "Vol_Change",
    "RSI_lag1",
    "Return_lag1",
]
TAKE_PROFIT_PERCENT = Decimal("0.015")
STOP_LOSS_PERCENT = Decimal("0.01")
CANCEL_TIMEOUT_SECONDS = 10
CANCEL_POLL_INTERVAL_SECONDS = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("spy_trading_bot")

load_dotenv(PROJECT_ROOT / ".env")


def create_trading_client() -> TradingClient:
    """Create an Alpaca paper-trading client from environment credentials."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment."
        )
    return TradingClient(api_key, secret_key, paper=True)


def load_model() -> xgb.XGBClassifier:
    """Load the native XGBoost JSON model and verify its feature order."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Export the Kaggle model as JSON first."
        )

    model = xgb.XGBClassifier()
    try:
        model.load_model(str(MODEL_PATH))
    except Exception as exc:
        logger.exception("Could not load XGBoost model path=%s", MODEL_PATH)
        raise RuntimeError(f"Unable to load model at {MODEL_PATH}") from exc

    trained_features = model.get_booster().feature_names
    if trained_features and trained_features != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature names/order do not match execution features: "
            f"model={trained_features}, execution={FEATURE_COLUMNS}"
        )

    logger.info("Loaded model path=%s", MODEL_PATH)
    return model


def _normalize_market_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance columns and validate required OHLCV fields."""
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    if isinstance(data.columns, pd.MultiIndex):
        for level in range(data.columns.nlevels):
            level_names = data.columns.get_level_values(level)
            if required_columns.issubset(set(level_names)):
                data.columns = level_names
                break
        else:
            raise ValueError("Could not identify OHLCV columns in yfinance response.")

    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Market data is missing columns: {sorted(missing_columns)}")
    return data.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()


def get_latest_features(ticker: str = "SPY") -> tuple[pd.DataFrame, float]:
    """Download recent OHLCV data and calculate one aligned model row."""
    try:
        data = yf.download(
            ticker,
            period="60d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data.empty:
            raise ValueError(f"yfinance returned no data for {ticker}.")

        data = _normalize_market_columns(data)
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
        data.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
        if data.empty:
            raise ValueError(f"No valid OHLCV rows remain for {ticker}.")

        close = data["Close"]
        data["RSI"] = ta.momentum.rsi(close, window=14)
        data["MACD"] = ta.trend.macd_diff(close)
        data["ATR"] = ta.volatility.average_true_range(
            data["High"], data["Low"], close
        )
        data["Return_1D"] = close.pct_change(1)
        data["Vol_Change"] = data["Volume"].pct_change(1)
        data["RSI_lag1"] = data["RSI"].shift(1)
        data["Return_lag1"] = data["Return_1D"].shift(1)

        data.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
        data.dropna(subset=FEATURE_COLUMNS + ["Close"], inplace=True)
        if data.empty:
            raise ValueError(f"Not enough valid history to calculate features for {ticker}.")

        latest_features = data.loc[:, FEATURE_COLUMNS].tail(1).astype(float)
        reference_price = float(data["Close"].iloc[-1])
        if not math.isfinite(reference_price) or reference_price <= 0:
            raise ValueError(f"Invalid reference price for {ticker}: {reference_price}.")
        return latest_features, reference_price
    except Exception as exc:
        logger.exception("Failed to fetch or prepare market data symbol=%s", ticker)
        raise RuntimeError(f"Unable to prepare market features for {ticker}.") from exc


def _round_price(price: Decimal) -> float:
    """Round a price to the nearest cent using conventional half-up rounding."""
    return float(price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _get_open_orders(trading_client: TradingClient, symbol: str) -> list:
    request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol],
        nested=True,
    )
    return trading_client.get_orders(filter=request)


def _cancel_open_orders(trading_client: TradingClient, symbol: str) -> None:
    """Cancel all open orders for a symbol and wait for cancellation to settle."""
    open_orders = _get_open_orders(trading_client, symbol)
    for order in open_orders:
        logger.info(
            "Canceling open order symbol=%s order_id=%s status=%s",
            symbol,
            order.id,
            order.status,
        )
        trading_client.cancel_order_by_id(order.id)

    if not open_orders:
        return

    deadline = time.monotonic() + CANCEL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        remaining_orders = _get_open_orders(trading_client, symbol)
        if not remaining_orders:
            logger.info("All open orders canceled symbol=%s", symbol)
            return
        time.sleep(CANCEL_POLL_INTERVAL_SECONDS)

    remaining_ids = [str(order.id) for order in remaining_orders]
    raise TimeoutError(
        f"Timed out waiting for open orders to cancel for {symbol}: {remaining_ids}"
    )


def _close_position_if_open(trading_client: TradingClient, symbol: str) -> None:
    positions = [
        position
        for position in trading_client.get_all_positions()
        if position.symbol == symbol
    ]
    if not positions:
        logger.info("No open position to close symbol=%s", symbol)
        return

    position = positions[0]
    logger.info(
        "Closing position symbol=%s qty=%s side=%s",
        symbol,
        position.qty,
        position.side,
    )
    order = trading_client.close_position(symbol)
    logger.info(
        "Position close submitted symbol=%s order_id=%s status=%s",
        symbol,
        order.id,
        order.status,
    )


def execute_trade(
    symbol: str = "SPY",
    allocation_dollars: float = 1000,
    trading_client: TradingClient | None = None,
) -> None:
    """Predict direction and manage a paper position with attached bracket exits."""
    if allocation_dollars <= 0:
        raise ValueError("allocation_dollars must be greater than zero.")

    client = trading_client or create_trading_client()
    account = client.get_account()
    logger.info(
        "Account status=%s equity=%s buying_power=%s paper=true",
        account.status,
        account.equity,
        account.buying_power,
    )

    model = load_model()
    latest_features, reference_price = get_latest_features(symbol)
    prediction = int(model.predict(latest_features)[0])
    probabilities = model.predict_proba(latest_features)[0]
    upward_probability = float(probabilities[1])
    logger.info(
        "Prediction symbol=%s reference_price=%.2f class=%d upward_probability=%.4f",
        symbol,
        reference_price,
        prediction,
        upward_probability,
    )

    if prediction != 1:
        logger.info("Non-buy signal; canceling orders and closing any position symbol=%s", symbol)
        _cancel_open_orders(client, symbol)
        _close_position_if_open(client, symbol)
        return

    open_orders = _get_open_orders(client, symbol)
    positions = [
        position
        for position in client.get_all_positions()
        if position.symbol == symbol
    ]
    if open_orders or positions:
        logger.warning(
            "Skipping buy to avoid duplicate exposure symbol=%s open_orders=%d positions=%d",
            symbol,
            len(open_orders),
            len(positions),
        )
        return

    quantity = math.floor(allocation_dollars / reference_price)
    if quantity <= 0:
        raise ValueError(
            f"Allocation ${allocation_dollars:.2f} is below the price of one "
            f"whole {symbol} share at ${reference_price:.2f}; Alpaca bracket "
            "orders do not support fractional shares."
        )

    reference_decimal = Decimal(str(reference_price))
    take_profit_price = _round_price(
        reference_decimal * (Decimal("1") + TAKE_PROFIT_PERCENT)
    )
    stop_loss_price = _round_price(
        reference_decimal * (Decimal("1") - STOP_LOSS_PERCENT)
    )
    logger.info(
        "Submitting bracket buy symbol=%s qty=%.2f reference_price=%.2f "
        "take_profit=%.2f stop_loss=%.2f",
        symbol,
        float(quantity),
        reference_price,
        take_profit_price,
        stop_loss_price,
    )

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_loss_price),
    )
    try:
        order = client.submit_order(order_request)
    except Exception as exc:
        logger.exception("Bracket order submission failed symbol=%s", symbol)
        raise RuntimeError(f"Unable to submit bracket order for {symbol}.") from exc

    logger.info(
        "Bracket order submitted symbol=%s order_id=%s status=%s "
        "reference_price_is_estimate=true",
        symbol,
        order.id,
        order.status,
    )


def main() -> int:
    try:
        execute_trade(symbol="SPY", allocation_dollars=1000)
    except Exception:
        logger.exception("Trading execution failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())