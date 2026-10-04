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
STOCK_FEATURE_COLUMNS = [
    "RSI",
    "MACD",
    "ATR",
    "Return_1D",
    "Vol_Change",
    "RSI_lag1",
    "Return_lag1",
]
CRYPTO_FEATURE_COLUMNS = [
    "SMA_10",
    "SMA_50",
    "RSI",
    "MACD",
    "MACD_signal",
    "BB_high",
    "BB_low",
]
TAKE_PROFIT_PERCENT = Decimal("0.015")
STOP_LOSS_PERCENT = Decimal("0.01")
CANCEL_TIMEOUT_SECONDS = 10
CANCEL_POLL_INTERVAL_SECONDS = 0.5

# Each asset uses its own market-data ticker, Alpaca symbol, model, and allocation.
SYMBOLS_CONFIG = [
    {
        "symbol": "SPY",
        "type": "stock",
        "yfinance_ticker": "SPY",
        "alpaca_symbol": "SPY",
        "model_path": "Model/xgb_trading_model.json",
        "allocation_dollars": 1000,
        "feature_columns": STOCK_FEATURE_COLUMNS,
    },
    {
        "symbol": "BTC",
        "type": "crypto",
        "yfinance_ticker": "BTC-USD",
        "alpaca_symbol": "BTC/USD",
        "model_path": "Model/xgb_btc_model.json",
        "allocation_dollars": 500,
        "feature_columns": CRYPTO_FEATURE_COLUMNS,
    },
    {
        "symbol": "ETH",
        "type": "crypto",
        "yfinance_ticker": "ETH-USD",
        "alpaca_symbol": "ETH/USD",
        "model_path": "Model/xgb_eth_model.json",
        "allocation_dollars": 500,
        "feature_columns": CRYPTO_FEATURE_COLUMNS,
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("multi_asset_trading_bot")

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


def load_model(
    model_path: Path,
    expected_features: list[str],
) -> xgb.XGBClassifier:
    """Load one native XGBoost JSON model and validate its feature contract."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = xgb.XGBClassifier()
    try:
        model.load_model(str(model_path))
    except Exception as exc:
        logger.exception("Could not load XGBoost model path=%s", model_path)
        raise RuntimeError(f"Unable to load model at {model_path}") from exc

    booster = model.get_booster()
    trained_features = booster.feature_names
    if trained_features and trained_features != expected_features:
        raise ValueError(
            "Model feature names/order do not match execution features: "
            f"model={trained_features}, execution={expected_features}"
        )
    if booster.num_features() != len(expected_features):
        raise ValueError(
            f"Model expects {booster.num_features()} features; "
            f"execution provides {len(expected_features)}."
        )

    logger.info("Loaded XGBoost model path=%s", model_path)
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


def get_latest_features(
    ticker: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, float]:
    """Download recent OHLCV data and calculate one aligned model input row."""
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

        # Crypto models use a separate SMA, MACD signal, and Bollinger schema.
        data["SMA_10"] = close.rolling(window=10).mean()
        data["SMA_50"] = close.rolling(window=50).mean()
        data["MACD_signal"] = ta.trend.macd_signal(close)
        data["BB_high"] = ta.volatility.bollinger_hband(close)
        data["BB_low"] = ta.volatility.bollinger_lband(close)

        missing_features = set(feature_columns).difference(data.columns)
        if missing_features:
            raise ValueError(
                f"Unsupported model features for {ticker}: "
                f"{sorted(missing_features)}"
            )

        # Keep model inputs finite and in the model's trained feature order.
        data.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
        data.dropna(subset=feature_columns + ["Close"], inplace=True)
        if data.empty:
            raise ValueError(
                f"Not enough valid history to calculate features for {ticker}."
            )

        latest_features = data.loc[:, feature_columns].tail(1).astype(float)
        reference_price = float(data["Close"].iloc[-1])
        if not math.isfinite(reference_price) or reference_price <= 0:
            raise ValueError(
                f"Invalid reference price for {ticker}: {reference_price}."
            )
        return latest_features, reference_price
    except Exception as exc:
        logger.exception("Failed to fetch or prepare market data ticker=%s", ticker)
        raise RuntimeError(f"Unable to prepare market features for {ticker}.") from exc


def _round_price(price: Decimal) -> float:
    """Round a price to the nearest cent using conventional half-up rounding."""
    return float(price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _get_open_orders(trading_client: TradingClient, symbol: str) -> list:
    request = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        symbols=[symbol],
    )
    return trading_client.get_orders(filter=request)


def _cancel_open_orders(trading_client: TradingClient, symbol: str) -> None:
    """Cancel open symbol orders and wait briefly for cancellation to settle."""
    deadline = time.monotonic() + CANCEL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        open_orders = _get_open_orders(trading_client, symbol)
        if not open_orders:
            logger.info("No remaining open orders symbol=%s", symbol)
            return

        # Refresh after each cancellation because canceling one bracket exit can
        # automatically cancel its paired exit order.
        order = open_orders[0]
        logger.info(
            "Canceling open order symbol=%s order_id=%s status=%s",
            symbol,
            order.id,
            order.status,
        )
        try:
            trading_client.cancel_order_by_id(order.id)
        except Exception:
            # A fill/cancel race is harmless if the order has already left the
            # open set; propagate the error if the order is still active.
            remaining_ids = {
                str(open_order.id)
                for open_order in _get_open_orders(trading_client, symbol)
            }
            if str(order.id) in remaining_ids:
                raise
            logger.info("Order no longer open after cancel race order_id=%s", order.id)
        time.sleep(CANCEL_POLL_INTERVAL_SECONDS)

    remaining_orders = _get_open_orders(trading_client, symbol)
    remaining_ids = [str(order.id) for order in remaining_orders]
    raise TimeoutError(
        f"Timed out waiting for open orders to cancel for {symbol}: {remaining_ids}"
    )


def _get_open_position(trading_client: TradingClient, symbol: str):
    """Find a position by Alpaca's symbol, including slash-separated crypto pairs."""
    positions = trading_client.get_all_positions()
    return next((position for position in positions if position.symbol == symbol), None)


def _close_position_if_open(trading_client: TradingClient, symbol: str) -> None:
    position = _get_open_position(trading_client, symbol)
    if position is None:
        logger.info("No open position to close symbol=%s", symbol)
        return

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


def execute_asset(
    asset: dict,
    trading_client: TradingClient,
    model: xgb.XGBClassifier,
) -> None:
    """Evaluate one configured asset and submit its appropriate paper order."""
    asset_type = asset["type"]
    if asset_type not in {"stock", "crypto"}:
        raise ValueError(f"Unsupported asset type: {asset_type}")

    logical_symbol = asset["symbol"]
    alpaca_symbol = asset["alpaca_symbol"]
    feature_columns = asset["feature_columns"]
    latest_features, reference_price = get_latest_features(
        asset["yfinance_ticker"],
        feature_columns,
    )
    prediction = int(model.predict(latest_features)[0])
    probabilities = model.predict_proba(latest_features)[0]
    upward_probability = float(probabilities[1])
    logger.info(
        "Asset prediction symbol=%s type=%s yfinance_ticker=%s "
        "alpaca_symbol=%s reference_price=%.2f prediction=%d "
        "upward_probability=%.4f",
        logical_symbol,
        asset_type,
        asset["yfinance_ticker"],
        alpaca_symbol,
        reference_price,
        prediction,
        upward_probability,
    )

    if prediction != 1:
        logger.info(
            "Non-buy signal; cleaning up orders and position symbol=%s",
            alpaca_symbol,
        )
        _cancel_open_orders(trading_client, alpaca_symbol)
        _close_position_if_open(trading_client, alpaca_symbol)
        return

    # Do not increase exposure when this asset already has an order or position.
    open_orders = _get_open_orders(trading_client, alpaca_symbol)
    position = _get_open_position(trading_client, alpaca_symbol)
    if open_orders or position is not None:
        logger.warning(
            "Skipping buy to avoid duplicate exposure symbol=%s open_orders=%d "
            "has_position=%s",
            alpaca_symbol,
            len(open_orders),
            position is not None,
        )
        return

    allocation = float(asset["allocation_dollars"])
    if allocation <= 0:
        raise ValueError(f"Allocation must be positive for {logical_symbol}.")

    if asset_type == "crypto":
        # Crypto market orders use USD notional rather than fractional share qty.
        order_request = MarketOrderRequest(
            symbol=alpaca_symbol,
            notional=allocation,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
        logger.info(
            "Submitting crypto market buy symbol=%s notional_usd=%.2f",
            alpaca_symbol,
            allocation,
        )
    else:
        # Alpaca equity bracket orders require whole shares; round down to budget.
        quantity = math.floor(allocation / reference_price)
        if quantity <= 0:
            raise ValueError(
                f"Allocation ${allocation:.2f} is below the price of one whole "
                f"{alpaca_symbol} share at ${reference_price:.2f}."
            )

        reference_decimal = Decimal(str(reference_price))
        take_profit_price = _round_price(
            reference_decimal * (Decimal("1") + Decimal("0.015"))
        )
        stop_loss_price = _round_price(
            reference_decimal * (Decimal("1") - Decimal("0.01"))
        )
        order_request = MarketOrderRequest(
            symbol=alpaca_symbol,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
            stop_loss=StopLossRequest(stop_price=stop_loss_price),
        )
        logger.info(
            "Submitting equity bracket buy symbol=%s qty=%d reference_price=%.2f "
            "take_profit=%.2f stop_loss=%.2f reference_is_estimate=true",
            alpaca_symbol,
            quantity,
            reference_price,
            take_profit_price,
            stop_loss_price,
        )

    try:
        order = trading_client.submit_order(order_request)
    except Exception as exc:
        logger.exception("Order submission failed symbol=%s", alpaca_symbol)
        raise RuntimeError(f"Unable to submit order for {alpaca_symbol}.") from exc

    logger.info(
        "Order submitted symbol=%s order_id=%s status=%s",
        alpaca_symbol,
        order.id,
        order.status,
    )


def main() -> int:
    """Run configured assets sequentially; return non-zero on asset failures."""
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

    # Process assets serially to keep API usage and runner time predictable.
    had_asset_failure = False
    for asset in SYMBOLS_CONFIG:
        model_path = PROJECT_ROOT / asset["model_path"]
        if not model_path.is_file() and asset["type"] == "crypto":
            logger.warning(
                "Model file not found for %s, skipping... path=%s",
                asset["yfinance_ticker"],
                model_path,
            )
            continue

        try:
            model = load_model(model_path, asset["feature_columns"])
            execute_asset(asset, trading_client, model)
        except Exception:
            had_asset_failure = True
            logger.exception(
                "Asset processing failed symbol=%s alpaca_symbol=%s",
                asset["symbol"],
                asset["alpaca_symbol"],
            )

    if had_asset_failure:
        logger.error("One or more assets failed during this execution.")
        return 1

    logger.info("Multi-asset trading execution completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())