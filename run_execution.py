"""Execution orchestration for regression and legacy classifier models.

Runs against the Alpaca paper account by default. Live trading requires
ALPACA_PAPER=false plus ALPACA_LIVE_CONFIRM and ALPACA_LIVE_MAX_CAPITAL (see the
"Live-trading safeguards" block below). A kill switch and a daily-loss circuit
breaker stop new entries without blocking exits.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
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
# Crypto has no bracket orders on Alpaca, so exits are checked by the bot itself:
# close a crypto position when price falls this many ATRs below its entry price.
STOP_LOSS_ATR_MULTIPLE = 1.5
# After any filled SELL on a symbol (stop-loss, HOLD close, manual close, SPY
# bracket exit), do not re-enter for this many hours. Set to 0 to disable.
COOLDOWN_HOURS = 6
# Round trips cost ~0.5% in fees (Alpaca crypto charges ~0.25% per side), so a new
# position is only opened when the prediction clears the model's own threshold by
# a margin. Exits still use the model's threshold, which leaves a hysteresis band.
CLASSIFIER_ENTRY_MIN = 0.60  # minimum up-probability to open a position
REGRESSOR_ENTRY_MIN = 0.006  # minimum predicted 3-day return (0.6%) to open one
# A HOLD signal may not close a position opened less than this many hours ago.
# The crypto stop-loss is exempt. Set to 0 to disable.
MIN_HOLD_HOURS = 8
# The models are trained on finished daily candles, so build signals from
# completed bars only (today's still-forming bar is dropped before features).
USE_COMPLETED_BARS = True
_SPY_CLOSE_CACHE: pd.Series | None = None


def is_paper_mode() -> bool:
    """Paper trading unless ALPACA_PAPER is explicitly set to "false"."""
    return os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"


def mode_label() -> str:
    return "PAPER" if is_paper_mode() else "LIVE"


# --- Live-trading safeguards ---------------------------------------------------
# Going live needs ALL of: ALPACA_PAPER=false, ALPACA_LIVE_CONFIRM set to the exact
# phrase below, and ALPACA_LIVE_MAX_CAPITAL set to a positive dollar amount. The bot
# then sizes everything as if the account held at most that much money.
LIVE_CONFIRM_ENV = "ALPACA_LIVE_CONFIRM"
LIVE_CONFIRM_PHRASE = "I_ACCEPT_REAL_MONEY_RISK"
LIVE_MAX_CAPITAL_ENV = "ALPACA_LIVE_MAX_CAPITAL"
# Kill switch: set TRADING_HALTED=true, or create a file named HALT_TRADING in the
# project root. New entries stop; exits, stop-losses and cleanups keep working.
HALT_ENV = "TRADING_HALTED"
HALT_FILE = PROJECT_ROOT / "HALT_TRADING"
# Circuit breaker: stop opening positions once equity is down this much versus the
# previous trading day's closing equity. Override with MAX_DAILY_LOSS_PCT (e.g. 0.02).
DEFAULT_MAX_DAILY_LOSS_PCT = 0.03
MAX_DAILY_LOSS_ENV = "MAX_DAILY_LOSS_PCT"


def live_capital_cap() -> float | None:
    """Return the live-mode capital cap in USD, or None in paper mode.

    Raises ValueError if live mode is requested without the confirmation phrase or
    without a valid positive capital cap, so a half-configured live run never starts.
    """
    if is_paper_mode():
        return None
    if os.getenv(LIVE_CONFIRM_ENV, "").strip() != LIVE_CONFIRM_PHRASE:
        raise ValueError(
            f"Live trading refused: set {LIVE_CONFIRM_ENV}={LIVE_CONFIRM_PHRASE} "
            "to confirm you accept the risk of real-money trading"
        )
    raw = os.getenv(LIVE_MAX_CAPITAL_ENV, "").strip()
    try:
        cap = float(raw)
    except ValueError:
        cap = float("nan")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError(
            f"Live trading refused: {LIVE_MAX_CAPITAL_ENV} must be a positive dollar "
            f"amount (got {raw!r})"
        )
    return cap


def kill_switch_active() -> bool:
    if os.getenv(HALT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return HALT_FILE.exists()


def max_daily_loss_pct() -> float:
    raw = os.getenv(MAX_DAILY_LOSS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_DAILY_LOSS_PCT
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", MAX_DAILY_LOSS_ENV, raw)
        return DEFAULT_MAX_DAILY_LOSS_PCT
    return value if math.isfinite(value) and 0 < value < 1 else DEFAULT_MAX_DAILY_LOSS_PCT


def entries_halt_reason(equity: float, last_equity: float | None) -> str | None:
    """Why new positions must not be opened right now, or None if trading may proceed."""
    if kill_switch_active():
        return f"kill switch active ({HALT_ENV} or {HALT_FILE.name})"
    if last_equity and last_equity > 0 and math.isfinite(equity):
        change = equity / last_equity - 1.0
        limit = max_daily_loss_pct()
        if change <= -limit:
            return (
                f"daily loss circuit breaker: equity {change:+.2%} vs previous close "
                f"(limit -{limit:.2%})"
            )
    return None


def apply_capital_cap(
    trading_client: TradingClient,
    cap: float | None,
    equity: float,
    buying_power: float,
) -> tuple[float, float]:
    """Limit sizing inputs to the live capital cap (no-op in paper mode)."""
    if cap is None:
        return equity, buying_power
    invested = sum(
        get_position_value(trading_client, asset["alpaca_symbol"]) for asset in ASSETS
    )
    return min(equity, cap), min(buying_power, max(0.0, cap - invested))


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


def drop_partial_bar(data: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop today's still-forming daily bar so signals use completed candles only.

    Falls back to returning the data unchanged (with a warning) if the index is
    not a DatetimeIndex, so a format surprise never breaks trading.
    """
    if not USE_COMPLETED_BARS or len(data) < 2:
        return data
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning("Cannot detect partial bar, index is not dates label=%s", label)
        return data
    last = data.index[-1]
    if last.tzinfo is not None:
        last = last.tz_localize(None)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    if last.normalize() >= today:
        logger.info("Dropped partial daily bar label=%s bar_date=%s", label, last.date())
        return data.iloc[:-1]
    return data


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
        normalized = drop_partial_bar(
            normalize_market_columns(response), "SPY-benchmark"
        )
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
        # Latest traded price (including today's forming bar) is used for sizing
        # and the bracket levels; the signal itself uses completed bars only.
        latest_price = float(
            pd.to_numeric(market_data["Close"], errors="coerce").dropna().iloc[-1]
        )
        signal_data = drop_partial_bar(market_data, ticker)
        spy_close = (
            pd.to_numeric(signal_data["Close"], errors="coerce")
            if ticker == "SPY"
            else get_spy_close_history()
        )
        frame = build_inference_features(signal_data, feature_columns, spy_close)
        if frame.empty:
            raise ValueError(f"Insufficient valid feature history for {ticker}")
        latest_features = frame.loc[:, feature_columns].tail(1).astype(float)
        reference_price = latest_price
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


def buying_blocked(ticker: str) -> tuple[bool, str]:
    """Check Model/model_status.json, written by retrain.py, for a buying block.

    Fails open: a missing or unreadable file never stops trading, so behaviour is
    unchanged until retrain.py has run and recorded a verdict for the asset.
    """
    path = MODEL_DIR / "model_status.json"
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        entry = (status.get("assets") or {}).get(ticker) or {}
    except FileNotFoundError:
        return False, ""
    except Exception:
        logger.warning("Could not read model status file path=%s", path, exc_info=True)
        return False, ""
    if entry.get("buying") == "blocked":
        return True, str(entry.get("reason", "blocked by model status"))
    return False, ""


def entry_margin_required(model, result: dict) -> float:
    """Prediction needed to open a new position (never below the model's own threshold)."""
    floor = (
        REGRESSOR_ENTRY_MIN
        if isinstance(model, XGBRegressorModel)
        else CLASSIFIER_ENTRY_MIN
    )
    return max(float(result["threshold"]), floor)


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


def stop_loss_triggered(
    position, reference_price: float, current_atr: float
) -> tuple[bool, float, float, float]:
    """Check the ATR stop. Returns (hit, entry_price, current_price, stop_price)."""
    entry_price = float(position.avg_entry_price)
    current_price = float(position.current_price or reference_price)
    stop_price = entry_price - STOP_LOSS_ATR_MULTIPLE * current_atr
    return current_price <= stop_price, entry_price, current_price, stop_price


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

    # Bot-side stop-loss for crypto (SPY already has an exchange-side bracket stop).
    if asset["type"] == "crypto":
        held = get_open_position(trading_client, alpaca_symbol)
        if held is not None:
            hit, entry_price, current_price, stop_price = stop_loss_triggered(
                held, reference_price, current_atr
            )
            if hit:
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
                return

    if result["signal"] != "BUY":
        # Don't let a marginal signal flip close a position that was only just
        # opened. This also leaves SPY's bracket stop/take-profit orders in place.
        # The crypto stop-loss above is unaffected.
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
        return

    if halt_reason:
        logger.warning(
            "Skipping buy: entries halted ticker=%s reason=%s",
            asset["ticker"],
            halt_reason,
        )
        return

    blocked, block_reason = buying_blocked(asset["ticker"])
    if blocked:
        logger.warning(
            "Skipping buy: blocked by model status ticker=%s reason=%s",
            asset["ticker"],
            block_reason,
        )
        return

    if asset["type"] != "crypto" and not us_market_open(trading_client):
        logger.info(
            "Skipping buy: US market is closed ticker=%s (no after-hours orders sized from stale prices)",
            asset["ticker"],
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

    required = entry_margin_required(model, result)
    if float(result["raw_prediction"]) < required:
        logger.info(
            "Skipping buy: below entry margin ticker=%s prediction=%.4f required=%.4f",
            asset["ticker"],
            float(result["raw_prediction"]),
            required,
        )
        return

    if COOLDOWN_HOURS > 0:
        recent_exit = get_recent_filled_order(
            trading_client, alpaca_symbol, OrderSide.SELL, COOLDOWN_HOURS
        )
        if recent_exit is not None:
            logger.info(
                "Skipping buy during cooldown ticker=%s last_exit_order=%s filled_at=%s "
                "cooldown_hours=%s",
                asset["ticker"],
                recent_exit.id,
                recent_exit.filled_at,
                COOLDOWN_HOURS,
            )
            return

    # Independent exposure cap: never let one asset exceed its equity fraction,
    # even if the duplicate-position check above is ever bypassed.
    existing_value = get_position_value(trading_client, alpaca_symbol)
    exposure_room = equity * MAX_POSITION_EQUITY_FRACTION - existing_value
    if exposure_room <= 0:
        logger.warning(
            "Skipping buy: exposure cap reached ticker=%s held=%.2f cap=%.2f",
            asset["ticker"],
            existing_value,
            equity * MAX_POSITION_EQUITY_FRACTION,
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
    max_position_value = min(exposure_room, buying_power)
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute trades from model signals")
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
    """Execute requested assets serially using the configured Alpaca account."""
    arguments = _parse_args(argv)
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
    try:
        last_equity = float(getattr(account, "last_equity", None) or 0) or None
    except (TypeError, ValueError):
        last_equity = None
    halt_reason = entries_halt_reason(float(account.equity), last_equity)
    if halt_reason:
        logger.error("Entries halted: %s", halt_reason)
        send_telegram_message(
            f"<b>ENTRIES HALTED</b> [{mode_label()}]\n{escape(halt_reason)}\n"
            "Exits and stop-losses still run."
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
    if had_asset_failure:
        logger.error("One or more requested assets failed execution")
        return 1
    logger.info("%s execution completed assets=%s", mode_label(), [item["ticker"] for item in arguments.assets])
    return 0


if __name__ == "__main__":
    sys.exit(main())