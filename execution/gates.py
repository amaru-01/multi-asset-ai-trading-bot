"""Entry gates: every check a BUY signal must pass before a position may be opened."""

import json
import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide

from config import MODEL_DIR
from models import XGBRegressorModel

from .broker import (
    get_open_orders,
    get_open_position,
    get_recent_filled_order,
    us_market_open,
)
from .safeguards import BUY_ASSETS_ENV, buy_enabled

logger = logging.getLogger("multi_asset_trading_bot")

# After any filled SELL on a symbol (stop-loss, HOLD close, manual close, SPY
# bracket exit), do not re-enter for this many hours. Set to 0 to disable.
COOLDOWN_HOURS = 6
# Round trips cost ~0.5% in fees (Alpaca crypto charges ~0.25% per side), so a new
# position is only opened when the prediction clears the model's own threshold by
# a margin. Exits still use the model's threshold, which leaves a hysteresis band.
CLASSIFIER_ENTRY_MIN = 0.60  # minimum up-probability to open a position
REGRESSOR_ENTRY_MIN = 0.006  # minimum predicted 3-day return (0.6%) to open one


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


def _switches_allow_entry(
    asset: dict,
    trading_client: TradingClient,
    halt_reason: str | None,
) -> bool:
    """Halt, per-asset switch, model-status block and market-hours checks."""
    if halt_reason:
        logger.warning(
            "Skipping buy: entries halted ticker=%s reason=%s",
            asset["ticker"],
            halt_reason,
        )
        return False

    if not buy_enabled(asset):
        logger.info(
            "Skipping buy: buying disabled for asset ticker=%s "
            "(config buying_enabled / %s)",
            asset["ticker"],
            BUY_ASSETS_ENV,
        )
        return False

    blocked, block_reason = buying_blocked(asset["ticker"])
    if blocked:
        logger.warning(
            "Skipping buy: blocked by model status ticker=%s reason=%s",
            asset["ticker"],
            block_reason,
        )
        return False

    if asset["type"] != "crypto" and not us_market_open(trading_client):
        logger.info(
            "Skipping buy: US market is closed ticker=%s (no after-hours orders sized from stale prices)",
            asset["ticker"],
        )
        return False
    return True


def _order_checks_pass(
    asset: dict,
    trading_client: TradingClient,
    model,
    result: dict,
) -> bool:
    """Duplicate-position, entry-margin and cooldown checks."""
    alpaca_symbol = asset["alpaca_symbol"]
    open_orders = get_open_orders(trading_client, alpaca_symbol)
    position = get_open_position(trading_client, alpaca_symbol)
    if open_orders or position is not None:
        logger.warning(
            "Skipping duplicate buy ticker=%s open_orders=%d has_position=%s",
            asset["ticker"],
            len(open_orders),
            position is not None,
        )
        return False

    required = entry_margin_required(model, result)
    if float(result["raw_prediction"]) < required:
        logger.info(
            "Skipping buy: below entry margin ticker=%s prediction=%.4f required=%.4f",
            asset["ticker"],
            float(result["raw_prediction"]),
            required,
        )
        return False

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
            return False
    return True


def entry_allowed(
    asset: dict,
    trading_client: TradingClient,
    model,
    result: dict,
    halt_reason: str | None,
) -> bool:
    """Run every gate a BUY signal must pass, in order, stopping at the first failure."""
    return _switches_allow_entry(asset, trading_client, halt_reason) and _order_checks_pass(
        asset, trading_client, model, result
    )
