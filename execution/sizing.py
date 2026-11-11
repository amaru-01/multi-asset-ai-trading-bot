"""Position sizing: ATR risk budget, confidence scaling and the per-asset exposure cap."""

import logging
import math

from alpaca.trading.client import TradingClient

from models import XGBRegressorModel

from .broker import get_position_value

logger = logging.getLogger("multi_asset_trading_bot")

RISK_FACTOR = 0.01
MAX_POSITION_EQUITY_FRACTION = 0.25


def size_position(
    asset: dict,
    trading_client: TradingClient,
    model,
    result: dict,
    equity: float,
    buying_power: float,
    reference_price: float,
    current_atr: float,
) -> float | None:
    """Dollar value to buy, or None when the per-asset exposure cap leaves no room."""
    # Independent exposure cap: never let one asset exceed its equity fraction,
    # even if the duplicate-position check above is ever bypassed.
    existing_value = get_position_value(trading_client, asset["alpaca_symbol"])
    exposure_room = equity * MAX_POSITION_EQUITY_FRACTION - existing_value
    if exposure_room <= 0:
        logger.warning(
            "Skipping buy: exposure cap reached ticker=%s held=%.2f cap=%.2f",
            asset["ticker"],
            existing_value,
            equity * MAX_POSITION_EQUITY_FRACTION,
        )
        return None

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
    return allocation
