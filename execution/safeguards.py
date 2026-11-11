"""Live-trading safeguards: capital cap, kill switch, loss breaker, buy-enabled switch."""

import logging
import math
import os

from alpaca.trading.client import TradingClient

from config import ASSETS, PROJECT_ROOT

from .broker import get_position_value
from .mode import is_paper_mode

logger = logging.getLogger("multi_asset_trading_bot")

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
BUY_ASSETS_ENV = "BUY_ASSETS"


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


def _asset_names(asset: dict) -> set[str]:
    return {
        asset["ticker"].upper(),
        asset["symbol"].upper(),
        asset["alpaca_symbol"].upper(),
    }


def buy_enabled(asset: dict) -> bool:
    """May the bot open NEW positions in this asset?

    BUY_ASSETS (comma-separated tickers such as "BTC-USD,ETH-USD") overrides the
    "buying_enabled" flag in config.py when set and non-empty. Use TRADING_HALTED to
    stop buying everything. This never affects managing a position already held.
    """
    raw = os.getenv(BUY_ASSETS_ENV, "").strip()
    if not raw:
        return bool(asset.get("buying_enabled", True))
    wanted = {item.strip().upper() for item in raw.split(",") if item.strip()}
    return bool(wanted & _asset_names(asset))


def warn_unknown_buy_assets() -> None:
    """Log names in BUY_ASSETS that match no configured asset (likely typos)."""
    raw = os.getenv(BUY_ASSETS_ENV, "").strip()
    if not raw:
        return
    known = set().union(*(_asset_names(asset) for asset in ASSETS))
    unknown = sorted(
        item.strip() for item in raw.split(",")
        if item.strip() and item.strip().upper() not in known
    )
    if unknown:
        logger.warning("%s names no configured asset: %s", BUY_ASSETS_ENV, unknown)


def kill_switch_source() -> str | None:
    """Name what triggered the kill switch, or None if it is off."""
    if os.getenv(HALT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return f"{HALT_ENV} environment variable"
    if HALT_FILE.exists():
        return f"{HALT_FILE.name} file"
    return None


def kill_switch_active() -> bool:
    return kill_switch_source() is not None


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
    source = kill_switch_source()
    if source:
        return f"kill switch active ({source})"
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
