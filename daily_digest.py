"""Daily market digest script for Monday-Saturday snapshots."""

import json
import logging
import os
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import yfinance as yf
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from config import (
    ALPACA_API_KEY_ENV,
    ALPACA_SECRET_KEY_ENV,
    ASSETS,
    MODEL_DIR,
    PROJECT_ROOT,
    TELEGRAM_INSIGHTS_BOT_TOKEN_ENV,
    TELEGRAM_INSIGHTS_CHAT_ID_ENV,
)
from telegram_notifier import send_telegram_message

logger = logging.getLogger("multi_asset_trading_bot.daily_digest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
load_dotenv(PROJECT_ROOT / ".env")


def get_crypto_24h_stats(ticker: str) -> dict:
    """Fetch 24h market stats for a crypto ticker."""
    try:
        response = yf.download(
            ticker,
            period="2d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if response is None or response.empty or len(response) < 2:
            logger.warning("Insufficient data for ticker=%s", ticker)
            return None

        latest = response.iloc[-1]
        previous = response.iloc[-2]

        current_price = float(latest["Close"].iloc[0])
        previous_price = float(previous["Close"].iloc[0])
        pct_change = ((current_price - previous_price) / previous_price) * 100
        high = float(latest["High"].iloc[0])
        low = float(latest["Low"].iloc[0])

        return {
            "ticker": ticker,
            "current_price": current_price,
            "pct_change": pct_change,
            "high": high,
            "low": low,
        }
    except Exception as exc:
        logger.exception("Failed to fetch crypto stats ticker=%s", ticker)
        return None


def get_alpaca_account_summary() -> dict:
    """Fetch account balance and position summary from Alpaca."""
    try:
        api_key = os.getenv(ALPACA_API_KEY_ENV)
        secret_key = os.getenv(ALPACA_SECRET_KEY_ENV)
        if not api_key or not secret_key:
            logger.warning("Alpaca credentials not configured")
            return None

        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()

        total_market_value = 0.0
        total_unrealized_pl = 0.0
        active_positions = []

        for position in positions:
            qty = float(position.qty)
            if qty > 0:
                market_value = float(position.market_value)
                unrealized_pl = float(position.unrealized_pl)
                total_market_value += market_value
                total_unrealized_pl += unrealized_pl
                active_positions.append(
                    {
                        "symbol": position.symbol,
                        "qty": qty,
                        "market_value": market_value,
                        "unrealized_pl": unrealized_pl,
                    }
                )

        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "active_position_count": len(active_positions),
            "total_market_value": total_market_value,
            "total_unrealized_pl": total_unrealized_pl,
            "positions": active_positions,
        }
    except Exception as exc:
        logger.exception("Failed to fetch Alpaca account summary")
        return None


def get_model_signal_bias(asset: dict) -> dict:
    """Load model artifact and extract current signal bias."""
    try:
        model_path = Path(asset["model_path"])
        if not model_path.is_absolute():
            model_path = MODEL_DIR / model_path

        if not model_path.is_file():
            logger.warning("Model file missing ticker=%s path=%s", asset["ticker"], model_path)
            return None

        with open(model_path, "r") as f:
            model_data = json.load(f)

        threshold = model_data.get("decision_threshold", 0.0)
        return {
            "ticker": asset["ticker"],
            "threshold": threshold,
            "model_type": model_data.get("model_type", "Unknown"),
        }
    except Exception as exc:
        logger.exception("Failed to load model artifact ticker=%s", asset["ticker"])
        return None


def format_daily_digest(crypto_stats: list, account_summary: dict, model_signals: list) -> str:
    """Format the daily digest HTML message."""
    lines = ["<b>☀️ DAILY CRYPTO DIGEST</b>"]

    for stats in crypto_stats:
        if stats:
            lines.append(
                f"• {escape(stats['ticker'])}: ${stats['current_price']:,.2f} "
                f"(24h: {stats['pct_change']:+.1f}%) | "
                f"Range: ${stats['low']:,.0f} - ${stats['high']:,.0f}"
            )

    lines.append("")
    lines.append("<b>💰 PORTFOLIO STATUS</b>")

    if account_summary:
        if account_summary["positions"]:
            for pos in account_summary["positions"]:
                lines.append(
                    f"• Active Position: {pos['qty']:.8f} {escape(pos['symbol'])} "
                    f"(${pos['market_value']:,.2f} Value)"
                )
        else:
            lines.append("• Active Position: None")

        lines.append(
            f"• Unrealized P/L: ${account_summary['total_unrealized_pl']:+,.2f}"
        )
        lines.append(f"• Account Equity: ${account_summary['equity']:,.2f}")
    else:
        lines.append("• Account data unavailable")

    lines.append("")
    lines.append("<b>📊 CURRENT SIGNALS</b>")

    for signal in model_signals:
        if signal:
            signal_bias = "BUY" if signal["threshold"] > 0 else "HOLD"
            lines.append(
                f"• {escape(signal['ticker'])}: {signal_bias} "
                f"(Threshold: {signal['threshold']:+.2%})"
            )

    lines.append("")
    lines.append(f"<i>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>")

    return "\n".join(lines)


def main() -> int:
    """Generate and send daily digest to Telegram insights bot."""
    logger.info("Starting daily digest generation")

    crypto_stats = []
    for ticker in ["BTC-USD", "ETH-USD"]:
        stats = get_crypto_24h_stats(ticker)
        if stats:
            crypto_stats.append(stats)

    account_summary = get_alpaca_account_summary()

    model_signals = []
    for asset in ASSETS:
        if asset["type"] == "crypto":
            signal = get_model_signal_bias(asset)
            if signal:
                model_signals.append(signal)

    message = format_daily_digest(crypto_stats, account_summary, model_signals)

    bot_token = os.getenv(TELEGRAM_INSIGHTS_BOT_TOKEN_ENV)
    chat_id = os.getenv(TELEGRAM_INSIGHTS_CHAT_ID_ENV)

    if send_telegram_message(message, parse_mode="HTML", bot_token=bot_token, chat_id=chat_id):
        logger.info("Daily digest sent successfully")
        return 0
    else:
        logger.error("Failed to send daily digest")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
