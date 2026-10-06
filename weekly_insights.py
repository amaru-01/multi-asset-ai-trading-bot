"""Weekly insights newsletter script for Sunday deep-dive analysis."""

import json
import logging
import os
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
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

logger = logging.getLogger("multi_asset_trading_bot.weekly_insights")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
load_dotenv(PROJECT_ROOT / ".env")


def get_7day_market_stats(ticker: str) -> dict:
    """Calculate 7-day returns, range, and realized volatility for a ticker."""
    try:
        response = yf.download(
            ticker,
            period="2w",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if response is None or response.empty or len(response) < 7:
            logger.warning("Insufficient 7-day data for ticker=%s", ticker)
            return None

        recent_7d = response.tail(7)
        start_price = float(recent_7d.iloc[0]["Close"].iloc[0])
        end_price = float(recent_7d.iloc[-1]["Close"].iloc[0])
        pct_return = ((end_price - start_price) / start_price) * 100

        high = float(recent_7d["High"].max())
        low = float(recent_7d["Low"].min())

        daily_returns = recent_7d["Close"].pct_change().dropna()
        volatility = float(daily_returns.std() * np.sqrt(252) * 100)

        return {
            "ticker": ticker,
            "start_price": start_price,
            "end_price": end_price,
            "pct_return": pct_return,
            "high": high,
            "low": low,
            "volatility": volatility,
        }
    except Exception as exc:
        logger.exception("Failed to calculate 7-day stats ticker=%s", ticker)
        return None


def get_alpaca_weekly_performance() -> dict:
    """Query Alpaca trade history for weekly P/L and win rate."""
    try:
        api_key = os.getenv(ALPACA_API_KEY_ENV)
        secret_key = os.getenv(ALPACA_SECRET_KEY_ENV)
        if not api_key or not secret_key:
            logger.warning("Alpaca credentials not configured")
            return None

        client = TradingClient(api_key, secret_key, paper=True)

        week_ago = datetime.utcnow() - timedelta(days=7)
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=week_ago,
            limit=100,
        )
        orders = client.get_orders(filter=request)

        total_realized_pl = 0.0
        winning_trades = 0
        total_trades = 0

        for order in orders:
            if order.side.value == "sell" and order.filled_at:
                total_trades += 1
                realized_pl = float(order.filled_qty or 0) * (
                    float(order.filled_avg_price or 0) - float(order.limit_price or 0)
                )
                total_realized_pl += realized_pl
                if realized_pl > 0:
                    winning_trades += 1

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": win_rate,
            "total_realized_pl": total_realized_pl,
        }
    except Exception as exc:
        logger.exception("Failed to fetch Alpaca weekly performance")
        return None


def get_model_summary() -> list:
    """Summarize active model versions and ROC-AUC scores from Model/*.json."""
    summaries = []
    try:
        for asset in ASSETS:
            model_path = Path(asset["model_path"])
            if not model_path.is_absolute():
                model_path = MODEL_DIR / model_path

            if not model_path.is_file():
                logger.warning("Model file missing ticker=%s path=%s", asset["ticker"], model_path)
                continue

            with open(model_path, "r") as f:
                model_data = json.load(f)

            summaries.append(
                {
                    "ticker": asset["ticker"],
                    "model_type": model_data.get("model_type", "Unknown"),
                    "roc_auc": model_data.get("roc_auc", 0.0),
                    "accuracy": model_data.get("accuracy", 0.0),
                    "threshold": model_data.get("decision_threshold", 0.0),
                }
            )
    except Exception as exc:
        logger.exception("Failed to load model summaries")
    return summaries


def format_weekly_insights(
    market_stats: list, weekly_performance: dict, model_summaries: list
) -> str:
    """Format the weekly insights HTML newsletter."""
    lines = ["<b>📊 WEEKLY MARKET INSIGHTS</b>"]
    lines.append("")

    lines.append("<b>7-Day Performance</b>")
    for stats in market_stats:
        if stats:
            lines.append(
                f"• {escape(stats['ticker'])}: {stats['pct_return']:+.2f}% "
                f"| Range: ${stats['low']:,.0f} - ${stats['high']:,.0f} "
                f"| Vol: {stats['volatility']:.1f}%"
            )

    lines.append("")
    lines.append("<b>💼 Trading Performance (Last 7 Days)</b>")

    if weekly_performance:
        lines.append(f"• Total Trades: {weekly_performance['total_trades']}")
        lines.append(f"• Winning Trades: {weekly_performance['winning_trades']}")
        lines.append(f"• Losing Trades: {weekly_performance['losing_trades']}")
        lines.append(
            f"• Win Rate: {weekly_performance['win_rate']:.1f}%"
        )
        lines.append(
            f"• Realized P/L: ${weekly_performance['total_realized_pl']:+,.2f}"
        )
    else:
        lines.append("• Trading data unavailable")

    lines.append("")
    lines.append("<b>🤖 Model Performance Summary</b>")

    for summary in model_summaries:
        lines.append(
            f"• {escape(summary['ticker'])}: ROC-AUC {summary['roc_auc']:.4f} "
            f"| Accuracy {summary['accuracy']:.2%} "
            f"| Threshold {summary['threshold']:+.2%}"
        )

    lines.append("")
    lines.append("<b>📈 Key Takeaways</b>")

    btc_stats = next((s for s in market_stats if s and s["ticker"] == "BTC-USD"), None)
    eth_stats = next((s for s in market_stats if s and s["ticker"] == "ETH-USD"), None)

    if btc_stats:
        if btc_stats["pct_return"] > 0:
            lines.append(f"• BTC positive momentum ({btc_stats['pct_return']:+.2f}%)")
        else:
            lines.append(f"• BTC under pressure ({btc_stats['pct_return']:+.2f}%)")

    if eth_stats:
        if eth_stats["pct_return"] > 0:
            lines.append(f"• ETH showing strength ({eth_stats['pct_return']:+.2f}%)")
        else:
            lines.append(f"• ETH consolidating ({eth_stats['pct_return']:+.2f}%)")

    if weekly_performance and weekly_performance["win_rate"] > 50:
        lines.append(f"• Trading win rate above 50% ({weekly_performance['win_rate']:.1f}%)")
    elif weekly_performance:
        lines.append(f"• Trading win rate below 50% ({weekly_performance['win_rate']:.1f}%)")

    lines.append("")
    lines.append(f"<i>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>")

    return "\n".join(lines)


def main() -> int:
    """Generate and send weekly insights to Telegram insights bot."""
    logger.info("Starting weekly insights generation")

    market_stats = []
    for ticker in ["BTC-USD", "ETH-USD", "SPY"]:
        stats = get_7day_market_stats(ticker)
        if stats:
            market_stats.append(stats)

    weekly_performance = get_alpaca_weekly_performance()
    model_summaries = get_model_summary()

    message = format_weekly_insights(market_stats, weekly_performance, model_summaries)

    bot_token = os.getenv(TELEGRAM_INSIGHTS_BOT_TOKEN_ENV)
    chat_id = os.getenv(TELEGRAM_INSIGHTS_CHAT_ID_ENV)

    if send_telegram_message(message, parse_mode="HTML", bot_token=bot_token, chat_id=chat_id):
        logger.info("Weekly insights sent successfully")
        return 0
    else:
        logger.error("Failed to send weekly insights")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
