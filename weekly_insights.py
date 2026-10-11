"""Weekly insights newsletter script for Sunday deep-dive analysis."""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
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


def _clean_daily_closes(response) -> pd.DataFrame | None:
    """Flatten yfinance columns and drop today's still-forming daily bar."""
    if response is None or response.empty:
        return None
    if isinstance(response.columns, pd.MultiIndex):
        response = response.copy()
        response.columns = response.columns.get_level_values(0)
    response = response.dropna(subset=["Close"])
    if response.empty:
        return None
    index = pd.DatetimeIndex(response.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    response = response.copy()
    response.index = index.normalize()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    return response[response.index < today]


def compute_7day_stats(ticker: str, frame: pd.DataFrame) -> dict | None:
    """7-day return (calendar days), range and annualised volatility from completed bars."""
    if frame is None or len(frame) < 5:
        return None
    last_date = frame.index[-1]
    window = frame[frame.index > last_date - pd.Timedelta(days=7)]
    before = frame[frame.index <= last_date - pd.Timedelta(days=7)]
    if before.empty or len(window) < 3:
        return None
    start_price = float(before["Close"].iloc[-1])
    end_price = float(window["Close"].iloc[-1])
    closes = pd.concat([before["Close"].iloc[[-1]], window["Close"]])
    daily_returns = closes.pct_change().dropna()
    return {
        "ticker": ticker,
        "start_price": start_price,
        "end_price": end_price,
        "pct_return": (end_price - start_price) / start_price * 100,
        "high": float(window["High"].max()),
        "low": float(window["Low"].min()),
        "volatility": float(daily_returns.std() * np.sqrt(252) * 100),
    }


def get_7day_market_stats(ticker: str) -> dict | None:
    """Download about a month of daily bars and summarise the last 7 days."""
    try:
        response = yf.download(
            ticker,
            period="1mo",  # yfinance does not accept "2w"
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        stats = compute_7day_stats(ticker, _clean_daily_closes(response))
        if stats is None:
            logger.warning("Insufficient 7-day data for ticker=%s", ticker)
        return stats
    except Exception:
        logger.exception("Failed to calculate 7-day stats ticker=%s", ticker)
        return None


def _norm_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "").replace("-", "").upper()


def compute_realized_pl(orders, window_start: datetime) -> dict:
    """FIFO-match filled buys to filled sells; count sells filled inside the window.

    Prices only: exchange fees (about 0.25% per crypto side) are not subtracted,
    so real results are slightly lower than shown.
    """
    filled = []
    for order in orders:
        if not getattr(order, "filled_at", None):
            continue
        qty = float(order.filled_qty or 0)
        price = float(order.filled_avg_price or 0)
        if qty <= 0 or price <= 0:
            continue
        side = order.side.value if hasattr(order.side, "value") else str(order.side)
        filled.append((order.filled_at, _norm_symbol(order.symbol), side.lower(), qty, price))
    filled.sort(key=lambda row: row[0])

    lots: dict[str, list[list[float]]] = {}
    total = 0.0
    wins = losses = unmatched = 0
    window_start = window_start if window_start.tzinfo else window_start.replace(tzinfo=timezone.utc)
    for filled_at, symbol, side, qty, price in filled:
        book = lots.setdefault(symbol, [])
        if side == "buy":
            book.append([qty, price])
            continue
        remaining, pnl, matched = qty, 0.0, 0.0
        while remaining > 1e-12 and book:
            lot = book[0]
            take = min(remaining, lot[0])
            pnl += take * (price - lot[1])
            matched += take
            lot[0] -= take
            remaining -= take
            if lot[0] <= 1e-12:
                book.pop(0)
        if filled_at < window_start:
            continue
        if matched <= 0:
            unmatched += 1
            continue
        total += pnl
        wins += pnl > 0
        losses += pnl <= 0
    trades = wins + losses
    return {
        "total_trades": trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": (wins / trades * 100) if trades else 0.0,
        "total_realized_pl": total,
        "unmatched_sells": unmatched,
    }


def get_alpaca_weekly_performance() -> dict | None:
    """Realized P/L and win rate for round trips closed in the last 7 days."""
    try:
        api_key = os.getenv(ALPACA_API_KEY_ENV)
        secret_key = os.getenv(ALPACA_SECRET_KEY_ENV)
        if not api_key or not secret_key:
            logger.warning("Alpaca credentials not configured")
            return None

        paper = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"
        client = TradingClient(api_key, secret_key, paper=paper)

        now = datetime.now(timezone.utc)
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=now - timedelta(days=60),  # older buys are needed for cost basis
            limit=500,
        )
        orders = client.get_orders(filter=request)
        return compute_realized_pl(orders, now - timedelta(days=7))
    except Exception:
        logger.exception("Failed to fetch Alpaca weekly performance")
        return None


def get_model_summary() -> list:
    """Read the latest retraining verdicts from Model/model_status.json."""
    status_path = Path(MODEL_DIR) / "model_status.json"
    try:
        data = json.loads(status_path.read_text())
    except FileNotFoundError:
        logger.warning("No model status file at %s", status_path)
        return []
    except Exception:
        logger.exception("Could not read model status")
        return []

    summaries = []
    for asset in ASSETS:
        entry = (data.get("assets") or {}).get(asset["ticker"])
        if not entry:
            continue
        candidate = entry.get("candidate") or {}
        summaries.append(
            {
                "ticker": asset["ticker"],
                "buying": entry.get("buying", "allowed"),
                "live_model": entry.get("live_model", "unknown"),
                "roc_auc": candidate.get("roc_auc"),
                "wf_positive": candidate.get("wf_positive_folds"),
                "wf_folds": candidate.get("wf_folds"),
                "trades": candidate.get("backtest_trades"),
                "net": candidate.get("backtest_mean_net"),
                "checked_at": str(entry.get("checked_at", ""))[:10],
            }
        )
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
            f"• Realized P/L (before fees): ${weekly_performance['total_realized_pl']:+,.2f}"
        )
    else:
        lines.append("• Trading data unavailable")

    lines.append("")
    lines.append("<b>🤖 Model Status (latest weekly retrain)</b>")
    if not model_summaries:
        lines.append("• No retraining status available yet")
    for summary in model_summaries:
        icon = "🔴 BLOCKED" if summary["buying"] == "blocked" else "🟢 OK"
        auc = f"{summary['roc_auc']:.3f}" if summary["roc_auc"] is not None else "--"
        wf = (
            f"{summary['wf_positive']}/{summary['wf_folds']}"
            if summary["wf_folds"] is not None
            else "--"
        )
        net = (
            f"{summary['net']:+.2%} over {summary['trades']} trades"
            if summary["net"] is not None
            else "no trades"
        )
        lines.append(
            f"• {escape(summary['ticker'])}: {icon} | candidate AUC {auc} | WF {wf} | {net}"
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

    blocked = [s["ticker"] for s in model_summaries if s["buying"] == "blocked"]
    if blocked:
        lines.append(f"• New buys blocked for: {escape(', '.join(blocked))}")

    if weekly_performance and weekly_performance["total_trades"] < 5:
        lines.append(
            f"• Only {weekly_performance['total_trades']} closed trades; too few to judge win rate"
        )
    elif weekly_performance and weekly_performance["win_rate"] > 50:
        lines.append(f"• Trading win rate above 50% ({weekly_performance['win_rate']:.1f}%)")
    elif weekly_performance:
        lines.append(f"• Trading win rate below 50% ({weekly_performance['win_rate']:.1f}%)")

    lines.append("")
    lines.append(f"<i>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>")

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