"""Best-effort Telegram notifications for trading and retraining workflows."""

import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


logger = logging.getLogger("multi_asset_trading_bot.telegram")
TELEGRAM_API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT_SECONDS = 10


def send_telegram_message(message: str, parse_mode: str = "HTML") -> bool:
    """Send a Telegram message; return False without raising on any delivery issue."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        logger.info("Telegram notification skipped: credentials are not configured")
        return False
    if not message:
        logger.warning("Telegram notification skipped: message is empty")
        return False

    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
    ).encode("utf-8")
    request = Request(
        f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        if response_data.get("ok") is not True:
            logger.warning(
                "Telegram API rejected notification description=%s",
                response_data.get("description", "unknown error"),
            )
            return False
        return True
    except HTTPError as exc:
        logger.warning("Telegram HTTP request failed status=%s", exc.code)
    except (URLError, TimeoutError, OSError) as exc:
        logger.warning(
            "Telegram network request failed error_type=%s",
            type(exc).__name__,
        )
    except Exception as exc:
        logger.warning(
            "Unexpected Telegram notification error error_type=%s",
            type(exc).__name__,
        )
    return False
