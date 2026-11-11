"""Alert on Telegram when the scheduled trading runs stop succeeding.

If cron-job.org stops calling GitHub, or every run starts failing, the bot silently
stops managing positions. This script looks at the workflow history through the GitHub
API and sends a Telegram message when there has been no successful run for too long.

Standard library only, so the watchdog workflow needs no ``pip install``.

Run:  python -m monitoring.stale_runs   (reads its settings from environment variables)

  GITHUB_TOKEN, GITHUB_REPOSITORY  provided by GitHub Actions
  STALE_TRADING_HOURS              alert after this many hours without a success (default 6)
  WATCH_RISK_CHECK=true            also watch risk_check.yml
  STALE_RISK_MINUTES               ... alert after this many minutes (default 60)
  WATCH_BRANCH                     branch whose runs count (default main)
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from html import escape
from urllib.request import Request, urlopen

from notifications.telegram import send_telegram_message

logger = logging.getLogger("monitoring")

API_BASE = "https://api.github.com"
REQUEST_TIMEOUT_SECONDS = 20
# The watchdog workflow runs hourly. An alert fires in the first hour after the limit
# is crossed and again every REPEAT_MINUTES while the problem lasts, not every hour.
WATCH_INTERVAL_MINUTES = 60
REPEAT_MINUTES = 360
DEFAULT_TRADING_HOURS = 6.0
DEFAULT_RISK_MINUTES = 60.0


def parse_time(text: str) -> datetime:
    """Parse a GitHub timestamp such as 2026-11-11T08:00:00Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def age_minutes(then: datetime, now: datetime) -> float:
    return (now - then).total_seconds() / 60


def should_alert(
    age: float,
    threshold: float,
    repeat: float = REPEAT_MINUTES,
    window: float = WATCH_INTERVAL_MINUTES,
) -> bool:
    """True in the first `window` minutes after `threshold`, then again every `repeat`."""
    if age < threshold:
        return False
    return (age - threshold) % repeat < window


def format_age(minutes: float) -> str:
    hours, mins = divmod(int(minutes), 60)
    return f"{hours}h {mins:02d}m" if hours else f"{mins}m"


def api_get(path: str, token: str) -> dict:
    request = Request(
        API_BASE + path,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "trading-bot-watchdog",
        },
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def latest_run(repo: str, workflow: str, token: str, branch: str, success_only: bool):
    """Most recent run of a workflow (optionally only successful ones), or None."""
    query = f"?per_page=1&branch={branch}" + ("&status=success" if success_only else "")
    data = api_get(f"/repos/{repo}/actions/workflows/{workflow}/runs{query}", token)
    runs = data.get("workflow_runs") or []
    return runs[0] if runs else None


def describe_latest(run) -> str:
    if not run:
        return "No runs found at all."
    outcome = run.get("conclusion") or run.get("status") or "unknown"
    when = run.get("updated_at") or run.get("created_at") or "?"
    url = run.get("html_url")
    text = f"Latest run: {escape(str(outcome))} at {escape(str(when))}"
    return f"{text}\n{escape(url)}" if url else text


def check_workflow(
    label: str,
    workflow: str,
    threshold_minutes: float,
    repo: str,
    token: str,
    branch: str,
    now: datetime,
    fetch=latest_run,
) -> str | None:
    """Return an alert message if `workflow` has gone too long without a success."""
    success = fetch(repo, workflow, token, branch, True)
    latest = fetch(repo, workflow, token, branch, False)
    if success is None:
        # Never succeeded on this branch. Measure from the latest run if there is one;
        # with no runs at all, alert only in one hour of each repeat period.
        if latest is None:
            if now.hour % (REPEAT_MINUTES // 60) != 0:
                return None
            return (
                f"<b>BOT WATCHDOG</b>\nNo '{escape(workflow)}' runs exist on "
                f"{escape(branch)}. Is the scheduler pointed at this repository?"
            )
        age = age_minutes(parse_time(latest["created_at"]), now)
        last_success_text = "never"
    else:
        age = age_minutes(parse_time(success["updated_at"]), now)
        last_success_text = success["updated_at"]
    if not should_alert(age, threshold_minutes):
        return None
    return (
        f"<b>BOT WATCHDOG: {escape(label.upper())} RUNS STOPPED</b>\n"
        f"No successful '{escape(workflow)}' run for {format_age(age)} "
        f"(limit {format_age(threshold_minutes)}).\n"
        f"Last success: {escape(last_success_text)}\n"
        f"{describe_latest(latest)}\n"
        "Check cron-job.org and the Actions tab. Stop-losses and exits may not be "
        "running."
    )


def _number(name: str, default: float) -> float:
    """A positive number from the environment; blank or invalid values use the default."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not value > 0 or value == float("inf"):
        logger.warning("Ignoring invalid %s=%r", name, raw)
        return default
    return value


def main(now: datetime | None = None, fetch=latest_run) -> int:
    """Check every watched workflow. Returns 1 if a check or an alert delivery failed."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.getenv("GITHUB_TOKEN", "")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not token or not repo:
        logger.error("GITHUB_TOKEN and GITHUB_REPOSITORY must be set")
        return 1
    now = now or datetime.now(timezone.utc)
    branch = os.getenv("WATCH_BRANCH", "").strip() or "main"

    checks = [
        ("trading", "trading_bot.yml",
         _number("STALE_TRADING_HOURS", DEFAULT_TRADING_HOURS) * 60),
    ]
    if os.getenv("WATCH_RISK_CHECK", "").strip().lower() == "true":
        checks.append(
            ("risk check", "risk_check.yml",
             _number("STALE_RISK_MINUTES", DEFAULT_RISK_MINUTES))
        )

    failed = False
    for label, workflow, threshold in checks:
        try:
            message = check_workflow(
                label, workflow, threshold, repo, token, branch, now, fetch
            )
        except Exception:
            failed = True
            logger.exception("Could not check %s", workflow)
            continue
        if message is None:
            logger.info("%s is running normally", workflow)
            continue
        logger.warning("Stale: %s", workflow)
        if not send_telegram_message(message):
            # The helper is best-effort and silent. An undelivered alert must be loud.
            failed = True
            logger.error("Alert for %s could not be delivered to Telegram", workflow)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
