"""Tests for the stalled-run watchdog (monitoring/stale_runs.py)."""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring import stale_runs as sr  # noqa: E402

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
THRESHOLD = 360.0  # six hours, in minutes


def stamp(minutes_ago):
    return (NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(minutes_ago, conclusion="success", url="https://example.test/run/1"):
    return {
        "created_at": stamp(minutes_ago + 2), "updated_at": stamp(minutes_ago),
        "conclusion": conclusion, "status": "completed", "html_url": url,
    }


def fake_fetch(success=None, latest=None, calls=None):
    def fetch(repo, workflow, token, branch, success_only):
        if calls is not None:
            calls.append((workflow, branch, success_only))
        return success if success_only else latest
    return fetch


class TimeHelperTests(unittest.TestCase):
    def test_parse_time_and_age(self):
        then = sr.parse_time("2026-11-11T05:00:00Z")
        self.assertEqual(then, datetime(2026, 11, 11, 5, 0, tzinfo=timezone.utc))
        self.assertEqual(sr.age_minutes(then, NOW), 420.0)

    def test_format_age(self):
        self.assertEqual(sr.format_age(432), "7h 12m")
        self.assertEqual(sr.format_age(45), "45m")
        self.assertEqual(sr.format_age(60), "1h 00m")


class ShouldAlertTests(unittest.TestCase):
    def test_quiet_before_the_limit(self):
        self.assertFalse(sr.should_alert(359, THRESHOLD))
        self.assertFalse(sr.should_alert(0, THRESHOLD))

    def test_alerts_in_the_first_hour_after_the_limit(self):
        self.assertTrue(sr.should_alert(360, THRESHOLD))
        self.assertTrue(sr.should_alert(419.9, THRESHOLD))

    def test_does_not_repeat_every_hour(self):
        for age in (420, 480, 600, 700):
            self.assertFalse(sr.should_alert(age, THRESHOLD), age)

    def test_repeats_every_six_hours(self):
        self.assertTrue(sr.should_alert(725, THRESHOLD))
        self.assertTrue(sr.should_alert(1085, THRESHOLD))
        self.assertFalse(sr.should_alert(800, THRESHOLD))

    def test_works_for_short_limits(self):
        self.assertTrue(sr.should_alert(65, 60))
        self.assertFalse(sr.should_alert(130, 60))
        self.assertTrue(sr.should_alert(425, 60))  # 60 + 360 + 5


class CheckWorkflowTests(unittest.TestCase):
    def check(self, success=None, latest=None, threshold=THRESHOLD, now=NOW):
        return sr.check_workflow(
            "trading", "trading_bot.yml", threshold, "o/r", "tok", "main", now,
            fake_fetch(success, latest),
        )

    def test_recent_success_is_quiet(self):
        self.assertIsNone(self.check(success=run(90), latest=run(90)))

    def test_stale_success_alerts_with_the_details(self):
        message = self.check(
            success=run(380), latest=run(30, "failure", "https://example.test/run/9")
        )
        self.assertIn("TRADING RUNS STOPPED", message)
        self.assertIn("trading_bot.yml", message)
        self.assertIn("6h 20m", message)
        self.assertIn("limit 6h 00m", message)
        self.assertIn(f"Last success: {stamp(380)}", message)
        self.assertIn("Latest run: failure", message)
        self.assertIn("https://example.test/run/9", message)
        self.assertIn("cron-job.org", message)

    def test_stale_but_outside_the_alert_window_is_quiet(self):
        self.assertIsNone(self.check(success=run(500), latest=run(500)))

    def test_never_succeeded_measures_from_the_latest_run(self):
        message = self.check(success=None, latest=run(400, "failure"))
        self.assertIn("Last success: never", message)
        self.assertIn("Latest run: failure", message)

    def test_never_succeeded_but_a_run_is_recent_is_quiet(self):
        self.assertIsNone(self.check(success=None, latest=run(30, "failure")))

    def test_no_runs_at_all_alerts_only_in_one_hour_of_each_period(self):
        at_six = NOW.replace(hour=6)
        at_seven = NOW.replace(hour=7)
        message = self.check(success=None, latest=None, now=at_six)
        self.assertIn("No 'trading_bot.yml' runs exist", message)
        self.assertIsNone(self.check(success=None, latest=None, now=at_seven))

    def test_html_in_values_is_escaped(self):
        message = self.check(
            success=run(380), latest=run(30, "<b>x</b>", "https://example.test/?a=1&b=2")
        )
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", message)
        self.assertIn("a=1&amp;b=2", message)

    def test_in_progress_latest_run_is_described_by_its_status(self):
        latest = dict(run(5), conclusion=None, status="in_progress")
        self.assertIn("Latest run: in_progress", self.check(success=run(380), latest=latest))


class MainTests(unittest.TestCase):
    def setUp(self):
        keys = (
            "GITHUB_TOKEN", "GITHUB_REPOSITORY", "STALE_TRADING_HOURS", "WATCH_RISK_CHECK",
            "STALE_RISK_MINUTES", "WATCH_BRANCH",
        )
        env = {k: v for k, v in os.environ.items() if k not in keys}
        env.update({"GITHUB_TOKEN": "tok", "GITHUB_REPOSITORY": "o/r"})
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(sr, "send_telegram_message", return_value=True)
        self.send = patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, success=None, latest=None, calls=None):
        return sr.main(now=NOW, fetch=fake_fetch(success, latest, calls))

    def test_missing_credentials_is_an_error(self):
        del os.environ["GITHUB_TOKEN"]
        self.assertEqual(self.run_main(), 1)
        self.send.assert_not_called()

    def test_healthy_bot_sends_nothing(self):
        self.assertEqual(self.run_main(success=run(60), latest=run(60)), 0)
        self.send.assert_not_called()

    def test_stalled_bot_sends_one_alert(self):
        self.assertEqual(self.run_main(success=run(380), latest=run(30, "failure")), 0)
        self.send.assert_called_once()
        self.assertIn("RUNS STOPPED", self.send.call_args.args[0])

    def test_an_undelivered_alert_fails_the_run(self):
        self.send.return_value = False
        self.assertEqual(self.run_main(success=run(380), latest=run(30)), 1)

    def test_api_failure_fails_the_run_without_alerting(self):
        def boom(*args, **kwargs):
            raise OSError("api down")

        self.assertEqual(sr.main(now=NOW, fetch=boom), 1)
        self.send.assert_not_called()

    def test_only_the_trading_workflow_is_watched_by_default(self):
        calls = []
        self.run_main(success=run(60), latest=run(60), calls=calls)
        self.assertEqual({workflow for workflow, _, _ in calls}, {"trading_bot.yml"})
        self.assertEqual({branch for _, branch, _ in calls}, {"main"})

    def test_risk_check_can_be_watched_with_its_own_limit(self):
        os.environ["WATCH_RISK_CHECK"] = "true"
        calls = []
        # 90 minutes old: fine for trading (6h) but past the risk check's 60 minute limit
        self.assertEqual(self.run_main(success=run(90), latest=run(90), calls=calls), 0)
        self.assertEqual({w for w, _, _ in calls}, {"trading_bot.yml", "risk_check.yml"})
        self.send.assert_called_once()
        self.assertIn("RISK CHECK RUNS STOPPED", self.send.call_args.args[0])

    def test_limits_can_be_overridden(self):
        os.environ["STALE_TRADING_HOURS"] = "2"
        self.assertEqual(self.run_main(success=run(125), latest=run(125)), 0)
        self.send.assert_called_once()

    def test_invalid_limit_falls_back_to_the_default(self):
        for bad in ("banana", "-3", "0", "inf", ""):
            os.environ["STALE_TRADING_HOURS"] = bad
            self.send.reset_mock()
            self.run_main(success=run(125), latest=run(125))
            self.send.assert_not_called()  # 125 minutes is well inside the default 6 hours

    def test_branch_can_be_overridden(self):
        os.environ["WATCH_BRANCH"] = "release"
        calls = []
        self.run_main(success=run(60), latest=run(60), calls=calls)
        self.assertEqual({branch for _, branch, _ in calls}, {"release"})


class ApiTests(unittest.TestCase):
    def test_latest_run_asks_for_one_run_on_the_branch(self):
        with mock.patch.object(sr, "api_get", return_value={"workflow_runs": [{"id": 1}]}) as get:
            self.assertEqual(sr.latest_run("o/r", "trading_bot.yml", "tok", "main", True), {"id": 1})
        path = get.call_args.args[0]
        self.assertIn("/repos/o/r/actions/workflows/trading_bot.yml/runs", path)
        self.assertIn("per_page=1", path)
        self.assertIn("branch=main", path)
        self.assertIn("status=success", path)

    def test_latest_run_without_the_success_filter(self):
        with mock.patch.object(sr, "api_get", return_value={"workflow_runs": []}) as get:
            self.assertIsNone(sr.latest_run("o/r", "w.yml", "tok", "main", False))
        self.assertNotIn("status=success", get.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
