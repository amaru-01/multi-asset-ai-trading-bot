"""The buy/block decision per asset and the Model/model_status.json file."""

import json
import logging
from datetime import datetime, timezone

import numpy as np

from config import MODEL_DIR

from .settings import (
    BLOCK_BUYING_ON_FAILURE,
    MIN_BACKTEST_TRADES,
    MIN_FORWARD_TRADES,
    REQUIRED_PASS_STREAK,
    STATUS_FILE_NAME,
)

logger = logging.getLogger("multi_asset_trading_bot")

CANDIDATE_KEYS = (
    "gate_path",
    "roc_auc",
    "accuracy",
    "majority_baseline",
    "backtest_trades",
    "backtest_mean_net",
    "wf_folds",
    "wf_positive_folds",
    "wf_mean_auc",
    "decision_threshold",
    "entry_bar",
    "prediction_quantiles",
)


def _base_status(
    deployed: bool,
    metrics: dict,
    forward: dict | None,
    challenger: dict | None,
    tracking_since: str | None,
) -> dict:
    """The record every decision starts from: candidate summary, dates, forward result."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    candidate = {key: metrics.get(key) for key in CANDIDATE_KEYS}
    status = {
        "checked_at": now,
        "candidate": candidate,
        "tracking_since": tracking_since or now[:10],
        "deployed_candidate": bool(deployed),
    }
    if challenger:
        status["classifier_challenger"] = challenger
    if forward and forward.get("evaluated"):
        status["forward"] = {
            "since": status["tracking_since"],
            "rows": forward["rows"],
            "trades": forward["backtest"]["trades"],
            "mean_net": forward["backtest"]["mean_net"],
            "win_rate": forward["backtest"]["win_rate"],
        }
    return status


def _apply_model_evidence(
    status: dict,
    deployed: bool,
    incumbent: dict | None,
    previous: dict,
) -> tuple[bool, bool]:
    """(2) Judge the candidate or the live model. Returns (evidence_pass, fail_open)."""
    detail = ""
    evidence_pass = False
    fail_open = False
    if deployed:
        status.update(
            live_model="3-Day Regressor (retrained)",
            buying="allowed",
            reason="candidate passed the holdout, walk-forward and net-of-cost gates",
        )
        evidence_pass = True
    elif incumbent is None or not incumbent.get("evaluated"):
        why = (incumbent or {}).get("reason", "not evaluated")
        fail_open = True
        status.update(
            live_model="existing model file (not scored)",
            buying="allowed",
            reason=f"candidate rejected; existing model could not be scored ({why})",
        )
    else:
        backtest = incumbent["backtest"]
        model_type = incumbent["model_type"]
        if backtest["trades"] and backtest["mean_net"] == backtest["mean_net"]:
            detail = (
                f"{backtest['trades']} trades, mean net {backtest['mean_net']:+.2%} per trade"
            )
            if backtest["trades"] < MIN_BACKTEST_TRADES:
                detail += f"; fewer than the {MIN_BACKTEST_TRADES} needed to trust it"
        else:
            detail = f"only {backtest['trades']} trades (too few to judge)"
        status["incumbent"] = {
            "model_type": model_type,
            "threshold": incumbent["threshold"],
            "entry_bar": incumbent["entry_bar"],
            "rows": incumbent["rows"],
            "backtest": backtest,
        }
        if backtest["passes"]:
            status.update(
                live_model=f"{model_type} (existing, unverified)",
                buying="allowed",
                reason=(
                    f"candidate rejected; existing {model_type} is profitable on the holdout "
                    f"({detail}) but may have trained on that data"
                ),
            )
            # A model deployed last week was trained on this very window, so its
            # in-sample pass proves nothing new.
            evidence_pass = not previous.get("deployed_candidate", False)
        else:
            status.update(
                live_model=f"{model_type} (existing)",
                buying="blocked" if BLOCK_BUYING_ON_FAILURE else "allowed",
                reason=(
                    f"no model passed: candidate rejected and existing {model_type} is not "
                    f"proven by the net-of-cost back-test ({detail})"
                ),
            )
    return evidence_pass, fail_open


def _apply_forward_test(status: dict, evidence_pass: bool) -> tuple[bool, bool]:
    """(1) A forward test with enough trades is real out-of-sample evidence.

    Returns (forward_decided, evidence_pass).
    """
    fwd = status.get("forward")
    forward_decided = False
    if fwd and fwd["trades"] >= MIN_FORWARD_TRADES and fwd["mean_net"] == fwd["mean_net"]:
        text = (
            f"forward test since {fwd['since']}: {fwd['trades']} trades, "
            f"mean net {fwd['mean_net']:+.2%} per trade"
        )
        forward_decided = True
        if fwd["mean_net"] > 0:
            status.update(buying="allowed", reason=f"{text} (out of sample, positive)")
            evidence_pass = True
        else:
            status.update(
                buying="blocked" if BLOCK_BUYING_ON_FAILURE else "allowed",
                reason=f"{text} (out of sample, losing)",
            )
            evidence_pass = False
    return forward_decided, evidence_pass


def _apply_streak_rule(
    status: dict,
    previous: dict,
    evidence_pass: bool,
    fail_open: bool,
    forward_decided: bool,
) -> None:
    """(3) Streak rule for assets that were blocked before this run."""
    streak = int(previous.get("pass_streak", 0)) + 1 if evidence_pass else 0
    status["pass_streak"] = streak
    was_blocked = previous.get("buying") == "blocked"
    if (
        was_blocked
        and status["buying"] == "allowed"
        and not fail_open
        and not forward_decided
        and streak < REQUIRED_PASS_STREAK
    ):
        status.update(
            buying="blocked",
            reason=(
                f"pass streak {streak}/{REQUIRED_PASS_STREAK}; buying reopens after "
                f"{REQUIRED_PASS_STREAK} consecutive weekly passes"
            ),
        )
    status["evidence_pass"] = bool(evidence_pass)


def decide_status(
    deployed: bool,
    metrics: dict,
    incumbent: dict | None,
    previous: dict | None = None,
    forward: dict | None = None,
    challenger: dict | None = None,
    tracking_since: str | None = None,
) -> dict:
    """Summarise which model is live for an asset and whether buying is allowed.

    Order of evidence: (1) a forward test with enough trades overrides everything;
    otherwise (2) a candidate that passed, or a live model that passes the in-sample
    back-test; (3) a previously blocked asset needs REQUIRED_PASS_STREAK consecutive
    passes from (2) before buying reopens.
    """
    previous = previous or {}
    status = _base_status(deployed, metrics, forward, challenger, tracking_since)
    evidence_pass, fail_open = _apply_model_evidence(status, deployed, incumbent, previous)
    forward_decided, evidence_pass = _apply_forward_test(status, evidence_pass)
    _apply_streak_rule(status, previous, evidence_pass, fail_open, forward_decided)
    return status


def json_safe(value):
    """Convert numpy types and NaN/inf to plain JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def read_previous_assets() -> dict:
    """Last run's per-asset status (empty when the file is missing or unreadable)."""
    try:
        data = json.loads((MODEL_DIR / STATUS_FILE_NAME).read_text(encoding="utf-8"))
        return data.get("assets", {}) or {}
    except Exception:
        return {}


def tracking_start(previous: dict) -> str:
    """Date from which forward results count: the last run, or today if there is none.

    The live model was last saved before the previous run, so every completed bar after
    that date is unseen by it.
    """
    for key in ("tracking_since", "checked_at"):
        value = str(previous.get(key) or "")[:10]
        if len(value) == 10:
            return value
    return datetime.now(timezone.utc).date().isoformat()


def write_model_status(statuses: dict) -> bool:
    """Merge this run's per-asset status into Model/model_status.json."""
    path = MODEL_DIR / STATUS_FILE_NAME
    try:
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("assets", {})
        except Exception:
            existing = {}
        existing.update(statuses)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "assets": existing,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("Model status written path=%s assets=%s", path, sorted(statuses))
        return True
    except Exception:
        logger.exception("Could not write model status file")
        return False
