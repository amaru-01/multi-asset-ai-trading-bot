"""The Telegram retraining report."""

from html import escape

import numpy as np

from config import ASSETS
from notifications.telegram import send_telegram_message

from .settings import MIN_FORWARD_TRADES


def send_retraining_report(
    results: list[dict],
    updated_assets: list[str],
    statuses: dict,
) -> None:
    """Send per-asset validation metrics, live-model status and buying state to Telegram."""
    latest_by_ticker = {result["ticker"]: result for result in results}
    table_lines = [
        f"{'Ticker':<7} {'AUC':>5} {'WF':>3} {'Trd':>3} {'Net%':>5} Buy"
    ]
    detail_lines = []
    for asset in ASSETS:
        ticker = asset["ticker"]
        label = escape(ticker, quote=False)
        result = latest_by_ticker.get(ticker)
        status = statuses.get(ticker)
        if status is None:
            buying = "--"
        else:
            buying = "🔴" if status["buying"] == "blocked" else "🟢"
        if result is None or "error" in result:
            table_lines.append(
                f"{label:<7} {'--':>5} {'--':>3} {'--':>3} {'--':>5} ERR"
            )
            continue
        net = result["backtest_mean_net"]
        net_text = f"{net * 100:+.2f}" if np.isfinite(net) else "--"
        if result["passed"]:
            gate = f"Path {result['gate_path']}"
        elif result["gate_path"] == "FAIL-EVIDENCE":
            gate = "No edge"
        else:
            gate = "Fail"
        table_lines.append(
            f"{label:<7} {result['roc_auc']:>5.3f} "
            f"{result['wf_positive_folds']}/{result['wf_folds']:<1} "
            f"{result['backtest_trades']:>3} {net_text:>5} {buying}"
        )
        if status is not None:
            marker = "updated" if ticker in updated_assets else "kept"
            detail_lines.append(
                f"{label}: {escape(status['live_model'], quote=False)} ({marker}). "
                f"{escape(status['reason'][:600], quote=False)} [{gate}]"
            )
            fwd = status.get("forward")
            if fwd:
                mean = fwd["mean_net"]
                mean_text = f"{mean:+.2%}" if mean == mean else "n/a"
                detail_lines.append(
                    f"  Forward test since {fwd['since']}: {fwd['trades']} trades, "
                    f"mean net {mean_text} (counts at {MIN_FORWARD_TRADES}+ trades)"
                )
            chal = status.get("classifier_challenger")
            if chal and chal.get("evaluated"):
                detail_lines.append(
                    f"  Classifier challenger: AUC {chal['roc_auc']:.3f}, "
                    f"WF {chal['wf_positive_folds']}/{chal['wf_folds']}, "
                    f"{chal['backtest_trades']} trades, "
                    f"{'passed' if chal['passed'] else 'did not pass'} (report-only)"
                )
    message = (
        "<b>MODEL RETRAINING REPORT</b>\n"
        "<pre>" + "\n".join(table_lines) + "</pre>\n"
        "<i>WF = walk-forward folds with AUC&gt;0.5, Trd = back-test trades, "
        "Net% = mean net per trade, 🟢 buying allowed / 🔴 blocked</i>"
    )
    if detail_lines:
        message += "\n" + "\n".join(detail_lines)
    send_telegram_message(message, parse_mode="HTML")
