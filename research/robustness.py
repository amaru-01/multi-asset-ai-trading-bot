"""Robustness check: does the walk-forward result survive different settings and luck?

Builds on backtest.py. For each asset it:
  1. Re-runs the walk-forward backtest across several start dates and threshold
     percentiles (predictions are cached, so slippage sweeps are cheap).
  2. Sweeps slippage and the refit frequency at the base configuration.
  3. Runs a random-timing test: the strategy's own number of trades and holding
     periods are dropped on random days, with identical costs. The p-value is the share
     of random runs that earned at least as much as the strategy. A small p-value means
     the timing is hard to explain by luck; it is NOT proof of a future edge.

The headline comparison is against buy-and-hold scaled to the strategy's time in the
market, because a strategy that is only invested 30% of the time should be judged
against 30% of the market, not 100%.

Usage:
    python -m research.robustness                    # BTC-USD and ETH-USD (SPY barely trades)
    python -m research.robustness --assets SPY
    python -m research.robustness --random-trials 5000
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.backtest import (
    ANNUALIZATION,
    ROUND_TRIP_FEE,
    download,
    metrics,
    simulate,
    walk_forward_predictions,
)
from config import ASSETS, FORWARD_HORIZON_DAYS, PROJECT_ROOT
from features.pipeline import add_regression_features

logger = logging.getLogger("robustness")

DEFAULT_STARTS = ("2019-01-01", "2021-01-01", "2023-01-01")
DEFAULT_PERCENTILES = (55, 65, 75)
DEFAULT_SLIPPAGE_BPS = (5, 10, 20)
BASE_START, BASE_PERCENTILE, BASE_REFIT, BASE_SLIPPAGE = "2019-01-01", 65, 63, 5


def prepare_frame(asset: dict, spy_close: pd.Series) -> pd.DataFrame:
    frame = add_regression_features(download(asset["ticker"]), spy_close)
    close = frame["Close"]
    frame["Target_Return"] = (close.shift(-FORWARD_HORIZON_DAYS) - close) / close
    return frame.dropna(subset=asset["feature_columns"])


def random_timing_returns(
    interval_returns: np.ndarray,
    durations: list[int],
    one_side_cost: float,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Total returns of `trials` random placements of the same holding periods.

    The same number of trades and the same holding lengths are placed at random,
    non-overlapping positions inside the window, paying the same per-trade costs.
    """
    length = len(interval_returns)
    k = len(durations)
    free = length - sum(durations)
    if k == 0 or free < 0:
        return np.full(trials, np.nan)
    totals = np.empty(trials)
    for t in range(trials):
        order = rng.permutation(durations)
        # Stars and bars: k+1 gaps (before, between, after) that sum to `free`.
        cuts = np.sort(rng.integers(0, free + 1, size=k))
        gaps = np.diff(np.concatenate(([0], cuts, [free])))
        position = np.zeros(length)
        cursor = int(gaps[0])
        for hold, gap in zip(order, gaps[1:]):
            position[cursor : cursor + hold] = 1.0
            cursor += int(hold) + int(gap)
        previous = np.concatenate(([0.0], position[:-1]))
        daily = position * interval_returns - one_side_cost * np.abs(position - previous)
        totals[t] = np.prod(1.0 + daily) - 1.0
    return totals


def holding_periods(position: np.ndarray) -> list[int]:
    """Lengths of consecutive runs of 1s in a 0/1 position array."""
    runs, count = [], 0
    for value in position:
        if value > 0:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return runs


def evaluate(frame, asset, start, percentile, refit, slippage_bps, min_train, cache):
    """Run one configuration. Predictions are cached by (start, percentile, refit)."""
    first = max(int(frame.index.searchsorted(pd.Timestamp(start))), min_train)
    key = (start, percentile, refit)
    if key not in cache:
        cache[key] = walk_forward_predictions(
            frame, asset["feature_columns"], asset, first, refit, percentile
        )
    predictions, thresholds = cache[key]
    one_side = ROUND_TRIP_FEE[asset["type"]] / 2 + slippage_bps / 10_000
    result = simulate(
        frame["Open"].to_numpy(float), float(frame["Close"].iloc[-1]),
        predictions, thresholds, first, one_side,
    )
    ppy = ANNUALIZATION[asset["type"]]
    exposure = float(np.mean(result["position"]))
    strategy = metrics(result["strategy_returns"], ppy)
    buy_hold = metrics(result["buy_hold_returns"], ppy)
    matched = metrics(result["buy_hold_returns"] * exposure, ppy)
    return {
        "result": result,
        "one_side": one_side,
        "exposure": exposure,
        "trades": len(result["trades"]),
        "strategy": strategy["total_return"],
        "buy_hold": buy_hold["total_return"],
        "matched": matched["total_return"],
        "sharpe": strategy["sharpe"],
        "bh_sharpe": buy_hold["sharpe"],
    }


def pct(value: float) -> str:
    return f"{value:+.0%}" if np.isfinite(value) else "n/a"


def run_asset(asset, spy_close, args, rng) -> list[dict]:
    frame = prepare_frame(asset, spy_close)
    cache: dict = {}
    rows: list[dict] = []
    ticker = asset["ticker"]

    def record(label, start, percentile, refit, slippage, with_random):
        out = evaluate(frame, asset, start, percentile, refit, slippage, args.min_train, cache)
        p_value = float("nan")
        if with_random and out["trades"]:
            draws = random_timing_returns(
                out["result"]["buy_hold_returns"],
                holding_periods(out["result"]["position"]),
                out["one_side"], args.random_trials, rng,
            )
            p_value = float(np.mean(draws >= out["strategy"]))
        row = {
            "ticker": ticker, "test": label, "start": start, "percentile": percentile,
            "refit": refit, "slippage_bps": slippage, "trades": out["trades"],
            "exposure": out["exposure"], "strategy": out["strategy"],
            "matched_bh": out["matched"], "buy_hold": out["buy_hold"],
            "excess_vs_matched": out["strategy"] - out["matched"],
            "sharpe": out["sharpe"], "bh_sharpe": out["bh_sharpe"], "random_p": p_value,
        }
        rows.append(row)
        logger.info(
            "%s %-9s start=%s p%-2d refit=%-3d slip=%-2d | trades %3d exp %3.0f%% | "
            "strat %s vs matched %s (B&H %s) | random p=%s",
            ticker, label, start, percentile, refit, slippage, out["trades"],
            out["exposure"] * 100, pct(out["strategy"]), pct(out["matched"]),
            pct(out["buy_hold"]), f"{p_value:.3f}" if np.isfinite(p_value) else "n/a",
        )

    for start in args.starts:
        for percentile in args.percentiles:
            record("grid", start, percentile, BASE_REFIT, BASE_SLIPPAGE, True)
    for slippage in args.slippage:
        if slippage != BASE_SLIPPAGE:
            record("slippage", BASE_START, BASE_PERCENTILE, BASE_REFIT, slippage, False)
    record("refit-21", BASE_START, BASE_PERCENTILE, 21, BASE_SLIPPAGE, True)
    return rows


def verdict(table: pd.DataFrame) -> list[str]:
    lines = []
    for ticker, group in table.groupby("ticker"):
        grid = group[group["test"] == "grid"]
        n = len(grid)
        beats = int((grid["excess_vs_matched"] > 0).sum())
        sig = int((grid["random_p"] < 0.05).sum())
        both = int(((grid["excess_vs_matched"] > 0) & (grid["random_p"] < 0.05)).sum())
        slip = group[group["test"].isin(["grid", "slippage"])]
        slip = slip[(slip["start"] == BASE_START) & (slip["percentile"] == BASE_PERCENTILE)]
        survives_slippage = bool((slip["excess_vs_matched"] > 0).all()) if len(slip) else False
        lines.append(
            f"{ticker}: beat exposure-matched buy-and-hold in {beats}/{n} configs; "
            f"beat random timing (p<0.05) in {sig}/{n}; both in {both}/{n}; "
            f"still ahead of matched at every slippage level: {survives_slippage}"
        )
    return lines


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--assets", nargs="*", default=["BTC-USD", "ETH-USD"])
    parser.add_argument("--starts", nargs="*", default=list(DEFAULT_STARTS))
    parser.add_argument("--percentiles", nargs="*", type=float, default=list(DEFAULT_PERCENTILES))
    parser.add_argument("--slippage", nargs="*", type=float, default=list(DEFAULT_SLIPPAGE_BPS))
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--random-trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "backtest_results"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    selected = [a for a in ASSETS if a["ticker"] in args.assets]
    if not selected:
        logger.error("No matching assets")
        return 1
    spy_close = download("SPY")["Close"]
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for asset in selected:
        try:
            rows.extend(run_asset(asset, spy_close, args, rng))
        except Exception:
            logger.exception("Robustness run failed for %s", asset["ticker"])
    if not rows:
        return 1
    table = pd.DataFrame(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "robustness.csv", index=False)
    print("\nROBUSTNESS SUMMARY")
    for line in verdict(table):
        print(line)
    print(f"\nFull grid saved to {out_dir / 'robustness.csv'}")
    print("A small random p-value means timing is hard to explain by luck alone; it does "
          "not guarantee future performance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
