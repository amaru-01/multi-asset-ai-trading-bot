"""Year-by-year breakdown of the walk-forward strategy.

The robustness check showed a timing edge for BTC from 2019 and 2021 starts but not from
2023. That could mean the edge faded, or that it came from a few special periods such as
dodging the 2022 crash. This script splits the same walk-forward result by calendar year
and compares each year with buy-and-hold scaled to the strategy's time in the market and
with random placement of the same trades.

Run:  python -m research.yearly
      python -m research.yearly --assets BTC-USD --split-year 2023 --slippage-bps 10
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import ASSETS, PROJECT_ROOT
from research.backtest import ROUND_TRIP_FEE, download, simulate, walk_forward_predictions
from research.robustness import holding_periods, prepare_frame, random_timing_returns

logger = logging.getLogger("yearly")


def period_row(
    label: str,
    mask: np.ndarray,
    strategy_returns: np.ndarray,
    buy_hold_returns: np.ndarray,
    position: np.ndarray,
    entries: np.ndarray,
    one_side_cost: float,
    trials: int,
    rng: np.random.Generator,
) -> dict:
    """Strategy versus the fair comparisons over the days selected by `mask`."""
    exposure = float(np.mean(position[mask]))
    strategy = float(np.prod(1.0 + strategy_returns[mask]) - 1.0)
    buy_hold = float(np.prod(1.0 + buy_hold_returns[mask]) - 1.0)
    matched = float(np.prod(1.0 + buy_hold_returns[mask] * exposure) - 1.0)
    durations = holding_periods(position[mask])
    p_value = float("nan")
    if durations:
        draws = random_timing_returns(
            buy_hold_returns[mask], durations, one_side_cost, trials, rng
        )
        if not np.isnan(draws).all():
            p_value = float(np.mean(draws >= strategy))
    return {
        "period": label,
        "days": int(mask.sum()),
        "exposure": exposure,
        "trades": int(entries[mask].sum()),
        "strategy": strategy,
        "matched_bh": matched,
        "buy_hold": buy_hold,
        "excess_vs_matched": strategy - matched,
        "random_p": p_value,
    }


def yearly_rows(
    index: pd.DatetimeIndex,
    strategy_returns: np.ndarray,
    buy_hold_returns: np.ndarray,
    position: np.ndarray,
    one_side_cost: float,
    split_year: int,
    trials: int,
    rng: np.random.Generator,
) -> list[dict]:
    """One row per calendar year, then 'before <split_year>' and '<split_year> onward'."""
    years = np.asarray(index.year)
    previous = np.concatenate(([0.0], position[:-1]))
    entries = (position > 0) & (previous == 0)
    rows = []
    for year in sorted(set(years.tolist())):
        rows.append(
            period_row(
                str(year), years == year, strategy_returns, buy_hold_returns, position,
                entries, one_side_cost, trials, rng,
            )
        )
    for label, mask in (
        (f"before {split_year}", years < split_year),
        (f"{split_year} onward", years >= split_year),
    ):
        if mask.any():
            rows.append(
                period_row(
                    label, mask, strategy_returns, buy_hold_returns, position, entries,
                    one_side_cost, trials, rng,
                )
            )
    return rows


def run_asset(asset: dict, spy_close: pd.Series, args: argparse.Namespace, rng) -> pd.DataFrame:
    frame = prepare_frame(asset, spy_close)
    first = max(int(frame.index.searchsorted(pd.Timestamp(args.start))), args.min_train)
    predictions, thresholds = walk_forward_predictions(
        frame, asset["feature_columns"], asset, first, args.refit_every, args.percentile
    )
    one_side = ROUND_TRIP_FEE[asset["type"]] / 2 + args.slippage_bps / 10_000
    result = simulate(
        frame["Open"].to_numpy(float), float(frame["Close"].iloc[-1]),
        predictions, thresholds, first, one_side,
    )
    rows = yearly_rows(
        frame.index[first + 1:], result["strategy_returns"], result["buy_hold_returns"],
        result["position"], one_side, args.split_year, args.random_trials, rng,
    )
    table = pd.DataFrame(rows)
    table.insert(0, "ticker", asset["ticker"])
    return table


def print_table(ticker: str, table: pd.DataFrame, args: argparse.Namespace) -> None:
    pct = lambda v: f"{v:+.0%}" if np.isfinite(v) else "n/a"  # noqa: E731
    print(f"\n{ticker}  (walk-forward from {args.start}, threshold p{args.percentile:g}, "
          f"slippage {args.slippage_bps:g} bps/side)")
    print(f"{'Period':<14}{'Days':>5}{'In mkt':>8}{'Trades':>7}{'Strategy':>10}"
          f"{'Matched':>9}{'Excess':>8}{'B&H':>8}{'Random p':>10}")
    for _, r in table.iterrows():
        gap = "\n" if str(r["period"]).startswith("before") else ""
        p_text = f"{r['random_p']:.3f}" if np.isfinite(r["random_p"]) else "n/a"
        print(f"{gap}{r['period']:<14}{r['days']:>5}{r['exposure']:>8.0%}{r['trades']:>7}"
              f"{pct(r['strategy']):>10}{pct(r['matched_bh']):>9}"
              f"{pct(r['excess_vs_matched']):>8}{pct(r['buy_hold']):>8}{p_text:>10}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--assets", nargs="*", default=["BTC-USD", "ETH-USD"])
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--split-year", type=int, default=2023)
    parser.add_argument("--percentile", type=float, default=65)
    parser.add_argument("--refit-every", type=int, default=63)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
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
    tables = []
    for asset in selected:
        try:
            table = run_asset(asset, spy_close, args, rng)
        except Exception:
            logger.exception("Yearly breakdown failed for %s", asset["ticker"])
            continue
        tables.append(table)
        print_table(asset["ticker"], table, args)
    if not tables:
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(out_dir / "yearly.csv", index=False)
    print(f"\nSaved to {out_dir / 'yearly.csv'}")
    print("Matched = buy-and-hold scaled to the strategy's time in the market. Random p = share "
          "of random placements of the same trades that did at least as well. One row per "
          "year is a small sample: read the pattern, not any single year.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
