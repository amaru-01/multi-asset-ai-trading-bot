"""Walk-forward backtest of the live trading rule versus buy-and-hold.

How it avoids flattering the strategy:
  * Models are retrained on an expanding window and only ever predict bars that come
    after the data they saw (a FORWARD_HORIZON_DAYS purge removes target overlap).
  * The decision threshold is calibrated on the last 25% of each training window,
    never on the data being predicted.
  * Signals use completed daily bars and fill at the NEXT bar's open.
  * Every position change pays the asset's fee plus slippage.
  * The strategy is compared with buy-and-hold over the identical window, and with a
    buy-and-hold scaled to the strategy's time in the market (a fairer risk match).

Simplifications (all documented in the printed notes): the strategy is fully invested
when long, with no ATR position sizing, stop-loss, cooldown or minimum-hold rule.

Usage:
    python -m research.backtest
    python -m research.backtest --start 2020-01-01 --slippage-bps 10 --assets BTC-USD ETH-USD
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import ASSETS, FORWARD_HORIZON_DAYS, PROJECT_ROOT
from features.pipeline import add_regression_features, normalize_market_columns
from models import XGBRegressorModel

logger = logging.getLogger("backtest")

# Round-trip fee as a fraction of the position (mirrors retrain.py).
ROUND_TRIP_FEE = {"crypto": 0.005, "stock": 0.0}
ANNUALIZATION = {"crypto": 365, "stock": 252}
# Live entry bar for regressors (mirrors REGRESSOR_ENTRY_MIN in run_execution.py).
ENTRY_MIN = 0.006


def download(ticker: str) -> pd.DataFrame:
    response = yf.download(
        ticker, period="max", interval="1d", auto_adjust=False,
        progress=False, threads=False,
    )
    if response is None or response.empty:
        raise ValueError(f"No data returned for {ticker}")
    return normalize_market_columns(response)


def walk_forward_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    asset: dict,
    first_test_row: int,
    refit_every: int,
    percentile: float,
    horizon: int = FORWARD_HORIZON_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return out-of-sample (prediction, threshold) arrays aligned to `frame` rows.

    Rows before `first_test_row` stay NaN. Each block of `refit_every` rows is predicted
    by a model fit only on rows whose targets were already known at the block start.
    """
    n = len(frame)
    predictions = np.full(n, np.nan)
    thresholds = np.full(n, np.nan)
    for start in range(first_test_row, n, refit_every):
        end = min(start + refit_every, n)
        train = frame.iloc[: max(0, start - horizon)].dropna(subset=["Target_Return"])
        cut = int(len(train) * 0.75)
        fit_rows = train.iloc[: max(0, cut - horizon)]
        validation = train.iloc[cut:]
        if len(fit_rows) < 100 or validation.empty:
            continue
        calibrator = XGBRegressorModel(asset["symbol"], feature_columns).fit(
            fit_rows[feature_columns], fit_rows["Target_Return"]
        )
        threshold = float(
            np.percentile(calibrator.predict_values(validation[feature_columns]), percentile)
        )
        model = XGBRegressorModel(asset["symbol"], feature_columns).fit(
            train[feature_columns], train["Target_Return"]
        )
        predictions[start:end] = model.predict_values(frame.iloc[start:end][feature_columns])
        thresholds[start:end] = threshold
    return predictions, thresholds


def simulate(
    opens: np.ndarray,
    final_close: float,
    predictions: np.ndarray,
    thresholds: np.ndarray,
    first: int,
    one_side_cost: float,
    entry_min: float = ENTRY_MIN,
) -> dict:
    """Run the entry/exit rule over bars `first..n-1` and return daily returns and trades.

    A signal computed from bar i's completed data acts at bar i+1's open. A position is
    opened when the prediction reaches max(threshold, entry_min) and closed when it falls
    below the threshold. `one_side_cost` is charged on every entry and every exit. A
    position still open at the end is marked to `final_close` without an exit cost.
    """
    n = len(opens)
    ext = np.append(opens, final_close)
    interval_return = ext[1:] / ext[:-1] - 1.0  # return from open[j] to open[j+1]
    position = np.zeros(n)
    holding = False
    entry_price = 0.0
    trades: list[float] = []
    for i in range(first, n - 1):
        prediction, threshold = predictions[i], thresholds[i]
        if np.isnan(prediction) or np.isnan(threshold):
            position[i + 1] = float(holding)
            continue
        if not holding and prediction >= max(threshold, entry_min):
            holding, entry_price = True, opens[i + 1]
        elif holding and prediction < threshold:
            holding = False
            trades.append(opens[i + 1] / entry_price - 1.0 - 2 * one_side_cost)
        position[i + 1] = float(holding)
    if holding:
        trades.append(final_close / entry_price - 1.0 - one_side_cost)

    window = slice(first + 1, n)
    pos = position[window]
    previous = np.concatenate(([0.0], pos[:-1]))
    strategy = pos * interval_return[window] - one_side_cost * np.abs(pos - previous)
    return {
        "strategy_returns": strategy,
        "buy_hold_returns": interval_return[window],
        "position": pos,
        "trades": trades,
    }


def metrics(returns: np.ndarray, periods_per_year: int) -> dict:
    equity = np.cumprod(1.0 + returns)
    total = float(equity[-1] - 1.0) if len(equity) else float("nan")
    years = len(returns) / periods_per_year
    peak = np.maximum.accumulate(equity)
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    return {
        "total_return": total,
        "cagr": float(equity[-1] ** (1 / years) - 1.0) if years > 0 and equity[-1] > 0 else float("nan"),
        "sharpe": float(np.mean(returns) / std * np.sqrt(periods_per_year)) if std > 0 else float("nan"),
        "max_drawdown": float(np.min(equity / peak - 1.0)) if len(equity) else float("nan"),
    }


def run_asset(asset: dict, spy_close: pd.Series, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    history = download(asset["ticker"])
    frame = add_regression_features(history, spy_close)
    close = frame["Close"]
    frame["Target_Return"] = (close.shift(-FORWARD_HORIZON_DAYS) - close) / close
    columns = asset["feature_columns"]
    frame = frame.dropna(subset=columns)
    first = max(int(frame.index.searchsorted(pd.Timestamp(args.start))), args.min_train)
    if first >= len(frame) - args.refit_every:
        raise ValueError(f"Not enough data after {args.start} for {asset['ticker']}")

    predictions, thresholds = walk_forward_predictions(
        frame, columns, asset, first, args.refit_every, args.percentile
    )
    fee = ROUND_TRIP_FEE[asset["type"]]
    one_side = fee / 2 + args.slippage_bps / 10_000
    result = simulate(
        frame["Open"].to_numpy(float), float(frame["Close"].iloc[-1]),
        predictions, thresholds, first, one_side,
    )
    ppy = ANNUALIZATION[asset["type"]]
    strategy = metrics(result["strategy_returns"], ppy)
    buy_hold = metrics(result["buy_hold_returns"], ppy)
    exposure = float(np.mean(result["position"]))
    matched = metrics(result["buy_hold_returns"] * exposure, ppy)
    trades = np.array(result["trades"])
    window_index = frame.index[first + 1:]
    row = {
        "ticker": asset["ticker"],
        "window": f"{window_index[0].date()} to {window_index[-1].date()}",
        "bars": len(window_index),
        "trades": len(trades),
        "win_rate": float(np.mean(trades > 0)) if len(trades) else float("nan"),
        "avg_trade_net": float(np.mean(trades)) if len(trades) else float("nan"),
        "exposure": exposure,
        "strategy_return": strategy["total_return"],
        "buy_hold_return": buy_hold["total_return"],
        "exposure_matched_bh_return": matched["total_return"],
        "strategy_sharpe": strategy["sharpe"],
        "buy_hold_sharpe": buy_hold["sharpe"],
        "strategy_max_dd": strategy["max_drawdown"],
        "buy_hold_max_dd": buy_hold["max_drawdown"],
    }
    curves = pd.DataFrame(
        {
            "strategy": np.cumprod(1.0 + result["strategy_returns"]),
            "buy_hold": np.cumprod(1.0 + result["buy_hold_returns"]),
            "position": result["position"],
        },
        index=window_index,
    )
    return row, curves


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", default="2019-01-01", help="first out-of-sample date")
    parser.add_argument("--min-train", type=int, default=500, help="minimum training rows")
    parser.add_argument("--refit-every", type=int, default=63, help="bars between refits")
    parser.add_argument("--percentile", type=float, default=65, help="threshold percentile")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="slippage per side")
    parser.add_argument("--assets", nargs="*", help="tickers to run (default: all)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "backtest_results"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    selected = [a for a in ASSETS if not args.assets or a["ticker"] in args.assets]
    if not selected:
        logger.error("No matching assets")
        return 1
    spy_close = download("SPY")["Close"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for asset in selected:
        try:
            row, curves = run_asset(asset, spy_close, args)
        except Exception:
            logger.exception("Backtest failed for %s", asset["ticker"])
            continue
        rows.append(row)
        curves.to_csv(out_dir / f"equity_{asset['symbol']}.csv")
    if not rows:
        return 1

    table = pd.DataFrame(rows).set_index("ticker")
    table.to_csv(out_dir / "summary.csv")
    pct = lambda v: f"{v:+.1%}" if np.isfinite(v) else "n/a"  # noqa: E731
    num = lambda v: f"{v:.2f}" if np.isfinite(v) else "n/a"  # noqa: E731
    print("\nWALK-FORWARD BACKTEST (net of fees and slippage; next-bar-open fills)")
    print(f"slippage {args.slippage_bps:g} bps/side, refit every {args.refit_every} bars, "
          f"threshold p{args.percentile:g}\n")
    for ticker, r in table.iterrows():
        print(f"{ticker}  [{r['window']}, {int(r['bars'])} bars]")
        print(f"  Strategy      {pct(r['strategy_return']):>9}  Sharpe {num(r['strategy_sharpe']):>5}  "
              f"MaxDD {pct(r['strategy_max_dd']):>7}")
        print(f"  Buy & hold    {pct(r['buy_hold_return']):>9}  Sharpe {num(r['buy_hold_sharpe']):>5}  "
              f"MaxDD {pct(r['buy_hold_max_dd']):>7}")
        print(f"  B&H x exposure{pct(r['exposure_matched_bh_return']):>9}  "
              f"(same {r['exposure']:.0%} time in market)")
        print(f"  Trades {int(r['trades'])}, win rate {pct(r['win_rate'])}, "
              f"avg net/trade {pct(r['avg_trade_net'])}\n")
    print("Notes: fully invested when long; no ATR sizing, stop-loss, cooldown or minimum-hold;")
    print("open position at the end is marked to the last close. Results are not predictions.")
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
