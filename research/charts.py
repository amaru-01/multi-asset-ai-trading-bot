"""Draw the README charts from the CSV files the research scripts write.

Run the research scripts first, then:  python -m research.charts
Needs matplotlib, which is only used here and is not a runtime dependency:  pip install matplotlib
Reads backtest_results/ (git-ignored) and writes PNGs to docs/img/ (committed).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("backtest_results")
OUT = Path("docs/img")
ASSETS = ["SPY", "BTC", "ETH"]
NAMES = {"SPY": "SPY", "BTC": "BTC-USD", "ETH": "ETH-USD"}
STRATEGY, MARKET, GOOD, BAD, MUTED = "#1f6feb", "#8b949e", "#2da44e", "#cf222e", "#d0d7de"
SIGNIFICANT = 0.05


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=MUTED, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def equity_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, asset in zip(axes, ASSETS):
        data = pd.read_csv(RESULTS / f"equity_{asset}.csv", parse_dates=["Date"], index_col="Date")
        ax.plot(data.index, data["buy_hold"], color=MARKET, label="Buy and hold", linewidth=1.4)
        ax.plot(data.index, data["strategy"], color=STRATEGY, label="Strategy", linewidth=1.6)
        ax.set_yscale("log")
        ax.set_title(f"{NAMES[asset]}: growth of 1.0 (log scale)", fontsize=11)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        style(ax)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Walk-forward backtest: the strategy against simply holding", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "equity_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def risk_summary():
    summary = pd.read_csv(RESULTS / "summary.csv").set_index("ticker")
    tickers = [NAMES[a] for a in ASSETS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("Total return", "strategy_return", "buy_hold_return", 100, "{:+.0f}%"),
        ("Sharpe ratio", "strategy_sharpe", "buy_hold_sharpe", 1, "{:.2f}"),
        ("Worst drawdown", "strategy_max_dd", "buy_hold_max_dd", 100, "{:.0f}%"),
    ]
    width = 0.38
    for ax, (title, strat, hold, scale, fmt) in zip(axes, panels):
        xs = range(len(tickers))
        a = [summary.loc[t, strat] * scale for t in tickers]
        b = [summary.loc[t, hold] * scale for t in tickers]
        bars_s = ax.bar([x - width / 2 for x in xs], a, width, color=STRATEGY, label="Strategy")
        bars_h = ax.bar([x + width / 2 for x in xs], b, width, color=MARKET, label="Buy and hold")
        for bar, value in list(zip(bars_s, a)) + list(zip(bars_h, b)):
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom" if value >= 0 else "top", fontsize=8,
                        xytext=(0, 2 if value >= 0 else -2), textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(list(xs), tickers)
        ax.set_title(title, fontsize=11)
        style(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("Return, risk-adjusted return and drawdown", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "risk_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def yearly_excess():
    data = pd.read_csv(RESULTS / "yearly.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, ticker in zip(axes, ["BTC-USD", "ETH-USD"]):
        # yearly.csv holds both assets back to back; each starts with its first calendar year
        block = data[data["ticker"] == ticker] if "ticker" in data else data
        years = block[block["period"].str.fullmatch(r"\d{4}")]
        values = years["excess_vs_matched"] * 100
        colors = [GOOD if v > 0 else BAD for v in values]
        bars = ax.bar(years["period"], values, color=colors)
        for bar, p, v in zip(bars, years["random_p"], values):
            label = f"{v:+.0f}" + ("  *" if p < SIGNIFICANT else "")
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9,
                        xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("points vs matched buy-and-hold")
        ax.set_title(f"{ticker}: edge over a buy-and-hold with the same time in the market", fontsize=11)
        style(ax)
    fig.text(0.99, 0.005, "* beat 95% of random placements of the same trades (p < 0.05)",
             ha="right", fontsize=9, color="#57606a")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "yearly_excess.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def robustness():
    data = pd.read_csv(RESULTS / "robustness.csv")
    grid = data[data["test"] == "grid"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, ticker in zip(axes, ["BTC-USD", "ETH-USD"]):
        rows = grid[grid["ticker"] == ticker].reset_index(drop=True)
        labels = [f"from {s[:4]}, p{p}" for s, p in zip(rows["start"], rows["percentile"])]
        values = rows["excess_vs_matched"] * 100
        colors = [GOOD if p < SIGNIFICANT else (STRATEGY if v > 0 else BAD)
                  for p, v in zip(rows["random_p"], values)]
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_title(f"{ticker}: edge over matched buy-and-hold", fontsize=11)
        ax.set_xlabel("points of return")
        ax.grid(axis="x", color=MUTED, linewidth=0.6, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GOOD, STRATEGY, BAD)]
    fig.legend(handles, ["Beat random timing (p < 0.05)", "Ahead, but could be luck", "Behind"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Robustness: nine settings per asset (start date x entry threshold)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    global RESULTS, OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    RESULTS, OUT = args.results, args.out
    OUT.mkdir(parents=True, exist_ok=True)
    for draw in (equity_curves, risk_summary, yearly_excess, robustness):
        draw()
        print("wrote", draw.__name__)


if __name__ == "__main__":
    main()
