"""Prepare the legacy SPY feature CSV through the shared feature pipeline."""

import yfinance as yf

from config import FEATURE_CSV_PATH
from features.pipeline import create_legacy_spy_dataset, normalize_market_columns


def main() -> None:
    print("Fetching historical market data...")
    response = yf.download(
        "SPY",
        start="2020-01-01",
        end="2026-01-01",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if response is None or response.empty:
        raise RuntimeError("yfinance returned no SPY history")
    history = normalize_market_columns(response)
    print("Engineering legacy features...")
    dataset = create_legacy_spy_dataset(history)
    dataset.to_csv(FEATURE_CSV_PATH)
    print(f"Success! Dataset saved locally as '{FEATURE_CSV_PATH}'.")


if __name__ == "__main__":
    main()
