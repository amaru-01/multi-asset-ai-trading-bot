# AI Trading Bot Data Pipeline

This project prepares a machine-learning dataset for a stock trading strategy using SPY (SPDR S&P 500 ETF Trust) historical market data. It downloads daily price data, engineers technical indicators, and exports a clean CSV file suitable for model training.

## Overview

The repository focuses on the data preparation stage of an AI trading bot. It:

- fetches historical SPY market data from Yahoo Finance
- calculates momentum, trend, and volatility indicators
- creates lagged features to reduce look-ahead bias
- labels each row based on whether the next day’s return is positive
- saves the dataset as a CSV for downstream training and experimentation

## Included Files

- `fetch_data.py` — script that downloads data and creates the feature dataset
- `prepare_dataset.py` — equivalent pipeline script for preparing the same dataset
- `trading_features_spy.csv` — generated output containing the feature set and target

## What the Pipeline Computes

The dataset includes the following engineered features:

- RSI (14-day Relative Strength Index)
- MACD difference
- ATR (Average True Range)
- 1-day return
- volume change
- lagged RSI and return values
- target label: 1 if tomorrow’s return is positive, otherwise 0

This converts the raw OHLCV market data into a supervised learning problem for a binary classification model.

## Requirements

Install the required Python packages:

```bash
pip install yfinance pandas ta
```

Recommended environment:

- Python 3.9+
- pandas
- yfinance
- ta

## Usage

Run either of the data preparation scripts:

```bash
python fetch_data.py
```

or

```bash
python prepare_dataset.py
```

Both scripts will:

1. download SPY data from 2020-01-01 to 2026-01-01
2. compute technical indicators
3. clean invalid rows
4. generate `trading_features_spy.csv`

## Output Dataset

The exported CSV contains the final feature columns:

- RSI
- MACD
- ATR
- Return_1D
- Vol_Change
- RSI_lag1
- Return_lag1
- Target

The target variable is defined as:

- `1` if the next day’s return is positive
- `0` if the next day’s return is negative or zero

## Notes

- This project currently focuses on data preparation rather than full model training or live trading execution.
- The scripts are intentionally simple and easy to extend for experimentation with different feature sets, models, or time horizons.
- The generated dataset is meant to be used as input for downstream ML training in a larger AI trading system.

## Suggested Next Steps

- train a classifier on the generated dataset
- evaluate model performance with cross-validation
- tune technical indicators and lag features
- expand the pipeline to support additional tickers or multi-asset strategies

## License

This project does not currently include a specific license file. If you plan to share or publish it, add a license that matches your intended usage.
