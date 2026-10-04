# SPY AI Trading Bot

<p align="center">
	<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
	<img src="https://img.shields.io/badge/Model-XGBoost-EC6B23" alt="XGBoost model">
	<img src="https://img.shields.io/badge/Broker-Alpaca%20Paper-5CB85C" alt="Alpaca paper trading">
	<img src="https://img.shields.io/badge/Data-Yahoo%20Finance-6001D2" alt="Yahoo Finance data">
</p>

A Python research project for predicting the next-day direction of SPY (SPDR S&P 500 ETF Trust). It builds technical features from historical prices, uses an XGBoost classifier trained in Kaggle, and connects predictions to an Alpaca **paper-trading** account.

> This is an educational prototype, not investment advice. Backtest results are historical and do not predict future performance. The execution script is configured for Alpaca paper trading, not live trading.

## Backtest Snapshot

The chart below is the Kaggle backtest output included with the project.

![Kaggle out-of-sample equity curve comparing the XGBoost strategy with SPY buy-and-hold](Output_model/Equirt_curve_output.png)

| Reported metric | Result |
| --- | ---: |
| Strategy total return | 4.57% |
| SPY buy-and-hold return | 20.40% |
| Annualized Sharpe ratio | 0.31 |

These figures are transcribed from the supplied output image. In that run, the strategy returned less than buy-and-hold; this result should be evaluated with additional validation, costs, and risk measures before drawing conclusions.

## Workflow

1. **Prepare data:** download SPY daily prices and calculate features with `fetch_data.py` or `prepare_dataset.py`.
2. **Train:** upload the generated CSV to Kaggle and train the XGBoost classifier there. The training notebook is not included in this repository.
3. **Predict and paper trade:** `run_execution.py` downloads recent SPY data, recreates the model features, loads the saved classifier, and sends a paper order based on its prediction.

## Project Files

| Path | Purpose |
| --- | --- |
| `fetch_data.py` | Downloads historical SPY prices, engineers features, and exports the dataset. |
| `prepare_dataset.py` | Alternate script for generating the same feature dataset. |
| `run_execution.py` | Loads the saved model and runs the Alpaca paper-trading workflow. |
| `Model/xgb_trading_model.pkl` | Serialized XGBoost classifier used for prediction. |
| `Output_model/Equirt_curve_output.png` | Equity curve and backtest metrics from Kaggle. |
| `requirements.txt` | Python package dependencies. |
| `.gitignore` | Excludes local environments, secrets, caches, and generated CSV data. |

## Features and Target

The dataset contains these model inputs:

- `RSI` — 14-day Relative Strength Index
- `MACD` — MACD difference
- `ATR` — Average True Range
- `Return_1D` — one-day close-to-close return
- `Vol_Change` — one-day volume change
- `RSI_lag1`, `Return_lag1` — previous-day indicator values

`Target` is `1` when the following day's return is positive and `0` otherwise. The scripts drop rows made incomplete by rolling indicators and shifts.

## Tools

- **Python**, **pandas** — data preparation and tabular features
- **yfinance** — historical and recent market data
- **ta** — technical indicators
- **XGBoost**, **scikit-learn**, **joblib** — classification model and serialization
- **Kaggle Notebooks** — model training and backtest output
- **Alpaca Trading API** (`alpaca-py`) — paper account and order submission
- **python-dotenv** — local API credential loading

## Setup

Use Python 3.10 or newer. From the project folder, create and activate a virtual environment, then install the packages listed in `requirements.txt`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root with Alpaca **paper** credentials:

```dotenv
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
```

Keep `.env` private and do not commit credentials. The checked-in execution client is initialized with `paper=True`.

## Run

Generate the feature CSV:

```powershell
python fetch_data.py
```

Alternatively, use `python prepare_dataset.py`. Both scripts download SPY daily data from 2020-01-01 through 2025-12-31 and write `trading_features_spy.csv` in the project root. The CSV is ignored by Git and can be uploaded to Kaggle for training.

After placing the trained model at `Model/xgb_trading_model.pkl` and configuring paper credentials, run:

```powershell
python run_execution.py
```

The execution script may submit a paper buy order or close a paper position based on the model signal. Review its behavior and verify the Alpaca account is in paper mode before running it.

## Limitations

- The Kaggle training notebook and reproducible train/test split are not included here.
- Reported backtest performance is limited to the supplied screenshot; fees, slippage, and other risk statistics are not documented in the repository.
- A classifier prediction is not a guarantee of future price direction.
