# Multi-Asset AI Trading Bot

<p align="center">
	<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
	<img src="https://img.shields.io/badge/Model-XGBoost-EC6B23" alt="XGBoost model">
	<img src="https://img.shields.io/badge/Broker-Alpaca%20Paper-5CB85C" alt="Alpaca paper trading">
	<img src="https://img.shields.io/badge/Data-Yahoo%20Finance-6001D2" alt="Yahoo Finance data">
</p>

A Python research and paper-trading system for SPY, Bitcoin, and Ethereum. It engineers shared market features, trains per-asset XGBoost regressors to estimate three-day forward returns, validates candidates on chronological holdouts, and connects eligible signals to an Alpaca **paper-trading** account.

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

1. **Engineer features:** `features/pipeline.py` normalizes daily OHLCV data and calculates the shared regression and legacy feature contracts.
2. **Retrain:** `python retrain.py` downloads SPY, BTC-USD, and ETH-USD, selects a prediction cutoff using a training-only chronological validation slice, evaluates untouched holdouts, and exports each asset immediately when it passes Path A or Path B.
3. **Execute:** `python run_execution.py` evaluates all configured assets in Alpaca paper mode. Use `python run_execution.py --symbol BTC-USD` to run only one asset.
4. **Notify:** Telegram BUY, cleanup, and retraining reports are sent when `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are configured. Notification failures do not interrupt trading or retraining.

## Project Files

| Path | Purpose |
| --- | --- |
| `config.py` | Asset definitions, artifact paths, feature references, model settings, and environment variable names. |
| `features/contract.py` | Current five-feature regression schema and legacy classifier schemas. |
| `features/pipeline.py` | Shared OHLCV normalization, regression feature engineering, and legacy dataset generation. |
| `models/base_model.py` | Common fit, signal, and artifact interface. |
| `models/regressor.py` | Three-day return regressor and persisted per-model decision threshold. |
| `models/classifier.py` | Compatibility wrapper for legacy classifier artifacts. |
| `retrain.py` | Per-asset training, holdout gates, percentile cutoff selection, and Telegram report. |
| `run_execution.py` | Symbol-scoped Alpaca paper execution and Telegram trade/cleanup reports. |
| `telegram_notifier.py` | Best-effort Telegram Bot API helper using `urllib`. |
| `fetch_data.py`, `prepare_dataset.py` | Legacy SPY one-day CSV preparation through the shared feature pipeline. |
| `Model/*.json` | Native XGBoost model artifacts, updated per asset when its gate passes. |
| `Output_model/` | Archived Kaggle charts and evaluation outputs. |

## Features and Target

The current regression contract uses the same five ordered inputs for each asset: `SMA_20_50_Ratio`, `ATR_Normalized`, `RSI_14`, `ROC_5`, and `SPY_ROC_5`. The shared pipeline computes the benchmark feature from SPY closes and uses each asset's own OHLCV for the remaining indicators.

The model target is the three-day forward return, `(Close[t+3] - Close[t]) / Close[t]`. Holdout direction is positive when that return exceeds `0.5%`; predicted returns are converted to signals using the cutoff stored in each model's JSON attributes. Rolling-window rows and the final three unlabeled rows are excluded from training.

Legacy classifier artifacts remain loadable with their historical feature contracts until replaced by a passing regressor artifact.

## Tools

- **Python**, **pandas** — data preparation and tabular features
- **yfinance** — historical and recent market data
- **ta** — technical indicators
- **XGBoost**, **scikit-learn** — regression/classification models and native JSON serialization
- **Telegram Bot API** — optional HTML notifications
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
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

Keep `.env` private and do not commit credentials. Store the same variable names as GitHub Actions secrets for scheduled runs. The execution client is initialized with `paper=True`.

## Run

Run per-asset chronological retraining and holdout evaluation:

```powershell
python retrain.py
```

Each artifact is updated immediately when that asset passes either gate: Path A requires accuracy more than 1.5 percentage points above its holdout majority baseline and ROC-AUC above 0.52; Path B requires ROC-AUC at least 0.54 and accuracy no worse than 2 points below the baseline. A failed asset retains its existing artifact. The command returns nonzero if any asset fails or cannot be evaluated.

Run paper execution for all assets or select one:

```powershell
python run_execution.py
python run_execution.py --symbol BTC-USD
```

Regressor BUY signals use the artifact's persisted return cutoff. Crypto orders use USD notional; SPY orders use whole shares and ATR-based bracket exits. Legacy classifiers continue through the compatibility wrapper. Live prices are estimates and fills may differ because of market movement and slippage.

On a HOLD signal, execution cancels pending symbol orders and closes any open position. Telegram notifications are best effort and never block order handling. Review account settings and signals before running; execution uses Alpaca paper mode.

## Limitations

- Historical Kaggle training notebooks and charts are archival; current models are trained by `retrain.py`.
- Model quality gates can fail because market returns are noisy. In that case the previous model remains active for that asset.
- Reported backtest performance is limited to the supplied screenshot; fees, slippage, and other risk statistics are not documented in the repository.
- Regression outputs and percentile cutoffs are not guarantees of future returns.

## Kaggle Model Prediction Curves

The following charts were generated in the Kaggle notebook on **October 4, 2026**.

### SPY

![SPY out-of-sample strategy backtest curve](Output_model/Equirt_curve_output.png)

### Bitcoin (BTC)

![Bitcoin price and technical indicator curve](Output_model/Visual_BTC_Curve.png)

### Ethereum (ETH)

![Ethereum price and technical indicator curve](Output_model/Visual_ETH_Curve.png)
