"""Central configuration for assets, models, paths, and environment names."""

from pathlib import Path

from features.contract import (
    CRYPTO_FEATURE_COLUMNS,
    ETH_FEATURE_COLUMNS,
    LEGACY_CRYPTO_FEATURE_COLUMNS,
    LEGACY_STOCK_FEATURE_COLUMNS,
    STOCK_FEATURE_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "Model"
OUTPUT_MODEL_DIR = PROJECT_ROOT / "Output_model"
FEATURE_CSV_PATH = PROJECT_ROOT / "trading_features_spy.csv"

ALPACA_API_KEY_ENV = "ALPACA_API_KEY"
ALPACA_SECRET_KEY_ENV = "ALPACA_SECRET_KEY"
TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
TELEGRAM_INSIGHTS_BOT_TOKEN_ENV = "TELEGRAM_INSIGHTS_BOT_TOKEN"
TELEGRAM_INSIGHTS_CHAT_ID_ENV = "TELEGRAM_INSIGHTS_CHAT_ID"

FORWARD_HORIZON_DAYS = 3
REGRESSION_RETURN_THRESHOLD = 0.005
LEGACY_CLASSIFIER_THRESHOLD = 0.55
MIN_OUT_OF_SAMPLE_ACCURACY = 0.515
MIN_OUT_OF_SAMPLE_ROC_AUC = 0.52
MIN_ACCURACY_OVER_BASELINE = 0.015
STRONG_RANKING_ROC_AUC = 0.54
MAX_BASELINE_ACCURACY_DEFICIT = 0.02
TEST_FRACTION = 0.2
MIN_TRAINING_ROWS = 250
RANDOM_STATE = 42

# "buying_enabled" decides whether the bot may OPEN new positions in an asset. It never
# stops the bot from managing a position it already holds (stop-loss, HOLD cleanup,
# bracket exits). Based on the walk-forward backtest and robustness check, only BTC-USD
# showed any timing edge: SPY almost never trades, and ETH-USD lost to exposure-matched
# buy-and-hold in 8 of 9 settings. Override at run time with the BUY_ASSETS variable.
ASSETS = [
    {
        "ticker": "SPY",
        "symbol": "SPY",
        "type": "stock",
        "buying_enabled": False,
        "alpaca_symbol": "SPY",
        "model_path": MODEL_DIR / "xgb_trading_model.json",
        "feature_columns": list(STOCK_FEATURE_COLUMNS),
        "model_parameters": {
            "max_depth": 2,
            "learning_rate": 0.008,
            "n_estimators": 90,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
        },
    },
    {
        "ticker": "BTC-USD",
        "symbol": "BTC",
        "type": "crypto",
        "buying_enabled": True,
        "alpaca_symbol": "BTC/USD",
        "model_path": MODEL_DIR / "xgb_btc_model.json",
        "feature_columns": list(CRYPTO_FEATURE_COLUMNS),
        "model_parameters": {
            "max_depth": 3,
            "learning_rate": 0.015,
            "n_estimators": 120,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
        },
    },
    {
        "ticker": "ETH-USD",
        "symbol": "ETH",
        "type": "crypto",
        "buying_enabled": False,
        "alpaca_symbol": "ETH/USD",
        "model_path": MODEL_DIR / "xgb_eth_model.json",
        "feature_columns": list(ETH_FEATURE_COLUMNS),
        "model_parameters": {
            "max_depth": 3,
            "learning_rate": 0.015,
            "n_estimators": 120,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
        },
    },
]

LEGACY_CLASSIFIER_CONTRACTS = (
    LEGACY_STOCK_FEATURE_COLUMNS,
    LEGACY_CRYPTO_FEATURE_COLUMNS,
)
