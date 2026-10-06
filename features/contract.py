"""Feature names used by current regression and legacy classifier artifacts."""

STOCK_FEATURE_COLUMNS = [
    "SMA_20_50_Ratio",
    "ATR_Normalized",
    "RSI_14",
    "ROC_5",
    "SPY_ROC_5",
]
CRYPTO_FEATURE_COLUMNS = list(STOCK_FEATURE_COLUMNS)
ETH_FEATURE_COLUMNS = list(STOCK_FEATURE_COLUMNS)

LEGACY_STOCK_FEATURE_COLUMNS = [
    "RSI",
    "MACD",
    "ATR",
    "Return_1D",
    "Vol_Change",
    "RSI_lag1",
    "Return_lag1",
]
LEGACY_CRYPTO_FEATURE_COLUMNS = [
    "SMA_10",
    "SMA_50",
    "RSI",
    "MACD",
    "MACD_signal",
    "BB_high",
    "BB_low",
]

FEATURE_COLUMNS_BY_TICKER = {
    "SPY": STOCK_FEATURE_COLUMNS,
    "BTC-USD": CRYPTO_FEATURE_COLUMNS,
    "ETH-USD": ETH_FEATURE_COLUMNS,
}
