import yfinance as yf
import pandas as pd
import ta

# 1. Fetch historical data
print("Fetching market data...")
ticker = "SPY"
df = yf.download(ticker, start="2020-01-01", end="2026-01-01", interval="1d")

# Clean multi-index columns if returned by yfinance
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Engineer features
print("Engineering features...")
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
df['MACD'] = ta.trend.macd_diff(df['Close'])
df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
df['Return_1D'] = df['Close'].pct_change(1)
df['Vol_Change'] = df['Volume'].pct_change(1)

# Lagged features (previous period values)
df['RSI_lag1'] = df['RSI'].shift(1)
df['Return_lag1'] = df['Return_1D'].shift(1)

# 3. Create target label (1 if tomorrow's return > 0, else 0)
df['Tomorrow_Return'] = df['Close'].pct_change(1).shift(-1)
df['Target'] = (df['Tomorrow_Return'] > 0).astype(int)

# 4. Drop NaN rows from rolling indicators
df.dropna(inplace=True)

# Select final feature columns
features = ['RSI', 'MACD', 'ATR', 'Return_1D', 'Vol_Change', 'RSI_lag1', 'Return_lag1', 'Target']
final_df = df[features]

# 5. Export for Cloud Training
output_filename = "trading_features_spy.csv"
final_df.to_csv(output_filename)
print(f"Dataset successfully created and saved locally to '{output_filename}'!")