import yfinance as yf
import pandas as pd
import ta

def main():
    print("Fetching historical market data...")
    ticker = "SPY"
    
    # Download 5 years of daily price history
    df = yf.download(ticker, start="2020-01-01", end="2026-01-01", interval="1d")

    # Clean multi-index headers if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print("Engineering features...")

    # Technical Indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['Return_1D'] = df['Close'].pct_change(1)
    df['Vol_Change'] = df['Volume'].pct_change(1)

    # Lagged features (yesterday's values to avoid future leak)
    df['RSI_lag1'] = df['RSI'].shift(1)
    df['Return_lag1'] = df['Return_1D'].shift(1)

    # Target variable: 1 if tomorrow's price goes up, 0 if down
    df['Tomorrow_Return'] = df['Close'].pct_change(1).shift(-1)
    df['Target'] = (df['Tomorrow_Return'] > 0).astype(int)

    # Drop rows with NaN values created by indicators and shifting
    df.dropna(inplace=True)

    # Filter to essential ML input features + Target
    features = ['RSI', 'MACD', 'ATR', 'Return_1D', 'Vol_Change', 'RSI_lag1', 'Return_lag1', 'Target']
    final_df = df[features]

    # Save to CSV for cloud training
    output_filename = "trading_features_spy.csv"
    final_df.to_csv(output_filename)
    print(f"Success! Dataset saved locally as '{output_filename}'.")

if __name__ == "__main__":
    main()