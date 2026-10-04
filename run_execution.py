import os
import joblib
import pandas as pd
import yfinance as yf
import ta
from dotenv import load_dotenv

# Alpaca SDK imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 1. Load environment variables
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize Alpaca Paper Trading Client
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def get_latest_features(ticker="SPY"):
    """Fetch recent price data and compute model features."""
    df = yf.download(ticker, period="60d", interval="1d")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Compute technical indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['Return_1D'] = df['Close'].pct_change(1)
    df['Vol_Change'] = df['Volume'].pct_change(1)

    df['RSI_lag1'] = df['RSI'].shift(1)
    df['Return_lag1'] = df['Return_1D'].shift(1)

    df.dropna(inplace=True)

    feature_cols = ['RSI', 'MACD', 'ATR', 'Return_1D', 'Vol_Change', 'RSI_lag1', 'Return_lag1']
    latest_row = df[feature_cols].iloc[[-1]]
    
    return latest_row, df['Close'].iloc[-1]

def execute_trade(symbol="SPY", allocation_dollars=1000):
    """Evaluates AI model prediction and sends paper trading orders to Alpaca."""
    # Check account status
    account = trading_client.get_account()
    print(f"Alpaca Paper Equity: ${float(account.equity):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")

    # Load trained model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model", "xgb_trading_model.pkl")
    model = joblib.load(model_path)
    latest_features, current_price = get_latest_features(symbol)

    # Make prediction
    prediction = model.predict(latest_features)[0]
    probability = model.predict_proba(latest_features)[0][1]

    print("\n=== EXECUTION ENGINE REPORT ===")
    print(f"Target Symbol: {symbol}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Model Directional Probability (UP): {probability:.2%}")

    # Calculate position size (fractional shares)
    shares_to_buy = round(allocation_dollars / current_price, 2)

    # Trade logic
    if prediction == 1:
        print(f"\n[SIGNAL: BUY] Attempting to buy ~{shares_to_buy} shares (${allocation_dollars})...")
        
        # Prepare Market Buy Order
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=shares_to_buy,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        # Submit Order
        order = trading_client.submit_order(order_data)
        print(f"SUCCESS: Order submitted! Order ID: {order.id}")

    else:
        print("\n[SIGNAL: HOLD / EXIT] Model predicts flat or down move.")
        # Check if we hold an existing position and close it
        try:
            position = trading_client.get_open_position(symbol)
            print(f"Closing existing position of {position.qty} shares...")
            trading_client.close_position(symbol)
            print("SUCCESS: Position closed.")
        except Exception:
            print("No existing position to close. Staying in cash.")

if __name__ == "__main__":
    execute_trade(symbol="SPY", allocation_dollars=1000)