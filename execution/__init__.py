"""Order execution for the trading bot, split by responsibility.

mode          paper/live mode
broker        Alpaca client, orders and positions
market_data   yfinance downloads and model inputs
safeguards    live-trading cap, kill switch, daily-loss breaker, buy-enabled switch
risk          crypto stop-loss and the --risk-only mode
gates         every check a BUY signal must pass before a position is opened
sizing        ATR risk budget, confidence scaling and the exposure cap
orders        build and submit market orders, then report them
trading       per-asset flow: stop-loss, HOLD cleanup, gates, sizing, order

The command-line entry point stays ``python run_execution.py``.
"""
