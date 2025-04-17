import os
import sys
import logging
import numpy as np
from datetime import datetime

# Utility Libraries
from timedelta import Timedelta
import talib as ta
from finbert_utils import estimate_sentiment
from quantconnect.data import YahooDataBacktesting
from datetime import datetime
import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

# Alpaca/Lumibot
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from alpaca_trade_api import REST

# Alpaca Credentials
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

# The dictionary for the Alpaca broker
ALPACA_CREDS = {
  "API_KEY": API_KEY,
  "API_SECRET": API_SECRET,
  "PAPER": True,
}

class run_strategy(self):
    def __init__(self):
        self.start_date = datetime(2023, 11, 1)
        self.end_date = datetime(2023, 12, 31)
        self.portfolio = ["SPY"] # ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "PLTR", "ARKK", "SQ"]
        self.broker = Alpaca(ALPACA_CREDS)
        self.benchmark = ["SPY"]
        self.parameters = {"symbols": self.portfolio, 
                           "cash_at_risk": 1.0,  # default risk
                           "stable_allocation": 0.25}  # 25% into SPY
        self.strategy = AdvancedMLTrader(name = "enhanced_ml_trader",
                                         broker = self.broker,
                                         benchmark = self.benchmark,
                                         parameters = self.parameters,
                                         debug=True)
    
    # Run a backtest with Yahoo data
    def run(self):
        self.strategy.backtest(YahooDataBacktesting, self.start_date, self.end_date, parameters = self.parameters)








    # Bridge clustering model and trading strategy
    joblib.dump(kmeans, "regime_model.pkl")
    joblib.dump(scaler, "scaler.pkl")


    features_today = pd.DataFrame({
        "daily_return": ...,
        "volatility_5d": ...,
        "RSI": ...,
        ...
    }, index=[0])

    X_scaled_today = scaler.transform(features_today)
    regime_today = kmeans.predict(X_scaled_today)[0]

    if regime_today == 0:
        # Use standard strategy
    elif regime_today == 1:
        # Only trade long
    elif regime_today == 2:
        # Use tighter stop losses
