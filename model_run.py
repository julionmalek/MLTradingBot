'''
RUN MODEL
'''

# --- 1. IMPORT MODULES ---
if True:
    import os
    import sys
    import logging
    import numpy as np
    from datetime import datetime
    import pandas as pd
    from zoneinfo import ZoneInfo
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import joblib
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    from hmmlearn import hmm
    import streamlit as st
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from timedelta import Timedelta
    import talib as ta
    from lumibot.brokers import Alpaca
    from lumibot.strategies.strategy import Strategy
    from lumibot.traders import Trader
    from alpaca_trade_api import REST
    from bot_class import AdvancedMLTrader
    from dashboard import run_dashboard
    from lumibot.backtesting import YahooDataBacktesting

# --- 2. SETUP LOGGING (Console + standard format) ---
if True:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("lumibot").setLevel(logging.DEBUG)
    logging.getLogger("alpaca").setLevel(logging.DEBUG)
    logging.getLogger("asyncio").setLevel(logging.DEBUG)
    root_lvl = logging.getLogger().getEffectiveLevel()
    lumibot_lvl = logging.getLogger("lumibot").getEffectiveLevel()
    alpaca_lvl = logging.getLogger("alpaca").getEffectiveLevel()
    asyncio_lvl = logging.getLogger("asyncio").getEffectiveLevel()
    print("ROOT LOGGER LEVEL:", root_lvl)        # e.g. 10 == DEBUG, 20 == INFO, ...
    print("LUMIBOT LOGGER LEVEL:", lumibot_lvl)
    print("ALPACA LOGGER LEVEL:", alpaca_lvl)
    print("ASYNCIO LOGGER LEVEL:", asyncio_lvl)
    # Set the logging level to suppress debug messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('lumibot.data_sources.yahoo_data').setLevel(logging.WARNING)
    logging.getLogger('fsevents').setLevel(logging.WARNING)
    logging.getLogger('websockets.client').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    logging.getLogger('alpaca.trading.stream').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


# --- 3. ALPACA SETUP ---
if True:
    API_KEY = os.environ.get("ALPACA_API_KEY")
    API_SECRET = os.environ.get("ALPACA_API_SECRET")
    BASE_URL = "https://paper-api.alpaca.markets"
    ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True,
    }
    # Create the broker instance using Alpaca credentials
    broker = Alpaca(ALPACA_CREDS)

# --- 4. DEFINE SETTINGS ---
start = datetime(2022, 1, 1, tzinfo=ZoneInfo("America/New_York"))
end = datetime(2023, 12, 31, tzinfo=ZoneInfo("America/New_York"))
stock_symbols = ["SPY", "AAPL"] #, "MSFT", "GOOGL", "AMZN",  # stable performers
                 #"TSLA", "NVDA", "PLTR", "ARKK", "SQ"     # volatile picks]

# Import data
raw_data = YahooDataBacktesting(datetime_start=end, datetime_end=start)

# Clean data
length = (end - start).days
data_source = raw_data.get_bars(assets = stock_symbols, length = length, timestep = "day")

# Import etc for clustering
start1 = datetime(2021, 1, 1, tzinfo=ZoneInfo("America/New_York"))
end1 = datetime(2022, 12, 31, tzinfo=ZoneInfo("America/New_York"))
raw_data = YahooDataBacktesting(datetime_start=end1, datetime_end=start1)
length1 = (end1 - start1).days
cluster_training_data = raw_data.get_bars(assets = stock_symbols, length = length1, timestep = "day")

# Define the strategy and its parameters
strategy = AdvancedMLTrader(start = start,
                            end = end,
                            cluster_training_data = cluster_training_data,
                            data_source = data_source,
                            name="enhanced_ml_trader",
                            broker=broker,
                            benchmark="SPY",  # Explicitly setting benchmark
                            parameters={"symbols": stock_symbols,
                                        "cash": 5000,
                                        "risk aversion": 1.0}, # Start with 5000$
                            debug=True)  # Enable debug mode

# --- 4. Train model ---
# Generate features
features, _, targets = strategy.generate_features(compute_garch=True, return_raw_targets = True)


# Extract target variables
returns, volatility = targets['daily_return'], targets['volatility_5d']

# Train model
print(f"Number of observations: {len(features)}")

strategy.predict_and_update(features = features.drop(columns = ['daily_return', 'volatility_5d']),
                                     targets = pd.concat([returns, volatility], axis=1))

print(f"Trained on {len(strategy.predictions)} time points.")

# --- 5. Launch the dashboard ---
run_dashboard(strategy)
