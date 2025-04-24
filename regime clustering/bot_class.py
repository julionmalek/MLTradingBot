"""
Enhanced ML + Technical Strategy with Lumibot

Features:
- Clusters observations into different "regimes" based on technical indicators (RSI, SMA, MACD, ADX, Bollinger Bands)
- Uses FinBERT sentiment analysis to gauge bullish/bearish sentiment. (NOT YET)
"""

# --- 1. IMPORT MODULES ---
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
from finbert_utils import estimate_sentiment
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from alpaca_trade_api import REST

# --- 2. SETUP LOGGING (Console + standard format) ---
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

# --- 3. ALPACA SETUP ---
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_CREDS = {
  "API_KEY": API_KEY,
  "API_SECRET": API_SECRET,
  "PAPER": True,
}

# --- Proportions / Allocation for each symbol ---
# This dictionary says how much of your total *available cash* you want in each symbol
proportions = {
    # Stable and Well-Performing Stocks
    "AAPL": 0.10,  
    "MSFT": 0.10,  
    "GOOGL": 0.10,  
    "AMZN": 0.10,  
    "SPY": 0.20,   # S&P 500 ETF (stable base allocation)

    # More Volatile / Growth Stocks
    "TSLA": 0.15,  
    "NVDA": 0.10,  
    "PLTR": 0.05,  
    "ARKK": 0.05,  
    "SQ": 0.05,    
}


class AdvancedMLTrader(Strategy):
    """
    A multi-factor strategy combining:
      - Sentiment (FinBERT)
      - RSI, SMAs, MACD, ADX
      - Bollinger Bands & Stochastic for additional signals
      - A dynamic risk allocation based on SPY RSI

    The strategy invests 25% of capital into SPY at init (buy & hold),
    then trades other symbols based on technical + sentiment conditions.
    """

    def initialize(self, symbols, cash_at_risk=1.0, stable_allocation=0.25):
        """
        :param symbols: The list of symbols to trade.
        :param cash_at_risk: The fraction of total cash we are willing to risk.
        :param stable_allocation: The fraction of our total cash to put into SPY initially.
        """
        self.symbols = symbols
        self.cash_at_risk = cash_at_risk
        self.stable_allocation = stable_allocation

        # This determines how frequently we run the iteration
        self.sleeptime = "12H"  # every 12 hours in live trading

        # We will track if we already allocated to SPY
        self.spy_initialized = False

        # Keep track of last action for each symbol
        self.last_trade = {symbol: None for symbol in symbols}

        # Alpaca API for real-time account/positions
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

        logging.error("Initialization complete. Attempting to buy SPY with stable allocation.")
        self.initialize_spy()
        logging.error("this is an errorq")


    def initialize_spy(self):
        """Invest stable_allocation% of cash into SPY at the beginning and hold."""
        if self.spy_initialized:
            logging.error("SPY already initialized. Skipping.")
            return

        account_info = self.api.get_account()
        current_cash = self.get_cash()
        spy_cash = current_cash * self.stable_allocation

        logging.error(f"Current Cash: {current_cash:.2f}, allocating {spy_cash:.2f} to SPY.")

        spy_price = self.get_last_price("SPY")
        logging.error(f"Retrieved SPY price: {spy_price}")

        if spy_price is None:
            logging.error("Failed to get SPY price; cannot initialize SPY.")
            return

        spy_quantity = round(spy_cash / spy_price, 0)
        logging.error(f"Buying {spy_quantity} shares of SPY at approx. ${spy_price:.2f} each.")

        if spy_quantity > 0:
            order = self.create_order(
                "SPY",
                spy_quantity,
                "buy",
                type="market",
            )
            self.submit_order(order)
            logging.error("SPY initial allocation order submitted.")
        else:
            logging.warning("Calculated SPY quantity is 0. Not placing order.")

        self.spy_initialized = True

    def position_sizing(self, symbol):
        """
        Calculates how many shares to buy given the proportion in `proportions`
        and the current account cash.
        """
        account_info = self.api.get_account()
        cash = self.get_cash()
        symbol_prop = proportions.get(symbol, 0)

        allocated_cash = cash * symbol_prop
        last_price = self.get_last_price(symbol)

        logging.error(f"[position_sizing] Symbol={symbol}, "
                     f"Cash={cash:.2f}, "
                     f"AllocPct={symbol_prop}, "
                     f"AllocCash={allocated_cash:.2f}, "
                     f"LastPrice={last_price}")

        if last_price is None or last_price <= 0:
            logging.error(f"[position_sizing] Invalid price ({last_price}) for {symbol}.")
            return cash, None, 0

        quantity = round(allocated_cash / last_price, 0)
        return cash, last_price, quantity

    def dynamic_risk_allocation(self):
        """
        RSI (Relative Strength Index)
        → Measures recent price changes to detect overbought (>70) or oversold (<30) conditions — often used to time entry/exit points.

        Dynamically adjust self.cash_at_risk based on SPY RSI:
          - If SPY RSI > 70 => reduce risk
          - If SPY RSI < 30 => increase risk
          - Otherwise leave risk at 1 (default).
        """
        prices = self.get_historical_prices("SPY", length=14, timestep="day").df
        if len(prices) < 14:
            logging.warning("Not enough SPY data for RSI => skip dynamic_risk_allocation.")
            return

        spy_rsi_val = ta.RSI(prices["close"], timeperiod=14)[-1]

        if spy_rsi_val is None or np.isnan(spy_rsi_val):
            logging.warning("SPY RSI returned NaN => skip dynamic_risk_allocation.")
            return

        logging.error(f"[dynamic_risk_allocation] Current SPY RSI={spy_rsi_val:.2f}")

        if spy_rsi_val > 70:
            self.cash_at_risk = 0.5  # example: reduce risk to 50% if overbought
            logging.error("SPY RSI>70 => Decreasing risk to 0.5")
        elif spy_rsi_val < 30:
            self.cash_at_risk = 1.2  # example: slightly increase risk if oversold
            logging.error("SPY RSI<30 => Increasing risk to 1.2")
        else:
            self.cash_at_risk = 1.0  # normal
            logging.error("SPY RSI in normal range => risk=1.0")

    def get_dates(self, offset_days=3):
        """
        Returns (today_str, offset_str) for retrieving recent news for sentiment.
        """
        today = self.get_datetime()
        past = today - Timedelta(days=offset_days)
        return today.strftime("%Y-%m-%d"), past.strftime("%Y-%m-%d")

    def get_sentiment(self, symbol):
        """
        Sentiment (FinBERT)
        → Uses an NLP model trained on financial text to score news or tweets as positive/neutral/negative — gauges the market’s tone toward a stock.

        Uses FinBERT to analyze recent news headlines. Returns (probability, sentiment).
        """
        today_str, past_str = self.get_dates()
        try:
            news_items = self.api.get_news(symbol=symbol, start=past_str, end=today_str)
            headlines = [item.__dict__["_raw"]["headline"] for item in news_items]
            probability, sentiment = estimate_sentiment(headlines)
        except Exception as e:
            logging.error(f"[get_sentiment] Sentiment analysis failed for {symbol}: {e}")
            return 0.5, "neutral"

        logging.error(f"[get_sentiment] {symbol} => Sentiment='{sentiment}', Prob={probability:.2f}")
        return probability, sentiment

    # --------------------------
    #  Technical Indicators
    # --------------------------
    def calculate_technical_indicators(self, symbol):
        """
        Returns (rsi, sma20, sma50) for the last data point.
        RSI (Relative Strength Index)
        → Measures recent price changes to detect overbought (>70) or oversold (<30) conditions — often used to time entry/exit points.
        
        SMA (Simple Moving Average)
        → Smooths price data over a defined window (e.g., 50-day SMA) to identify overall trends or crossovers (e.g., short SMA crossing above long SMA = bullish signal).

        """
        try:
            hist_df = self.get_historical_prices(symbol, length=50, timestep="day").df
            if len(hist_df) < 50:
                logging.warning(f"[calc_tech_indicators] Not enough data ({len(hist_df)}) for {symbol}.")
                return None, None, None

            close_prices = hist_df["close"]
            rsi_val = ta.RSI(close_prices, timeperiod=14)[-1]
            sma20_val = ta.SMA(close_prices, timeperiod=20)[-1]
            sma50_val = ta.SMA(close_prices, timeperiod=50)[-1]

            logging.debug(f"[calc_tech_indicators] {symbol} => RSI={rsi_val:.2f}, "
                          f"SMA20={sma20_val:.2f}, SMA50={sma50_val:.2f}")
            return rsi_val, sma20_val, sma50_val
        except Exception as ex:
            logging.error(f"[calc_tech_indicators] Error for {symbol}: {ex}")
            return None, None, None

    def calculate_momentum_indicators(self, symbol):
        """
        MACD (Moving Average Convergence Divergence)
        → Tracks momentum via two EMAs (e.g., 12-day and 26-day) — signals generated when MACD crosses the signal line (momentum shifts).

        ADX (Average Directional Index)
        → Measures the strength (but not direction) of a trend; higher ADX = stronger trend, usually above 25.

        Returns (macd_val, macd_signal_val, adx_val) for the last data point.
        """
        try:
            hist_df = self.get_historical_prices(symbol, length=50, timestep="day").df
            if len(hist_df) < 50:
                logging.warning(f"[calc_momentum] Not enough data ({len(hist_df)}) for {symbol}.")
                return None, None, None

            high = hist_df["high"]
            low = hist_df["low"]
            close = hist_df["close"]

            macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            adx_series = ta.ADX(high, low, close, timeperiod=14)

            # Last data points
            macd_val = macd.iloc[-1]
            macd_signal_val = macd_signal.iloc[-1]
            adx_val = adx_series.iloc[-1]

            logging.debug(f"[calc_momentum] {symbol} => MACD={macd_val:.2f}, "
                          f"Signal={macd_signal_val:.2f}, ADX={adx_val:.2f}")
            return macd_val, macd_signal_val, adx_val
        except Exception as e:
            logging.error(f"[calc_momentum] Error for {symbol}: {e}")
            return None, None, None

    def calculate_volatility_indicators(self, symbol):
        """
        Bollinger Bands
        → Envelops price with upper/lower bands based on standard deviation — price touching the band often signals a volatility breakout or mean reversion.

        Stochastic Oscillator
        → Compares current price to its range over a period — useful for identifying momentum shifts, especially in ranging markets.

        Returns (atr_val, upper_bb, lower_bb, stoch_k, stoch_d) as a sample of added signals.
        - ATR for volatility
        - Bollinger Bands
        - Stochastic (K, D)
        """
        try:
            hist_df = self.get_historical_prices(symbol, length=50, timestep="day").df
            if len(hist_df) < 20:
                logging.warning(f"[calc_volatility_indicators] Not enough data for {symbol}.")
                return None, None, None, None, None

            high = hist_df["high"]
            low = hist_df["low"]
            close = hist_df["close"]

            # ATR (timeperiod=14)
            atr_series = ta.ATR(high, low, close, timeperiod=14)
            atr_val = atr_series.iloc[-1]

            # Bollinger Bands (timeperiod=20)
            upper_bb, mid_bb, lower_bb = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

            # Stochastic
            slowk, slowd = ta.STOCH(
                high, low, close,
                fastk_period=14, slowk_period=3, slowk_matype=0,
                slowd_period=3, slowd_matype=0
            )
            stoch_k = slowk.iloc[-1]
            stoch_d = slowd.iloc[-1]

            logging.debug(f"[calc_volatility_indicators] {symbol} => ATR={atr_val:.2f}, "
                          f"UpperBB={upper_bb.iloc[-1]:.2f}, LowerBB={lower_bb.iloc[-1]:.2f}, "
                          f"StochK={stoch_k:.2f}, StochD={stoch_d:.2f}")
            return atr_val, upper_bb.iloc[-1], lower_bb.iloc[-1], stoch_k, stoch_d
        except Exception as e:
            logging.error(f"[calc_volatility_indicators] Error for {symbol}: {e}")
            return None, None, None, None, None


    # Clustering
    def cluster_analysis(self, n_components=3, n_regimes=3, rolling_window=90, return_combined=False):
        """
        Run PCA, HMM clustering, and return regime information with adjustable parameters.
        
        Parameters:
            n_components (int): Number of principal components for PCA.
            n_regimes (int): Number of regimes (HMM states).
            rolling_window (int): Size of rolling window for daily returns and features.
            return_combined (bool): Whether to return both combined and averaged features.

        Returns:
            averaged_features (pd.DataFrame): DataFrame with averaged technical indicators and regimes.
            combined (pd.DataFrame): Combined stock data (individual stock returns).
        """
        # Set date range (same as backtest)
        start = datetime(2022, 1, 1, tzinfo=ZoneInfo("America/New_York"))
        end = datetime(2023, 12, 31, tzinfo=ZoneInfo("America/New_York"))
        length = (end - start).days

        stock_symbols = ["SPY", "AAPL"] #, "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "PLTR", "ARKK", "SQ"]

        # Instantiate the data source
        data_source = YahooDataBacktesting(datetime_start=end, datetime_end=start)
        data = data_source.get_bars(assets=stock_symbols,
                                    length=length,
                                    timestep="day")
        
        stock_dfs = []

        # Iterate through stock symbols and create data for each symbol
        for key, df_raw in data.items():
            df_raw = data[key]
            df = df_raw.df                                          # Convert to DataFrame
            df['symbol'] = str(key)                                 # Add symbol info in case you need it
            df.columns = df.columns.str.lower()                     # Separate dataframe for each symbol
            df = df[(df.index >= start) & (df.index <= end)].copy() # Ensure datetime index, filter by the start and end date

            # Calculate technical indicators. Note that we can add more.
            df['daily_return'] = df['close'].pct_change() # Daily return
            df['volatility_5d'] = df['daily_return'].rolling(5).std() # 5-day rolling volatility
            df['RSI'] = ta.RSI(df['close'], timeperiod=14) # RSI measures recent price changes to detect overbought (>70) or oversold (<30) conditions — often used to time entry/exit points.
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9) # MACD (Moving Average Convergence Divergence), track momentum, signals a shift in momentum.
            df['ADX'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14) # ADX (Average Directional Index). Higher ADX = stronger trend (not direction), usually above 25.
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['boll_width'] = df['BB_upper'] - df['BB_lower'] # Envelops price with upper/lower bands based on standard deviation. Price touching the band signals volatility breakout/mean reversion.

            # Keep only the necessary features
            df = df[['daily_return', 'volatility_5d', 'RSI', 'MACD', 'ADX', 'boll_width']]
            stock_dfs.append(df)

        # Merge all stock features by date and drop missing values
        combined = pd.concat(stock_dfs, axis=1, keys=stock_symbols)
        combined = combined.dropna()  # Drop rows with NaN values

        # Flatten multi-index columns (e.g., ('AAPL', 'MACD') → 'AAPL_MACD')
        combined.columns = [f"{symbol}_{feat}" for symbol, feat in combined.columns]

        # Apply rolling window (for smoothing, if necessary)
        averaged_features = pd.DataFrame(index=combined.index)
        feature_types = ['daily_return', 'volatility_5d', 'RSI', 'MACD', 'ADX', 'boll_width']
        for ft in feature_types:
            cols = [col for col in combined.columns if ft in col]
            averaged_features[ft] = combined[cols].mean(axis=1)

        # Apply rolling window to features
        averaged_features['daily_return_rolling'] = averaged_features['daily_return'].rolling(window=rolling_window).mean()

        # Check for NaNs before applying PCA and drop them if found
        if averaged_features.isnull().sum().sum() > 0:
            print("Warning: NaNs detected in averaged_features, dropping rows with NaNs.")
            averaged_features = averaged_features.dropna()

        # Normalize the features so they have mean 0 and standard deviation 1, so that metrics are scale independent
        scaler = StandardScaler().fit(averaged_features)
        X_scaled = scaler.transform(averaged_features)

        # Apply PCA with the specified number of components. Reduces the feature space to a few key dimensions that capture the most variation in the data.
        # Finds new axes (principal components) that are linear combinations of your original features, ranked by how much variance they explain.
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Apply HMM clustering with the specified number of regimes. Learns a set of hidden “regimes” that the system switches between over time, assuming that observations are Gaussian.
        # Uses the Expectation-Maximization algorithm to fit state transition probabilities and emission distributions, capturing how regimes evolve and produce the observed PCA features.
        model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", random_state=42)
        model.fit(X_pca)  # Fit the model to the data

        # Now use predict() to get the hidden states (regimes). Assign each timestamp to one of these regimes. Decoding hidden sequence, assigning each time step in your dataset to one of those inferred regimes.
        hidden_states = model.predict(X_pca)

        # Add the regimes to the averaged features DataFrame
        averaged_features['regime'] = hidden_states

        # Save models and results
        joblib.dump(scaler, 'regime_scaler.pkl')
        joblib.dump(pca, 'regime_pca.pkl')
        joblib.dump(model, 'regime_hmm.pkl')
        averaged_features.to_csv("portfolio_pca_hmm_regimes.csv")

        # Return either averaged features only or both averaged and combined data
        if return_combined:
            return averaged_features, combined, pca, X_pca
        else:
            return averaged_features, pca, X_pca


    def plot_cluster(self):
        """
        Interactive Streamlit dashboard for regime analysis.
        Includes dynamic sliders for PCA components, regime count, and rolling window size.
        Also adds visualizations for PCA (2D/3D) and PCA component loadings.
        """
        st.set_page_config(layout="wide")

        st.title("Regime Detection Dashboard")

        # Sidebar controls
        st.sidebar.header("Model Parameters")
        n_components = st.sidebar.slider("Number of PCA Components", 1, 10, 3)
        n_regimes = st.sidebar.slider("Number of Regimes (HMM States)", 2, 6, 3)
        rolling_window = st.sidebar.slider("Rolling Window (Days)", 30, 180, 90)

        # Load and rerun the clustering logic with current parameters
        averaged_features, combined, pca_model, X_pca = self.cluster_analysis(
            n_components=n_components,
            n_regimes=n_regimes,
            rolling_window=rolling_window,
            return_combined=True
        )

        tab1, tab2 = st.tabs(["📈 Model Fit Evaluation", "🔮 Model Prediction"])

        # --- TAB 1: Model Fit Evaluation ---
        with tab1:
            st.subheader("Returns and Regimes")

            # Plot average return
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=averaged_features.index,
                y=averaged_features['daily_return'],
                mode='lines',
                name='Avg Daily Return',
                line=dict(color='black')
            ))

            # Plot each individual stock return
            return_cols = [col for col in combined.columns if 'daily_return' in col]
            for col in return_cols:
                fig.add_trace(go.Scatter(
                    x=combined.index,
                    y=combined[col],
                    mode='lines',
                    name=col,
                    line=dict(width=0.7),
                    opacity=0.4
                ))

            # Add shaded regimes
            for regime in np.unique(averaged_features['regime']):
                mask = averaged_features['regime'] == regime
                regime_df = averaged_features[mask]
                fig.add_vrect(
                    x0=regime_df.index.min(),
                    x1=regime_df.index.max(),
                    fillcolor=f"rgba({regime * 50 % 255}, {regime * 100 % 255}, {regime * 150 % 255}, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text=f"Regime {regime}",
                    annotation_position="top left"
                )

            fig.update_layout(
                title="Returns with Regime Overlay",
                xaxis_title="Date",
                yaxis_title="Return",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- PCA Component Loadings Bar Graph ---
            st.subheader("PCA Component Loadings")
            try:
                num_features = pca_model.components_.shape[1]
                feature_names = combined.iloc[:, :num_features].columns
                components_df = pd.DataFrame(
                    pca_model.components_,
                    columns=feature_names,
                    index=[f"PC{i+1}" for i in range(pca_model.n_components_)]
                )
                st.dataframe(components_df)

                # Plot explained variance as bar chart
                explained_variance = pca_model.explained_variance_ratio_
                fig_variance = plt.figure(figsize=(8, 6))
                plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
                plt.title("Explained Variance by PCA Components")
                plt.xlabel("Principal Components")
                plt.ylabel("Explained Variance Ratio")
                plt.xticks(range(1, len(explained_variance) + 1))
                st.pyplot(fig_variance)

            except Exception as e:
                st.error(f"Error displaying PCA components: {e}")

            # --- PCA 2D Visualization ---
            st.subheader("PCA 2D Visualization")
            pca_2d = PCA(n_components=2)
            pca_2d_data = pca_2d.fit_transform(averaged_features.drop(columns=['regime']))

            # Ensure the number of rows matches
            if len(averaged_features) != len(pca_2d_data):
                st.error("Mismatch between the number of rows in PCA data and regime data.")
                return

            # Create 2D scatter plot with regime colors
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(pca_2d_data[:, 0], pca_2d_data[:, 1], c=averaged_features['regime'], cmap="viridis", s=100, alpha=0.7)
            plt.title("2D PCA Projection with Regimes")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.colorbar(label="Regimes")
            st.pyplot(plt.gcf())

            # --- PCA 3D Visualization ---
            st.subheader("PCA 3D Visualization")
            pca_3d = PCA(n_components=3)
            pca_3d_data = pca_3d.fit_transform(averaged_features.drop(columns=['regime']))

            # Ensure the number of rows matches
            if len(averaged_features) != len(pca_3d_data):
                st.error("Mismatch between the number of rows in PCA data and regime data.")
                return

            fig_3d = plt.figure(figsize=(10, 8))
            ax = fig_3d.add_subplot(111, projection='3d')
            scatter_3d = ax.scatter(pca_3d_data[:, 0], pca_3d_data[:, 1], pca_3d_data[:, 2], c=averaged_features['regime'], cmap="viridis", s=100, alpha=0.7)
            ax.set_title("3D PCA Projection with Regimes")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_zlabel("PCA Component 3")
            fig_3d.colorbar(scatter_3d, ax=ax, label="Regimes")
            st.pyplot(fig_3d)

        # --- TAB 2: Model Prediction Placeholder ---
        with tab2:
            st.subheader("Model Prediction")
            st.write("Coming soon... (rolling forecasting, reinforcement learning, etc.)")




    # -----------------------------------------
    #  Main logic on each trading iteration
    # -----------------------------------------
    def on_trading_iteration(self):
        # 1) Adjust risk based on SPY RSI
        self.dynamic_risk_allocation()

        # 2) Evaluate each symbol
        for symbol in self.symbols:
            # Skip re-initializing SPY or trying to rebalance it every iteration
            if symbol == "SPY":
                continue

            cash, last_price, quantity = self.position_sizing(symbol)
            if not last_price:
                logging.error(f"[{symbol}] Missing price data; skipping this iteration.")
                continue

            # --- Get sentiment ---
            probability, sentiment = self.get_sentiment(symbol)

            # --- Calculate technicals ---
            rsi, sma20, sma50 = self.calculate_technical_indicators(symbol)
            macd_val, macd_signal, adx_val = self.calculate_momentum_indicators(symbol)
            atr_val, upper_bb, lower_bb, stoch_k, stoch_d = self.calculate_volatility_indicators(symbol)

            if any(x is None for x in [rsi, sma20, sma50, macd_val, macd_signal, adx_val,
                                       atr_val, upper_bb, lower_bb, stoch_k, stoch_d]):
                logging.error(f"[{symbol}] Incomplete indicators => skipping trade logic.")
                continue

            # Log the technical snapshot
            logging.error(
                f"[{symbol}] Tech snapshot => "
                f"Sentiment={sentiment}, Prob={probability:.2f}, "
                f"RSI={rsi:.2f}, SMA20={sma20:.2f}, SMA50={sma50:.2f}, "
                f"MACD={macd_val:.2f}, MACDSignal={macd_signal:.2f}, ADX={adx_val:.2f}, "
                f"ATR={atr_val:.2f}, BB_Upper={upper_bb:.2f}, BB_Lower={lower_bb:.2f}, "
                f"StochK={stoch_k:.2f}, StochD={stoch_d:.2f}"
            )

            # Check current holdings
            current_position = self.get_position(symbol)
            logging.error(f"[{symbol}] Current Position: {current_position}")
            current_quantity = current_position.quantity if current_position else 0
            logging.error(f"[{symbol}] Current Quantity: {current_quantity}")

            # Example Buy Criteria (feel free to tweak):
            #   - Have enough cash for at least 1 share
            #   - Positive sentiment with high confidence
            #   - RSI < 60 (somewhat oversold)
            #   - 20-day SMA above 50-day (positive trend)
            #   - MACD > Signal line, ADX>25 => strong trend
            #   - StochK < 25 => oversold
            
            logging.error(f"[{symbol}] Cash={cash:.2f}, LastPrice={last_price:.2f}")
            if (
                cash > last_price
                and sentiment == "positive" and probability > 0.7
                and rsi < 70
                and sma20 > sma50
                and macd_val > macd_signal
                and adx_val < 25
                and stoch_k < 70
                and current_quantity == 0
            ):
                logging.error(f"[{symbol}] Buy criteria met. Placing BUY order for {quantity} shares.")
                buy_order = self.create_order(
                    symbol,
                    quantity,
                    "buy",
                    type="market",
                )
                self.submit_order(buy_order)
                self.last_trade[symbol] = "buy"

            # Example Sell Criteria:
            #   - Negative sentiment with decent confidence
            #   - RSI > 80 => overbought
            #   - SMA20 < SMA50 => downward trend
            #   - MACD < signal => negative momentum
            #   - ADX>20 => a trending environment (selling into a downward trend)
            #   - StochK > 80 => overbought
            elif (
                sentiment == "negative" and probability > 0.7
                and rsi > 80
                and sma20 < sma50
                and macd_val < macd_signal
                and adx_val > 20
                and stoch_k > 80
                and current_quantity > 0
            ):
                logging.warning(f"[{symbol}] Sell criteria met. SELLING all holdings.")
                sell_order = self.create_order(symbol, current_quantity, "sell")
                self.submit_order(sell_order)
                self.last_trade[symbol] = "sell"



# ------------------
#  Backtest Section
# ------------------
# Set the date range for backtesting
start_date = datetime(2023, 11, 1)
end_date = datetime(2023, 12, 31)

# Create the broker instance using Alpaca credentials
broker = Alpaca(ALPACA_CREDS)

# Define the strategy and its parameters
strategy = AdvancedMLTrader(
    name="enhanced_ml_trader",
    broker=broker,
    benchmark="SPY",  # Explicitly setting benchmark

    parameters={
        "symbols": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "SPY",  # stable performers
            "TSLA", "NVDA", "PLTR", "ARKK", "SQ"     # volatile picks
        ],
        "cash_at_risk": 1.0,  # default risk
        "stable_allocation": 0.25,  # 25% into SPY
    },
    debug=True,  # Enable debug mode

)




# TRY TO GENERATE CLUSTER ANALYSIS PLOT
strategy.plot_cluster()


'''
# Run a backtest with Yahoo data
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={
        # Redundant here, but you can re-specify or override:
        "symbols": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "SPY",
            "TSLA", "NVDA", "PLTR", "ARKK", "SQ"
        ],
        "cash_at_risk": 1.0,
        "stable_allocation": 0.25,

    },
)

    # If you want to run live/paper trading instead of backtesting, uncomment:
    # trader = Trader()
    # trader.add_strategy(strategy)
    # trader.run_all()

'''