"""
Enhanced ML + Technical Strategy with Lumibot

Features:
- Allocates a portion of capital (25%) to SPY on initialize (buy & hold).
- Uses FinBERT sentiment analysis to gauge bullish/bearish sentiment.
- Incorporates multiple technical indicators for buy/sell decisions:
    - RSI
    - SMA(20) vs SMA(50)
    - MACD
    - ADX
    - Bollinger Bands
    - Stochastic Oscillator
- Logs extensively for debugging/troubleshooting.
- Demonstrates dynamic risk allocation based on SPY's RSI.
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime


# Utility Libraries
from timedelta import Timedelta
import talib as ta
from finbert_utils import estimate_sentiment

# --- Logging Setup (Console + standard format) ---
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

# Lumibot / Alpaca
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from alpaca_trade_api import REST

# --- Alpaca Credentials ---
# Ideally, load these from environment variables or a .env file
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

# The dictionary for the Alpaca broker
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
