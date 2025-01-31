from lumibot.brokers import Alpaca
#from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from finbert_utils import estimate_sentiment
import talib
import numpy as np
import logging
import os
import sys
from lumibot.traders import Trader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Test logging
logging.info("This is an INFO log.")
logging.debug("This is a DEBUG log.")
logging.warning("This is a WARNING log.")
logging.error("This is an ERROR log.")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output logs to terminal
    ],
)

def log_message(self, message, level="info"):
    """
    Logs messages using Python's logging module.
    """
    logger = logging.getLogger(self.__class__.__name__)
    
    # Map levels to logging methods
    log_levels = {
        "debug": logger.debug,
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
        "critical": logger.critical,
    }

    # Log the message
    log_func = log_levels.get(level.lower(), logger.info)
    log_func(message)

API_KEY = "PKLCNW0XLBL0U9AFUBRZ" 
API_SECRET = "06zB23MhXVAVlR3XRsBcQSzUnYNefy60bMMGx6Ge" 
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True,
}

proportions = {
    # Stable and Well-Performing Stocks
    "AAPL": 0.10,  # Apple
    "MSFT": 0.10,  # Microsoft
    "GOOGL": 0.10,  # Google
    "AMZN": 0.10,  # Amazon
    "SPY": 0.20,  # S&P 500 ETF (stable base allocation)

    # Volatile Stocks
    "TSLA": 0.15,  # Tesla
    "NVDA": 0.10,  # NVIDIA
    "PLTR": 0.05,  # Palantir
    "ARKK": 0.05,  # ARK Innovation ETF
    "SQ": 0.05,  # Block (Square)
}

class AdvancedMLTrader(Strategy):
    def initialize(self, symbols: list, cash_at_risk: float = 1, stable_allocation: float = 0.25):
        self.symbols = symbols
        self.cash_at_risk = cash_at_risk
        self.stable_allocation = stable_allocation
        self.sleeptime = "12H"
        self.spy_initialized = False  # Flag to ensure SPY is only initialized once
        self.last_trade = {symbol: None for symbol in symbols}
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.initialize_spy()

    def position_sizing(self, symbol: str):
        cash = self.get_cash()
        allocated_cash = cash * proportions.get(symbol, 0)
        last_price = self.get_last_price(symbol)

        #logging.error(f"Cash: {cash}, Allocated Cash: {allocated_cash}, Last Price for {symbol}: {last_price}")

        if last_price is None:
            logging.error(f"Failed to retrieve last price for {symbol}")
            return cash, None, None

        quantity = round(allocated_cash / last_price, 0)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d")

    def get_sentiment(self, symbol: str):
        today, three_days_prior = self.get_dates()
        try:
            news = self.api.get_news(symbol=symbol, start=three_days_prior, end=today)
            news = [ev.__dict__["_raw"]["headline"] for ev in news]
            probability, sentiment = estimate_sentiment(news)
        except Exception as e:
            logging.error(f"Sentiment analysis failed for {symbol}: {e}")
            probability, sentiment = 0.5, "neutral"
        logging.error(f"Sentiment for {symbol}: {sentiment} with probability {probability}")
        return probability, sentiment


    def dynamic_risk_allocation(self):
        spy_prices = self.get_historical_prices("SPY", length=14, timestep="day").df["close"]
        spy_rsi = talib.RSI(spy_prices, timeperiod=14)[-1]

        if len(spy_prices) < 14:  # Ensure enough data for RSI calculation
            logging.error("Insufficient data for SPY RSI calculation. Skipping dynamic risk allocation.")
            return

        spy_rsi = talib.RSI(spy_prices, timeperiod=14)[-1]
        
        if np.isnan(spy_rsi):  # Handle NaN case
            logging.error("SPY RSI calculation returned NaN. Skipping dynamic risk allocation.")
            return

        if spy_rsi > 70:
            self.cash_at_risk = 1  # Reduce risk
        elif spy_rsi < 30:
            self.cash_at_risk = 1  # Increase risk
        else:
            self.cash_at_risk = 1

        logging.error(f"Dynamic Risk Allocation: cash_at_risk adjusted to {self.cash_at_risk} based on SPY RSI {spy_rsi}")

    def initialize_spy(self):
        """Invest 30% of cash into SPY at the beginning and hold."""
        if self.spy_initialized:
            logging.error("SPY already initialized. Skipping.")
            return

        spy_cash = self.get_cash() * self.stable_allocation
        logging.error(f"Allocating {spy_cash} to SPY")
        spy_price = self.get_last_price("SPY")
        logging.error(f"SPY Price: {spy_price}")
        if spy_price is not None:
            spy_quantity = round(spy_cash / spy_price, 0)
            logging.error(f"SPY Quantity: {spy_quantity}")
            order = self.create_order("SPY", spy_quantity, "buy", type="trailing_stop", trail_percent=0.30)
            self.submit_order(order)
            logging.error(f"Initialized SPY: Allocated {spy_quantity} shares at price {spy_price}")
            self.spy_initialized = True
        else:
            logging.error("Failed to initialize SPY: Could not fetch price.")

    def calculate_technical_indicators(self, symbol: str):
        try:
            prices_df = self.get_historical_prices(asset=symbol, length=50, timestep="day").df
            if len(prices_df) == 0:
                logging.info(f"No historical data for {symbol}. Skipping technical indicators calculation.")
                return None, None, None

            close_prices = prices_df["close"]
            if len(close_prices) < 14:  # Ensure sufficient data for RSI calculation
                logging.info(f"Not enough data to calculate RSI for {symbol}. Skipping.")
                return None, None, None

            rsi = talib.RSI(np.array(close_prices), timeperiod=14)[-1]
            sma_20 = talib.SMA(np.array(close_prices), timeperiod=20)[-1] if len(close_prices) >= 20 else None
            sma_50 = talib.SMA(np.array(close_prices), timeperiod=50)[-1] if len(close_prices) >= 50 else None

            return rsi, sma_20, sma_50
        except Exception as e:
            logging.error(f"Error in calculate_technical_indicators for {symbol}: {e}")
            return None, None, None

    def calculate_momentum_indicators(self, symbol: str):
        try:
            # Get historical prices
            prices_df = self.get_historical_prices(asset=symbol, length=50, timestep="day").df

            # Ensure there is enough data for calculation
            if len(prices_df) < 50:
                logging.error(f"Not enough data for {symbol} to calculate momentum indicators. Skipping.")
                return None, None, None

            # Extract high, low, and close prices
            high = prices_df["high"]
            low = prices_df["low"]
            close = prices_df["close"]

            # Calculate MACD
            macd, macdsignal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

            # Calculate ADX
            adx_series = talib.ADX(high, low, close, timeperiod=14)

            # Check if the ADX series has enough data and isn't empty
            if len(adx_series) == 0 or np.isnan(adx_series.iloc[-1]):
                logging.error(f"ADX calculation returned empty or NaN for {symbol}. Skipping.")
                return (
                    macd.iloc[-1] if len(macd) > 0 else None,
                    macdsignal.iloc[-1] if len(macdsignal) > 0 else None,
                    None,
                )

            # Ensure MACD series also has data
            if len(macd) == 0 or len(macdsignal) == 0:
                logging.error(f"MACD calculation returned empty for {symbol}. Skipping.")
                return None, None, None

            # Return the last values of MACD, signal line, and ADX
            return macd.iloc[-1], macdsignal.iloc[-1], adx_series.iloc[-1]

        except Exception as e:
            logging.error(f"Error in calculate_momentum_indicators for {symbol}: {e}")
            return None, None, None

    def calculate_atr(self, symbol: str):
        try:
            # Get historical prices
            prices_df = self.get_historical_prices(asset=symbol, length=50, timestep="day").df

            # Ensure there is enough data
            if len(prices_df) < 14:  # ATR requires at least 14 periods
                logging.error(f"Not enough data for {symbol} to calculate ATR. Skipping.")
                return None

            # Extract high, low, and close prices
            high = prices_df["high"]
            low = prices_df["low"]
            close = prices_df["close"]

            # Calculate ATR
            atr_series = talib.ATR(high, low, close, timeperiod=14)

            # Check if ATR series is empty or NaN
            if atr_series.empty or np.isnan(atr_series.iloc[-1]):
                logging.error(f"ATR calculation returned empty or NaN for {symbol}. Skipping.")
                return None

            # Return the last ATR value
            return atr_series.iloc[-1]

        except Exception as e:
            logging.error(f"Error in calculate_atr for {symbol}: {e}")
            return None

    def on_trading_iteration(self):
        self.dynamic_risk_allocation()

        for symbol in self.symbols:
            if symbol == "SPY":  # Skip additional SPY trades since it's rebalanced
                current_position = self.get_position(symbol)
                current_position_quantity = current_position.quantity if current_position else 0
                continue

            cash, last_price, quantity = self.position_sizing(symbol)

            if last_price is None:
                logging.info(f"Skipping {symbol} due to missing price data")
                continue

            probability, sentiment = self.get_sentiment(symbol)
            rsi, sma_20, sma_50 = self.calculate_technical_indicators(symbol)
            macd, macdsignal, adx = self.calculate_momentum_indicators(symbol)
            atr = self.calculate_atr(symbol)

            # Check if technical indicators are valid before proceeding
            if any(indicator is None for indicator in [rsi, sma_20, sma_50, macd, macdsignal, adx, atr]):
                logging.error(f"Skipping {symbol} due to insufficient data for technical indicators.")
                continue

            # Check current holdings before making trades
            current_position = self.get_position(symbol)
            current_position_quantity = current_position.quantity if current_position else 0

            # Buy logic
            if (
                cash > last_price and sentiment == "positive" and probability > 0.8
                and rsi < 70 and sma_20 > sma_50 and macd > macdsignal and adx > 25
                and current_position_quantity == 0
            ):
                logging.error(f"Placing buy order for {symbol}: {quantity} units")
                order = self.create_order(
                    symbol, quantity, "buy", type="trailing_stop", trail_percent=0.05
                )
                self.submit_order(order)
                self.last_trade[symbol] = "buy"

            # Sell logic
            elif (
                sentiment == "negative" and probability > 0.7 and rsi > 85
                and sma_20 < sma_50 and macd < macdsignal and adx > 20
                and current_position_quantity > 0
            ):
                logging.error(f"Selling all holdings for {symbol}")
                self.sell_all()
                self.last_trade[symbol] = "sell"

# Backtesting and running the strategy
#start_date = datetime(2020, 1, 1)
#end_date = datetime(2024, 12, 31)

broker = Alpaca(ALPACA_CREDS)
strategy = AdvancedMLTrader(
    name="advanced_ml_trader",
    broker=broker,
    parameters = {
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY",  # Stable performers
                "TSLA", "NVDA", "PLTR", "ARKK", "SQ"],   # Volatile stocks
    "cash_at_risk": 1,
}
)
#strategy.backtest(
 #   YahooDataBacktesting,
 #   start_date,
 #   end_date,
  #  parameters = {
  #  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY",  # Stable performers
  #              "TSLA", "NVDA", "PLTR", "ARKK", "SQ"],   # Volatile stocks
  #  "cash_at_risk": 1,
#}
#)

trader = Trader()
trader.add_strategy(strategy)
trader.run_all()