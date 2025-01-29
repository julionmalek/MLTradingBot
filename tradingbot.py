from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.brokers import Binance
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from finbert_utils import estimate_sentiment
import talib
import numpy as np
import logging

# Configure logging to display INFO level logs in the terminal
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG to see more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

API_KEY = "PKWHJN0D9KIYXVJBSAP6"
API_SECRET = "Y4nUWLqtCs3rF2huXfAGudLcxjAkR8bxccQdhHvK"
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True,
}

# Asset proportions for a diversified portfolio
proportions = {
    "AAPL": 0.15,
    "MSFT": 0.15,
    "GOOGL": 0.15,
    "AMZN": 0.15,
    "TSLA": 0.15,
    "BTC": 0.15,
    "ETH": 0.10,  # Crypto assets
}

class AdvancedMLTrader(Strategy):
    def initialize(self, symbols: list, cash_at_risk: float = 0.5):
        self.symbols = symbols
        self.cash_at_risk = cash_at_risk
        self.sleeptime = "24H"
        self.last_trade = {symbol: None for symbol in symbols}
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self, symbol: str):
        cash = self.get_cash()
        allocated_cash = cash * proportions.get(symbol, 0)
        last_price = self.get_last_price(symbol)

        self.log_message(f"Cash: {cash}, Allocated Cash: {allocated_cash}, Last Price for {symbol}: {last_price}")

        if last_price is None:
            self.log_message(f"Failed to retrieve last price for {symbol}")
            return cash, None, None  # Return None for invalid data

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
            self.log_message(f"Sentiment analysis failed for {symbol}: {e}")
            probability, sentiment = 0.5, "neutral"
        return probability, sentiment

    def calculate_technical_indicators(self, symbol: str):
        # Fetch historical prices
        try:
            prices = self.get_historical_prices(
                asset=symbol,
                length=50,  # Fetch the last 50 data points
                timestep="day",  # Use daily data
            )
        except Exception as e:
            self.log_message(f"Failed to fetch historical prices for {symbol}: {e}")
            return None, None, None

        # Extract close prices
        try:
            if hasattr(prices, "df"):
                close_prices = prices.df["close"]
            elif isinstance(prices, list):
                close_prices = [bar.close for bar in prices]
            else:
                raise TypeError(f"Unsupported Bars format for {symbol}")
        except Exception as e:
            self.log_message(f"Error extracting close prices for {symbol}: {e}")
            return None, None, None

        # Ensure enough data is available
        if len(close_prices) < 20:
            self.log_message(f"Not enough historical data for {symbol}")
            return None, None, None

        # Calculate technical indicators
        rsi = talib.RSI(np.array(close_prices), timeperiod=14)[-1]
        sma_20 = talib.SMA(np.array(close_prices), timeperiod=20)[-1]
        sma_50 = talib.SMA(np.array(close_prices), timeperiod=50)[-1]

        return rsi, sma_20, sma_50

    def on_trading_iteration(self):
        for symbol in self.symbols:
            cash, last_price, quantity = self.position_sizing(symbol)

            if last_price is None:
                self.log_message(f"Skipping {symbol} due to missing price data")
                continue  # Skip this symbol if the price is missing

            probability, sentiment = self.get_sentiment(symbol)
            rsi, sma_20, sma_50 = self.calculate_technical_indicators(symbol)

            if cash > last_price and rsi is not None:
                if sentiment == "positive" and probability > 0.8 and rsi < 70 and sma_20 > sma_50:
                    if self.last_trade[symbol] == "sell":
                        self.sell_all()
                    order = self.create_order(
                        symbol,
                        quantity,
                        "buy",
                        type="bracket",
                        take_profit_price=last_price * 1.2,  # Target 20% profit
                        stop_loss_price=last_price * 0.95,   # Risk 5%
                    )
                    self.submit_order(order)
                    self.last_trade[symbol] = "buy"
                elif sentiment == "negative" and probability > 0.8 and rsi > 30 and sma_20 < sma_50:
                    if self.last_trade[symbol] == "buy":
                        self.sell_all()
                    order = self.create_order(
                        symbol,
                        quantity,
                        "sell",
                        type="bracket",
                        take_profit_price=last_price * 0.8,  # Target 20% loss reduction
                        stop_loss_price=last_price * 1.05,
                    )
                    self.submit_order(order)
                    self.last_trade[symbol] = "sell"

start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)

broker = Alpaca(ALPACA_CREDS)
strategy = AdvancedMLTrader(
    name="advanced_ml_trader",
    broker=broker,
    parameters={"symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC", "ETH"], "cash_at_risk": 0.5},
    debug=True,  # Enable debug mode
)
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC", "ETH"], "cash_at_risk": 0.5},
)
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
