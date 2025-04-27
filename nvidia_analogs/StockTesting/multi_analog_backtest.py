import os, sys, logging
import pandas as pd
from datetime import timedelta
from pathlib import Path

from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from alpaca_trade_api import REST

# ─── Directory Setup ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

# ─── Data Directory ───────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
ANA_DIR = DATA_DIR / "AnalogFullBacktests"
TRADING_FEE = 0.001  # 0.1% trading fee

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(stream=sys.stdout,
                    level=logging.ERROR,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("MultiAnalog")

class MultiAnalogStrategy(Strategy):
    """
    Dynamically allocates funds across multiple symbols based on analog signals and avg returns.
    """

    def initialize(self, **kwargs):
        self.symbols = kwargs.get("symbols", ["NVDA", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"])
        #self.symbols = kwargs.get("symbols", ["NVDA"])
        self.signal_dir = kwargs.get("signal_dir", ANA_DIR)
        self.invest_frac = kwargs.get("invest_frac", 0.8)
        self.horizon_days = kwargs.get("horizon_days", 5)

        if not self.symbols:
            logger.error("⚠️  No symbols provided; strategy will not trade.")

        self.signal_data = {}
        for sym in self.symbols:
            path = self.signal_dir / f"{sym}_analog_backtest.csv"
            if path.exists():
                df = pd.read_csv(path, parse_dates=["date"], index_col="date")
                df.index = df.index.normalize()
                self.signal_data[sym] = df
            else:
                logger.error(f"⚠️  Missing file for {sym}: {path}")
                self.signal_data[sym] = pd.DataFrame()

        self.entry_dates = {}

        self.api = REST(
            base_url="https://paper-api.alpaca.markets",
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_API_SECRET")
        )

        logger.error(f"Initialized with symbols={self.symbols}, invest_frac={self.invest_frac}, horizon_days={self.horizon_days}")

    def on_trading_iteration(self):
        today = self.get_datetime().date()
        ts = pd.Timestamp(today)

        # First, sell any positions that have reached their horizon
        for symbol in self.symbols:
            pos = self.get_position(symbol)
            qty = pos.quantity if pos else 0
            if qty > 0:
                entry = self.entry_dates.get(symbol)
                if entry and ts >= entry + pd.Timedelta(days=self.horizon_days):
                    logger.error(f"{symbol}: Horizon passed → SELL {qty}")
                    order = self.create_order(symbol, qty, "sell", type="market")
                    self.submit_order(order)
                    self.entry_dates.pop(symbol, None)
                continue

        # Determine symbols eligible to buy
        available_cash = self.get_cash()
        investable_cash = available_cash * self.invest_frac

        candidates = []
        for sym in self.symbols:
            if self.get_position(sym):
                continue  # Skip if already holding

            df = self.signal_data.get(sym)
            if df is None or ts not in df.index:
                continue

            row = df.loc[ts]
            if row.get('signal', False):
                score = row.get('avg_analog_ret', 0)
                if score > 0:
                    candidates.append((sym, score))

        if not candidates:
            return

        # Allocate funds proportionally
        total_score = sum(score for _, score in candidates)
        if total_score <= 0:
            return

        for sym, score in candidates:
            weight = score / total_score
            allocation = investable_cash * weight * (1 - TRADING_FEE)  # Apply fee adjustment to buying power
            price = self.get_last_price(sym)
            if price and price > 0:
                qty = int(allocation // price)
                if qty > 0:
                    logger.error(f"{sym}: signal=TRUE, avg_ret={score:.2%} → BUY {qty}")
                    order = self.create_order(sym, qty, "buy", type="market")
                    self.submit_order(order)
                    self.entry_dates[sym] = ts

# ─── Entrypoint ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import datetime

    broker = Alpaca({
        "API_KEY": os.getenv("ALPACA_API_KEY"),
        "API_SECRET": os.getenv("ALPACA_API_SECRET"),
        "PAPER": True
    })

    strat = MultiAnalogStrategy(
        name="MultiAnalog",
        broker=broker,
        benchmark="SPY",
        parameters={
            "symbols": ["NVDA", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"],
            #"symbols": ["NVDA"],
            "signal_dir": DATA_DIR,
            "invest_frac": 0.8,
            "horizon_days": 5
        },
        debug=True
    )

    strat.backtest(
        YahooDataBacktesting,
        datetime(2015, 4, 20),
        datetime(2025, 4, 23)
    )
