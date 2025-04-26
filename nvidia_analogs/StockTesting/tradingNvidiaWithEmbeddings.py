# analog_backtest_strategy.py
import os, sys, logging
import pandas as pd
from datetime import timedelta

from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from alpaca_trade_api import REST

# ─── Logging ─────────────────────────────────────────────────────
logging.basicConfig(stream=sys.stdout,
                    level=logging.ERROR,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("AnalogBT")

class AnalogBacktestStrategy(Strategy):
    """
    Backtests analog-based signals on NVDA.
    """

    def initialize(self, **kwargs):
        # pull with defaults so we never KeyError
        self.symbols       = kwargs.get("symbols", ["NVDA"])
        self.signal_csv    = kwargs.get("analog_csv", "../data/analog_full_backtest2.csv")
        self.invest_frac   = kwargs.get("investment_fraction", 0.5)
        self.horizon_days  = kwargs.get("horizon_days", 5)


        if not self.symbols:
            logger.error("⚠️  No symbols provided in parameters; strategy will not trade.")

        if not self.signal_csv:
            logger.error("⚠️  No analog_csv provided; cannot load signals.")

        # load signals if we have a path
        self.signals = pd.Series(dtype=bool)
        if self.signal_csv:
            df = pd.read_csv(self.signal_csv, parse_dates=["date"], index_col="date")
            df.index = df.index.normalize()
            self.signals = df["signal"].astype(bool)

        # track entry dates
        self.entry_dates = {}

        # Alpaca REST client
        self.api = REST(
            base_url="https://paper-api.alpaca.markets",
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_API_SECRET")
        )

        logger.error(f"Initialized with symbols={self.symbols}, "
                     f"signal_csv={self.signal_csv}, invest_frac={self.invest_frac}, "
                     f"horizon_days={self.horizon_days}")

    def on_trading_iteration(self):
        today = self.get_datetime().date()
        ts = pd.Timestamp(today)

        for symbol in self.symbols:
            pos = self.get_position(symbol)
            qty = pos.quantity if pos else 0

            # 1) if long and horizon passed → sell
            if qty > 0:
                entry = self.entry_dates.get(symbol)
                if entry and ts >= entry + pd.Timedelta(days=self.horizon_days):
                    logger.error(f"{symbol}: horizon passed → SELL {qty}")
                    order = self.create_order(symbol, qty, "sell", type="market")
                    self.submit_order(order)
                    self.entry_dates.pop(symbol, None)
                continue

            # 2) if flat and signal==True → buy
            if ts in self.signals.index and self.signals.loc[ts]:
                cash = self.get_cash()
                price = self.get_last_price(symbol)
                invest = cash * self.invest_frac
                buy_qty = int(invest // price)
                if buy_qty > 0:
                    logger.error(f"{symbol}: signal=TRUE → BUY {buy_qty}")
                    order = self.create_order(symbol, buy_qty, "buy", type="market")
                    self.submit_order(order)
                    self.entry_dates[symbol] = ts
                else:
                    logger.error(f"{symbol}: signal=TRUE but insufficient cash ({cash})")

# ─── Entrypoint ─────────────────────────────────────────────────────
if __name__=="__main__":
    from datetime import datetime

    broker = Alpaca({
        "API_KEY":    os.getenv("ALPACA_API_KEY"),
        "API_SECRET": os.getenv("ALPACA_API_SECRET"),
        "PAPER":      True
    })

    strat = AnalogBacktestStrategy(
        name="AnalogBT",
        broker=broker,
        benchmark="NVDA",
        parameters={
            "symbols":             ["NVDA"],
            "analog_csv":          "nvidia_analogs/data/analog_full_backtest2.csv",
            "investment_fraction": 0.5,
            "horizon_days":        5
        },
        debug=True
    )

    strat.backtest(
        YahooDataBacktesting,
        datetime(2020, 4, 20),
        datetime(2025, 4, 11)
    )
