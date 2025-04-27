# test_sentiment_from_news.py

import os
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from nvidia_analogs.DataServices.sentiment_utils import estimate_sentiment  # ← from the module we made

# ─── Alpaca API Setup ───────────────────────────────────────────────
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

# ─── Functions ──────────────────────────────────────────────────────
def get_dates():
    today = datetime.now()
    three_days_prior = today - Timedelta(days=3)
    return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

def get_sentiment(symbol="NVDA"):
    today, three_days_prior = get_dates()
    try:
        news_items = api.get_news(
            symbol=symbol,
            start=three_days_prior,
            end=today
        )
        headlines = [n.__dict__["_raw"]["headline"] for n in news_items]
        print(f"Fetched {len(headlines)} headlines for {symbol}")
        
        probability, sentiment = estimate_sentiment(headlines)
        return probability, sentiment

    except Exception as e:
        print(f"Error fetching news: {e}")
        return 0.0, "neutral"

# ─── Main Execution ─────────────────────────────────────────────────
if __name__ == "__main__":
    symbol = "NVDA"  # You can change this to any stock symbol
    probability, sentiment = get_sentiment(symbol)
    print(f"Sentiment for {symbol}: {sentiment} ({probability:.2f})")
