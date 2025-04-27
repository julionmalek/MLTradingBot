# build_sentiment_features.py

import os
import pandas as pd
from datetime import datetime
from timedelta import Timedelta
from alpaca_trade_api import REST
from nvidia_analogs.DataServices.sentiment_utils import estimate_sentiment

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

def fetch_sentiment_for_dates(symbol: str, dates: pd.Series) -> pd.DataFrame:
    """
    Fetch news sentiment for a list of dates and a symbol.
    Returns a DataFrame with date, sentiment probability, and sentiment label (as int).
    """
    recs = []
    for date in dates:
        today = pd.to_datetime(date).normalize()
        three_days_prior = today - Timedelta(days=3)
        print(f"Fetching sentiment for {symbol} from {three_days_prior} to {today}")

        try:
            news_items = api.get_news(
                symbol=symbol,
                start=three_days_prior.strftime('%Y-%m-%d'),
                end=today.strftime('%Y-%m-%d')
            )
            headlines = [n.__dict__["_raw"]["headline"] for n in news_items]
            probability, sentiment = estimate_sentiment(headlines)

            sentiment_label = {
                "positive": 1,
                "neutral": 0,
                "negative": -1
            }.get(sentiment.lower(), 0)

            recs.append({
                "date": today,
                "news_sentiment_prob": probability,
                "news_sentiment_label": sentiment_label
            })

        except Exception as e:
            print(f"Error fetching sentiment for {symbol} on {today}: {e}")
            recs.append({
                "date": today,
                "news_sentiment_prob": 0.0,
                "news_sentiment_label": 0
            })

    return pd.DataFrame(recs)
