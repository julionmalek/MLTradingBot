# fetch_prices.py
import yfinance as yf
import pandas as pd

# ←— add as many tickers as you like here
TICKERS = ["NVDA","AAPL","MSFT","TSLA","GOOGL","AMZN"]

def fetch_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(start=start_date, end=end_date, interval="1d")
    df = df[["Open","High","Low","Close","Volume"]].reset_index()
    df.rename(columns={"Date":"date"}, inplace=True)
    return df

if __name__=="__main__":
    START="2010-01-01"
    END  ="2025-04-23"
    for sym in TICKERS:
        df = fetch_prices(sym, START, END)
        df.to_csv(f"../data/{sym}_prices.csv", index=False)
        print(f"✓ Fetched {sym}: {len(df)} rows")
