import yfinance as yf
import pandas as pd

def fetch_nvidia(start_date: str, end_date: str) -> pd.DataFrame:
    ticker = yf.Ticker("NVDA")
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.reset_index(inplace=True)
    df.rename(columns={'Date':'date'}, inplace=True)
    return df

if __name__ == "__main__":
    df = fetch_nvidia("2022-04-20", "2025-04-21")
    df.to_csv("data/nvidia_prices.csv", index=False)
    print("Fetched and saved NVDA data.")