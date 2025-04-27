import pandas as pd
import talib as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
from nvidia_analogs.DataServices.fetch_macro import fetch_macro
from nvidia_analogs.DataServices.build_sentiment_features import fetch_sentiment_for_dates
import sys
from pathlib import Path

# list of tickers (used externally)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # e.g. MLTradingBot/nvidia_analogs
sys.path.insert(0, str(PROJECT_ROOT.parent))  # MLTradingBot


# ─── where our feature parquet files live ─────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
PCSV_DIR = DATA_DIR / "pricesCSV"
FEATURES_DIR = DATA_DIR / "featuresParquets"


TICKERS = ["NVDA","AAPL","MSFT","TSLA","GOOGL","AMZN"]

def compute_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    # ensure 'date' is a naive datetime index at midnight
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(None).dt.normalize()

    # 1) Technical indicators
    c = df['Close'].values
    df['SMA_20'] = ta.SMA(c, timeperiod=20)
    df['RSI_14'] = ta.RSI(c, timeperiod=14)
    macd, _, _  = ta.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD']   = macd

    # 2) Fetch & merge macro
    start = df['date'].min().strftime('%Y-%m-%d')
    end   = df['date'].max().strftime('%Y-%m-%d')
    macro = fetch_macro(start, end)
    macro['date'] = pd.to_datetime(macro['date'], utc=True).dt.tz_convert(None).dt.normalize()
    df = df.merge(macro, on='date', how='left')
    macro_cols = [col for col in macro.columns if col != 'date']
    for col in macro_cols:
        df[col] = df[col].ffill().bfill()

    # 3) Fetch & merge sentiment
    #sent = build_sentiment_features(df['date'])
    #sent['date'] = pd.to_datetime(sent['date'], utc=True).dt.tz_convert(None).dt.normalize()
    #df = df.merge(sent, on='date', how='left').fillna({'news_sentiment': 0.0})

    #df['news_sentiment'] = 0.0

    sent = fetch_sentiment_for_dates(symbol=sym, dates=df['date'])
    sent['date'] = pd.to_datetime(sent['date'], utc=True).dt.tz_convert(None).dt.normalize()
    df = df.merge(sent, on='date', how='left').fillna({'news_sentiment_prob': 0.0, 'news_sentiment_label': 0})
    print(f"✓ Fetched sentiment for {len(sent)} dates")
    print(f"✓ Merged sentiment for {len(df)} rows")


    # 4) Drop any rows missing core price/techs
    core = ['Open','High','Low','Close','Volume','SMA_20','RSI_14','MACD']
    df = df.dropna(subset=core).reset_index(drop=True)

    # 5) Build & scale
    features = core + macro_cols + ['news_sentiment_prob', 'news_sentiment_label']
    X = df[features].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    Xs = np.nan_to_num(Xs)

    out = pd.DataFrame(Xs, columns=features)
    out['date'] = df['date']
    return out

if __name__ == '__main__':
    import os
    for sym in TICKERS:
        path = PCSV_DIR / f"{sym}_prices.csv"
        prices = pd.read_csv(path, parse_dates=['date'])
        feat   = compute_features(prices)
        out_path = FEATURES_DIR / f"{sym}_features_with_sentiment.parquet"
        feat.to_parquet(out_path, index=False)        
        print(f"✓ {sym} → {len(feat)} feature rows")
