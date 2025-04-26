import pandas as pd
import talib as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
from fetch_macro import fetch_macro
from fetch_sentiment import build_sentiment_features

# list of tickers (used externally)
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

    df['news_sentiment'] = 0.0


    # 4) Drop any rows missing core price/techs
    core = ['Open','High','Low','Close','Volume','SMA_20','RSI_14','MACD']
    df = df.dropna(subset=core).reset_index(drop=True)

    # 5) Build & scale
    features = core + macro_cols + ['news_sentiment']
    X = df[features].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    Xs = np.nan_to_num(Xs)

    out = pd.DataFrame(Xs, columns=features)
    out['date'] = df['date']
    return out

if __name__ == '__main__':
    import os
    for sym in TICKERS:
        path = os.path.join('../data', f'{sym}_prices.csv')
        prices = pd.read_csv(path, parse_dates=['date'])
        feat   = compute_features(prices)
        feat.to_parquet(f'../data/{sym}_features.parquet', index=False)
        print(f"✓ {sym} → {len(feat)} feature rows")
