import pandas as pd
import talib as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
from fetch_macro import fetch_macro
from fetch_sentiment import build_sentiment_features

def compute_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(None).dt.normalize()

    # 1) Technical indicators
    c = df['Close'].values
    df['SMA_20'] = ta.SMA(c, timeperiod=20)
    df['RSI_14'] = ta.RSI(c, timeperiod=14)
    macd,_,_    = ta.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD']  = macd

    # 2) Fetch & merge macro
    start, end = df['date'].min(), df['date'].max()
    macro = fetch_macro(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    macro['date'] = pd.to_datetime(macro['date']).dt.normalize()
    df = df.merge(macro, on='date', how='left')
    macro_cols = [c for c in macro.columns if c!='date']
    print(f"[compute_features] macro_cols: {macro_cols}")

    # forward/backfill each macro
    for col in macro_cols:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    # 3) Fetch & merge sentiment
    sentiment_df = build_sentiment_features(df['date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()
    df = df.merge(sentiment_df, on='date', how='left')
    df['news_sentiment'] = df['news_sentiment'].fillna(0.0)

    # 4) Drop any rows missing core price/techs
    core = ['Open','High','Low','Close','Volume','SMA_20','RSI_14','MACD']
    df = df.dropna(subset=core).reset_index(drop=True)

    # 5) Build & scale
    features = core + macro_cols + ['news_sentiment']
    print(f"[compute_features] final features: {features}")
    X = df[features].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    Xs = np.nan_to_num(Xs, nan=0.0)

    out = pd.DataFrame(Xs, columns=features)
    out['date'] = df['date']
    print(f"[compute_features] output shape: {out.shape}")
    return out

if __name__ == "__main__":
    prices = pd.read_csv("data/nvidia_prices.csv", parse_dates=['date'])
    feat = compute_features(prices)
    feat.to_parquet("data/nvidia_features.parquet", index=False)
    print("[compute_features] Saved data/nvidia_features.parquet")
