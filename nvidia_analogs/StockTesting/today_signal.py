# today_signal.py

import pandas as pd
import yfinance as yf
import talib as ta

from datetime import date, timedelta
from pinecone import Pinecone
from DataServices.fetch_macro import fetch_macro
from nvidia_analogs.streamlitFrontend.compute_forward_returns import compute_forward_returns
from config import PN_API_KEY, PN_ENV, INDEX_NAME

# ─── Live-build helper ─────────────────────────────────────────
def build_live_vector(ticker="NVDA", lookback_days=365):
    """
    Build today's feature vector from scratch:
      - technicals from yfinance
      - macro from FRED
      - (dummy 0 for sentiment)
    Returns: raw_vec(list), today(pd.Timestamp), row_df(pd.DataFrame)
    """
    # 1) price history for techs
    hist = yf.Ticker(ticker).history(period=f"{lookback_days}d")[['Open','High','Low','Close','Volume']]
    hist.index = pd.to_datetime(hist.index, utc=True).tz_convert(None).normalize()
    hist.index.name = 'date'
    df = hist.reset_index()

    # 2) technicals
    close = df['Close'].values
    sma20 = ta.SMA(close, timeperiod=20)[-1]
    rsi14 = ta.RSI(close, timeperiod=14)[-1]
    macd_v = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0][-1]

    today = df['date'].iloc[-1]

    # 3) macro (forward-fill)
    start = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    macro = fetch_macro(start, end)
    macro['date'] = pd.to_datetime(macro['date']).dt.normalize()
    if today in macro['date'].values:
        mrow = macro.loc[macro['date']==today].iloc[0]
    else:
        mrow = macro[macro['date']<today].iloc[-1]
    mvals = mrow.drop('date').tolist()

    # 4) assemble (append dummy sentiment)
    raw = [
        df['Open'].iloc[-1], df['High'].iloc[-1],
        df['Low'].iloc[-1],  df['Close'].iloc[-1],
        df['Volume'].iloc[-1],
        sma20, rsi14, macd_v
    ] + mvals + [0.0]

    # build a one-row DataFrame for inspection and later persistence
    cols = (
        ['Open','High','Low','Close','Volume','SMA_20','RSI_14','MACD'] +
        list(mrow.drop('date').index) +
        ['news_sentiment']
    )
    row_df = pd.DataFrame([raw], columns=cols, index=[today])
    return raw, today, row_df

# ─── Main ────────────────────────────────────────────────────────
if __name__=="__main__":
    TOP_K, HORIZON_DAYS, PROFIT_THRESH = 10, 5, 0.03
    PARQUET_PATH = "data/nvidia_features.parquet"

    # 1) load precomputed features calendar
    feat_df = (
        pd.read_parquet(PARQUET_PATH)
          .assign(date=lambda df: pd.to_datetime(df['date']).dt.normalize())
    )

    # 2) define today
    today = pd.Timestamp(date.today()).normalize()

    # 3) get or build vector
    if today in feat_df['date'].values:
        print(f"✔️ Found precomputed features for {today.date()}")
        vec = (
            feat_df.loc[feat_df['date']==today]
                   .drop(columns='date')
                   .iloc[0]
                   .tolist()
        )
    else:
        print(f"⚠️ No precomputed features for {today.date()}, building live vector…")
        vec, today, row_df = build_live_vector()

        # **Append to Parquet**  
        row_df_reset = row_df.reset_index().rename(columns={'index':'date'})
        # preserve schema order
        row_df_reset['date'] = pd.to_datetime(row_df_reset['date'])
        feat_df = pd.concat([feat_df, row_df_reset], ignore_index=True)
        feat_df.to_parquet(PARQUET_PATH, index=False)
        print("▶️  Appended live vector to Parquet.")

        # **Upsert** into Pinecone
        pc    = Pinecone(api_key=PN_API_KEY, environment=PN_ENV)
        index = pc.Index(INDEX_NAME)
        # use ISO date as the vector ID
        index.upsert(vectors=[
            (today.isoformat(), [float(x) for x in row_df.iloc[0].tolist()],
             {"date": today.isoformat()})
        ])
        print("▶️  Upserted live vector to Pinecone.")

    # 4) ensure pure Python floats
    vec = [float(x) for x in vec]

    # 5) Pinecone query
    pc    = Pinecone(api_key=PN_API_KEY, environment=PN_ENV)
    index = pc.Index(INDEX_NAME)
    resp  = index.query(vector=vec, top_k=TOP_K, include_metadata=True)

    # 6) parse analog dates
    analog_dates = [
        pd.to_datetime(m.metadata['date'], utc=True)
          .tz_convert(None)
          .normalize()
        for m in resp.matches
    ]

    # 7) compute analog forward returns
    df_ret = compute_forward_returns(analog_dates, horizons=[HORIZON_DAYS])
    if df_ret.empty:
        print("⚠️  No analog forward-return data available.")
        exit(1)
    avg_analog_ret = df_ret['return_pct'].mean() / 100.0

    # 8) compute actual forward return only if data exists
    price_hist = yf.Ticker("NVDA").history(
        start=today.strftime("%Y-%m-%d"),
        end=(today + timedelta(days=HORIZON_DAYS+1)).strftime("%Y-%m-%d")
    )['Close']
    price_hist.index = pd.to_datetime(price_hist.index).tz_localize(None).normalize()

    future_date = today + timedelta(days=HORIZON_DAYS)
    if future_date in price_hist.index:
        actual_ret = price_hist.loc[future_date] / price_hist.loc[today] - 1.0
        actual_str = f"{actual_ret:.2%}"
    else:
        actual_ret = None
        actual_str = "N/A (live)"

    # 9) decide signal
    signal = (avg_analog_ret >= PROFIT_THRESH)

    # 10) report
    print(f"🏷  Analysis for {today.date()}")
    print(f" • Avg analog {HORIZON_DAYS}-day return: {avg_analog_ret:.2%}")
    print(f" • Actual   {HORIZON_DAYS}-day return: {actual_str}")
    print(f" → SIGNAL = {'BUY' if signal else 'NO BUY'} (threshold {PROFIT_THRESH:.2%})")
