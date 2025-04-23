import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta

from pinecone import Pinecone
from config import PN_API_KEY, PN_ENV, INDEX_NAME
from fetch_macro import fetch_macro
from fetch_sentiment import build_sentiment_features
from compute_forward_returns import compute_forward_returns
from visualize_analogs import (
    plot_similarity,
    plot_forward_returns,
    plot_feature_profiles
)
from datetime import date as dt_date, timedelta


st.title("NVDA Analog Explorer")

# --- Helper to build today's feature vector ----------------
def build_today_vector(ticker="NVDA", lookback_days=365):
    # 1) fetch enough history for technicals
    hist = yf.Ticker(ticker).history(period=f"{lookback_days}d")
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    hist = hist[['Open','High','Low','Close','Volume']]\
           .reset_index().rename(columns={'Date':'date'})

    close = hist['Close'].values
    sma20 = ta.SMA(close, timeperiod=20)[-1]
    rsi14 = ta.RSI(close, timeperiod=14)[-1]
    macd,_,_ = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_val = macd[-1]

    if np.isnan(sma20) or np.isnan(rsi14) or np.isnan(macd_val):
        raise RuntimeError("Insufficient history to compute technicals – increase lookback_days")

    # 2) macro up through today
    today = hist['date'].iloc[-1]
    start = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    macro = fetch_macro(start, end)
    macro['date'] = pd.to_datetime(macro['date']).dt.normalize()

    exact = macro[macro['date'] == today]
    if not exact.empty:
        row_macro = exact.iloc[0]
    else:
        prev = macro[macro['date'] < today]
        row_macro = prev.iloc[-1]  # last available

    macro_vals = row_macro.drop('date').tolist()

    # 3) sentiment
    sent = build_sentiment_features([today])['news_sentiment'].iloc[0]

    # 4) assemble
    vec = [
        hist['Open'].iloc[-1], hist['High'].iloc[-1],
        hist['Low'].iloc[-1],  hist['Close'].iloc[-1],
        hist['Volume'].iloc[-1],
        sma20, rsi14, macd_val
    ] + macro_vals + [sent]

    return vec, today


# --- Load historical features --------------------------------
feat_df = pd.read_parquet("data/nvidia_features.parquet")
feat_df["date"] = pd.to_datetime(feat_df["date"]).dt.normalize()
hist_dates = feat_df["date"].dt.date
min_date, max_date = hist_dates.min(), hist_dates.max()

# --- Sidebar controls ----------------------------------------
st.sidebar.header("Settings")
use_live = st.sidebar.checkbox(
    "Use Today's Live Data",
    value=True
)

if use_live:
    selected_date = dt_date.today()
else:
    selected_date = st.sidebar.date_input(
        "Pick analysis date", value=max_date,
        min_value=min_date, max_value=max_date
    )

top_k   = st.sidebar.slider("Number of analogs (top‑K)", 1, 50, 10)
horizons= st.sidebar.multiselect(
    "Forward‑return horizons (days)",
    [1,5,20,60,120], default=[1,5,20]
)

# --- Build query vector --------------------------------------
if use_live:
    query_vec, actual_date = build_today_vector()
    st.info(f"🔎 Running live analysis for **{actual_date.date()}**")
    FEATURE_NAMES = [
    'Open','High','Low','Close','Volume',
    'SMA_20','RSI_14','MACD',
    'CPI','UNRATE','FEDFUNDS','PCEPI','M2SL','GS10','VIXCLS',
    'news_sentiment'
    ]
    st.subheader("Today's Feature Data")
    today_df = pd.DataFrame([query_vec], columns=FEATURE_NAMES)
    today_df['date'] = actual_date
    # reorder to put date first
    today_df = today_df[['date'] + FEATURE_NAMES]
    st.dataframe(today_df)
else:
    if selected_date not in hist_dates.values:
        st.error(f"No feature data for {selected_date}.")
        st.stop()
    row = feat_df[feat_df["date"].dt.date == selected_date].iloc[0]
    query_vec = row.drop("date").tolist()
    actual_date = selected_date
    st.subheader(f"Backtest analysis for {selected_date}")

# --- Pinecone query ------------------------------------------
pc    = Pinecone(api_key=PN_API_KEY, environment=PN_ENV)
index = pc.Index(INDEX_NAME)
# ensure plain floats
query_vec = [float(v) for v in query_vec]
resp = index.query(vector=query_vec, top_k=top_k, include_metadata=True)

analog_dates = [
    pd.to_datetime(m.metadata["date"]).normalize()
    for m in resp.matches
]
sims = [m.score for m in resp.matches]

# --- Similarity plot -----------------------------------------
st.subheader("Similarity Scores")
st.pyplot(plot_similarity(analog_dates, sims))

# --- Forward returns ----------------------------------------
returns_df = compute_forward_returns(analog_dates, horizons=horizons)
st.subheader("Forward‑Return Data")
st.dataframe(returns_df)

# aggregate stats
if not returns_df.empty:
    summary = returns_df.groupby("horizon")["return_pct"].agg(
        mean="mean", std="std", median="median", min="min", max="max"
    ).reset_index()
    st.subheader("Aggregated Forward‑Return Statistics")
    st.dataframe(
        summary.style.format({
            "mean":"{:.2f}%", "std":"{:.2f}%",
            "median":"{:.2f}%", "min":"{:.2f}%", "max":"{:.2f}%"
        })
    )

# forward-return charts
st.subheader("Forward‑Return Bar Charts")
for fig in plot_forward_returns(returns_df):
    st.pyplot(fig)

# --- Feature table & profiles --------------------------------
st.subheader("Feature Vectors (query + analogs)")
table_df = feat_df.set_index("date")
selected_ts = [pd.Timestamp(actual_date)] + analog_dates
mask = table_df.index.normalize().isin(selected_ts)
display = table_df.loc[mask]
st.dataframe(display)

st.subheader("Feature Profiles")
feature_names = display.columns.tolist()
vectors = [display.iloc[0].tolist()] + [display.iloc[i].tolist() for i in range(1,len(display))]
st.pyplot(plot_feature_profiles(vectors[0], vectors[1:], feature_names))
