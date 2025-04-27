#!/usr/bin/env python3
"""
analog_backtest_multi.py

For each ticker in TICKERS:
  - Load its feature vectors (data/{TICKER}_features.parquet)
  - Load its prices      (data/{TICKER}_prices.csv)
  - Query Pinecone (namespace=TICKER) for top-K analog dates *before* each query date
  - Compute avg analog forward return vs actual forward return
  - Emit summary and write data/{TICKER}_backtest.csv
"""

import pandas as pd
from pinecone import Pinecone
from datetime import timedelta, datetime
from nvidia_analogs.config import PN_API_KEY, PN_ENV, INDEX_NAME2
from nvidia_analogs.streamlitFrontend.compute_forward_returns import compute_forward_returns
from pathlib import Path

PN_API_KEY = "pcsk_36gTWZ_M3B4d5VeAZmn1Gt2jymGyX7uQAacupuiDD5EtnKmfx6AkP9UgGNxsU6zLXNPNco"
PN_ENV    = "us-east1"  # e.g. "us-west1-gcp", "us-east1-gcp"


SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # e.g. MLTradingBot/nvidia_analogs

# ─── Configuration ────────────────────────────────────────────────────────────
#TICKERS      = ["NVDA","AAPL","MSFT","TSLA","GOOGL","AMZN"]
TICKERS      = ["NVDA"]
TOP_K        = 10
HORIZON_DAYS = 5
PROFIT_THRESH= 0.03  # 3% analog threshold
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR      = PROJECT_ROOT / "data" / "AnalogFullBacktests"
FEAT_DIR = DATA_DIR / "featuresParquets"
PRICE_DIR= DATA_DIR / "pricesCSV"

# if you didn’t actually split into subdirs, fall back to root data/
if not FEAT_DIR.exists():  FEAT_DIR  = DATA_DIR
if not PRICE_DIR.exists(): PRICE_DIR = DATA_DIR

# ─── Pinecone setup ───────────────────────────────────────────────────────────
pc    = Pinecone(api_key=PN_API_KEY, environment=PN_ENV)
index = pc.Index(INDEX_NAME2)

# ─── Backtest per‐ticker ───────────────────────────────────────────────────────
for ticker in TICKERS:
    print(f"\n=== Backtesting {ticker} ===")

    # 1) load features & prices
    feat_path  = FEAT_DIR  / f"{ticker}_features_with_sentiment.parquet"
    price_path = PRICE_DIR / f"{ticker}_prices.csv"

    if not feat_path.exists() or not price_path.exists():
        print(f" ⚠️  Missing files for {ticker} → {feat_path} or {price_path}")
        continue

    feat_df = pd.read_parquet(feat_path)
    feat_df["date"] = pd.to_datetime(feat_df["date"]).dt.normalize()

    price_df = (
        pd.read_csv(price_path, parse_dates=["date"])
          .set_index("date")
          .sort_index()
    )
    price_df.index = (
        pd.to_datetime(price_df.index, utc=True)
          .tz_convert(None)
          .normalize()
    )

    records = []

    # 2) loop all query dates
    for query_date in feat_df["date"]:
        # 2a) price0 on or before query_date
        prior = price_df.loc[price_df.index <= query_date, "Close"]
        if prior.empty:
            continue
        price0 = prior.iloc[-1]

        # 2b) build & query Pinecone (overfetch to allow filtering)
        vec = feat_df.loc[feat_df["date"] == query_date].drop(columns="date").iloc[0].tolist()
        resp = index.query(
            vector=vec,
            top_k=TOP_K*2,
            include_metadata=True,
            namespace=ticker
        )

        # 2c) keep only matches *before* query_date
        good = []
        for m in resp.matches:
            adate = pd.to_datetime(m.metadata["date"]).normalize()
            if adate < query_date:
                good.append((adate, m.score))
                if len(good) >= TOP_K:
                    break
        if not good:
            continue

        analog_dates = [d for d,_ in good]

        # 3) avg analog forward return
        df_ret = compute_forward_returns(analog_dates, horizons=[HORIZON_DAYS])
        if df_ret.empty:
            continue
        avg_analog_ret = df_ret["return_pct"].mean() / 100.0

        # 4) actual forward return
        cutoff = query_date + timedelta(days=HORIZON_DAYS)
        future = price_df.loc[price_df.index >= cutoff, "Close"]
        if future.empty:
            continue
        price1     = future.iloc[0]
        actual_ret = (price1 / price0) - 1.0

        # 5) record
        records.append({
            "date":           query_date,
            "avg_analog_ret": avg_analog_ret,
            "actual_ret":     actual_ret,
            "signal":         avg_analog_ret >= PROFIT_THRESH
        })

    # 6) assemble & dump
    if not records:
        print(f" ⚠️  No backtest records for {ticker}.")
        continue

    results = pd.DataFrame(records).set_index("date").sort_index()
    out_csv = OUT_DIR / f"{ticker}_analog_backtest2.csv"
    results.to_csv(out_csv)
    print(f" ✔️  Wrote {out_csv}")
    print("    • Corr(analog,actual):", results["avg_analog_ret"].corr(results["actual_ret"]))
    print("    • Signal==True stats:\n",  results[results["signal"]]["actual_ret"].describe(), sep="")
    print("    • Signal==False stats:\n", results[~results["signal"]]["actual_ret"].describe(), sep="")
