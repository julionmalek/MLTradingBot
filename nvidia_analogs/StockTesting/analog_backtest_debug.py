# analog_backtest_debug.py
import pandas as pd
from pinecone import Pinecone
from datetime import timedelta

from config import PN_API_KEY, PN_ENV, INDEX_NAME
from nvidia_analogs.streamlitFrontend.compute_forward_returns import compute_forward_returns

# 1) Load features & prices
feat_df = pd.read_parquet("data/nvidia_features.parquet")
feat_df['date'] = pd.to_datetime(feat_df['date']).dt.normalize()

price_df = (
    pd.read_csv("data/nvidia_prices.csv", parse_dates=['date'])
      .set_index('date')
      .sort_index()
)
# fix: parse index as UTC, drop tz, then normalize
price_df.index = (
    pd.to_datetime(price_df.index, utc=True)
      .tz_convert(None)
      .normalize()
)

# 2) Pinecone client
pc    = Pinecone(api_key=PN_API_KEY, environment=PN_ENV)
index = pc.Index(INDEX_NAME)

# Backtest params
TOP_K         = 10
HORIZON_DAYS  = 5
PROFIT_THRESH = 0.03

records = []

# 3) Loop over all possible “query” dates
for query_date in feat_df['date']:
    # ensure we can find price0 at or before query_date
    past = price_df.loc[price_df.index <= query_date, 'Close']
    if past.empty:
        continue
    price0 = past.iloc[-1]

    # build & run the analog query
    vec = feat_df.loc[feat_df['date'] == query_date] \
                 .drop(columns='date').iloc[0].tolist()
    resp = index.query(vector=vec, top_k=TOP_K, include_metadata=True)
    analog_dates = [pd.to_datetime(m.metadata['date']).normalize()
                    for m in resp.matches]

    # average analog forward return
    df_ret = compute_forward_returns(analog_dates, horizons=[HORIZON_DAYS])
    if df_ret.empty:
        continue
    avg_analog_ret = df_ret['return_pct'].mean() / 100.0

    # find price1 at or after query_date + horizon
    cutoff = query_date + timedelta(days=HORIZON_DAYS)
    future = price_df.loc[price_df.index >= cutoff, 'Close']
    if future.empty:
        continue
    price1 = future.iloc[0]
    actual_ret = (price1 / price0) - 1.0

    records.append({
        'date':           query_date,
        'avg_analog_ret': avg_analog_ret,
        'actual_ret':     actual_ret,
        'signal':         avg_analog_ret >= PROFIT_THRESH
    })

# 4) assemble results
results = pd.DataFrame(records)
if not results.empty:
    results.set_index('date', inplace=True)
    print("Overall correlation (analog → actual):",
          results['avg_analog_ret'].corr(results['actual_ret']))
    print(f"\nWhen signal==True:\n{results[results['signal']]['actual_ret'].describe()}")
    print(f"\nWhen signal==False:\n{results[~results['signal']]['actual_ret'].describe()}")
    results.to_csv("data/analog_full_backtest2.csv")
    print("▶️  Wrote data/analog_full_backtest2.csv")
else:
    print("⚠️  No backtest records generated—check your data windows.")
