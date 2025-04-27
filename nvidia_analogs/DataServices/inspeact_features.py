# inspect_features.py
import pandas as pd
import sys
from pathlib import Path

# ─── make sure our project root is on sys.path ────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # e.g. MLTradingBot/nvidia_analogs
sys.path.insert(0, str(PROJECT_ROOT.parent))  # MLTradingBot


TICKERS = ["NVDA","AAPL","MSFT","TSLA","GOOGL","AMZN"]


# ─── where our feature parquet files live ─────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
FPARQUETS_DIR = DATA_DIR / "featuresParquets"



# 1) Load the pickle
path = FPARQUETS_DIR / f"NVDA_features_with_sentiment.parquet"
df = pd.read_parquet(path)

# 2) List out the macro & sentiment columns
macro_cols = ['CPI','UNRATE','FEDFUNDS']#,'PCEPI','M2SL','GS10','VIXCLS']
all_cols   = ['date'] + macro_cols + ['news_sentiment_prob', 'news_sentiment_label']

# 3) Show the first 10 rows
print("=== Feature Preview ===")
print(df[all_cols].head(100).to_string(index=False))

# 4) Show summary statistics for those columns
print("\n=== Summary Statistics ===")
print(df[macro_cols + ['news_sentiment_prob', 'news_sentiment_label']].describe().transpose().to_string())
