# inspect_features.py
import pandas as pd

# 1) Load the pickle
df = pd.read_parquet("../data/nvidia_features.parquet")

# 2) List out the macro & sentiment columns
macro_cols = ['CPI','UNRATE','FEDFUNDS']#,'PCEPI','M2SL','GS10','VIXCLS']
all_cols   = ['date'] + macro_cols + ['news_sentiment']

# 3) Show the first 10 rows
print("=== Feature Preview ===")
print(df[all_cols].head(10).to_string(index=False))

# 4) Show summary statistics for those columns
print("\n=== Summary Statistics ===")
print(df[macro_cols + ['news_sentiment']].describe().transpose().to_string())
