# ingest_to_pinecone.py

import os
import sys
from pathlib import Path
import pandas as pd
from pinecone import Pinecone, ServerlessSpec

# ─── make sure our project root is on sys.path ────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # e.g. MLTradingBot/nvidia_analogs
sys.path.insert(0, str(PROJECT_ROOT.parent))  # MLTradingBot

from nvidia_analogs.config import PN_API_KEY, PN_ENV, INDEX_NAME

# ─── where our feature parquet files live ─────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

TICKERS = ["NVDA","AAPL","MSFT","TSLA","GOOGL","AMZN"]

# ─── infer vector dimension from the first ticker ─────────────────────────────
sample_path = DATA_DIR / f"{TICKERS[0]}_features.parquet"
if not sample_path.exists():
    raise FileNotFoundError(f"Expected feature file not found: {sample_path}")
sample = pd.read_parquet(sample_path)
dim = sample.drop(columns=["date"]).shape[1]

# ─── initialize Pinecone ──────────────────────────────────────────────────────
pc = Pinecone(api_key=PN_API_KEY, environment=PN_ENV)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PN_ENV)
    )

index = pc.Index(INDEX_NAME)

# ─── upsert each ticker’s features into its own namespace ─────────────────────
for sym in TICKERS:
    feat_path = DATA_DIR / f"{sym}_features.parquet"
    if not feat_path.exists():
        print(f"⚠️  Skipping {sym}, feature file not found at {feat_path}")
        continue

    df = pd.read_parquet(feat_path)
    vectors = [
        (str(row["date"]), row.drop("date").tolist(), {"date": str(row["date"])})
        for _, row in df.iterrows()
    ]

    # batch‐upsert into the symbol’s namespace
    for i in range(0, len(vectors), 100):
        batch = vectors[i : i + 100]
        index.upsert(vectors=batch, namespace=sym)

    print(f"✔️  Upserted {len(vectors)} vectors for {sym} in namespace “{sym}”")

print("🎉  All tickers ingested!")
