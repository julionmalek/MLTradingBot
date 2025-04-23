import os
from dotenv import load_dotenv
load_dotenv()

PN_API_KEY = os.getenv("PINECONE_API_KEY")
PN_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "nvidia-analogs"

# Macroeconomic data (FRED)
FRED_SOURCE = "fred"

# News Sentiment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# FinBERT model for sentiment
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")