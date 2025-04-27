import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")  # always try to load it
# Functions to fetch fresh env vars
def get_pinecone_api_key():
    return os.getenv("PINECONE_API_KEY")

def get_pinecone_env():
    return os.getenv("PINECONE_ENV")


PN_API_KEY = os.getenv("PINECONE_API_KEY")
PN_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "nvidia-analogs"
INDEX_NAME2 = "nvidia-analogs2"

# Macroeconomic data (FRED)
FRED_SOURCE = "fred"

# News Sentiment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# FinBERT model for sentiment
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")