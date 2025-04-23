import os
import pandas as pd
import requests
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import timedelta

load_dotenv()

NEWS_API_KEY  = os.getenv("NEWS_API_KEY")
NEWS_API_URL  = "https://newsapi.org/v2/everything"
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")

# one‑time init
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def score_texts(texts: list[str]) -> float:
    if not texts:
        print("  [score_texts] no texts → 0.0")
        return 0.0
    try:
        results = sentiment_pipeline(texts)
    except Exception as e:
        print(f"  [score_texts] pipeline error: {e}")
        return 0.0

    vals = []
    for r in results:
        lbl = r['label'].lower()
        sc  = r['score']
        vals.append(sc if 'positive' in lbl else -sc)
    avg = sum(vals)/len(vals) if vals else 0.0
    print(f"  [score_texts] {len(vals)} scores → avg {avg:.4f}")
    return avg

def build_sentiment_features(dates: list[pd.Timestamp]) -> pd.DataFrame:
    recs = []
    print(f"[fetch_sentiment] fetching for {len(dates)} dates")
    for d in dates:
        day = pd.to_datetime(d).normalize()
        frm = day.strftime("%Y-%m-%d")
        to  = (day + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"[fetch_sentiment] {frm}…", end="")

        params = {
            'q':'NVIDIA OR NVDA',
            'from':frm,'to':to,
            'language':'en','pageSize':100,
            'apiKey':NEWS_API_KEY
        }
        try:
            r = requests.get(NEWS_API_URL, params=params, timeout=10)
            arts = r.json().get('articles',[])
            print(f" {len(arts)} articles", end="")
        except Exception as e:
            print(f" API error: {e}", end="")
            arts = []

        texts = [a.get('title','') + '. ' + a.get('description','') for a in arts]
        score = score_texts(texts)
        recs.append({'date':day,'news_sentiment':score})

    df = pd.DataFrame(recs)
    print(f"\n[fetch_sentiment] done → {len(df)} rows")
    return df
