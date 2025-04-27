# sentiment_utils.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple

# ─── Model Setup ─────────────────────────────────────────────────────
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

labels = ["positive", "negative", "neutral"]

# ─── Estimate Sentiment ──────────────────────────────────────────────
def estimate_sentiment(texts: List[str]) -> Tuple[float, str]:
    """
    Given a list of news texts, returns (probability, sentiment).
    """
    if not texts:
        return 0.0, "neutral"
    
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]

    summed_logits = torch.sum(logits, dim=0)
    probs = torch.nn.functional.softmax(summed_logits, dim=-1)

    sentiment_idx = torch.argmax(probs)
    probability = probs[sentiment_idx].item()
    sentiment = labels[sentiment_idx]
    
    return probability, sentiment

# ─── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment([
        "markets responded negatively to the news!",
        "traders were displeased!"
    ])
    print(tensor, sentiment)
    print("CUDA Available:", torch.cuda.is_available())
