import re

import requests

from llm_runtime import get_ollama_model, get_ollama_url

OLLAMA_URL = get_ollama_url()
OLLAMA_MODEL = get_ollama_model()
ALLOWED_SENTIMENTS = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}
POSITIVE_HINTS = {
    "excellent", "great", "good", "fast", "recommend", "satisfied",
    "perfect", "on time", "quality", "love", "happy",
}
NEGATIVE_HINTS = {
    "late", "delay", "damaged", "bad", "poor", "complaint", "unhappy",
    "broken", "wrong", "slow", "issue", "problem", "never again",
}


def normalize_sentiment(raw_sentiment):
    cleaned = (raw_sentiment or "").strip().lower()
    if cleaned in ALLOWED_SENTIMENTS:
        return ALLOWED_SENTIMENTS[cleaned]
    if "positive" in cleaned:
        return "Positive"
    if "negative" in cleaned:
        return "Negative"
    if any(token in cleaned for token in ["neutral", "mixed", "improvement", "suggestion"]):
        return "Neutral"
    return "Neutral"


def _keyword_sentiment(review_text: str) -> str:
    text = re.sub(r"\s+", " ", (review_text or "").strip().lower())
    positive_hits = sum(hint in text for hint in POSITIVE_HINTS)
    negative_hits = sum(hint in text for hint in NEGATIVE_HINTS)

    if negative_hits > positive_hits:
        return "Negative"
    if positive_hits > negative_hits:
        return "Positive"
    return "Neutral"


def analyze_sentiment(review_text):
    prompt = f"""
Classify the sentiment of this customer review.

Return ONLY one word:
Positive
Negative
Neutral

Review:
{review_text}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        result = response.json()
        sentiment = result["response"].strip()
        return normalize_sentiment(sentiment)
    except Exception as error:
        print("Sentiment Error:", error)
        return _keyword_sentiment(review_text)


def batch_sentiment(reviews):
    return [analyze_sentiment(review) for review in reviews]
