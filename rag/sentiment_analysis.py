# =========================================
# SENTIMENT ANALYSIS USING OLLAMA
# =========================================

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
ALLOWED_SENTIMENTS = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
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


# =========================================
# ANALYZE SENTIMENT
# =========================================

def analyze_sentiment(review_text):

    prompt = f"""
Classify the sentiment of this customer review.

Return ONLY one word:

Positive
Negative
Neutral

Do not return any other label such as Mixed, Improvement, Complaint, Good, or Bad.

Review:
{review_text}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    sentiment = result["response"].strip()

    return normalize_sentiment(sentiment)


# =========================================
# BATCH SENTIMENT
# =========================================

def batch_sentiment(reviews):

    sentiments = []

    for r in reviews:

        sentiment = analyze_sentiment(r)

        sentiments.append(sentiment)

    return sentiments


# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    test_reviews = [

        "Delivery was late and product damaged",

        "Excellent quality and fast shipping",

        "Product arrived but packaging average"

    ]

    results = batch_sentiment(test_reviews)

    print("\nSentiment Results:\n")

    for r, s in zip(test_reviews, results):

        print(f"{r} → {s}")
