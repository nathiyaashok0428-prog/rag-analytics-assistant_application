from collections import Counter
import re


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "were", "have", "had",
    "was", "are", "but", "not", "you", "your", "our", "they", "them", "their",
    "would", "could", "should", "about", "into", "after", "before", "during",
    "because", "while", "there", "here", "when", "where", "what", "which",
    "who", "whom", "been", "being", "very", "more", "most", "much", "many",
    "than", "then", "only", "just", "also", "still", "over", "under", "again",
    "product", "products", "customer", "customers", "review", "reviews", "order",
    "orders", "store", "seller", "item", "items", "service", "company", "client",
}

THEME_KEYWORDS = {
    "delivery": {"delivery", "shipping", "delay", "late", "arrived", "courier"},
    "quality": {"quality", "damaged", "broken", "defect", "defective", "poor"},
    "support": {"support", "service", "response", "contact", "help"},
    "packaging": {"packaging", "package", "box", "packed"},
    "refund": {"refund", "return", "exchange", "cancel", "cancellation"},
    "price": {"price", "cost", "expensive", "cheap", "value"},
    "satisfaction": {"satisfaction", "satisfied", "happy", "unhappy", "disappointed"},
}


def extract_themes(reviews, top_k=3):
    if not reviews:
        return []

    joined = " ".join(review.lower() for review in reviews)

    matched_themes = []
    for theme, keywords in THEME_KEYWORDS.items():
        score = sum(joined.count(keyword) for keyword in keywords)
        if score > 0:
            matched_themes.append((theme, score))

    if matched_themes:
        matched_themes.sort(key=lambda item: item[1], reverse=True)
        return [theme for theme, _ in matched_themes[:top_k]]

    tokens = re.findall(r"[a-zA-Z]{4,}", joined)
    counts = Counter(token for token in tokens if token not in STOPWORDS)
    return [word for word, _ in counts.most_common(top_k)]
