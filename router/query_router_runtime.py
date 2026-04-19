import re

import requests

from llm_runtime import get_ollama_model, get_ollama_url

OLLAMA_URL = get_ollama_url()
OLLAMA_MODEL = get_ollama_model()
ALLOWED_ROUTES = {"SQL", "RAG", "HYBRID", "UNKNOWN"}

SQL_KEYWORDS = {
    "sales", "revenue", "orders", "payment", "city", "state", "category",
    "trend", "top", "count", "average", "avg", "monthly", "year", "seller",
    "product", "products", "delay", "volume", "payment type", "order volume",
}
RAG_KEYWORDS = {
    "complaint", "complaints", "review", "reviews", "feedback", "delivery",
    "quality", "packaging", "satisfaction", "unhappy", "issue", "issues",
    "speed", "late", "damaged", "sentiment", "customer unhappy",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _looks_meaningful(query: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", query)
    if len(tokens) < 2:
        return False
    return any(len(token) > 2 for token in tokens)


def _keyword_route(user_query: str) -> str:
    query = _normalize(user_query)
    if not _looks_meaningful(query):
        return "UNKNOWN"

    sql_hits = sum(keyword in query for keyword in SQL_KEYWORDS)
    rag_hits = sum(keyword in query for keyword in RAG_KEYWORDS)

    if sql_hits and rag_hits:
        return "HYBRID"
    if rag_hits:
        return "RAG"
    if sql_hits:
        return "SQL"
    return "UNKNOWN"


def _ollama_route(user_query: str) -> str:
    prompt = f"""
You are an intelligent query classifier.

Classify the user query into:
SQL -> structured analytics queries
RAG -> customer feedback queries
HYBRID -> needs both analytics and review feedback
UNKNOWN -> if query is unclear or not meaningful

Return ONLY one word:
SQL
RAG
HYBRID
UNKNOWN

User Query:
{user_query}
"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=30,
    )
    result = response.json()["response"].strip().upper()
    return result if result in ALLOWED_ROUTES else "UNKNOWN"


def route_query(user_query):
    keyword_result = _keyword_route(user_query)
    if keyword_result != "UNKNOWN":
        return keyword_result

    try:
        return _ollama_route(user_query)
    except Exception as error:
        print("Router Error:", error)
        return keyword_result


def extract_subquery(user_query, subquery_type):
    keyword_route = _keyword_route(user_query)

    if subquery_type == "SQL":
        if keyword_route == "HYBRID":
            return user_query
        instruction = (
            "Rewrite the user's request as only the structured analytics part that should be answered with SQL. "
            "Keep metrics, dimensions, and filters. Remove review-summary wording. "
            "For hybrid questions, ask for numeric business metrics only, such as sales, revenue, counts, averages, or ratings by category, city, state, or time. "
            "Do not ask for raw review text, concatenated comments, complaint excerpts, or sentiment explanations in the SQL sub-question."
        )
    else:
        if keyword_route in {"RAG", "UNKNOWN"}:
            return user_query
        instruction = (
            "Rewrite the user's request as only the customer review analysis part that should be answered from review text. "
            "Keep complaint, satisfaction, sentiment, feedback, quality, and delivery wording."
        )

    prompt = f"""
You are helping decompose a business analytics question.

{instruction}

Rules:
- Return only one short rewritten query
- Return plain English only
- Do not write SQL
- Do not explain
- Do not add quotes

Original Question:
{user_query}

Rewritten Query:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        result = response.json()["response"].strip()
        return result or user_query
    except Exception:
        return user_query
