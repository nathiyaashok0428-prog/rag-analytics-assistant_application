# =========================================
# HYBRID ANSWER ENGINE
# SQL + RAG + LLM EXPLANATION
# =========================================

import pandas as pd

from sql.nl_to_sql import (
    generate_sql,
    execute_sql
)

from rag.retriever import (
    retrieve_reviews
)

from router.query_router import (
    route_query
)

import requests

from llm_runtime import get_ollama_model, get_ollama_url

# Ollama
OLLAMA_URL = get_ollama_url()
OLLAMA_MODEL = get_ollama_model()


# =========================================
# GENERATE FINAL ANSWER USING LLM
# =========================================

def generate_answer(context, user_query):

    prompt = f"""
You are an AI analytics assistant.

Use the provided data to answer the question.

Explain clearly in English.

User Question:
{user_query}

Data:
{context}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    answer = result["response"]

    return answer


# =========================================
# MAIN PIPELINE
# =========================================

def process_query(user_query):

    print("\nUser Query:", user_query)

    # Step 1 — Route query
    route = route_query(user_query)

    print("Route Selected:", route)

    # =========================
    # SQL PIPELINE
    # =========================

    if route == "SQL":

        sql = generate_sql(user_query)

        print("\nGenerated SQL:")
        print(sql)

        df = execute_sql(sql)

        if df.empty:

            return "No data found."

        context = df.head(10).to_string()

        answer = generate_answer(
            context,
            user_query
        )

        return answer


    # =========================
    # RAG PIPELINE
    # =========================

    else:

        reviews = retrieve_reviews(
            user_query
        )

        context = "\n".join(reviews)

        answer = generate_answer(
            context,
            user_query
        )

        return answer


# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    queries = [

        "Top 5 product categories by sales",

        "Why customers unhappy about delivery",

        "Average payment value",

        "Common complaints about product quality"

    ]

    for q in queries:

        result = process_query(q)

        print("\nFinal Answer:")
        print(result)

        print("\n" + "="*60)
