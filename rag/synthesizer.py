# =========================================
# ANSWER SYNTHESIZER
# Combines SQL + RAG + Sentiment
# =========================================

import textwrap

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def _looks_weak(answer):

    if not answer:
        return True

    cleaned = answer.strip()

    if len(cleaned) < 90:
        return True

    lowered = cleaned.lower()
    weak_phrases = [
        "based on the provided data",
        "the data suggests",
        "insufficient data",
    ]

    return sum(phrase in lowered for phrase in weak_phrases) >= 2


def _fallback_sql_summary(user_query, df):

    if df is None or df.empty:
        return (
            f'For "{user_query}", the SQL pipeline returned no structured rows. '
            "Try a more specific metric, timeframe, or dimension."
        )

    preview = df.head(5)
    columns = ", ".join(preview.columns.astype(str))
    sample_lines = []

    for index, row in preview.iterrows():
        values = [f"{col}={row[col]}" for col in preview.columns[: min(4, len(preview.columns))]]
        sample_lines.append(f"{index + 1}. " + ", ".join(values))

    return (
        f'For "{user_query}", the SQL pipeline returned {len(df)} rows with columns: {columns}.\n\n'
        "Sample rows from the result:\n"
        + "\n".join(sample_lines)
    )


def _format_theme_line(themes):

    if not themes:
        return ""

    return "\nTop themes: " + ", ".join(themes)


def _fallback_rag_summary(user_query, reviews, themes=None):

    if not reviews:
        return (
            f'For "{user_query}", no matching customer review snippets were retrieved.'
        )

    bullet_lines = [
        f'- {textwrap.shorten(review, width=180, placeholder="...")}'
        for review in reviews[:5]
    ]

    return (
        f'For "{user_query}", the answer is based on these retrieved customer review snippets:\n'
        + "\n".join(bullet_lines)
        + _format_theme_line(themes)
    )


def _fallback_hybrid_summary(user_query, df, reviews, themes=None):

    sql_part = _fallback_sql_summary(user_query, df)
    rag_part = _fallback_rag_summary(user_query, reviews, themes)

    return (
        f'For "{user_query}", combine the structured analytics result with the review evidence.\n\n'
        f"{sql_part}\n\n{rag_part}"
    )


# =========================================
# GENERATE FINAL RESPONSE
# =========================================

def synthesize_answer(user_query, context_data):

    prompt = f"""
You are an AI Business Analytics Assistant.

Your job:
Generate a clear and professional answer.

Use the provided data to answer.

User Question:
{user_query}

Data:
{context_data}

Instructions:

- Summarize insights clearly
- Highlight key findings
- Use simple English
- Provide meaningful explanation
- Avoid technical jargon
- Answer the user's exact question, not a generic overview
- If the data is insufficient, say what is missing
- Do not repeat the same boilerplate answer across different questions
- Write at least 3 short points or a short paragraph of 3 to 5 sentences
- Refer to the actual evidence in the provided data

Answer:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()

        return result["response"].strip()
    except Exception:
        return ""


# =========================================
# SYNTHESIZE SQL RESULTS
# =========================================

def synthesize_sql_result(user_query, df):

    if df.empty:
        return "No data available."

    context = df.head(10).to_string()

    answer = synthesize_answer(
        user_query,
        context
    )

    if _looks_weak(answer):
        return _fallback_sql_summary(user_query, df)

    return answer


# =========================================
# SYNTHESIZE RAG RESULTS
# =========================================

def synthesize_rag_result(user_query, reviews, themes=None):

    context = "\n".join(reviews)
    if themes:
        context += "\n\nTop complaint themes:\n- " + "\n- ".join(themes)

    answer = synthesize_answer(
        user_query,
        context
    )

    if _looks_weak(answer):
        return _fallback_rag_summary(user_query, reviews, themes)

    return answer


def synthesize_hybrid_result(user_query, df, reviews, themes=None):

    sql_context = "No SQL data available."
    if df is not None and not df.empty:
        sql_context = df.head(10).to_string()

    review_context = "No review evidence available."
    if reviews:
        review_context = "\n".join(reviews)
        if themes:
            review_context += "\n\nTop complaint themes:\n- " + "\n- ".join(themes)

    combined_context = f"""
Structured business data:
{sql_context}

Customer review evidence:
{review_context}
"""

    answer = synthesize_answer(
        user_query,
        combined_context
    )

    if _looks_weak(answer):
        return _fallback_hybrid_summary(user_query, df, reviews, themes)

    return answer


# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    test_query = "Why customers unhappy?"

    sample_reviews = [

        "Delivery was late and product damaged",

        "Poor packaging quality",

        "Customer support slow"

    ]

    result = synthesize_rag_result(
        test_query,
        sample_reviews
    )

    print("\nSynthesized Answer:\n")

    print(result)
