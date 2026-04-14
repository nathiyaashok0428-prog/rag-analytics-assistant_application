import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def route_query(user_query):

    prompt = f"""
You are an intelligent query classifier.

Classify the user query into:

SQL → structured analytics queries
Examples:
- Total sales
- Top products
- Revenue by city

RAG → customer feedback queries
Examples:
- Customer complaints
- Delivery issues
- Product feedback

HYBRID → needs BOTH analytics + feedback
Examples:
- Top products and customer complaints
- Sales trends and feedback

UNKNOWN → if query is unclear, random text, or meaningless
Examples:
- dsfsdfs
- random words
- unclear question

Return ONLY one word:

SQL
RAG
HYBRID
UNKNOWN

User Query:
{user_query}
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

        result = response.json()["response"]

        result = result.strip().upper()

        allowed = ["SQL", "RAG", "HYBRID", "UNKNOWN"]

        if result not in allowed:
            return "UNKNOWN"

        return result

    except Exception as e:

        print("Router Error:", e)

        return "UNKNOWN"


def extract_subquery(user_query, subquery_type):

    if subquery_type == "SQL":
        instruction = (
            "Rewrite the user's request as only the structured analytics part that should be answered with SQL. "
            "Keep metrics, dimensions, and filters. Remove review-summary wording. "
            "For hybrid questions, ask for numeric business metrics only, such as sales, revenue, counts, averages, or ratings by category, city, state, or time. "
            "Do not ask for raw review text, concatenated comments, complaint excerpts, or sentiment explanations in the SQL sub-question."
        )
    else:
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
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()["response"].strip()
        return result or user_query
    except Exception:
        return user_query
