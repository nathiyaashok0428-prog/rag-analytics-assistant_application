# =========================================
# NL TO SQL USING OLLAMA (FREE LOCAL LLM)
# =========================================

import json
import sqlite3
import re
from pathlib import Path
from functools import lru_cache
import pandas as pd
from llm_runtime import call_ollama

from sql.executor import execute_sql as run_sql_query

# ===============================
# DATABASE PATH
# ===============================

DB_PATH = "data/ecommerce.db"
SQL_CACHE_PATH = Path("data/sql_query_cache.json")
TABLE_COLUMNS = {
    "orders": {
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
        "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    },
    "order_items": {
        "order_id", "order_item_id", "product_id", "seller_id",
        "shipping_limit_date", "price", "freight_value",
    },
    "products": {
        "product_id", "product_category_name", "product_name_lenght",
        "product_description_lenght", "product_photos_qty", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm",
    },
    "customers": {
        "customer_id", "customer_unique_id", "customer_zip_code_prefix",
        "customer_city", "customer_state",
    },
    "payments": {"order_id", "payment_type", "payment_value"},
    "reviews": {
        "review_id", "order_id", "review_score", "review_comment_title",
        "review_comment_message", "review_creation_date", "review_answer_timestamp",
    },
    "sellers": {"seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"},
}
TABLE_ID_FALLBACKS = {
    "orders": "order_id",
    "order_items": "order_id",
    "products": "product_id",
    "customers": "customer_id",
    "payments": "order_id",
    "reviews": "order_id",
    "sellers": "seller_id",
}
SQL_RESERVED_WORDS = {
    "select", "from", "join", "inner", "left", "right", "full", "outer", "on",
    "where", "group", "order", "by", "limit", "having", "as", "and", "or",
}
ORDER_STATUS_NORMALIZATION = {
    "sold": "delivered",
    "completed": "delivered",
}


def normalize_query_key(user_query):
    return " ".join(user_query.strip().lower().split())


def load_sql_cache():
    if not SQL_CACHE_PATH.exists():
        return {}

    try:
        return json.loads(SQL_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_sql_cache(cache):
    SQL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQL_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def get_cached_sql(user_query):
    cache = load_sql_cache()
    return cache.get(normalize_query_key(user_query))


def persist_sql_cache(user_query, sql_query):
    normalized_key = normalize_query_key(user_query)
    cache = load_sql_cache()
    cache[normalized_key] = sql_query
    save_sql_cache(cache)

# ===============================
# DATABASE SCHEMA
# ===============================

SCHEMA = """
You are an expert SQL generator.

Database: SQLite

Tables:

orders(
    order_id,
    customer_id,
    order_status,
    order_purchase_timestamp,
    order_approved_at,
    order_delivered_carrier_date,
    order_delivered_customer_date,
    order_estimated_delivery_date
)

order_items(
    order_id,
    order_item_id,
    product_id,
    seller_id,
    shipping_limit_date,
    price,
    freight_value
)

products(
    product_id,
    product_category_name
)

customers(
    customer_id,
    customer_unique_id,
    customer_zip_code_prefix,
    customer_city,
    customer_state
)

payments(
    order_id,
    payment_type,
    payment_value
)

reviews(
    review_id,
    order_id,
    review_score,
    review_comment_title,
    review_comment_message,
    review_creation_date,
    review_answer_timestamp
)

sellers(
    seller_id,
    seller_zip_code_prefix,
    seller_city,
    seller_state
)

Rules:

- Generate valid SQLite SQL
- Use LIMIT 10
- Valid order_status values in this dataset include delivered, shipped, canceled, unavailable, invoiced, processing, created, approved
- Product categories are stored in Portuguese values such as eletronicos, eletroportateis, eletrodomesticos, consoles_games
- When joining tables, always qualify shared column names with their table name
- Prefer explicit table prefixes in SELECT, GROUP BY, ORDER BY, and WHERE clauses
- For mixed analytics and feedback questions, generate only the structured analytics part
- For hybrid questions, return numeric business metrics only, not raw review text or concatenated review comments
- Do not use MySQL-only syntax such as SEPARATOR inside GROUP_CONCAT
- Quote all string values in WHERE and IN clauses
- Return a single executable SQLite SELECT statement
- Do not explain
- Return only SQL
"""

# ===============================
# GENERATE SQL
# ===============================

def clean_sql_response(raw_sql):

    sql = raw_sql.strip()

    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"(?im)^\s*(final\s+sql\s+query|sql\s+query|corrected\s+sql|sql)\s*:\s*", "", sql).strip()
    sql = re.sub(r"(?im)^\s*(structured\s+sub-?question|rewritten\s+query)\s*:\s*.*$", "", sql).strip()

    match = re.search(r"(?is)\b(WITH|SELECT)\b.*?(;|$)", sql)
    if match:
        sql = match.group(0).strip()

    return sql


def sanitize_sql_identifiers(sql):
    sql = re.sub(r"(?is)^.*?\b(WITH|SELECT)\b", lambda m: m.group(1), sql)
    sql = re.sub(r"(?is)(;).*?$", r"\1", sql)

    sql = re.sub(
        r"EXTRACT\s*\(\s*YEAR\s+FROM\s+([A-Za-z_][A-Za-z0-9_\.]*)\s*\)",
        r"CAST(strftime('%Y', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"EXTRACT\s*\(\s*MONTH\s+FROM\s+([A-Za-z_][A-Za-z0-9_\.]*)\s*\)",
        r"CAST(strftime('%m', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"GROUP_CONCAT\(\s*([^,\)]+?)\s+ORDER BY\s+.+?\s+SEPARATOR\s+'([^']*)'\s*\)",
        r"GROUP_CONCAT(\1, '\2')",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"GROUP_CONCAT\(\s*([^,\)]+?)\s+SEPARATOR\s+'([^']*)'\s*\)",
        r"GROUP_CONCAT(\1, '\2')",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"GROUP_CONCAT\(\s*([^,\)]+?)\s+ORDER BY\s+.+?\)",
        r"GROUP_CONCAT(\1)",
        sql,
        flags=re.IGNORECASE,
    )

    for wrong_status, normalized_status in ORDER_STATUS_NORMALIZATION.items():
        sql = re.sub(
            rf"order_status\s*=\s*'{wrong_status}'",
            f"order_status = '{normalized_status}'",
            sql,
            flags=re.IGNORECASE,
        )

    for table_name, id_column in TABLE_ID_FALLBACKS.items():
        sql = re.sub(
            rf"\b{table_name}\.id\b",
            f"{table_name}.{id_column}",
            sql,
            flags=re.IGNORECASE,
        )

    alias_pairs = re.findall(
        r"\b(?:FROM|JOIN)\s+(orders|order_items|products|customers|payments|reviews|sellers)\s+([A-Za-z_][A-Za-z0-9_]*)",
        sql,
        flags=re.IGNORECASE,
    )
    table_aliases = {
        table.lower(): alias
        for table, alias in alias_pairs
        if alias.lower() not in SQL_RESERVED_WORDS
    }
    known_aliases = set(table_aliases.values())

    for table_name, alias in table_aliases.items():
        sql = re.sub(
            rf"\b{table_name}\.([A-Za-z_][A-Za-z0-9_]*)\b",
            rf"{alias}.\1",
            sql,
            flags=re.IGNORECASE,
        )

    def replace_three_part(match):
        table_name = match.group(1).lower()
        column_name = match.group(2)
        alias = table_aliases.get(table_name)
        if alias:
            return f"{alias}.{column_name}"
        return match.group(0)

    sql = re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\.(orders|order_items|products|customers|payments|reviews)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        replace_three_part,
        sql,
    )

    column_to_table = {}
    for table_name, columns in TABLE_COLUMNS.items():
        for column_name in columns:
            column_to_table.setdefault(column_name, []).append(table_name)

    def replace_unknown_alias(match):
        alias = match.group(1)
        column_name = match.group(2)

        if alias in known_aliases:
            return match.group(0)

        candidate_tables = column_to_table.get(column_name, [])
        if len(candidate_tables) != 1:
            return match.group(0)

        table_name = candidate_tables[0]
        fixed_alias = table_aliases.get(table_name)
        if fixed_alias:
            return f"{fixed_alias}.{column_name}"

        # If the query uses full table names instead of aliases,
        # repair bogus prefixes such as ON.column_name.
        return f"{table_name}.{column_name}"

    sql = re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        replace_unknown_alias,
        sql,
    )

    needs_customers = re.search(
        r"\b(customers\.(customer_city|customer_state)|customer_city|customer_state)\b",
        sql,
        flags=re.IGNORECASE,
    )
    has_customers_source = re.search(r"\b(?:FROM|JOIN)\s+customers\b", sql, flags=re.IGNORECASE)
    has_orders_source = re.search(r"\b(?:FROM|JOIN)\s+orders\b", sql, flags=re.IGNORECASE)

    if needs_customers and not has_customers_source and has_orders_source:
        sql = re.sub(
            r"\bFROM\s+orders\b",
            "FROM orders JOIN customers ON orders.customer_id = customers.customer_id",
            sql,
            count=1,
            flags=re.IGNORECASE,
        )

    needs_products = re.search(
        r"\b(products\.product_category_name|product_category_name)\b",
        sql,
        flags=re.IGNORECASE,
    )
    has_products_source = re.search(r"\b(?:FROM|JOIN)\s+products\b", sql, flags=re.IGNORECASE)
    has_order_items_source = re.search(r"\b(?:FROM|JOIN)\s+order_items\b", sql, flags=re.IGNORECASE)

    if needs_products and not has_products_source and has_order_items_source:
        sql = re.sub(
            r"\bFROM\s+order_items\b",
            "FROM order_items JOIN products ON order_items.product_id = products.product_id",
            sql,
            count=1,
            flags=re.IGNORECASE,
        )

    needs_sellers = re.search(
        r"\b(sellers\.(seller_city|seller_state)|seller_city|seller_state|seller_id)\b",
        sql,
        flags=re.IGNORECASE,
    )
    has_sellers_source = re.search(r"\b(?:FROM|JOIN)\s+sellers\b", sql, flags=re.IGNORECASE)

    if needs_sellers and not has_sellers_source and has_order_items_source:
        sql = re.sub(
            r"\bFROM\s+order_items\b",
            "FROM order_items JOIN sellers ON order_items.seller_id = sellers.seller_id",
            sql,
            count=1,
            flags=re.IGNORECASE,
        )

    # If the query filters by order_status but never joins orders,
    # add the missing join and qualify the column for SQLite.
    references_order_status = re.search(r"\border_status\b", sql, flags=re.IGNORECASE)
    has_orders_source = re.search(r"\b(?:FROM|JOIN)\s+orders\b", sql, flags=re.IGNORECASE)
    has_order_items_source = re.search(r"\b(?:FROM|JOIN)\s+order_items\b", sql, flags=re.IGNORECASE)

    if references_order_status and not has_orders_source and has_order_items_source:
        sql = re.sub(
            r"\bFROM\s+order_items\b",
            "FROM order_items JOIN orders ON order_items.order_id = orders.order_id",
            sql,
            count=1,
            flags=re.IGNORECASE,
        )

    if re.search(r"\border_status\b", sql, flags=re.IGNORECASE):
        sql = re.sub(
            r"(?<!\.)\border_status\b",
            "orders.order_status",
            sql,
            flags=re.IGNORECASE,
        )

    def quote_status_literals(raw_items):
        items = []
        for raw_item in raw_items.split(","):
            item = raw_item.strip()
            if not item:
                continue
            if item.startswith("'") and item.endswith("'"):
                items.append(item)
            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item):
                items.append(f"'{item}'")
            else:
                items.append(item)
        return f"orders.order_status IN ({', '.join(items)})"

    sql = re.sub(
        r"(?<!\.)((?:orders\.)?order_status)\s+IN\s*\(([^)]*)\)",
        lambda match: quote_status_literals(match.group(2)),
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"product_category_name\s*=\s*'electronics'",
        "product_category_name IN ('eletronicos', 'eletroportateis', 'eletrodomesticos', 'consoles_games')",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"product_category_name\s+LIKE\s+'%electronics%?'",
        "product_category_name IN ('eletronicos', 'eletroportateis', 'eletrodomesticos', 'consoles_games')",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"p\.product_category_name\s*=\s*'electronics'",
        "p.product_category_name IN ('eletronicos', 'eletroportateis', 'eletrodomesticos', 'consoles_games')",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"p\.product_category_name\s+LIKE\s+'%electronics%?'",
        "p.product_category_name IN ('eletronicos', 'eletroportateis', 'eletrodomesticos', 'consoles_games')",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"\bJOIN\s+sellers\s+ON\s+order_items\.seller_id\s*=\s*order_items\.seller_id",
        "JOIN sellers ON order_items.seller_id = sellers.seller_id",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"\bJOIN\s+sellers\s+ON\s+oi\.seller_id\s*=\s*oi\.seller_id",
        "JOIN sellers ON oi.seller_id = sellers.seller_id",
        sql,
        flags=re.IGNORECASE,
    )

    return sql


def _template_sql_fallback(user_query):

    lowered = user_query.lower()

    if any(token in lowered for token in ["total number of orders", "total orders", "how many orders", "number of orders"]):
        return """
SELECT
    COUNT(DISTINCT orders.order_id) AS total_orders
FROM orders
WHERE orders.order_status IN ('delivered', 'shipped');
""".strip()

    if any(token in lowered for token in ["average payment value", "avg payment", "mean payment", "average order payment"]):
        return """
SELECT
    AVG(payments.payment_value) AS average_payment_value
FROM payments;
""".strip()

    if any(token in lowered for token in ["revenue by payment type", "payment type revenue", "total revenue by payment type"]):
        return """
SELECT
    payments.payment_type AS payment_type,
    SUM(payments.payment_value) AS total_revenue
FROM payments
GROUP BY payments.payment_type
ORDER BY total_revenue DESC
LIMIT 10;
""".strip()

    if any(token in lowered for token in ["top selling cities", "top 10 selling cities", "best selling cities", "top cities by sales"]):
        return """
SELECT
    customers.customer_city AS customer_city,
    COUNT(DISTINCT orders.order_id) AS order_count,
    SUM(order_items.price) AS total_sales
FROM orders
JOIN customers ON orders.customer_id = customers.customer_id
JOIN order_items ON orders.order_id = order_items.order_id
WHERE orders.order_status IN ('delivered', 'shipped')
GROUP BY customers.customer_city
ORDER BY total_sales DESC, order_count DESC
LIMIT 10;
""".strip()

    if any(token in lowered for token in ["monthly order volume", "order volume", "monthly orders", "orders per month"]):
        year_match = re.search(r"\b(20\d{2})\b", lowered)
        year_filter = ""
        if year_match:
            year_filter = f"\nWHERE strftime('%Y', orders.order_purchase_timestamp) = '{year_match.group(1)}'"
        return f"""
SELECT
    strftime('%Y-%m', orders.order_purchase_timestamp) AS month,
    COUNT(DISTINCT orders.order_id) AS orders_count
FROM orders{year_filter}
GROUP BY strftime('%Y-%m', orders.order_purchase_timestamp)
ORDER BY month
LIMIT 10;
""".strip()

    if any(token in lowered for token in ["monthly sales trend", "sales trend", "monthly sales"]):
        return """
SELECT
    strftime('%Y-%m', orders.order_purchase_timestamp) AS month,
    SUM(order_items.price) AS total_sales
FROM orders
JOIN order_items ON orders.order_id = order_items.order_id
WHERE orders.order_status IN ('delivered', 'shipped')
GROUP BY strftime('%Y-%m', orders.order_purchase_timestamp)
ORDER BY month DESC
LIMIT 10;
""".strip()

    if any(token in lowered for token in ["top selling products", "best selling products", "top products", "best performers"]):
        return """
SELECT
    products.product_category_name AS product_category,
    COUNT(DISTINCT orders.order_id) AS order_count,
    SUM(order_items.price) AS total_sales
FROM orders
JOIN order_items ON orders.order_id = order_items.order_id
JOIN products ON order_items.product_id = products.product_id
WHERE orders.order_status IN ('delivered', 'shipped')
GROUP BY products.product_category_name
ORDER BY order_count DESC, total_sales DESC
LIMIT 10;
""".strip()

    if any(token in lowered for token in ["revenue by category", "category revenue"]):
        return """
SELECT
    products.product_category_name AS category,
    SUM(order_items.price) AS total_revenue
FROM orders
JOIN order_items ON orders.order_id = order_items.order_id
JOIN products ON order_items.product_id = products.product_id
WHERE orders.order_status IN ('delivered', 'shipped')
GROUP BY products.product_category_name
ORDER BY total_revenue DESC
LIMIT 10;
""".strip()

    if "electronics" in lowered and any(token in lowered for token in ["satisfaction", "review", "rating", "compare"]):
        return """
SELECT
    p.product_category_name AS product_category,
    SUM(oi.price) AS total_sales,
    AVG(r.review_score) AS avg_review_score
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN reviews r ON o.order_id = r.order_id
WHERE p.product_category_name IN ('eletronicos', 'eletroportateis', 'eletrodomesticos', 'consoles_games')
GROUP BY p.product_category_name
ORDER BY total_sales DESC, avg_review_score DESC
LIMIT 10;
""".strip()

    mentions_category = "categor" in lowered
    mentions_sales = any(token in lowered for token in ["best selling", "top selling", "top categories", "sales count", "highest sales", "total sales", "sales"])
    mentions_reviews = any(token in lowered for token in ["review", "rating", "score", "satisfaction"])

    if mentions_category and mentions_sales and mentions_reviews:
        return """
SELECT
    products.product_category_name AS category,
    AVG(reviews.review_score) AS avg_rating,
    COUNT(DISTINCT orders.order_id) AS sales_count
FROM orders
JOIN order_items ON orders.order_id = order_items.order_id
JOIN products ON order_items.product_id = products.product_id
LEFT JOIN reviews ON orders.order_id = reviews.order_id
WHERE orders.order_status IN ('delivered', 'shipped')
GROUP BY products.product_category_name
ORDER BY sales_count DESC, avg_rating DESC
LIMIT 10;
""".strip()

    if any(token in lowered for token in ["seller", "sellers"]) and any(token in lowered for token in ["worst review", "lowest revenue", "lowest sales", "worst reviews"]):
        return """
SELECT
    oi.seller_id AS seller_id,
    AVG(r.review_score) AS avg_review_score,
    SUM(oi.price) AS total_revenue,
    COUNT(DISTINCT o.order_id) AS order_count
FROM sellers s
JOIN order_items oi ON s.seller_id = oi.seller_id
JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN reviews r ON o.order_id = r.order_id
GROUP BY oi.seller_id
HAVING AVG(r.review_score) IS NOT NULL
ORDER BY avg_review_score ASC, total_revenue ASC
LIMIT 10;
""".strip()

    mentions_delay = any(token in lowered for token in ["delay", "late delivery", "delivery delay"])
    mentions_state = any(token in lowered for token in ["state", "states", "region"])
    if mentions_delay and mentions_state:
        return """
SELECT
    customers.customer_state AS customer_state,
    AVG(julianday(orders.order_delivered_customer_date) - julianday(orders.order_estimated_delivery_date)) AS avg_delivery_delay_days,
    COUNT(DISTINCT orders.order_id) AS order_count
FROM orders
JOIN customers ON orders.customer_id = customers.customer_id
WHERE orders.order_status = 'delivered'
  AND orders.order_delivered_customer_date IS NOT NULL
  AND orders.order_estimated_delivery_date IS NOT NULL
GROUP BY customers.customer_state
ORDER BY avg_delivery_delay_days DESC
LIMIT 10;
""".strip()

    return None


def _call_ollama(prompt):

    return call_ollama(prompt, timeout=45)


def _can_execute_sql(sql_query):

    conn = sqlite3.connect(DB_PATH)

    try:
        pd.read_sql_query(sql_query, conn)
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        conn.close()


def repair_sql(user_query, bad_sql, sql_error):

    prompt = f"""
You are fixing a SQLite query for this schema.

{SCHEMA}

User Question:
{user_query}

Broken SQL:
{bad_sql}

Execution Error:
{sql_error}

Instructions:
- Return one corrected SQLite SELECT query only
- Keep the query focused on the structured analytics part of the question
- Use valid aliases consistently across SELECT, JOIN, WHERE, and GROUP BY
- Remove markdown fences and explanations

Corrected SQL:
"""

    repaired_candidate = _call_ollama(prompt)
    if not repaired_candidate:
        return bad_sql

    repaired_sql = sanitize_sql_identifiers(clean_sql_response(repaired_candidate))

    return repaired_sql


@lru_cache(maxsize=256)
def _generate_sql_cached(user_query):
    # 1. Check persistent cache first (avoids repeated LLM calls for known queries)
    cached_sql = get_cached_sql(user_query)
    if cached_sql:
        cached_valid, _ = _can_execute_sql(cached_sql)
        if cached_valid:
            return cached_sql

    # 2. Call LLM first — this is the primary path
    prompt = f"""
{SCHEMA}

User Question:
{user_query}

SQL Query:
"""

    llm_response = _call_ollama(prompt)

    if llm_response:
        sql = sanitize_sql_identifiers(clean_sql_response(llm_response))
        is_valid, sql_error = _can_execute_sql(sql)

        if is_valid:
            persist_sql_cache(user_query, sql)
            return sql

        # 3. LLM gave invalid SQL — ask LLM to repair it
        repaired_sql = repair_sql(user_query, sql, sql_error)
        repaired_valid, _ = _can_execute_sql(repaired_sql)
        if repaired_valid:
            persist_sql_cache(user_query, repaired_sql)
            return repaired_sql

    # 4. LLM unavailable or both LLM attempts failed — use curated template as last resort
    fallback_sql = _template_sql_fallback(user_query)
    if fallback_sql:
        fallback_valid, _ = _can_execute_sql(fallback_sql)
        if fallback_valid:
            persist_sql_cache(user_query, fallback_sql)
            return fallback_sql

    # 5. Nothing worked — return None so the caller can show a proper error
    return None


def generate_sql(user_query):
    cleaned_query = user_query.strip()
    if not cleaned_query:
        return ""
    result = _generate_sql_cached(cleaned_query)
    # Return empty string when no SQL could be generated so the caller
    # can detect the failure and display a proper error to the user.
    return result if result is not None else ""


# ===============================
# EXECUTE SQL
# ===============================

def execute_sql(sql_query):
    return run_sql_query(sql_query)


# ===============================
# TEST SCRIPT
# ===============================

if __name__ == "__main__":

    print("Testing NL → SQL System\n")

    user_query = "Top 5 product categories by sales"

    sql = generate_sql(user_query)

    print("Generated SQL:\n")
    print(sql)

    result = execute_sql(sql)

    print("\nResult:")
    print(result.head())
