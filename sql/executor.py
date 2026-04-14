import sqlite3
import pandas as pd

# Database Path
DB_PATH = "data/ecommerce.db"


def execute_sql_with_error(sql_query):
    """
    Executes a SQL query on SQLite and returns both the result and any error.
    """

    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql_query(sql_query, conn)
        finally:
            conn.close()

        return df, None

    except Exception as exc:

        print("SQL Execution Error:", exc)

        return pd.DataFrame(), str(exc)


def execute_sql(sql_query):
    """
    Backward-compatible wrapper that returns only the DataFrame result.
    """

    df, _ = execute_sql_with_error(sql_query)
    return df
