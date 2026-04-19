# =========================================
# CHART GENERATOR
# =========================================

import pandas as pd
import plotly.graph_objects as go

from llm_runtime import call_ollama


METRIC_KEYWORDS = [
    "revenue",
    "sales",
    "sale",
    "count",
    "total",
    "amount",
    "value",
    "profit",
    "payment",
    "score",
    "rating",
    "avg",
    "average",
]
PIE_KEYWORDS = ["share", "percent", "percentage", "ratio", "distribution"]


def _apply_layout(fig, x_col, y_col):

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=str(x_col),
        yaxis_title=str(y_col),
    )
    fig.update_yaxes(tickformat=",.2f")

    return fig


def _select_chart_columns(df):

    if df.empty or len(df.columns) < 2:
        return None, None

    numeric_columns = []
    categorical_columns = []

    for column in df.columns:
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        if numeric_series.notna().sum() > 0:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    if not numeric_columns:
        return None, None

    x_col = categorical_columns[0] if categorical_columns else df.columns[0]

    def metric_score(column_name):
        lowered = str(column_name).lower()
        for idx, keyword in enumerate(METRIC_KEYWORDS):
            if keyword in lowered:
                return len(METRIC_KEYWORDS) - idx
        return 0

    y_candidates = [column for column in numeric_columns if column != x_col] or numeric_columns
    y_col = max(y_candidates, key=metric_score)

    if y_col == x_col and len(y_candidates) > 1:
        y_col = y_candidates[1]

    return x_col, y_col


def _prepare_chart_data(df):

    x_col, y_col = _select_chart_columns(df)

    if not x_col or not y_col:
        return None

    chart_df = df[[x_col, y_col]].copy()
    chart_df = chart_df.dropna(subset=[y_col])

    if chart_df.empty:
        return None

    chart_df[x_col] = chart_df[x_col].fillna("Unknown").astype(str)
    chart_df[y_col] = pd.to_numeric(chart_df[y_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[y_col])

    if chart_df.empty:
        return None

    return chart_df


def _choose_chart_type_rules(df):
    """Rule-based chart type selection used as fallback when LLM is unavailable."""

    chart_df = _prepare_chart_data(df)

    if chart_df is None:
        return None

    _, y_col = _select_chart_columns(df)
    first_col = str(chart_df.columns[0]).lower()
    metric_col = str(y_col).lower() if y_col is not None else str(chart_df.columns[1]).lower()
    unique_count = chart_df.iloc[:, 0].nunique()

    if "date" in first_col or "month" in first_col or "year" in first_col or "time" in first_col:
        return "line"

    if any(keyword in metric_col for keyword in PIE_KEYWORDS) and len(chart_df) <= 6 and unique_count <= 6:
        return "pie"

    return "bar"


def _choose_chart_type_llm(df, user_query=""):
    """Ask the LLM to choose the best chart type for the given data."""

    if df.empty or len(df.columns) < 2:
        return None

    col_names = ", ".join(str(c) for c in df.columns.tolist())
    sample_rows = df.head(3).to_string(index=False)

    prompt = f"""You are a data visualization expert.

Given a dataframe with columns: {col_names}

Sample data:
{sample_rows}

User question context: {user_query or "business analytics query"}

Choose the most appropriate chart type from these options:
- bar: for comparing categories or ranking items
- line: for trends over time (dates, months, years)
- pie: for showing proportional distribution (shares, percentages)

Return ONLY one word: bar, line, or pie
Do not explain. Do not add punctuation.
"""

    try:
        result = call_ollama(prompt, timeout=20)
        if result:
            chart_type = result.strip().lower().rstrip(".")
            if chart_type in {"bar", "line", "pie"}:
                return chart_type
    except Exception:
        pass

    return None


def choose_chart_type(df, user_query=""):
    """Select chart type using LLM first, falling back to rule-based logic."""

    chart_df = _prepare_chart_data(df)
    if chart_df is None:
        return None

    # Primary: LLM-based selection (per design document requirement)
    llm_choice = _choose_chart_type_llm(df, user_query)
    if llm_choice:
        return llm_choice

    # Fallback: keyword + column-type heuristics
    return _choose_chart_type_rules(df)


def generate_bar_chart(df):

    chart_df = _prepare_chart_data(df)

    if chart_df is None:
        return None

    chart_df = chart_df.sort_values(by=chart_df.columns[1], ascending=False)

    x_values = chart_df.iloc[:, 0].tolist()
    y_values = chart_df.iloc[:, 1].tolist()

    fig = go.Figure(
        data=[
            go.Bar(
                x=x_values,
                y=y_values,
                text=[f"{value:,.2f}" for value in y_values],
                textposition="outside",
            )
        ]
    )
    fig.update_xaxes(categoryorder="total descending")
    fig.update_yaxes(type="linear")
    return _apply_layout(fig, chart_df.columns[0], chart_df.columns[1])


def generate_line_chart(df):

    chart_df = _prepare_chart_data(df)

    if chart_df is None:
        return None

    fig = go.Figure(
        data=[
            go.Scatter(
                x=chart_df.iloc[:, 0].tolist(),
                y=chart_df.iloc[:, 1].tolist(),
                mode="lines+markers",
            )
        ]
    )
    return _apply_layout(fig, chart_df.columns[0], chart_df.columns[1])


def generate_pie_chart(df):

    chart_df = _prepare_chart_data(df)

    if chart_df is None:
        return None

    chart_df = chart_df.sort_values(by=chart_df.columns[1], ascending=False)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=chart_df.iloc[:, 0].tolist(),
                values=chart_df.iloc[:, 1].tolist(),
                textinfo="percent+label",
            )
        ]
    )
    return _apply_layout(fig, chart_df.columns[0], chart_df.columns[1])


def auto_chart(df, user_query=""):

    chart_type = choose_chart_type(df, user_query)

    if chart_type == "line":
        return generate_line_chart(df)

    if chart_type == "pie":
        return generate_pie_chart(df)

    if chart_type == "bar":
        return generate_bar_chart(df)

    return None
