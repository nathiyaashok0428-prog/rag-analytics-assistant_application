import textwrap
from html import escape
from copy import deepcopy

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bootstrap_assets import ensure_runtime_assets


st.set_page_config(
    page_title="AI Analytics Assistant",
    layout="wide",
)

ensure_runtime_assets()

from rag.retriever_runtime import retrieve_reviews
from rag.sentiment_runtime import batch_sentiment
from rag.synthesizer import (
    synthesize_hybrid_result,
    synthesize_rag_result,
    synthesize_sql_result,
)
from rag.theme_extractor import extract_themes
from router.query_router_runtime import extract_subquery, route_query
from sql.executor import execute_sql_with_error
from sql.nl_to_sql import generate_sql
from visualization.chart_generator import auto_chart


PROMPT_CARDS = [
    {
        "title": "Show monthly sales trend",
        "description": "Analyze sales performance over time",
        "icon": "&#128200;",
    },
    {
        "title": "Top selling products",
        "description": "Discover your best performers",
        "icon": "&#128722;",
    },
    {
        "title": "Customer delivery complaints",
        "description": "Review delivery feedback and issues",
        "icon": "&#128666;",
    },
    {
        "title": "Revenue by category",
        "description": "Break down earnings by product type",
        "icon": "&#128230;",
    },
]

def init_state() -> None:
    defaults = {
        "query_text": "",
        "current_result": None,
        "conversation": [],
        "chat_history": [],
        "history_search": "",
        "pending_query": None,
        "is_processing": False,
        "current_chat_id": None,
        "chat_counter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #020817;
            --panel: #0b1225;
            --sidebar: #060d1d;
            --border: #223252;
            --text: #edf2ff;
            --muted: #b9c6de;
            --subtle: #94a4c2;
            --blue: #3b82f6;
            --blue-soft: #172554;
            --accent: #1677ee;
            --accent-2: #0f5fc7;
            --shadow: 0 12px 34px rgba(2, 8, 23, 0.55);
            --chat-max-width: 1080px;
            --chat-desktop-gutter: 24rem;
        }

        .stApp {
            background: var(--bg);
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebarUserContent"] {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding-bottom: 8rem;
            padding: 1rem 1.5rem;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 18rem;
            max-width: 1240px;
        }

        .brand-shell {
            background: transparent;
            text-align: center;
            padding: 0.8rem 0 1.2rem;
            margin-bottom: 1.2rem;
        }

        .hero-logo {
            width: 116px;
            height: 116px;
            border-radius: 32px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #1e81f6 0%, #1268db 100%);
            color: white;
            font-size: 3rem;
            margin: 0 auto 1rem;
            box-shadow: 0 12px 26px rgba(22, 119, 238, 0.2);
        }

        .hero-title {
            color: var(--text);
            font-size: clamp(3.1rem, 5vw, 5.25rem);
            font-weight: 900;
            line-height: 1.02;
            letter-spacing: -0.04em;
            margin: 0 0 0.45rem;
        }

        .hero-subtitle {
            color: var(--subtle);
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 720px;
            margin: 0 auto;
        }

        .section-title {
            color: var(--text);
            font-size: clamp(2rem, 3vw, 2.7rem);
            font-weight: 850;
            letter-spacing: -0.02em;
            margin: 2rem 0 0.2rem;
            text-align: center;
        }

        .section-copy {
            color: var(--subtle);
            font-size: 0.98rem;
            margin-bottom: 1.8rem;
            text-align: center;
        }

        .sidebar-card {
            background: transparent;
            border: none;
            box-shadow: none;
            padding: 0.2rem 0 1rem;
            margin-bottom: 1rem;
        }

        .sidebar-logo {
            width: 48px;
            height: 48px;
            border-radius: 16px;
            background: linear-gradient(135deg, #1e81f6 0%, #1268db 100%);
            color: white;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .sidebar-title {
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 800;
            margin: 0;
        }

        .sidebar-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
            margin: 0.55rem 0 0;
        }

        .history-label {
            color: var(--subtle);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 1.25rem 0 0.65rem;
        }

        .prompt-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1.25rem 1.35rem;
            min-height: 132px;
            margin-bottom: 0.65rem;
            box-shadow: none;
        }

        .prompt-card-button button {
            min-height: 132px !important;
            border-radius: 22px !important;
            border: 1px solid var(--border) !important;
            background: var(--panel) !important;
            box-shadow: none !important;
            padding: 1.15rem 1.25rem !important;
            text-align: left !important;
            display: flex !important;
            align-items: flex-start !important;
            justify-content: flex-start !important;
            white-space: pre-line !important;
            line-height: 1.55 !important;
            cursor: pointer !important;
        }

        .prompt-card-button button:hover {
            border-color: #cbd4e1 !important;
            background: #fbfcff !important;
        }

        .prompt-icon {
            width: 48px;
            height: 48px;
            border-radius: 16px;
            background: var(--blue-soft);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--blue);
            font-size: 1.4rem;
            margin-bottom: 1rem;
        }

        .prompt-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.4;
            margin-bottom: 0.3rem;
        }

        .prompt-copy {
            color: var(--subtle);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.9rem;
            padding: 0.2rem 0 1rem;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .sidebar-collapse {
            color: var(--subtle);
            font-size: 1rem;
            font-weight: 700;
        }

        .sidebar-search {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            border-radius: 18px;
            background: #0e1730;
            color: var(--subtle);
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border);
        }

        .sidebar-profile {
            padding-top: 1.1rem;
            border-top: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.85rem;
            background: var(--sidebar);
            position: fixed;
            left: 1rem;
            bottom: 1rem;
            width: 15.5rem;
            z-index: 5;
        }

        .sidebar-profile-badge {
            width: 48px;
            height: 48px;
            border-radius: 999px;
            background: linear-gradient(135deg, #1e81f6 0%, #1268db 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
        }

        .sidebar-profile-name {
            color: var(--text);
            font-weight: 800;
            line-height: 1.2;
        }

        .sidebar-profile-mail {
            color: var(--subtle);
            font-size: 0.9rem;
            line-height: 1.2;
            margin-top: 0.15rem;
        }

        .chat-topbar {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 0 1rem;
        }

        .chat-avatar {
            width: 40px;
            height: 40px;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: white;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            box-shadow: 0 8px 20px rgba(0, 113, 227, 0.22);
        }

        .chat-title-line {
            color: var(--text);
            font-weight: 800;
            line-height: 1.2;
        }

        .chat-subtitle-line {
            color: var(--subtle);
            font-size: 0.88rem;
            line-height: 1.2;
        }

        .assistant-meta {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--subtle);
            font-size: 0.9rem;
            margin: 0.4rem 0 0.8rem;
        }

        .user-query {
            display: inline-block;
            margin: 0 0 1rem auto;
            max-width: min(760px, 92%);
            padding: 0.9rem 1.15rem;
            border-radius: 20px;
            background: #111c37;
            border: 1px solid var(--border);
            color: var(--text);
            box-shadow: 0 8px 22px rgba(2, 8, 23, 0.45);
        }

        .assistant-avatar-small {
            width: 28px;
            height: 28px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: white;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 800;
        }

        .pill {
            display: inline-block;
            background: var(--blue-soft);
            color: var(--blue);
            border: 1px solid rgba(22, 119, 238, 0.18);
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .result-title {
            color: var(--text);
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .result-copy {
            color: var(--subtle);
            margin-bottom: 0.2rem;
        }

        .insight-body,
        .insight-body p,
        .insight-body li {
            color: var(--text) !important;
            font-size: 1rem;
            line-height: 1.7;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: var(--panel);
            color: var(--text);
            font-weight: 700;
            min-height: 46px;
            box-shadow: none;
        }

        [data-testid="stSidebar"] .stButton > button {
            border-radius: 12px;
            min-height: 40px;
            box-shadow: none;
        }

        [data-testid="stSidebar"] .stTextInput,
        [data-testid="stSidebar"] .stTextInput > div,
        [data-testid="stSidebar"] div[data-baseweb="input"],
        [data-testid="stSidebar"] div[data-baseweb="base-input"] {
            box-shadow: none !important;
            border: none !important;
            outline: none !important;
            background: transparent !important;
        }

        [data-testid="stSidebar"] .stTextInput input {
            border-radius: 18px !important;
            border: 1px solid var(--border) !important;
            background: #0e1730 !important;
            color: var(--text) !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stSidebar"] .stTextInput input::placeholder {
            color: var(--subtle) !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] .stTextInput input:focus,
        [data-testid="stSidebar"] .stTextInput input:focus-visible {
            border-color: #d8dee8 !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stSidebar"] .stTextInput > div:focus-within,
        [data-testid="stSidebar"] div[data-baseweb="input"]:focus-within,
        [data-testid="stSidebar"] div[data-baseweb="base-input"]:focus-within,
        [data-testid="stSidebar"] .stTextInput *:focus,
        [data-testid="stSidebar"] .stTextInput *:focus-visible {
            box-shadow: none !important;
            outline: none !important;
            border: none !important;
        }

        [data-testid="stSidebar"] .stTextInput label {
            display: none !important;
        }

        .stButton > button:hover {
            border-color: #33518a;
            color: var(--blue);
            background: #101d3b;
        }

        [data-testid="stChatFloatingInputContainer"] {
            left: auto !important;
            right: auto !important;
            width: min(1120px, calc(100vw - 1.2rem)) !important;
            margin: 0 auto 0.35rem !important;
            padding: 0.45rem 0.45rem 0.65rem !important;
            border-radius: 22px !important;
            background: rgba(2, 8, 23, 0.88) !important;
            border: none !important;
            box-shadow: none !important;
        }

        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottomBlockContainer"] > div {
            background: #020817 !important;
            border: none !important;
            box-shadow: none !important;
        }

        [data-testid="stChatInput"] > div {
            max-width: var(--chat-max-width);
            margin: 0 auto;
            width: min(var(--chat-max-width), calc(100vw - var(--chat-desktop-gutter)));
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            display: flex;
            align-items: center;
        }

        [data-testid="stChatInput"] div[data-baseweb="textarea"],
        [data-testid="stChatInput"] div[data-baseweb="base-input"] {
            border: 1px solid var(--border) !important;
            border-radius: 20px !important;
            background: #0b1225 !important;
            box-shadow: 0 10px 26px rgba(2, 8, 23, 0.5) !important;
        }

        [data-testid="stChatInput"] textarea {
            min-height: 54px !important;
            border-radius: 20px !important;
            border: none !important;
            background: #0b1225 !important;
            color: var(--text) !important;
            box-shadow: none !important;
            padding: 0.9rem 3.3rem 0.9rem 1.1rem !important;
            line-height: 1.5 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--subtle) !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] textarea:focus,
        [data-testid="stChatInput"] textarea:focus-visible {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stChatInput"] div[data-baseweb="textarea"]:focus-within,
        [data-testid="stChatInput"] div[data-baseweb="base-input"]:focus-within {
            border-color: #33518a !important;
            box-shadow: 0 6px 20px rgba(2, 8, 23, 0.5) !important;
            outline: none !important;
        }

        [data-testid="stChatInput"] button {
            width: 36px !important;
            height: 36px !important;
            border-radius: 999px !important;
            background: #1e3a8a !important;
            color: #dbeafe !important;
            border: none !important;
        }

        [data-testid="stChatMessage"] {
            background: transparent !important;
        }

        [data-testid="stChatMessageContent"] {
            background: transparent !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] div,
        [data-testid="stChatMessageContent"] span {
            color: var(--text) !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] div,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] li {
            color: var(--text) !important;
        }

        [data-testid="stCaptionContainer"] {
            color: var(--subtle) !important;
        }

        [data-testid="stCodeBlock"] pre,
        [data-testid="stCodeBlock"] code {
            color: #dbeafe !important;
            background: #081022 !important;
        }

        [data-testid="stCodeBlock"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] div[role="gridcell"],
        [data-testid="stDataFrame"] div[role="columnheader"],
        [data-testid="stDataFrame"] div[role="rowheader"] {
            color: var(--text) !important;
            background: #0b1225 !important;
        }

        .stAlert {
            border-radius: 18px;
        }

        .stAlert [data-testid="stMarkdownContainer"] p,
        .stAlert [data-testid="stMarkdownContainer"] div,
        .stAlert [data-testid="stMarkdownContainer"] span {
            color: var(--text) !important;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-bottom: 15rem;
            }

            .hero-title {
                font-size: 2.6rem;
            }

            [data-testid="stChatInput"] > div {
                width: calc(100vw - 1.4rem);
            }

            [data-testid="stChatFloatingInputContainer"] {
                width: calc(100vw - 0.8rem) !important;
                padding: 0.4rem 0.35rem 0.55rem !important;
            }
        }

        @media (max-width: 1200px) {
            :root {
                --chat-desktop-gutter: 20rem;
            }
        }

        @media (max-width: 1024px) {
            :root {
                --chat-desktop-gutter: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def update_chat_history(current_query: str) -> None:
    history = st.session_state.chat_history

    if st.session_state.current_chat_id is None:
        st.session_state.chat_counter += 1
        st.session_state.current_chat_id = st.session_state.chat_counter

    chat_id = st.session_state.current_chat_id
    conversation_queries = [item["query"] for item in st.session_state.conversation]
    title_query = conversation_queries[0] if conversation_queries else current_query

    entry = {
        "id": chat_id,
        "title": textwrap.shorten(title_query, width=34, placeholder="..."),
        "query": current_query,
        "queries": conversation_queries,
    }

    filtered = [item for item in history if item.get("id") != chat_id]
    filtered.insert(0, entry)
    st.session_state.chat_history = filtered


def load_history_chat(entry: dict) -> None:
    st.session_state.current_chat_id = entry.get("id")
    st.session_state.query_text = entry.get("query", "")
    st.session_state.pending_query = None
    st.session_state.is_processing = False
    st.session_state.conversation = []
    st.session_state.current_result = None

    for query in entry.get("queries", []):
        run_query(query, update_history=False)


def build_base_result(user_query: str, route: str) -> dict:
    return {
        "query": user_query,
        "route": route,
        "sql_query_text": None,
        "rag_query_text": None,
        "sql": None,
        "df": None,
        "sql_error": None,
        "chart_error": None,
        "fig": None,
        "reviews": None,
        "themes": [],
        "sentiment_df": None,
        "sentiment_counts": None,
        "explanation": None,
        "error": None,
    }


def prepare_display_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None

    display_df = df.copy()

    if display_df.empty:
        return display_df

    object_columns = display_df.select_dtypes(include=["object"]).columns
    for column in object_columns:
        display_df[column] = display_df[column].fillna("Unknown")
        display_df[column] = display_df[column].replace({"None": "Unknown"})

    return display_df


def run_sql_pipeline(query_text: str) -> dict:
    sql = generate_sql(query_text)
    df, sql_error = execute_sql_with_error(sql) if sql else (pd.DataFrame(), "Could not generate a SQL query for this question. Please try rephrasing.")

    chart = None
    chart_error = None
    if not df.empty:
        try:
            chart = auto_chart(df, query_text)
        except Exception as exc:
            chart_error = str(exc)

    return {
        "sql": sql,
        "df": df,
        "sql_error": sql_error,
        "fig": chart,
        "chart_error": chart_error,
    }


def run_rag_pipeline(query_text: str) -> dict:
    reviews = retrieve_reviews(query_text)
    sentiments = batch_sentiment(reviews)
    sentiment_df = pd.DataFrame(
        {"Review": reviews, "Sentiment": sentiments}
    )
    sentiment_counts = (
        sentiment_df["Sentiment"].value_counts()
        if not sentiment_df.empty
        else pd.Series(dtype="int64")
    )
    themes = extract_themes(reviews)

    return {
        "reviews": reviews,
        "themes": themes,
        "sentiment_df": sentiment_df,
        "sentiment_counts": sentiment_counts,
    }


def run_query(user_query: str, update_history: bool = True) -> None:
    route = route_query(user_query)

    if route == "UNKNOWN":
        result = {
            "query": user_query,
            "route": route,
            "error": (
                "Unable to understand the query. Please ask a meaningful business "
                "question about sales, customers, products, or reviews."
            ),
        }
        st.session_state.current_result = result
        st.session_state.conversation.append(result)
        if update_history:
            update_chat_history(user_query)
        return

    with st.spinner("Analyzing your business question..."):
        result = build_base_result(user_query, route)

        if route in {"SQL", "HYBRID"}:
            sql_prompt = user_query
            if route == "HYBRID":
                sql_prompt = extract_subquery(user_query, "SQL")

            result["sql_query_text"] = sql_prompt
            result.update(run_sql_pipeline(sql_prompt))

            if route == "SQL":
                if result["sql_error"]:
                    result["explanation"] = (
                        "The SQL query could not be executed. Please rephrase the "
                        "question or try a more specific business metric."
                    )
                else:
                    result["explanation"] = (
                        synthesize_sql_result(user_query, result["df"])
                        if not result["df"].empty
                        else "No data returned for this query."
                    )

        if route in {"RAG", "HYBRID"}:
            rag_prompt = user_query
            if route == "HYBRID":
                rag_prompt = extract_subquery(user_query, "RAG")

            result["rag_query_text"] = rag_prompt
            result.update(run_rag_pipeline(rag_prompt))
            if route == "RAG":
                result["explanation"] = synthesize_rag_result(
                    user_query,
                    result["reviews"],
                    result["themes"],
                )
            else:
                result["explanation"] = synthesize_hybrid_result(
                    user_query,
                    result["df"],
                    result["reviews"],
                    result["themes"],
                )

        st.session_state.current_result = result
        st.session_state.conversation.append(result)
        if update_history:
            update_chat_history(user_query)


def queue_query(query: str) -> None:
    query = query.strip()
    if not query:
        return

    st.session_state.query_text = query
    st.session_state.pending_query = query
    st.session_state.is_processing = True


def process_pending_query() -> None:
    pending_query = st.session_state.get("pending_query")
    if not pending_query:
        return

    run_query(pending_query)
    st.session_state.pending_query = None
    st.session_state.is_processing = False


def submit_query(query: str) -> None:
    query = query.strip()
    st.session_state.query_text = query
    if query:
        run_query(query)


def reset_to_home() -> None:
    st.session_state.current_result = None
    st.session_state.conversation = []
    st.session_state.current_chat_id = None
    st.session_state.query_text = ""
    st.session_state.pending_query = None
    st.session_state.is_processing = False
    st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-header">
                    <div class="sidebar-brand">
                        <div class="sidebar-logo">✦</div>
                        <div class="sidebar-title">Nexus AI</div>
                    </div>
                    <div class="sidebar-collapse">◧</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("＋  New Chat", use_container_width=True):
            reset_to_home()

        st.markdown('<div class="history-label">Recents</div>', unsafe_allow_html=True)
        search_term = st.text_input(
            "Search history",
            key="history_search",
            placeholder="🔍 Search",
            label_visibility="collapsed",
        ).strip().lower()

        if not st.session_state.chat_history:
            st.caption("Your recent analytics questions will appear here.")
        else:
            visible_history = [
                item
                for item in st.session_state.chat_history
                if not search_term
                or search_term in item["query"].lower()
                or search_term in item["title"].lower()
            ]
            if not visible_history:
                st.caption("No matching history found.")
            for idx, item in enumerate(visible_history):
                if st.button(item["title"], key=f"history_{idx}", use_container_width=True):
                    load_history_chat(item)
                    st.rerun()

        st.markdown("<div style='height: 44vh; min-height: 240px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="sidebar-profile">
                <div class="sidebar-profile-badge">NA</div>
                <div>
                    <div class="sidebar-profile-name">Nathiya Ashok</div>
                    <div class="sidebar-profile-mail">nathiyaashok0428@gmail.com</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat_header() -> None:
    left, right = st.columns([6, 1])
    with left:
        st.markdown(
            """
            <div class="chat-topbar">
                <div class="chat-avatar">AI</div>
                <div>
                    <div class="chat-title-line">Nexus AI</div>
                    <div class="chat-subtitle-line">Analytics assistant</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Home", use_container_width=True):
            reset_to_home()


def render_assistant_marker(route: str) -> None:
    st.markdown(
        f"""
        <div class="assistant-meta">
            <div class="assistant-avatar-small">AI</div>
            <div>Nexus AI is using the {escape(route)} pipeline.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    cards_disabled = bool(st.session_state.get("is_processing"))

    st.markdown(
        """
        <div class="brand-shell">
            <div class="hero-logo">✦</div>
            <div class="hero-title">Nexus AI</div>
            <div class="hero-subtitle">Intelligent Analytics Assistant</div>
            <div class="hero-copy">
                An AI workspace to help you analyze data, generate insights, and make
                informed business decisions more effectively.
            </div>
        </div>
        <div class="section-title">What can I help you with today?</div>
        <div class="section-copy">
            Start a conversation, explore insights, or ask questions about your business data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2, gap="large")
    for idx, card in enumerate(PROMPT_CARDS):
        with cols[idx % 2]:
            st.markdown('<div class="prompt-card-button">', unsafe_allow_html=True)
            if st.button(
                f"{card['icon']}  {card['title']}\n{card['description']}",
                key=f"prompt_{idx}",
                use_container_width=True,
                disabled=cards_disabled,
            ):
                queue_query(card["title"])
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def render_input_area() -> None:
    query = st.chat_input("Ask anything about your business data...")
    if query:
        queue_query(query)
        st.rerun()


def render_result_card(title: str, copy: str | None = None, badge: str | None = None) -> None:
    if badge:
        st.markdown(f'<div class="pill">{badge}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-title">{title}</div>', unsafe_allow_html=True)
    if copy:
        st.markdown(f'<div class="result-copy">{copy}</div>', unsafe_allow_html=True)


def close_result_card() -> None:
    st.markdown("", unsafe_allow_html=True)


def render_single_result(result: dict) -> None:
    if not result:
        return

    if result["route"] == "UNKNOWN":
        st.markdown(
            f'<div style="display:flex; justify-content:flex-end;"><div class="user-query">{escape(result["query"])}</div></div>',
            unsafe_allow_html=True,
        )
        render_assistant_marker("UNKNOWN")
        render_result_card("Query needs refinement", result["error"], "Needs input")
        st.info(
            "Examples: Top 5 product categories by sales, Why customers are unhappy about delivery, Top products and customer complaints"
        )
        close_result_card()
        return

    st.markdown(
        f'<div style="display:flex; justify-content:flex-end;"><div class="user-query">{escape(result["query"])}</div></div>',
        unsafe_allow_html=True,
    )

    render_assistant_marker(result["route"])
    render_result_card(
        "Analysis ready",
        "Here is the analysis for your current question.",
        f'{result["route"]} pipeline',
    )
    close_result_card()

    if result["sql"]:
        sql_copy = "The structured query used for the analytics step."
        if result.get("sql_query_text") and result["route"] == "HYBRID":
            sql_copy = f'Structured sub-question: "{result["sql_query_text"]}"'
        render_result_card("Generated SQL", sql_copy)
        with st.expander("Show raw SQL", expanded=False):
            st.code(result["sql"], language="sql")
        if result.get("sql_error"):
            st.error(f"SQL execution failed: {result['sql_error']}")
        close_result_card()

    if isinstance(result["df"], pd.DataFrame):
        render_result_card("Structured results", "Data returned from the SQL pipeline.")
        if result.get("sql_error"):
            st.info("No SQL table or chart is shown because the generated query failed to run.")
        elif result["df"].empty:
            st.warning("No data returned for this query.")
        else:
            display_df = prepare_display_df(result["df"])
            st.dataframe(display_df, use_container_width=True)
            if result["fig"] is not None:
                if hasattr(result["fig"], "to_plotly_json"):
                    st.plotly_chart(result["fig"], use_container_width=True)
                else:
                    st.pyplot(result["fig"])
            elif result.get("chart_error"):
                st.info(f"Chart could not be generated for this result: {result['chart_error']}")
        close_result_card()

    if isinstance(result["sentiment_df"], pd.DataFrame):
        review_count = len(result["sentiment_df"].index)
        feedback_copy = f"Retrieved {review_count} review snippets and sentiment labels."
        if result.get("rag_query_text") and result["route"] == "HYBRID":
            feedback_copy = (
                f'Review sub-question: "{result["rag_query_text"]}" '
                f"Retrieved {review_count} review snippets and sentiment labels."
            )
        render_result_card(
            "Customer feedback",
            feedback_copy,
        )
        if result.get("themes"):
            st.caption("Top complaint themes: " + ", ".join(result["themes"]))
        st.dataframe(result["sentiment_df"], use_container_width=True)
        if result["sentiment_counts"] is not None and not result["sentiment_counts"].empty:
            sentiment_counts = (
                result["sentiment_df"]["Sentiment"]
                .value_counts()
                .reindex(["Negative", "Neutral", "Positive"], fill_value=0)
            )
            sentiment_counts = sentiment_counts[sentiment_counts > 0]
            sentiment_chart = go.Figure(
                data=[
                    go.Bar(
                        x=sentiment_counts.index.tolist(),
                        y=sentiment_counts.values.tolist(),
                        text=sentiment_counts.values.tolist(),
                        textposition="outside",
                    )
                ]
            )
            sentiment_chart.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Sentiment",
                yaxis_title="Count",
            )
            sentiment_chart.update_yaxes(dtick=1)
            st.plotly_chart(sentiment_chart, use_container_width=True)
        close_result_card()

    if result["explanation"]:
        insight_copy = "Synthesized summary based on your current question."
        if result["route"] == "HYBRID":
            insight_copy = "Synthesized summary using both SQL results and retrieved reviews."
        render_result_card("AI insight", insight_copy)
        paragraphs = [
            f"<p>{escape(line.strip())}</p>"
            for line in result["explanation"].splitlines()
            if line.strip()
        ]
        if paragraphs:
            st.markdown(
                '<div class="insight-body">' + "".join(paragraphs) + "</div>",
                unsafe_allow_html=True,
            )
        close_result_card()


def render_results() -> None:
    conversation = st.session_state.get("conversation", [])
    if conversation:
        for item in conversation:
            render_single_result(item)
        return

    result = st.session_state.current_result
    if result:
        render_single_result(result)


def main() -> None:
    init_state()
    inject_styles()
    process_pending_query()
    render_sidebar()
    has_conversation = bool(st.session_state.get("conversation"))
    if st.session_state.current_result is None and not has_conversation:
        render_hero()
        render_input_area()
    else:
        render_results()
        render_input_area()


if __name__ == "__main__":
    main()
