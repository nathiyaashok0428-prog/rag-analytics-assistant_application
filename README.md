# LLM-Powered Analytics Assistant with RAG

A Streamlit-based analytics assistant for the Olist Brazilian e-commerce dataset. The application supports three query paths:

- `SQL` for structured analytics over a SQLite database
- `RAG` for customer-review analysis using FAISS retrieval
- `HYBRID` for combined analytics + review-grounded insights

The app is designed for business-style natural-language questions such as sales trends, revenue analysis, customer complaints, sentiment, and cross-analysis between metrics and feedback.

## Features

- Natural language to SQL for business analytics
- Review retrieval with FAISS + sentence-transformers
- Sentiment analysis over retrieved review snippets
- Complaint/theme extraction
- Query routing into `SQL`, `RAG`, `HYBRID`, or `UNKNOWN`
- Plotly chart generation for SQL results
- Chat-style Streamlit UI with conversation history
- Persistent validated SQL cache for stable repeated queries across restarts

## Project Structure

```text
rag-analytics-assistant/
├── app.py
├── requirements.txt
├── data/
│   ├── ecommerce.db
│   ├── olist_loader.py
│   └── raw/
├── rag/
│   ├── embedder.py
│   ├── retriever.py
│   ├── sentiment_analysis.py
│   ├── synthesizer.py
│   ├── theme_extractor.py
│   ├── translator.py
│   ├── faiss_index.bin
│   └── review_chunks.pkl
├── router/
│   └── query_router.py
├── sql/
│   ├── executor.py
│   └── nl_to_sql.py
└── visualization/
    └── chart_generator.py
```

## Architecture

### 1. SQL Pipeline

For structured business questions:

1. Route question to `SQL`
2. Generate SQLite SQL with the LLM
3. Sanitize and repair SQL
4. Execute on `data/ecommerce.db`
5. Summarize the result in plain English
6. Render table + Plotly chart

### 2. RAG Pipeline

For review-focused questions:

1. Route question to `RAG`
2. Retrieve top review chunks from FAISS
3. Translate / clean snippets if needed
4. Run sentiment analysis
5. Extract top themes
6. Synthesize a business answer

### 3. Hybrid Pipeline

For questions requiring both analytics and customer feedback:

1. Route question to `HYBRID`
2. Split into SQL sub-question + RAG sub-question
3. Run both pipelines independently
4. Synthesize a combined grounded answer

## Tech Stack

- Python
- Streamlit
- SQLite
- pandas
- Plotly
- FAISS
- sentence-transformers
- requests
- Ollama + Mistral (local LLM)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama (local)

Make sure Ollama is running locally and the `mistral` model is available.

Example:

```bash
ollama run mistral
```

Optional runtime environment variables (local or cloud):

- `OLLAMA_URL` (default: `http://localhost:11434/api/generate`)
- `OLLAMA_MODEL` (default: `mistral`)

If Ollama is unavailable (common on Streamlit Community Cloud), the app still runs with deterministic fallbacks. Translation also falls back to a no-key Google translator implementation.

### 3. Run the app

```bash
streamlit run app.py
```

## Data Assets Required

These runtime assets are required for the app to function:

- `data/ecommerce.db`
- `rag/faiss_index.bin`
- `rag/review_chunks.pkl`

For local development, keep them in those exact paths.

For Streamlit Community Cloud, do not commit the large files to GitHub. Instead, provide downloadable URLs using either environment variables or Streamlit secrets:

- `ECOMMERCE_DB_URL`
- `FAISS_INDEX_URL`
- `REVIEW_CHUNKS_URL`

On startup, the app will automatically download any missing asset into the correct local path.

## Example Questions

### SQL

- Top 5 product categories by sales
- Total number of orders
- Average payment value
- Top 10 selling cities
- Total revenue by payment type
- What are the top 10 product categories by revenue?
- Show me monthly order volume for 2017
- Which state has the highest average delivery delay?

### RAG

- Why customers unhappy about delivery
- Customer complaints about packaging
- Why delivery delays happen
- Product quality issues reported
- Customer satisfaction feedback
- What do customers complain about most?
- What are customers saying about delivery speed?

### Hybrid

- Top selling products and customer complaints
- Sales trends and delivery feedback
- Best selling categories and customer reviews
- How do electronics category sales compare to customer satisfaction?
- Which sellers have the worst reviews and lowest revenue?

## Stability Notes

The project includes:

- SQL sanitization for SQLite compatibility
- SQL repair for common malformed LLM outputs
- validated deterministic fallbacks for important business query families
- persistent SQL cache in local runtime to keep previously validated queries stable across restarts

## Limitations

- The system is optimized for the Olist schema and common business-query families
- Fully arbitrary natural-language analytics questions are not guaranteed
- Local Ollama inference can be slower than hosted models
- Conversation history is session-based unless additional persistence is added

## Deployment Notes

For Streamlit Community Cloud or similar deployment:

- push the application code to GitHub
- do not commit `.env`
- do not commit local `__pycache__`
- do not commit `data/sql_query_cache.json`
- do not commit large runtime assets such as:
  - `data/ecommerce.db`
  - `rag/faiss_index.bin`
  - `rag/review_chunks.pkl`
- configure these secrets in Streamlit Cloud:
  - `ECOMMERCE_DB_URL`
  - `FAISS_INDEX_URL`
  - `REVIEW_CHUNKS_URL`

The deployed app can be created from `share.streamlit.io`, and the final public app URL will use the `*.streamlit.app` domain.

## Recommended Commit Set

Commit:

- `app.py`
- `requirements.txt`
- `sql/`
- `rag/`
- `router/`
- `visualization/`
- `README.md`
- `DESIGN_COVERAGE.md`
- `.gitignore`
- `bootstrap_assets.py`

Do not commit:

- `.env`
- `__pycache__/`
- local cache files
- large runtime assets
- `app_new.py` if you want the cleanest repo
