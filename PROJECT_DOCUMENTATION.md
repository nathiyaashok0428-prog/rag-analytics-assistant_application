# Project Documentation

## Project Name

LLM-Powered Analytics Assistant with RAG

## Objective

This project enables business users to ask natural-language questions over:

- structured e-commerce data in SQLite
- unstructured customer review text using RAG

The system routes each user question into one of three paths:

- `SQL`
- `RAG`
- `HYBRID`

and returns grounded, explainable results in a Streamlit interface.

## Business Problem

Traditional analytics workflows require:

- manual SQL writing
- separate qualitative review analysis
- technical skill from the end user

This project reduces that friction by turning business questions into:

- structured metrics
- feedback insights
- combined hybrid analysis

## Dataset

The implementation is built on the Olist Brazilian e-commerce dataset and uses these core entities:

- `orders`
- `order_items`
- `products`
- `customers`
- `payments`
- `reviews`
- `sellers`

## System Components

### 1. UI Layer

File:
- `app.py`
- `bootstrap_assets.py`

Responsibilities:
- Streamlit layout
- sidebar history
- prompt cards
- question submission
- rendering tables, charts, insights
- runtime asset bootstrapping for cloud deployment

### 2. Query Routing

File:
- `router/query_router.py`

Responsibilities:
- classify query into `SQL`, `RAG`, `HYBRID`, or `UNKNOWN`
- decompose hybrid queries into structured and review-oriented sub-questions

### 3. SQL Pipeline

Files:
- `sql/nl_to_sql.py`
- `sql/executor.py`

Responsibilities:
- generate SQL from natural language
- sanitize SQL for SQLite compatibility
- repair malformed SQL
- apply validated fallback templates for high-value business query families
- execute SQL and return pandas DataFrames
- persist validated SQL for restart stability

### 4. RAG Pipeline

Files:
- `rag/retriever.py`
- `rag/translator.py`
- `rag/sentiment_analysis.py`
- `rag/theme_extractor.py`
- `rag/synthesizer.py`

Responsibilities:
- retrieve semantically relevant review chunks
- clean and translate snippets when needed
- label sentiment
- extract top complaint / feedback themes
- synthesize natural-language answers

### 5. Visualization Layer

File:
- `visualization/chart_generator.py`

Responsibilities:
- choose chart shape from SQL result structure
- render Plotly charts
- support line, bar, and pie outputs where appropriate

## Query Flow

### SQL Query Flow

1. User asks a structured analytics question
2. Router returns `SQL`
3. SQL prompt is sent to the local LLM
4. SQL is sanitized and validated
5. If needed, repair/fallback logic is applied
6. SQL is executed in SQLite
7. DataFrame is summarized and charted

### RAG Query Flow

1. User asks a feedback-oriented question
2. Router returns `RAG`
3. Relevant review chunks are retrieved from FAISS
4. Sentiment and themes are extracted
5. Answer is synthesized from retrieved evidence

### Hybrid Query Flow

1. User asks a mixed business + review question
2. Router returns `HYBRID`
3. Query is split into:
   - SQL sub-question
   - RAG sub-question
4. Both pipelines run independently
5. Final answer is synthesized from both outputs

## Reliability Improvements Added

To make the project stable for demo use, the following reliability features were added:

- SQLite-specific SQL cleanup
- correction for invalid aliases and broken joins
- deterministic fallback SQL templates for key business questions
- persistent validated SQL cache in `data/sql_query_cache.json`
- metric-aware chart selection
- fallback summarization when the LLM answer is weak

## Example Supported Questions

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

### Hybrid

- Top selling products and customer complaints
- Sales trends and delivery feedback
- Best selling categories and customer reviews
- How do electronics category sales compare to customer satisfaction?
- Which sellers have the worst reviews and lowest revenue?

## Known Limitations

- The app is optimized for the Olist schema and project business-use cases
- It is not a universal NL-to-SQL engine
- Some business intents use deterministic fallback SQL templates for stability
- Local LLM inference may feel slow compared with hosted APIs
- Chat history is session-based in the UI layer unless extra persistence is added

## Deployment Notes

To deploy successfully, make sure the following are included:

- `app.py`
- `requirements.txt`
- source modules under `sql/`, `rag/`, `router/`, `visualization/`
- `data/ecommerce.db`
- `rag/faiss_index.bin`
- `rag/review_chunks.pkl`

Do not commit:

- `.env`
- `__pycache__`
- local runtime caches unless intentionally desired

## Demo Talking Points

- The app supports three pipelines: `SQL`, `RAG`, `HYBRID`
- Routing decides which pipeline to trigger
- SQL results are grounded in SQLite execution
- RAG results are grounded in retrieved review text
- Hybrid answers combine both structured and unstructured evidence
- Stability is improved through SQL sanitization, repair, fallback templates, and persistent validated query caching
