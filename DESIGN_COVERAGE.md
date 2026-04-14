# Design Document Coverage Review

This file maps the current implementation to the project design document for the **LLM-Powered Analytics Assistant with RAG**.

## Summary

The current implementation covers the core architecture described in the design document:

- `SQL` pipeline
- `RAG` pipeline
- `HYBRID` pipeline
- query routing
- chart generation
- Streamlit UI

The project is in a strong **prototype / demo-ready** state. It is not fully production-hardened, but the main scenarios in the design document are implemented and working.

---

## 1. Data Ingestion & Preparation

### Design Doc Expectation

- Load Olist CSVs into SQLite
- Prepare review text for retrieval
- Build FAISS index using sentence-transformers

### Current Implementation

- SQLite database present at `data/ecommerce.db`
- Loader script present at `data/olist_loader.py`
- Review retrieval assets present:
  - `rag/faiss_index.bin`
  - `rag/review_chunks.pkl`
- Retriever uses FAISS and sentence-transformers outputs

### Coverage

`Covered`

---

## 2. SQL Path

### Design Doc Expectation

- Schema-aware prompt
- Generate valid SQLite SQL
- Execute query
- Summarize result

### Current Implementation

- Implemented in `sql/nl_to_sql.py` and `sql/executor.py`
- Includes:
  - schema prompt
  - SQL sanitization
  - SQL repair
  - validated fallbacks for key business intents
  - persistent validated SQL cache across restarts
- Summarization handled in `rag/synthesizer.py`

### Coverage

`Covered`

### Notes

This part is stronger than a basic design-doc version because it now includes:

- SQLite-specific SQL cleanup
- deterministic fallback templates for high-risk questions
- persistent SQL stabilization across server restarts

---

## 3. RAG Path

### Design Doc Expectation

- Retrieve top review chunks
- Use retrieved text as grounding context
- Perform sentiment classification
- Extract complaint themes

### Current Implementation

- Retrieval in `rag/retriever.py`
- Translation cleanup in `rag/translator.py`
- Sentiment in `rag/sentiment_analysis.py`
- Theme extraction in `rag/theme_extractor.py`
- Response synthesis in `rag/synthesizer.py`

### Coverage

`Covered`

---

## 4. Hybrid Path

### Design Doc Expectation

- Run SQL and RAG together
- Merge both outputs
- Produce one combined answer

### Current Implementation

- Query decomposition in `router/query_router.py`
- SQL sub-query + RAG sub-query handled separately in `app.py`
- Combined answer synthesis in `rag/synthesizer.py`

### Coverage

`Covered`

---

## 5. Query Router

### Design Doc Expectation

- Classify into `SQL`, `RAG`, or `HYBRID`

### Current Implementation

- `router/query_router.py`
- Also includes `UNKNOWN` safety fallback

### Coverage

`Covered`

---

## 6. Auto Chart Generation

### Design Doc Expectation

- Choose a chart type
- Render with Plotly

### Current Implementation

- `visualization/chart_generator.py`
- Plotly-based chart rendering
- Metric-aware chart selection
- Integrated into `app.py`

### Coverage

`Covered`

### Notes

The design doc says chart type can be LLM-determined.  
Current implementation uses deterministic chart selection logic instead of an LLM prompt for every chart. This is a reasonable engineering choice for reliability and speed.

---

## 7. Streamlit UI

### Design Doc Expectation

- Web app interface
- Accept natural-language questions
- Show results inline

### Current Implementation

- `app.py`
- Chat-style UI
- Prompt cards
- Sidebar history
- SQL/RAG/HYBRID result rendering
- Plotly charts inline

### Coverage

`Covered`

---

## 8. Deployment Readiness

### Design Doc Expectation

- Deployable Streamlit app
- Code + docs + requirements

### Current Implementation

- `app.py` is the runtime entry point
- `requirements.txt` exists
- Local runtime assets are supported
- Missing runtime assets can now be downloaded at startup through environment variables or Streamlit secrets
- This repo now includes:
  - `README.md`
  - `.gitignore`
  - `DESIGN_COVERAGE.md`

### Coverage

`Covered for local/demo deployment`

### Remaining External Deliverables

These are outside the codebase itself and still need to be prepared if required by evaluation:

- live deployment URL
- demo video
- notebook / EDA artifact
- dataset Drive link

---

## Design Doc Gaps / Honest Caveats

These are the areas where the implementation is good, but not a perfect one-to-one match:

1. Chart type selection is deterministic rather than LLM-driven.
2. Some high-value business questions use validated fallback SQL templates.
   - This is acceptable for reliability in demo/prototype conditions.
   - It should be described as a **guarded fallback strategy**, not as random hardcoding.
3. Full arbitrary NL-to-SQL coverage is not guaranteed.
4. Conversation persistence across full server restarts is not implemented yet.
   - SQL stability across restarts is implemented through persistent SQL cache.

---

## Overall Verdict

### Core functionality

`Covered`

### Demo readiness

`Strong`

### Production readiness

`Prototype / pre-production`

The implementation covers the main design-document scenarios well enough for:

- mentor demo
- project review
- GitHub submission
- local deployment demonstration
