import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from bootstrap_assets import ensure_runtime_assets
from rag.translator import translate_to_english, translate_to_portuguese


INDEX_PATH = Path("rag/faiss_index.bin")
CHUNKS_PATH = Path("rag/review_chunks.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = None
index = None
chunks = None


def _ensure_loaded() -> None:
    global model, index, chunks

    if model is not None and index is not None and chunks is not None:
        return

    ensure_runtime_assets()

    if model is None:
        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)

    if index is None:
        print("Loading FAISS index...")
        index = faiss.read_index(str(INDEX_PATH))

    if chunks is None:
        with CHUNKS_PATH.open("rb") as file_handle:
            chunks = pickle.load(file_handle)
        print("Chunks loaded:", len(chunks))


def retrieve_reviews(query, top_k=5):
    _ensure_loaded()

    translated_query = translate_to_portuguese(query)
    query_embedding = model.encode([translated_query])
    _, indices = index.search(np.array(query_embedding), top_k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    translated_results = []

    for chunk in retrieved_chunks:
        english_chunk = translate_to_english(chunk)
        if english_chunk and english_chunk not in translated_results:
            translated_results.append(english_chunk)

    return translated_results
