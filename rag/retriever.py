# =========================================
# MULTILINGUAL RAG RETRIEVER
# Uses Translation + FAISS
# =========================================

import faiss
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer

# Import translator
from rag.translator import (
    translate_to_portuguese,
    translate_to_english
)

# =========================================
# LOAD EMBEDDING MODEL
# =========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================
# LOAD FAISS INDEX
# =========================================

print("Loading FAISS index...")

index = faiss.read_index(
    "rag/faiss_index.bin"
)

# =========================================
# LOAD TEXT CHUNKS
# =========================================

with open("rag/review_chunks.pkl", "rb") as f:

    chunks = pickle.load(f)

print("Chunks loaded:", len(chunks))


# =========================================
# RETRIEVE FUNCTION
# =========================================

def retrieve_reviews(query, top_k=5):

    print("\nOriginal Query:", query)

    # Step 1 — Translate query
    translated_query = translate_to_portuguese(query)

    print("Translated Query:", translated_query)

    # Step 2 — Encode query
    query_embedding = model.encode(
        [translated_query]
    )

    # Step 3 — Search FAISS
    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    # Step 4 — Retrieve chunks
    retrieved_chunks = [
        chunks[i]
        for i in indices[0]
    ]

    # Step 5 — Translate results back to English
    translated_results = []

    for chunk in retrieved_chunks:

        english_chunk = translate_to_english(chunk)

        if english_chunk and english_chunk not in translated_results:
            translated_results.append(english_chunk)

    return translated_results


# =========================================
# TEST BLOCK
# =========================================

if __name__ == "__main__":

    test_query = "late delivery problems"

    results = retrieve_reviews(test_query)

    print("\nTop Retrieved Reviews:\n")

    for r in results:

        print("-", r[:200])
