# =========================================
# RAG EMBEDDING PIPELINE
# =========================================

import sqlite3
import pandas as pd
import numpy as np
import faiss
import pickle

from sentence_transformers import SentenceTransformer

# ===============================
# LOAD DATABASE
# ===============================

print("Connecting to database...")

conn = sqlite3.connect("data/ecommerce.db")

query = """
SELECT review_comment_message
FROM reviews
WHERE review_comment_message != ''
"""

df = pd.read_sql(query, conn)

conn.close()

print("Reviews loaded:", df.shape)

# ===============================
# CHUNK TEXT
# ===============================

print("Chunking text...")

CHUNK_SIZE = 200

def chunk_text(text):

    words = text.split()

    chunks = []

    for i in range(0, len(words), CHUNK_SIZE):

        chunk = " ".join(words[i:i + CHUNK_SIZE])

        chunks.append(chunk)

    return chunks


all_chunks = []

for text in df["review_comment_message"]:

    chunks = chunk_text(text)

    all_chunks.extend(chunks)

print("Total chunks:", len(all_chunks))

# ===============================
# LOAD EMBEDDING MODEL
# ===============================

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ===============================
# GENERATE EMBEDDINGS
# ===============================

print("Generating embeddings...")

embeddings = model.encode(
    all_chunks,
    show_progress_bar=True
)

embeddings = np.array(embeddings)

print("Embeddings shape:", embeddings.shape)

# ===============================
# BUILD FAISS INDEX
# ===============================

print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index created")

# ===============================
# SAVE FILES
# ===============================

print("Saving files...")

faiss.write_index(
    index,
    "rag/faiss_index.bin"
)

with open("rag/review_chunks.pkl", "wb") as f:

    pickle.dump(all_chunks, f)

print("✅ Embedding pipeline completed")