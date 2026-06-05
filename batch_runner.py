"""
batch_runner.py  –  report-ready runner for Part 4 RAG improvements
Reads OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL from .env
Works with Groq (or any OpenAI-compatible endpoint).
Embeddings: local sentence-transformers  (unchanged from notebook)
"""

import os
import sys
import io
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv(override=True)

# ── Groq / OpenAI-compatible client (Improvement 4.5) ──────────────────────
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")

# ── Embedding model (unchanged) ─────────────────────────────────────────────
print("Loading embedding model…", flush=True)
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


def get_embedding(text: str) -> np.ndarray:
    emb = embedding_model.encode(text)
    emb = emb / np.linalg.norm(emb)
    return emb.astype("float32")


# ── Improvement 4.1: overlapping word-based chunking ─────────────────────────
def chunk_text(text: str, source: str, chunk_size: int = 120, overlap: int = 30):
    words = text.split()
    chunks, chunk_id, start = [], 0, 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append({"text": " ".join(chunk_words), "source": source, "chunk_id": chunk_id})
        chunk_id += 1
        start += chunk_size - overlap
    return chunks


# ── Load knowledge base ──────────────────────────────────────────────────────
data_dir = Path("data")
file_texts = {}
for fp in sorted(data_dir.glob("*.txt")):
    file_texts[fp.name] = fp.read_text(encoding="utf-8")

chunk_objects = []
for filename, text in file_texts.items():
    chunk_objects.extend(chunk_text(text, source=filename))
chunks = [c["text"] for c in chunk_objects]

print(f"Loaded {len(file_texts)} knowledge file(s): {list(file_texts.keys())}")
print(f"Created {len(chunk_objects)} chunk(s).\n")

# ── Build FAISS index ────────────────────────────────────────────────────────
chunk_embeddings = np.array([get_embedding(c) for c in chunks])
embedding_dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(chunk_embeddings)
print(f"FAISS index ready, {index.ntotal} vectors, dim={embedding_dim}\n")

# ── Improvement 4.2: configurable top_k ─────────────────────────────────────
top_k = 4


# ── Core RAG function ────────────────────────────────────────────────────────
def ask(question: str, verbose: bool = True) -> str:
    q_emb = np.array([get_embedding(question)])
    distances, indices = index.search(q_emb, top_k)

    retrieved = []
    for score, idx in zip(distances[0], indices[0]):
        retrieved.append({
            "score": float(score),
            "source": chunk_objects[idx]["source"],
            "chunk_id": chunk_objects[idx]["chunk_id"],
            "text": chunk_objects[idx]["text"],
        })

    if verbose:
        print(f"\nTop {top_k} retrieved chunks:")
        for i, r in enumerate(retrieved, 1):
            print(f"  [{i}] source={r['source']} | chunk_id={r['chunk_id']} | score={r['score']:.4f}")
            print(f"       {r['text'][:110]}…")

    # ── Improvement 4.3: improved prompt that refuses out-of-context questions
    context_blocks = []
    for i, r in enumerate(retrieved, 1):
        context_blocks.append(
            f"[Source {i}: {r['source']}, chunk {r['chunk_id']}]\n{r['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful assistant answering questions using only the provided context.\n\n"
        "Rules:\n"
        "1. Use only the information from the context.\n"
        '2. If the answer is not in the context, say: "There is no information available in the provided context."\n'
        "3. Keep the answer clear and concise.\n"
        "4. Mention the source file when possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    if verbose:
        print("\n── Constructed prompt ──────────────────────────────────────────────────────")
        print(prompt)
        print("────────────────────────────────────────────────────────────────────────────")

    # ── Improvement 4.5: Groq via standard chat completions (not Azure) ───────
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()

    if verbose:
        print(f"\nLLM Answer ({MODEL}):")
        print(answer)

    return answer


# ── Batch run (Q1-Q5) ────────────────────────────────────────────────────────
QUESTIONS = [
    ("Q1", "When is IT support available?"),
    ("Q2", "How many times can a student retake a failed exam?"),
    ("Q3", "What are the library opening hours?"),
    ("Q4", "What is the capital city of Australia?"),
    ("Q5", "როდის არის უნივერსიტეტის ბიბლიოთეკა ღია?"),
]

print("\n" + "=" * 80)
print("BATCH RUN – Q1 through Q5")
print("=" * 80)

answers = {}
for label, question in QUESTIONS:
    print(f"\n{'=' * 80}")
    print(f"{label}: {question}")
    ans = ask(question, verbose=True)
    answers[label] = {"question": question, "answer": ans}
    print()

# ── Improvement 4.4: interactive mode demo ───────────────────────────────────
print("\n" + "=" * 80)
print("INTERACTIVE MODE DEMO (Improvement 4.4)")
print("=" * 80)
print('Simulated input: "Where are final grades published?"\n')
demo_q = "Where are final grades published?"
print(f"Ask a question, or type 'exit': {demo_q}")
ask(demo_q, verbose=True)

print("\nExiting interactive mode.")
