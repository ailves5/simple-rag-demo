"""
RAG batch runner — Part 4 evaluation script.
Runs 5 test questions through the RAG pipeline and saves:
  report/run_log.txt  — verbose console output
  report/results.md   — clean writeup

LLM calls require AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT in .env.
If the key is absent the script continues in retrieval-only mode and marks
answers as [LLM UNAVAILABLE — no API key].
"""

import os
import sys
import io
import textwrap
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Configuration ────────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 120   # words per chunk
OVERLAP    = 30    # words of overlap between chunks
TOP_K      = 4

DATA_DIR    = Path("data")
REPORT_DIR  = Path("report")
REPORT_DIR.mkdir(exist_ok=True)

# Questions: (label, question_text, expected_source_hint)
QUESTIONS = [
    # 1 – answerable from it_support.txt (new file)
    ("Q1 [it_support.txt]",
     "When is IT support available?",
     "it_support.txt"),

    # 2 – answerable from student_handbook.txt (new file)
    ("Q2 [student_handbook.txt]",
     "How many times can a student retake a failed exam?",
     "student_handbook.txt"),

    # 3 – answerable from student_handbook.txt (new file)
    ("Q3 [student_handbook.txt]",
     "What are the library opening hours?",
     "student_handbook.txt"),

    # 4 – OUT-OF-SCOPE: nothing in the KB about this
    ("Q4 [out-of-scope]",
     "What is the capital city of Australia?",
     None),

    # 5 – Non-English: Georgian "When is the university library open?"
    #     The multilingual model should map this to the student_handbook chunk.
    ("Q5 [non-English / Georgian]",
     "როდის არის უნივერსიტეტის ბიბლიოთეკა ღია?",
     "student_handbook.txt"),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

class Tee:
    """Writes to both stdout and a string buffer."""
    def __init__(self):
        self._buf = io.StringIO()
    def write(self, text):
        sys.__stdout__.write(text)
        self._buf.write(text)
    def flush(self):
        sys.__stdout__.flush()
    def getvalue(self):
        return self._buf.getvalue()


def chunk_text(text, source, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    chunk_id = 0
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append({
            "text":     " ".join(words[start:end]),
            "source":   source,
            "chunk_id": chunk_id,
        })
        chunk_id += 1
        start += chunk_size - overlap
    return chunks


def get_embedding(text, model):
    emb = model.encode(text)
    emb = emb / np.linalg.norm(emb)
    return emb.astype("float32")


def build_prompt(question, retrieved_metadata):
    context_blocks = []
    for i, item in enumerate(retrieved_metadata, start=1):
        context_blocks.append(
            f"[Source {i}: {item['source']}, chunk {item['chunk_id']}]\n"
            f"{item['text']}"
        )
    context = "\n\n".join(context_blocks)
    return textwrap.dedent(f"""
        You are a helpful assistant answering questions using only the provided context.

        Rules:
        1. Use only the information from the context.
        2. If the answer is not in the context, say: "There is no information available in the provided context."
        3. Keep the answer clear and concise.
        4. Mention the source file when possible.

        Context:
        {context}

        Question:
        {question}

        Answer:
    """).strip()


# ── Setup LLM client (optional) ──────────────────────────────────────────────

api_key      = os.getenv("AZURE_OPENAI_API_KEY")
endpoint     = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment   = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
LLM_AVAILABLE = bool(api_key and endpoint and "your_key_here" not in api_key)

client = None
if LLM_AVAILABLE:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=endpoint)


def call_llm(prompt):
    if not LLM_AVAILABLE:
        return "[LLM UNAVAILABLE — no API key. Retrieval-only mode.]"
    try:
        response = client.responses.create(
            model=deployment,
            input=prompt,
            temperature=0,
        )
        return response.output_text
    except Exception as e:
        return f"[LLM ERROR: {e}]"


# ── Main ─────────────────────────────────────────────────────────────────────

tee = Tee()
sys.stdout = tee

print("=" * 72)
print("RAG BATCH RUNNER — Part 4")
print("=" * 72)
print(f"\nEmbedding model : {EMBEDDING_MODEL_NAME}")
print(f"Chunk size      : {CHUNK_SIZE} words  |  Overlap: {OVERLAP} words")
print(f"Top-k           : {TOP_K}")
print(f"LLM available   : {LLM_AVAILABLE}")
if not LLM_AVAILABLE:
    print("  *** Running in RETRIEVAL-ONLY mode — answers will show placeholder ***")
print()

# 1. Load knowledge files
print("── Loading knowledge base ──────────────────────────────────────────────")
file_texts = {}
for fp in sorted(DATA_DIR.glob("*.txt")):
    text = fp.read_text(encoding="utf-8")
    file_texts[fp.name] = text
    print(f"  {fp.name}: {len(text.split())} words")

# 2. Chunk
chunk_objects = []
for filename, text in file_texts.items():
    chunk_objects.extend(chunk_text(text, source=filename))
chunks = [c["text"] for c in chunk_objects]
print(f"\nTotal chunks    : {len(chunk_objects)}")
print("\nExample chunks (first 3):")
for c in chunk_objects[:3]:
    print(f"  [{c['source']} | chunk {c['chunk_id']}] "
          f"{c['text'][:80]}{'...' if len(c['text']) > 80 else ''}")

# 3. Embeddings
print("\n── Loading embedding model ─────────────────────────────────────────────")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"  Model loaded: {EMBEDDING_MODEL_NAME}")

chunk_embeddings = np.array(
    [get_embedding(c, embedding_model) for c in chunks]
).astype("float32")
embedding_dim = chunk_embeddings.shape[1]
print(f"  Embedding dim : {embedding_dim}")

# 4. FAISS index
index = faiss.IndexFlatIP(embedding_dim)
index.reset()
index.add(chunk_embeddings)
print(f"  FAISS index   : {index.ntotal} vectors (IndexFlatIP / cosine)")

# 5. Run queries
results_for_md = []  # collect structured results

for label, question, _ in QUESTIONS:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"{label}")
    print(f"Question: {question}")
    print(sep)

    q_emb = np.array([get_embedding(question, embedding_model)]).astype("float32")
    distances, indices = index.search(q_emb, TOP_K)

    retrieved = []
    for score, idx in zip(distances[0], indices[0]):
        retrieved.append({
            "score":    float(score),
            "source":   chunk_objects[idx]["source"],
            "chunk_id": chunk_objects[idx]["chunk_id"],
            "text":     chunk_objects[idx]["text"],
        })

    print(f"\nTop {TOP_K} retrieved chunks:")
    for i, item in enumerate(retrieved, start=1):
        print(f"  [{i}] {item['source']} | chunk {item['chunk_id']} "
              f"| score {item['score']:.4f}")
        print(f"       {item['text'][:110]}{'...' if len(item['text']) > 110 else ''}")

    prompt = build_prompt(question, retrieved)
    print("\n── RAG Prompt ──────────────────────────────────────────────────────────")
    print(prompt)

    answer = call_llm(prompt)
    print("\n── Answer ──────────────────────────────────────────────────────────────")
    print(answer)

    results_for_md.append({
        "label":     label,
        "question":  question,
        "retrieved": retrieved,
        "prompt":    prompt,
        "answer":    answer,
    })

sys.stdout = sys.__stdout__

# ── Save run_log.txt ─────────────────────────────────────────────────────────
log_path = REPORT_DIR / "run_log.txt"
log_path.write_text(tee.getvalue(), encoding="utf-8")
print(f"Saved: {log_path}")

# ── Save results.md ──────────────────────────────────────────────────────────

def md_table_row(*cols):
    return "| " + " | ".join(str(c) for c in cols) + " |"


md_lines = []
md_lines += [
    "# RAG System — Part 4 Results\n",
    f"> Generated by `run.py` — {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
    f"> LLM: {'Azure OpenAI (' + deployment + ')' if LLM_AVAILABLE else '**UNAVAILABLE** — retrieval-only run'}\n",
    f"> Embedding model: `{EMBEDDING_MODEL_NAME}`\n",
    "",
    "---\n",
    "## 1. Knowledge Base Description\n",
    "| File | Content summary | Words |",
    "|------|----------------|-------|",
]
kb_desc = {
    "knowledge.txt": "Original baseline file — geography facts, common myths "
                     "(Earth's oceans, Amazon rainforest, Mount Everest, etc.)",
    "it_support.txt": "**[Added Part 4]** University IT-support policies — "
                      "password reset, support hours, hardware/software procedures.",
    "student_handbook.txt": "**[Added Part 4]** University student policies — "
                            "course registration, exam retakes, library hours, grade portal.",
}
for fname, text in file_texts.items():
    desc = kb_desc.get(fname, "—")
    md_lines.append(md_table_row(f"`{fname}`", desc, len(text.split())))
md_lines += [
    "",
    f"**Chunking:** overlapping word-based chunks "
    f"(size={CHUNK_SIZE} words, overlap={OVERLAP} words) → **{len(chunk_objects)} chunks** total.\n",
    "",
    "---\n",
    "## 2. Example Questions and Generated Answers\n",
]

for r in results_for_md:
    md_lines += [
        f"### {r['label']}\n",
        f"**Query:** {r['question']}\n",
        "",
        f"**Top-{TOP_K} retrieved chunks:**\n",
        "| Rank | Source | Chunk | Score | Preview (first 90 chars) |",
        "|------|--------|-------|-------|--------------------------|",
    ]
    for i, item in enumerate(r["retrieved"], start=1):
        preview = item["text"][:90].replace("|", "\\|") + ("…" if len(item["text"]) > 90 else "")
        md_lines.append(md_table_row(
            i, f"`{item['source']}`", item["chunk_id"],
            f"{item['score']:.4f}", preview,
        ))
    md_lines += [
        "",
        "<details><summary>Full RAG prompt (click to expand)</summary>\n",
        "```",
        r["prompt"],
        "```\n",
        "</details>\n",
        "",
        f"**Generated answer:**\n",
        f"> {r['answer'].strip()}\n",
        "",
        "---\n",
    ]

md_lines += [
    "## 3. Improvements Implemented\n",
    "| # | Improvement | File / Location |",
    "|---|-------------|-----------------|",
    md_table_row(1, "**Overlapping word-based chunking** — "
                 "replaces `split('\\n\\n')` with sliding-window chunks "
                 f"(size={CHUNK_SIZE}, overlap={OVERLAP})",
                 "`simple_rag_demo.ipynb` — cell *2. Load all knowledge files*, "
                 "function `chunk_text()`"),
    md_table_row(2, "**Configurable `top_k`** — "
                 "retrieves 4 chunks instead of a hardcoded k=2",
                 "`simple_rag_demo.ipynb` — cell *6. Test questions*, `top_k = 4`"),
    md_table_row(3, "**Improved prompt design** — "
                 "system rules that restrict answers to provided context and "
                 "require the model to admit missing information",
                 "`simple_rag_demo.ipynb` — cell *6. Test questions*, `prompt = ...`"),
    md_table_row(4, "**Interactive Q&A loop** — "
                 "optional `while True` loop for manual testing",
                 "`simple_rag_demo.ipynb` — cell *7. Improvement 4*"),
    md_table_row(5, "**Multi-language embedding model** — "
                 "`paraphrase-multilingual-mpnet-base-v2` supports 50+ languages, "
                 "enabling non-English queries",
                 "`simple_rag_demo.ipynb` — cell *2. Load environment variables*, "
                 "`SentenceTransformer(\"paraphrase-multilingual-mpnet-base-v2\")`"),
    md_table_row(6, "**Two new domain KB files** — "
                 "`it_support.txt` and `student_handbook.txt`",
                 "`data/it_support.txt`, `data/student_handbook.txt`"),
    "",
    "---\n",
    "## 4. Performance Analysis\n",
]

# Build performance analysis from actual scores
for r in results_for_md:
    top_score = r["retrieved"][0]["score"]
    top_source = r["retrieved"][0]["source"]
    md_lines += [
        f"**{r['label']}** (`{r['question'][:60]}{'…' if len(r['question'])>60 else ''}`)",
        f"- Best score: **{top_score:.4f}** from `{top_source}`",
    ]
    if top_score > 0.5:
        md_lines.append("- ✅ High-confidence retrieval — relevant chunk found at top position.")
    elif top_score > 0.35:
        md_lines.append("- ⚠️  Moderate confidence — chunk retrieved but semantic overlap is partial.")
    else:
        md_lines.append("- ❌ Low confidence — no strongly matching chunk found (expected for out-of-scope queries).")
    md_lines.append("")

md_lines += [
    "**Overlap chunking effect:** Because knowledge files are short (< 200 words each), "
    "a single overlap pass creates several near-duplicate chunks per file. "
    "This slightly inflates the index but ensures boundary sentences aren't lost.\n",
    "",
    "**Multilingual retrieval (Q5):** The `paraphrase-multilingual-mpnet-base-v2` model "
    "maps semantically equivalent phrases across languages to nearby embedding space. "
    "If Q5 scores above 0.35 for `student_handbook.txt`, the multilingual capability "
    "is confirmed working even without any translation step.\n",
    "",
    "---\n",
    "## 5. Challenges Encountered\n",
    "- **No LLM API key in this environment** — the notebook targets Azure OpenAI "
    "(GPT-4o). The runner ran in retrieval-only mode; generated answers show a placeholder. "
    "All retrieval, chunking, and embedding behaviour is fully observable without the LLM.\n",
    "- **Small KB size** — with only three files (≈ 250 words total), the FAISS index holds "
    f"{len(chunk_objects)} chunks. Out-of-scope queries still return chunks (cosine similarity "
    "always finds *something*); the improved prompt guards against the LLM treating those as "
    "valid answers.\n",
    "- **Embedding cache** — the notebook sets `HF_HOME` to a Windows path (`E:\\\\...`) which "
    "is ignored on macOS/Linux. The model re-downloads to the default HuggingFace cache on "
    "first run; this is harmless but worth noting for portability.\n",
]

md_path = REPORT_DIR / "results.md"
md_path.write_text("\n".join(md_lines), encoding="utf-8")
print(f"Saved: {md_path}")
print("\nDone.")
