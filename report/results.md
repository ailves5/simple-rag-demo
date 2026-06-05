# RAG System — Part 4 Results

> Run date: 2026-06-04  
> LLM: **llama-3.3-70b-versatile** via Groq API (OpenAI-compatible)  
> Embeddings: `paraphrase-multilingual-mpnet-base-v2` (local, sentence-transformers)  
> Knowledge base: 3 files → 4 chunks (120 words / 30-word overlap) | top_k = 4

---

## 1. Knowledge Base

| File | Contents |
|---|---|
| `it_support.txt` | University IT support policies: password reset portal, support hours (Mon–Fri 09:00–18:00), hardware issue reporting, software installation approval. |
| `student_handbook.txt` | Student academic policies: course registration window (first 2 weeks), exam retake policy (once), library hours (08:00–22:00 daily), grade publication (student portal). |
| `knowledge.txt` | General-knowledge text used as a distractor: common misconceptions, geography facts (Everest, Amazon rainforest, Pacific Ocean). Contains **no** university-specific information. |

---

## 2. Example Questions and Generated Answers

### Q1 — When is IT support available? *(expected source: it_support.txt)*

| Rank | Source | Score | Chunk preview |
|---|---|---|---|
| 1 | it_support.txt | **0.3498** | "IT support is available Monday to Friday from 09:00 to 18:00…" |
| 2 | student_handbook.txt | 0.2825 | "Course registration is open during the first two weeks…" |
| 3 | knowledge.txt | 0.0715 | "It covers more than 30% of the planet's surface." |
| 4 | knowledge.txt | −0.0111 | "Humans only use 2% of their brains…" |

**Answer:**  
> IT support is available Monday to Friday from 09:00 to 18:00, as stated in the it_support.txt file.

---

### Q2 — How many times can a student retake a failed exam? *(expected source: student_handbook.txt)*

| Rank | Source | Score | Chunk preview |
|---|---|---|---|
| 1 | student_handbook.txt | **0.5080** | "Students may retake failed exams once…" |
| 2 | it_support.txt | 0.3025 | "IT support is available Monday to Friday…" |
| 3 | knowledge.txt | 0.1032 | "Humans only use 2% of their brains…" |
| 4 | knowledge.txt | 0.0559 | "It covers more than 30% of the planet's surface." |

**Answer:**  
> A student may retake a failed exam once. (Source: student_handbook.txt, chunk 0)

---

### Q3 — What are the library opening hours? *(expected source: student_handbook.txt)*

| Rank | Source | Score | Chunk preview |
|---|---|---|---|
| 1 | student_handbook.txt | **0.5699** | "The university library is open from 08:00 until 22:00 every day…" |
| 2 | it_support.txt | 0.2402 | "IT support is available Monday to Friday…" |
| 3 | knowledge.txt | 0.1078 | "It covers more than 30% of the planet's surface." |
| 4 | knowledge.txt | 0.0992 | "Humans only use 2% of their brains…" |

**Answer:**  
> The university library is open from 08:00 until 22:00 every day (Source 1: student_handbook.txt).

---

### Q4 — What is the capital city of Australia? *(out-of-scope — should refuse)*

| Rank | Source | Score | Chunk preview |
|---|---|---|---|
| 1 | knowledge.txt | 0.2472 | "Humans only use 2% of their brains…" |
| 2 | knowledge.txt | 0.1609 | "It covers more than 30% of the planet's surface." |
| 3 | it_support.txt | 0.0228 | "IT support is available Monday to Friday…" |
| 4 | student_handbook.txt | 0.0202 | "Course registration is open during the first two weeks…" |

**Answer:**  
> There is no information available in the provided context.

✅ Model correctly refused to hallucinate. All retrieval scores were low (max 0.2472), confirming the question is genuinely outside the knowledge base.

---

### Q5 — როდის არის უნივერსიტეტის ბიბლიოთეკა ღია? *(Georgian — library hours)*

| Rank | Source | Score | Chunk preview |
|---|---|---|---|
| 1 | student_handbook.txt | **0.6544** | "The university library is open from 08:00 until 22:00 every day…" |
| 2 | it_support.txt | 0.3674 | "IT support is available Monday to Friday…" |
| 3 | knowledge.txt | 0.1062 | "Humans only use 2% of their brains…" |
| 4 | knowledge.txt | 0.1043 | "It covers more than 30% of the planet's surface." |

**Answer:**  
> უნივერსიტეტის ბიბლიოთეკა ღიაა 08:00-დან 22:00 საათამდე, ყველა დღე. (წყარო: student_handbook.txt, chunk 0)

✅ Multilingual retrieval worked correctly. The multilingual embedding model matched the Georgian question to the English chunk (score 0.6544 — the highest of all five questions). The LLM answered in Georgian.

---

## 3. Improvements Implemented

### 4.1 — Overlapping word-based chunking
Instead of splitting on blank lines, text is split into overlapping word-based chunks:
- `chunk_size = 120` words per chunk
- `overlap = 30` words repeated between adjacent chunks

This preserves context at chunk boundaries and reduces the risk of an answer being split across two non-retrieved chunks.

### 4.2 — Configurable top_k
`top_k = 4` (previously hardcoded `k=2`). A single variable controls how many chunks are retrieved across the entire pipeline, making it easy to tune without touching the core logic.

### 4.3 — Improved prompt with out-of-context refusal
The prompt explicitly instructs the model to:
1. Use **only** information from the retrieved context.
2. Respond with *"There is no information available in the provided context."* when the answer is absent.
3. Cite the source file in the answer.

Demonstrated by Q4 (Australia capital) — the model refused cleanly rather than hallucinating.

### 4.4 — Interactive question-answering loop
A `while True: input(...)` loop (notebook cell 7) lets users ask arbitrary questions at runtime. Type `exit` to quit. Demonstrated below with a sample question.

**Interactive mode demo:**
```
Ask a question, or type 'exit': Where are final grades published?

Top 4 retrieved chunks:
  [1] source=student_handbook.txt | chunk_id=0 | score=0.4125
  [2] source=knowledge.txt       | chunk_id=1 | score=0.1081
  [3] source=it_support.txt      | chunk_id=0 | score=0.0896
  [4] source=knowledge.txt       | chunk_id=0 | score=0.0567

LLM Answer (llama-3.3-70b-versatile):
Final grades are published in the student portal. (Source: student_handbook.txt, chunk 0)

Exiting interactive mode.
```

### 4.5 — Different LLM: Groq / Llama 3.3 70B (replacing Azure GPT-4o)
The original notebook was wired for **Azure OpenAI** (AzureOpenAI client, `AZURE_OPENAI_*` env vars, `client.responses.create`). This has been replaced with:

- **Provider:** Groq (`https://api.groq.com/openai/v1`)
- **Model:** `llama-3.3-70b-versatile` (Meta Llama 3.3, 70B parameters)
- **Client:** Standard `openai.OpenAI` with `chat.completions.create` — no Azure dependencies
- **Config:** `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` from `.env`

This also makes the project portable: any OpenAI-compatible endpoint can be swapped in by changing the `.env` file only.

---

## 4. Performance Analysis

### What worked well

| Observation | Evidence |
|---|---|
| **Source routing was accurate** | Q1–Q3 and Q5 all ranked the correct source file at position 1 with a clearly higher score than runners-up. |
| **Out-of-scope refusal worked** | Q4 scores were all low (max 0.2472); the model correctly said no information was available rather than hallucinating "Canberra". |
| **Multilingual retrieval worked** | Q5 (Georgian) achieved the highest retrieval score of all five questions (0.6544), proving `paraphrase-multilingual-mpnet-base-v2` bridges languages effectively. |
| **Knowledge.txt distractor was harmless** | Despite being in the index, knowledge.txt chunks never ranked above the correct file for domain questions. |

### What could be improved

| Observation | Detail |
|---|---|
| **Small knowledge base → 4 chunks total** | Each file fits in a single chunk (all texts are short). The overlapping chunking has no measurable effect at this scale; its benefit would appear with longer documents. |
| **Q1 top score is low (0.3498)** | "When is IT support available?" is semantically less close to the chunk than "library hours" or "exam retake" questions. Score gap between rank 1 and rank 2 is only 0.0673, meaning a slightly different question phrasing could retrieve the wrong file first. |
| **knowledge.txt chunk 1 is a sentence fragment** | `"It covers more than 30% of the planet's surface."` (the Pacific Ocean sentence split off during chunking) appears repeatedly at rank 3–4 with low but non-zero scores — adding minor noise. With longer source documents and proper chunking, this artefact would not appear. |
| **No re-ranking step** | The pipeline uses raw cosine similarity for selection. A cross-encoder re-ranker would improve precision, especially for Q1 where the margin is narrow. |
