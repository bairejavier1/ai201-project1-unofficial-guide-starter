# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

This system covers student-generated reviews of professors at Florida International University (FIU) across three departments: Computer Science, Computer Engineering, and Information Systems. This knowledge is valuable because official university channels — course catalogs, syllabi, and department websites — only describe what a course covers, not what it's actually like to take it. Rate My Professors captures honest, first-hand student experiences including teaching quality, exam difficulty, grading fairness, and workload — information that is impossible to find through official channels and critical for students making course registration decisions.

---

## Documents

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Rate My Professors | CS — Caryl Rahn (Overall: 2.1/5) | documents/cs_prof_Caryl_Rahn.txt |
| 2 | Rate My Professors | CS — Gregory Reis (Overall: 4.9/5) | documents/cs_prof_Gregory_Reis.txt |
| 3 | Rate My Professors | CS — Ning Xie (Overall: 4.0/5) | documents/cs_prof_Ning_Xie.txt |
| 4 | Rate My Professors | CS — Tiana Solis (Overall: 2.9/5) | documents/cs_prof_Tiana_Solis.txt |
| 5 | Rate My Professors | CE — Vladimir Pozdin (Overall: 2.0/5) | documents/ce_prof_Vladimir_Pozdin.txt |
| 6 | Rate My Professors | CE — Nonnarit O-Larnnithipong (Overall: 4.9/5) | documents/ce_prof_Nonnarit_O-Larnnithipong.txt |
| 7 | Rate My Professors | CE — Fatima Boujarwah (Overall: 4.0/5) | documents/ce_prof_Fatima_Boujarwah.txt |
| 8 | Rate My Professors | IS — Andres Rodriguez (Overall: 5.0/5) | documents/is_prof_Andres_Rodriguez.txt |
| 9 | Rate My Professors | IS — David Palmer (Overall: 4.2/5) | documents/is_prof_David_Palmer.txt |
| 10 | Rate My Professors | IS — Rehan Akbar (Overall: 4.2/5) | documents/is_prof_Rehan_Akbar.txt |

---

## Chunking Strategy

**Chunk size:** 500 characters

**Overlap:** 50 characters

**Reasoning:** Each document consists of short, individual student reviews — typically 2 to 5 sentences long. A 500-character chunk is large enough to contain one complete review with its metadata (Quality, Difficulty, Course), which is the natural unit of meaning in this corpus. Going smaller (e.g. 200 characters) would split individual reviews mid-sentence, destroying the context needed to understand the student's opinion. Going larger (e.g. 1000+ characters) would merge multiple reviews into one chunk, diluting the signal when a query is looking for a specific type of feedback. A 50-character overlap ensures that if a review happens to straddle a chunk boundary, the key content still appears in at least one retrievable chunk.

---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers (runs locally, no API key required)

**Top-k:** 5

**Production tradeoff reflection:** For a production deployment, I would weigh several tradeoffs. all-MiniLM-L6-v2 is fast and free but has a 256-token context limit, which is fine for short reviews but would truncate longer documents. A model like text-embedding-3-small (OpenAI) or voyage-large-2 offers higher accuracy on domain-specific text and longer context windows, but introduces API cost and latency. If FIU had international students writing reviews in Spanish or Creole, multilingual support (e.g. paraphrase-multilingual-MiniLM-L12-v2) would become a priority. For this project, all-MiniLM-L6-v2 is the right choice: it is accurate enough for short English review text, runs locally with no cost or rate limits, and keeps the pipeline simple.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about Gregory Reis's exams and grading? | Students say his exams are not proctored, tests are online, and grading is lenient. He accepts late work and is very understanding. |
| 2 | Which Computer Science professor should students avoid and why? | Students strongly warn against Caryl Rahn due to strict and inconsistent grading, and accusations of sending students to conduct for AI use. |
| 3 | Is Vladimir Pozdin's class recommended for beginners? | No — students say the class is very fast-paced, assumes prior knowledge of circuits, and is not suitable for beginners. |
| 4 | What is the workload like for Andres Rodriguez's class? | Students describe the workload as light and manageable — quizzes, discussions, essays, and a midterm/final, with a very flexible and understanding professor. |
| 5 | How do students describe Rehan Akbar's teaching style? | Students say he reads directly off PowerPoint slides, rarely responds to emails, gives difficult exams with obscure questions, and provides little to no feedback. |

---

## Anticipated Challenges

1. **Chunk boundary splitting reviews:** Because reviews vary in length, some longer reviews may be split across two chunks. If the key opinion (e.g. "avoid at all costs") lands at the end of a review that starts in a previous chunk, retrieval may return only the second half without enough context for the LLM to generate a good answer. The 50-character overlap is designed to mitigate this but may not fully solve it for very long reviews.

2. **Professor name variations and typos:** Some professor names are long or unusual (e.g. Nonnarit O-Larnnithipong). If a user query uses a nickname, partial name, or misspelling, the embedding model may not match it to the correct document chunks. This is a known limitation of semantic search on proper nouns — the model treats them as tokens rather than understanding they refer to a person.

---

## Architecture

┌─────────────────────┐
│   Document Ingestion │  ← Python (ingest.py)
│   Load .txt files    │    os.listdir + open()
│   Clean text         │
└────────┬────────────┘
│
▼
┌─────────────────────┐
│   Chunking           │  ← Python (chunk.py)
│   500 char chunks    │    Custom sliding window
│   50 char overlap    │
└────────┬────────────┘
│
▼
┌─────────────────────┐
│  Embedding +         │  ← sentence-transformers
│  Vector Store        │    all-MiniLM-L6-v2
│                      │    ChromaDB (embed.py)
└────────┬────────────┘
│
▼
┌─────────────────────┐
│   Retrieval          │  ← ChromaDB query (retrieve.py)
│   Top-k = 5          │    Semantic similarity search
│   + source metadata  │
└────────┬────────────┘
│
▼
┌─────────────────────┐
│   Generation         │  ← Groq API (generate.py)
│   llama-3.3-70b      │    Grounded prompt template
│   Grounded response  │    Source attribution
└────────┬────────────┘
│
▼
┌─────────────────────┐
│   Query Interface    │  ← Gradio (app.py)
│   Web UI             │    Text input + answer output
│   Source display     │    + sources display
└─────────────────────┘

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
I will give Claude my Chunking Strategy section and Documents section from this planning.md, along with the structure of my .txt files (header block + REVIEW N format). I will ask Claude to implement ingest.py (loads all .txt files from the documents/ folder, extracts clean text) and chunk.py (splits text into 500-character chunks with 50-character overlap, attaches source filename as metadata). I will verify the output by printing 5 sample chunks and confirming each is readable, self-contained, and tagged with the correct source filename.

**Milestone 4 — Embedding and retrieval:**
I will give Claude my Retrieval Approach section and the architecture diagram, and ask it to implement embed.py (embeds all chunks using all-MiniLM-L6-v2 and stores them in ChromaDB with source metadata) and retrieve.py (accepts a query string, returns top-5 most relevant chunks with source names and distance scores). I will verify by running 3 of my evaluation plan questions and confirming the returned chunks are visibly relevant.

**Milestone 5 — Generation and interface:**
I will give Claude my grounding requirement (answer only from retrieved context, cite sources), the Groq model name (llama-3.3-70b-versatile), and my architecture diagram, and ask it to implement generate.py (calls Groq with a grounded prompt template) and app.py (Gradio UI with question input, answer output, and sources display). I will verify grounding by asking a question my documents don't cover and confirming the system declines to answer rather than hallucinating.