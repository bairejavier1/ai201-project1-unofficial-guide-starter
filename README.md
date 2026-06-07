# The Unofficial Guide — FIU Professor Reviews
### A RAG System for Student-Generated Knowledge at Florida International University

---

## Domain

This system covers student-generated reviews of professors at Florida International University (FIU) across three departments: Computer Science, Computer Engineering, and Information Systems. This knowledge is valuable because official university channels — course catalogs, syllabi, and department websites — only describe what a course covers, not what it is actually like to take it. Rate My Professors captures honest, first-hand student experiences including teaching quality, exam difficulty, grading fairness, and workload — information that is impossible to find through official channels and critical for students making course registration decisions.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Rate My Professors | Professor reviews | documents/cs_prof_Caryl Rahn.txt |
| 2 | Rate My Professors | Professor reviews | documents/cs_prof_Gregory Reis.txt |
| 3 | Rate My Professors | Professor reviews | documents/cs_prof_Ning_Xie.txt |
| 4 | Rate My Professors | Professor reviews | documents/cs_prof_Tiana_Solis.txt |
| 5 | Rate My Professors | Professor reviews | documents/ce_prof_Vladimir_Pozdin.txt |
| 6 | Rate My Professors | Professor reviews | documents/ce_prof_Nonnarit_O-Larnnithipong.txt |
| 7 | Rate My Professors | Professor reviews | documents/ce_prof_Fatima_Boujarwah.txt |
| 8 | Rate My Professors | Professor reviews | documents/is_prof_Andres_Rodriguez.txt |
| 9 | Rate My Professors | Professor reviews | documents/is_prof_David_Palmer.txt |
| 10 | Rate My Professors | Professor reviews | documents/is_prof_Rehan_Akbar.txt |

---

## Chunking Strategy

**Chunk size:** One complete review per chunk — natural boundary splitting using REVIEW N markers present in each document.

**Overlap:** None. Since each review is a self-contained unit of student opinion, there is no need for overlap. Reviews do not share content across boundaries.

**Why these choices fit your documents:** Each source document contains 5 student reviews separated by `REVIEW 1`, `REVIEW 2`, etc. markers. Initially a 400-character sliding window with 50-character overlap was implemented, producing 62 chunks. However, retrieval quality was poor because the professor header block (name, department, URL) consumed most of each character-based chunk, leaving little room for actual review content. The strategy was changed to extract each individual review as its own chunk using the REVIEW N markers as natural split points. A short professor context line (name, department, overall rating, review number) is prepended to every chunk so that every chunk knows which professor it belongs to. This produced cleaner, more semantically meaningful chunks that retrieved much more accurately — distance scores improved from 0.88–1.03 to 0.74–0.90 for specific professor queries.

**Final chunk count:** 50 (5 reviews × 10 professors)

**Sample chunks:**

```
--- Chunk 1 | Source: cs_prof_Gregory Reis.txt ---
Professor: Gregory Reis | Department: Computer Science | Overall Rating: 4.9/5 | Review #1
Quality: 5.0
Difficulty: 1.0
Course: CAP2752
Review: He is very kind! The work isn't hard and all the material is online for free.
He walks you through each exercise in class. He genuinely wants you to learn.
Highly recommend this class.

--- Chunk 2 | Source: cs_prof_Caryl Rahn.txt ---
Professor: Caryl Rahn | Department: Computer Science | Overall Rating: 2.1/5 | Review #2
Quality: 1.0
Difficulty: 3.0
Course: CGS3095
Review: Avoid like the plague. Capricious grader - Deducted points from a rubric
category for reasons unrelated to the specified grading criterion!

--- Chunk 3 | Source: ce_prof_Vladimir_Pozdin.txt ---
Professor: Vladimir Pozdin | Department: Computer Engineering | Overall Rating: 2.0/5 | Review #1
Quality: 1.0
Difficulty: 5.0
Course: EEL3110C
Review: Very unorganized class. Assignments not clearly defined of how he wants them.
Labs are awful and take 3+ hours due to lack of structure.

--- Chunk 4 | Source: is_prof_Andres_Rodriguez.txt ---
Professor: Andres Rodriguez | Department: Information Science | Overall Rating: 5/5 | Review #1
Quality: 5.0
Difficulty: 5.0
Course: ISM3011
Review: Literally the only professor I'll write a review for. Genuinely amazing teacher
who teaches through valuable and engaging life lessons in his lectures.

--- Chunk 5 | Source: is_prof_Rehan_Akbar.txt ---
Professor: Rehan Akbar | Department: Information Science | Overall Rating: 4.2/5 | Review #1
Quality: 1.0
Difficulty: 3.0
Course: CEN4010
Review: AVOID unless you're comfortable memorizing slides word for word. Tests are based
on PowerPoints, and you're expected to remember very specific details.
```

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (runs locally, no API key required)

**Production tradeoff reflection:** For a production deployment serving real FIU students, I would weigh several tradeoffs. `all-MiniLM-L6-v2` is fast and free but has a 256-token context limit, which is sufficient for short reviews but would truncate longer documents. A model like `text-embedding-3-small` (OpenAI) or `voyage-large-2` offers higher accuracy on domain-specific text and longer context windows, but introduces API cost and latency per query. If FIU had international students writing reviews in Spanish or Haitian Creole, multilingual support (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) would become a priority. For local development and this project scope, `all-MiniLM-L6-v2` is the right choice — it is accurate enough for short English review text, runs with no cost or rate limits, and keeps the pipeline simple and reproducible.

---

## Retrieval Test Results

**Query 1:** "What do students say about Gregory Reis exams and grading?"

Top returned chunks:
- `cs_prof_Gregory Reis.txt` — Review #5: "Online chapter quizzes and discussions, 1 online midterm, group project for final. AMAZING! Super nice and passionate professor, very understanding, accept late work and quizzes. 100% GPA boost." (distance: 0.74)
- `cs_prof_Gregory Reis.txt` — Review #4: "Incredible professor with a burning passion for his study, his students, and his class." (distance: 0.78)
- `cs_prof_Gregory Reis.txt` — Review #3: "Professor Reis is, in the simplest terms, simply the best." (distance: 0.81)

Why relevant: All three chunks are from the correct professor file and directly describe his assessment structure, grading flexibility, and workload — exactly what the query asks about.

---

**Query 2:** "Which Computer Science professor should students avoid and why?"

Top returned chunks:
- `cs_prof_Ning_Xie.txt` — Review with Quality 2.0: "He tends to be rude if you don't know things, and he makes it difficult to ask questions because of how intimidating he is." (distance: 0.87)
- `cs_prof_Tiana_Solis.txt` — Review with Quality 3.0: "The professor mainly lectured and had a strong accent, which made understanding harder at times." (distance: 0.88)
- `cs_prof_Caryl Rahn.txt` — Review with Quality 1.0: "Avoid like the plague. Capricious grader." (distance: 0.93)

Why relevant: Department filter correctly limited results to CS professors only. Chunks surface the most critical reviews, giving the LLM the context needed to identify which professors students warn against.

---

**Query 3:** "Is Vladimir Pozdin recommended for beginners?"

Top returned chunks:
- `ce_prof_Vladimir_Pozdin.txt` — Review #1: "Avoid at all cost. His course is not a beginner course but he assumes you should already know it." (distance: 0.90)
- `ce_prof_Vladimir_Pozdin.txt` — Review #2: "Very unorganized class. Labs are awful and take 3+ hours." (distance: 0.93)
- `ce_prof_Vladimir_Pozdin.txt` — Review #5: "His class is just not for people with no knowledge of circuits because is very fast paced." (distance: 0.99)

Why relevant: Professor name detection correctly locked retrieval to Vladimir Pozdin's file only. All returned chunks directly answer whether the class suits beginners — and unanimously say no.

---

## Grounded Generation

**System prompt grounding instruction:**

```
You are a helpful student advisor for Florida International University (FIU).
Your job is to answer questions about professors and courses using ONLY the
student reviews provided to you.

STRICT RULES you must follow:
1. Answer ONLY using information from the provided documents. Do not use any outside knowledge.
2. If the documents do not contain enough information to answer the question, say exactly:
   "I don't have enough information in my documents to answer that question."
3. Answer in clear, natural language without mentioning document numbers or review numbers
   in your response. Just answer the question directly and conversationally.
4. Be honest — if reviews are mixed, reflect that in your answer.
5. Never make up or assume information not present in the documents.
6. Do NOT add a Sources line at the end — sources will be added automatically.
```

**How source attribution is surfaced in the response:** Source attribution is handled programmatically — not left to the LLM. After generation, the pipeline checks which professor names appear in the answer text and maps them back to their source filenames. These are displayed in a separate "Retrieved From" section in the Gradio UI. If the system declines to answer (out-of-scope question), no sources are shown.

**Example grounded response:**

Question: "What do students say about Gregory Reis exams and grading?"

```
Answer:
Students seem to really appreciate Professor Gregory Reis' approach to exams
and grading. One review mentions that he accepts late work and quizzes, which
suggests that he's quite flexible and understanding. Another review states that
the workload is very manageable, with discussions taking the most time.
Additionally, one student mentions that he "doesn't expect you to be an expert"
and is "willing to teach anything from scratch", which implies that his exams
and grading are fair and focused on learning rather than stressing students out.

Retrieved From:
• cs_prof_Gregory Reis.txt
```

**Example out-of-scope refusal:**

Question: "What is the best restaurant near FIU campus?"

```
Answer:
I don't have enough information in my documents to answer that question.

Retrieved From:
None — question was out of scope.
```

---

## Query Interface

**Input field:** A text box labeled "Ask a Question" with placeholder text guiding users on what to ask. Users can type a question and either press Enter or click the "Ask" button to submit.

**Output fields:**
- **Answer** — a multi-line text box displaying the LLM's grounded, conversational response
- **Retrieved From** — a smaller text box listing the source `.txt` files the answer was drawn from

**Sample interaction transcript:**

```
User input: What is the workload like for Andres Rodriguez's class?

Answer:
Students describe Professor Andres Rodriguez's workload as very light and
manageable. His assignments typically include quizzes worth 10%, discussion
posts worth 20%, two essay projects worth 10%, and a midterm and final worth
30% of the grade. He is described as very flexible with homework and
understanding of students' personal lives. Multiple students describe his
class as one of the easiest they have taken and strongly recommend him.

Retrieved From:
• is_prof_Andres_Rodriguez.txt
```

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about Gregory Reis's exams and grading? | Exams are not proctored, tests are online, grading is lenient, accepts late work | Correctly described online midterm, accepts late work, manageable workload, lenient grading | Relevant | Accurate |
| 2 | Which Computer Science professor should students avoid and why? | Caryl Rahn — strict grading, AI accusations | Identified Ning Xie as primary concern due to rude behavior; mentioned Tiana Solis mixed reviews; did not surface Caryl Rahn as top result | Partially relevant | Partially accurate |
| 3 | Is Vladimir Pozdin's class recommended for beginners? | No — fast-paced, assumes prior knowledge | Correctly answered no — class assumes prior knowledge, very fast-paced, not suitable for beginners | Relevant | Accurate |
| 4 | What is the workload like for Andres Rodriguez's class? | Light — quizzes, discussions, essays, flexible professor | Correctly described light workload, quizzes, discussions, essays, flexible and understanding professor | Relevant | Accurate |
| 5 | How do students describe Rehan Akbar's teaching style? | Reads off slides, rarely responds to emails, difficult exams | Correctly described reading off slides, long grading times, difficult exams, little feedback | Relevant | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** "Which Computer Science professor should students avoid and why?"

**What the system returned:** The system identified Professor Ning Xie as the primary professor to avoid due to rude behavior and intimidating demeanor, with a secondary mention of Tiana Solis. It did not surface Caryl Rahn as the top result, even though Rahn has the lowest overall rating (2.1/5) among CS professors and has the most severe negative reviews — including accusations of sending students to conduct hearings and bragging about pushing students to withdraw.

**Root cause (tied to a specific pipeline stage):** This is a retrieval failure caused by semantic mismatch at the embedding stage. The query "Which CS professor should students avoid?" semantically matches words like "rude", "intimidating", and "avoid" — which appear prominently in Ning Xie's reviews. Caryl Rahn's most severe reviews use different language: "capricious grader", "provost", "plagiarism charges" — vocabulary that is less semantically similar to the word "avoid" as used in the query. The `all-MiniLM-L6-v2` model matched on surface-level warning language rather than on the severity of the underlying complaint.

**What you would change to fix it:** Adding metadata filtering by overall rating would allow the system to prioritize the lowest-rated professor when the query asks who to avoid. Alternatively, hybrid search combining BM25 keyword matching (which would catch "avoid" appearing literally in Rahn's reviews) with semantic search would improve recall for this type of query.

---

## Spec Reflection

**One way the spec helped you during implementation:** Writing the evaluation plan in `planning.md` before writing any code forced me to define 5 specific, testable questions upfront. This made it easy to verify the pipeline at each milestone — I could run the same 5 questions against retrieval, then generation, and immediately see whether each stage was producing useful output. Without the spec, I would have tested with vague ad-hoc questions and missed the failure case with Caryl Rahn entirely.

**One way your implementation diverged from the spec, and why:** The spec called for a 400-character sliding window chunking strategy with 50-character overlap. During implementation, testing showed that character-based chunks produced poor retrieval because the professor header block (name, department, URL) consumed most of each chunk's character budget, leaving little room for actual review content. The strategy was changed to extract each individual review as its own chunk using the REVIEW N markers as natural split points. This produced 50 chunks instead of 62, but with dramatically better retrieval quality — distance scores improved from 0.88–1.03 to 0.74–0.90 for specific professor queries.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* The Documents section and Chunking Strategy section from planning.md, along with the structure of the .txt files (header block + REVIEW N format). Asked Claude to implement ingest.py and an initial chunk.py using a 500-character sliding window with 50-character overlap.
- *What it produced:* A working ingest.py that loaded and cleaned all 10 .txt files correctly, and a chunk.py that split text by character count with overlap, producing 62 chunks.
- *What I changed or overrode:* After testing retrieval quality, I directed Claude to rewrite chunk.py entirely to use natural review boundary splitting instead of character-based splitting. I identified the problem (header block consuming chunk space) and instructed Claude to extract REVIEW N blocks as individual chunks with professor context prepended. This reduced chunk count from 62 to 50 but improved retrieval distance scores significantly.

**Instance 2**

- *What I gave the AI:* The Retrieval Approach section from planning.md, the architecture diagram, and the grounding requirement (answer only from retrieved context, cite sources). Asked Claude to implement retrieve.py and generate.py.
- *What it produced:* A retrieve.py using standard semantic search across all chunks, and a generate.py with a grounded system prompt and source attribution logic.
- *What I changed or overrode:* I identified two problems after testing: (1) retrieval was returning professors from the wrong department when the query specified a department, and (2) specific professor queries were returning results from other professors. I directed Claude to add a detect_department() function and a detect_professor() function to retrieve.py, implementing priority-based filtering: professor name detection takes highest priority, followed by department filtering, followed by broad cross-department search. I also directed Claude to add a two-pass positive review filter for broad "best professor" queries to ensure all three departments are represented with relevant results.