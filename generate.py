from retrieve import retrieve
from groq import Groq
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# The model we are using
MODEL = "llama-3.3-70b-versatile"


def build_prompt(query, chunks):
    """
    Builds a grounded prompt that instructs the LLM to answer
    ONLY from the retrieved chunks, not from general knowledge.
    """

    # Format each chunk with its source for the LLM to reference
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[Document {i+1} - Source: {chunk['source']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    # System prompt — this is the grounding instruction
    system_prompt = """You are a helpful student advisor for Florida International University (FIU).
Your job is to answer questions about professors and courses using ONLY the student reviews provided to you.

STRICT RULES you must follow:
1. Answer ONLY using information from the provided documents. Do not use any outside knowledge.
2. If the documents do not contain enough information to answer the question, say exactly: "I don't have enough information in my documents to answer that question."
3. Answer in clear, natural language without mentioning document numbers or review numbers in your response. Just answer the question directly and conversationally.
4. Be honest — if reviews are mixed, reflect that in your answer.
5. Never make up or assume information not present in the documents.
6. Do NOT add a Sources line at the end — sources will be added automatically."""

    # User prompt — contains the retrieved context and the question
    user_prompt = f"""Here are the relevant student reviews I found:

{context}

---

Based ONLY on the student reviews above, please answer this question:
{query}"""

    return system_prompt, user_prompt


def ask(query):
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from ChromaDB
    2. Build a grounded prompt
    3. Send to Groq LLM
    4. Return the answer with sources

    Returns a dict with:
      - 'answer': the LLM's grounded response
      - 'sources': list of source filenames used
      - 'chunks': the raw retrieved chunks (for debugging)
    """

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query)

    # Step 2: Build the grounded prompt
    system_prompt, user_prompt = build_prompt(query, chunks)

    # Step 3: Call Groq LLM
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,  # low temperature = more factual, less creative
        max_tokens=1000
    )

    # Step 4: Extract the answer
    answer = response.choices[0].message.content

    # Step 5: Handle sources smartly
    # If the system declined to answer, show no sources
    if "I don't have enough information" in answer:
        sources = []
    else:
        # Extract professor names mentioned in the answer
        # and only show sources whose professor name appears in the answer
        sources = []
        for chunk in chunks:
            source = chunk["source"]
            if source in sources:
                continue
            # Extract professor name from chunk text
            prof_name = chunk["text"].split("|")[0].replace("Professor:", "").strip()
            # Only include source if professor name is mentioned in the answer
            if prof_name.split()[-1] in answer or prof_name.split()[0] in answer:
                sources.append(source)

        # Fallback: if no sources matched, use top 2 retrieved sources
        if not sources:
            seen = []
            for chunk in chunks[:2]:
                if chunk["source"] not in seen:
                    seen.append(chunk["source"])
            sources = seen

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks
    }


# --- Run this file directly to test generation ---
if __name__ == "__main__":

    # Test with 2 of your evaluation questions
    test_queries = [
        "What do students say about Gregory Reis exams and grading?",
        "Which Computer Science professor should students avoid and why?"
    ]

    for query in test_queries:
        print(f"\nQuestion: {query}")
        print("=" * 60)

        result = ask(query)

        print(f"Answer:\n{result['answer']}")
        if result['sources']:
            print(f"\nSources: {', '.join(result['sources'])}")
        else:
            print("\nSources: None — question was out of scope")
        print("=" * 60)

    # Test out-of-scope query — system should decline to answer
    print("\nQuestion: What is the best restaurant near FIU campus?")
    print("=" * 60)
    out_of_scope = ask("What is the best restaurant near FIU campus?")
    print(f"Answer:\n{out_of_scope['answer']}")
    if out_of_scope['sources']:
        print(f"\nSources: {', '.join(out_of_scope['sources'])}")
    else:
        print("\nSources: None — question was out of scope")
    print("=" * 60)