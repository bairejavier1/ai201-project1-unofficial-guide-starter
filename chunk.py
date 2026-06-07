from ingest import load_documents

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Splits a text string into chunks of chunk_size characters,
    with an overlap of `overlap` characters between consecutive chunks.
    Skips any empty chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if len(chunk) > 0:
            chunks.append(chunk)

        # Move forward by (chunk_size - overlap) so chunks share some context
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents, chunk_size=400, overlap=50):
    """
    Takes a list of document dicts (from ingest.py) and splits each one
    into chunks. Returns a flat list of chunk dicts, each containing:
      - 'text': the chunk content
      - 'source': the original filename
      - 'chunk_index': position of this chunk within its document
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["filename"],
                "chunk_index": i
            })

    return all_chunks


# --- Run this file directly to test it ---
if __name__ == "__main__":
    # Step 1: Load documents
    documents = load_documents()

    # Step 2: Chunk them
    all_chunks = chunk_documents(documents)

    # Step 3: Report total chunk count
    print(f"\nTotal chunks created: {len(all_chunks)}")

    # Step 4: Print 5 sample chunks so we can verify quality
    print("\n--- 5 SAMPLE CHUNKS ---\n")
    sample_indices = [0, 5, 10, 15, 20]
    for i in sample_indices:
        if i < len(all_chunks):
            chunk = all_chunks[i]
            print(f"Chunk #{i} | Source: {chunk['source']} | Index in doc: {chunk['chunk_index']}")
            print(f"{chunk['text']}")
            print("-" * 60)