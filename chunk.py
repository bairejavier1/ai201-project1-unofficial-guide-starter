from ingest import load_documents
import re

def extract_reviews(text):
    """
    Strips the header block (Professor, Department, URL, etc.)
    and extracts each individual review as its own clean string.
    Returns a list of review strings.
    """
    reviews = []

    # Split on the REVIEW N pattern
    parts = re.split(r'---\s*\nREVIEW \d+', text)

    # parts[0] is the header block — skip it
    # parts[1:] are the individual reviews
    for part in parts[1:]:
        part = part.strip()
        if len(part) > 50:  # skip empty or near-empty splits
            reviews.append(part)

    return reviews


def chunk_documents(documents, chunk_size=400, overlap=50):
    """
    Takes a list of document dicts (from ingest.py).
    Extracts individual reviews from each document and
    treats each review as its own chunk, tagged with
    the professor name and source filename.

    Returns a flat list of chunk dicts, each containing:
      - 'text': the review content with professor context
      - 'source': the original filename
      - 'chunk_index': position of this chunk within its document
    """
    all_chunks = []

    for doc in documents:
        # Extract professor name from header for context
        name_match = re.search(r'Professor:\s*(.+)', doc["text"])
        dept_match = re.search(r'Department:\s*(.+)', doc["text"])
        rating_match = re.search(r'Overall Rating:\s*(.+)', doc["text"])

        professor_name = name_match.group(1).strip() if name_match else "Unknown"
        department = dept_match.group(1).strip() if dept_match else "Unknown"
        overall_rating = rating_match.group(1).strip() if rating_match else "Unknown"

        # Extract individual reviews
        reviews = extract_reviews(doc["text"])

        for i, review in enumerate(reviews):
            # Prepend professor context to every review chunk
            # This ensures every chunk knows WHO it's about
            chunk_text = (
                f"Professor: {professor_name} | "
                f"Department: {department} | "
                f"Overall Rating: {overall_rating}\n"
                f"{review}"
            )

            all_chunks.append({
                "text": chunk_text,
                "source": doc["filename"],
                "chunk_index": i
            })

    return all_chunks


# --- Run this file directly to test it ---
if __name__ == "__main__":
    documents = load_documents()
    all_chunks = chunk_documents(documents)

    print(f"\nTotal chunks created: {len(all_chunks)}")
    print("\n--- 5 SAMPLE CHUNKS ---\n")

    sample_indices = [0, 5, 10, 15, 20]
    for i in sample_indices:
        if i < len(all_chunks):
            chunk = all_chunks[i]
            print(f"Chunk #{i} | Source: {chunk['source']} | Index: {chunk['chunk_index']}")
            print(f"{chunk['text']}")
            print("-" * 60)