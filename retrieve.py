from sentence_transformers import SentenceTransformer
import chromadb

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "fiu_professor_reviews"
TOP_K = 5

# Load model and ChromaDB once when this module is imported
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME)
print("Ready!\n")


def detect_department(query):
    """
    Detects if the query mentions a specific department.
    Returns the department name or None if not specified.
    """
    query_lower = query.lower()

    if any(term in query_lower for term in [
        "computer science", "cs professor", "cs class"
    ]):
        return "Computer Science"

    if any(term in query_lower for term in [
        "computer engineering", "ce professor", "ce class"
    ]):
        return "Computer Engineering"

    if any(term in query_lower for term in [
        "information science", "information systems",
        "is professor", "is class"
    ]):
        return "Information Systems"

    return None


def retrieve(query, top_k=TOP_K):
    """
    Takes a user query string and returns the top_k most
    relevant chunks from ChromaDB using semantic search.
    If the query mentions a specific department, filters
    results to only that department.

    Returns a list of dicts, each containing:
      - 'text': the chunk content
      - 'source': the source filename
      - 'distance': similarity score (lower = more relevant)
    """

    # Step 1: Convert the query to a vector
    query_embedding = model.encode(query).tolist()

    # Step 2: Detect if query is department-specific
    department = detect_department(query)

    # Step 3: Search ChromaDB
    if department:
        # Filter by department prefix in source filename
        prefix_map = {
            "Computer Science": "cs_",
            "Computer Engineering": "ce_",
            "Information Systems": "is_"
        }
        prefix = prefix_map[department]

        # Get more results then filter, to ensure we get top_k after filtering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(50, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        # Filter to only chunks from the detected department
        chunks = []
        for i in range(len(results["documents"][0])):
            source = results["metadatas"][0][i]["source"]
            if source.startswith(prefix):
                chunks.append({
                    "text": results["documents"][0][i],
                    "source": source,
                    "distance": round(results["distances"][0][i], 4)
                })
            if len(chunks) >= top_k:
                break

        # Fallback to unfiltered if not enough results found
        if len(chunks) < 2:
            chunks = []
            for i in range(min(top_k, len(results["documents"][0]))):
                chunks.append({
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "distance": round(results["distances"][0][i], 4)
                })
    else:
        # No department filter — standard semantic search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": round(results["distances"][0][i], 4)
            })

    return chunks


def print_results(query, chunks):
    """
    Prints retrieval results in a readable format for testing.
    """
    print(f"Query: {query}")
    print("=" * 60)
    for i, chunk in enumerate(chunks):
        print(f"\nResult #{i+1}")
        print(f"Source: {chunk['source']}")
        print(f"Distance: {chunk['distance']} (lower = more relevant)")
        print(f"Text:\n{chunk['text']}")
        print("-" * 60)


# --- Run this file directly to test retrieval ---
if __name__ == "__main__":

    test_queries = [
        "What do students say about Gregory Reis exams and grading?",
        "Which Computer Science professor should students avoid and why?",
        "Is Vladimir Pozdin class recommended for beginners?"
    ]

    for query in test_queries:
        chunks = retrieve(query)
        print_results(query, chunks)
        print("\n")