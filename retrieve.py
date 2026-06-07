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


def retrieve(query, top_k=TOP_K):
    """
    Takes a user query string and returns the top_k most
    relevant chunks from ChromaDB using semantic search.

    Returns a list of dicts, each containing:
      - 'text': the chunk content
      - 'source': the source filename
      - 'distance': similarity score (lower = more relevant)
    """

    # Step 1: Convert the query to a vector using the same model
    query_embedding = model.encode(query).tolist()

    # Step 2: Search ChromaDB for the most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Step 3: Format results into a clean list
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

    # Test with 3 of your 5 evaluation plan questions
    test_queries = [
        "What do students say about Gregory Reis exams and grading?",
        "Which Computer Science professor should students avoid and why?",
        "Is Vladimir Pozdin class recommended for beginners?"
    ]

    for query in test_queries:
        chunks = retrieve(query)
        print_results(query, chunks)
        print("\n")