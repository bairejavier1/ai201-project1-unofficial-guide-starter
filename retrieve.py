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

# Map of professor name keywords to source filenames
PROFESSOR_FILES = {
    "gregory reis": "cs_prof_Gregory Reis.txt",
    "reis": "cs_prof_Gregory Reis.txt",
    "caryl rahn": "cs_prof_Caryl Rahn.txt",
    "rahn": "cs_prof_Caryl Rahn.txt",
    "ning xie": "cs_prof_Ning_Xie.txt",
    "xie": "cs_prof_Ning_Xie.txt",
    "tiana solis": "cs_prof_Tiana_Solis.txt",
    "solis": "cs_prof_Tiana_Solis.txt",
    "vladimir pozdin": "ce_prof_Vladimir_Pozdin.txt",
    "pozdin": "ce_prof_Vladimir_Pozdin.txt",
    "nonnarit": "ce_prof_Nonnarit_O-Larnnithipong.txt",
    "fatima boujarwah": "ce_prof_Fatima_Boujarwah.txt",
    "boujarwah": "ce_prof_Fatima_Boujarwah.txt",
    "andres rodriguez": "is_prof_Andres_Rodriguez.txt",
    "rodriguez": "is_prof_Andres_Rodriguez.txt",
    "david palmer": "is_prof_David_Palmer.txt",
    "palmer": "is_prof_David_Palmer.txt",
    "rehan akbar": "is_prof_Rehan_Akbar.txt",
    "akbar": "is_prof_Rehan_Akbar.txt",
}


def detect_professor(query):
    """
    Detects if the query mentions a specific professor name.
    Returns the source filename or None.
    """
    query_lower = query.lower()
    for name, filename in PROFESSOR_FILES.items():
        if name in query_lower:
            return filename
    return None


def detect_department(query):
    """
    Detects if the query mentions a specific department.
    Returns the department name, 'ALL' for broad questions,
    or None if not specified.
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

    # Detect broad cross-department questions
    if any(term in query_lower for term in [
        "all departments", "all three", "every department",
        "best professor", "worst professor", "compare",
        "across departments", "overall best", "overall worst"
    ]):
        return "ALL"

    return None


def retrieve(query, top_k=TOP_K):
    """
    Takes a user query string and returns the top_k most
    relevant chunks from ChromaDB using semantic search.

    Priority order:
    1. If a specific professor is named → filter to that professor only
    2. If a specific department is mentioned → filter to that department
    3. If a broad question → return top results from all departments
    4. Otherwise → standard semantic search across all chunks

    Returns a list of dicts, each containing:
      - 'text': the chunk content
      - 'source': the source filename
      - 'distance': similarity score (lower = more relevant)
    """

    # Step 1: Convert the query to a vector
    query_embedding = model.encode(query).tolist()

    # Step 2: Check for specific professor name first
    professor_file = detect_professor(query)

    # Step 3: Check for department if no professor detected
    department = None if professor_file else detect_department(query)

    # Step 4: Search ChromaDB based on detected context
    if professor_file:
        # Specific professor query — only return chunks from that professor
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(50, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            source = results["metadatas"][0][i]["source"]
            if source == professor_file:
                chunks.append({
                    "text": results["documents"][0][i],
                    "source": source,
                    "distance": round(results["distances"][0][i], 4)
                })
            if len(chunks) >= top_k:
                break

    elif department == "ALL":
        # Broad cross-department question
        # Get all results then pick top 2 positive reviews per department
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(50, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        dept_counts = {"cs_": 0, "ce_": 0, "is_": 0}
        chunks = []

        # First pass — only positive reviews (Quality 4.0 or 5.0)
        for i in range(len(results["documents"][0])):
            source = results["metadatas"][0][i]["source"]
            text = results["documents"][0][i]
            if "Quality: 4.0" in text or "Quality: 5.0" in text:
                for prefix in dept_counts:
                    if source.startswith(prefix) and dept_counts[prefix] < 2:
                        chunks.append({
                            "text": text,
                            "source": source,
                            "distance": round(results["distances"][0][i], 4)
                        })
                        dept_counts[prefix] += 1
            if len(chunks) >= 6:
                break

        # Second pass fallback — fill any department that got no results
        if len(chunks) < 6:
            for i in range(len(results["documents"][0])):
                source = results["metadatas"][0][i]["source"]
                text = results["documents"][0][i]
                for prefix in dept_counts:
                    if source.startswith(prefix) and dept_counts[prefix] < 2:
                        if not any(
                            c["source"] == source and c["text"] == text
                            for c in chunks
                        ):
                            chunks.append({
                                "text": text,
                                "source": source,
                                "distance": round(results["distances"][0][i], 4)
                            })
                            dept_counts[prefix] += 1
                if len(chunks) >= 6:
                    break

    elif department:
        # Specific department filter
        prefix_map = {
            "Computer Science": "cs_",
            "Computer Engineering": "ce_",
            "Information Systems": "is_"
        }
        prefix = prefix_map[department]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(50, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

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

        # Fallback to unfiltered if not enough results
        if len(chunks) < 2:
            chunks = []
            for i in range(min(top_k, len(results["documents"][0]))):
                chunks.append({
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "distance": round(results["distances"][0][i], 4)
                })

    else:
        # No filter — standard semantic search
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
        "Is Vladimir Pozdin class recommended for beginners?",
        "Who is the best professor among all three departments?"
    ]

    for query in test_queries:
        chunks = retrieve(query)
        print_results(query, chunks)
        print("\n")