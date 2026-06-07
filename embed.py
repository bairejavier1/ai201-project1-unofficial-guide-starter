from chunk import chunk_documents
from ingest import load_documents
from sentence_transformers import SentenceTransformer
import chromadb

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "fiu_professor_reviews"

def build_vector_store():
    """
    Loads documents, chunks them, embeds each chunk using
    all-MiniLM-L6-v2, and stores everything in ChromaDB
    with source metadata. This only needs to be run once.
    """

    # Step 1: Load and chunk all documents
    print("Loading documents...")
    documents = load_documents()

    print("\nChunking documents...")
    all_chunks = chunk_documents(documents)
    print(f"Total chunks to embed: {len(all_chunks)}")

    # Step 2: Load the embedding model (runs locally, no API key needed)
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded successfully!")

    # Step 3: Set up ChromaDB
    print(f"\nSetting up ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection if it exists (so we can re-run cleanly)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")

    collection = client.create_collection(COLLECTION_NAME)
    print(f"Created new collection: {COLLECTION_NAME}")

    # Step 4: Embed each chunk and store in ChromaDB
    print("\nEmbedding chunks and storing in ChromaDB...")
    for i, chunk in enumerate(all_chunks):
        # Convert chunk text to a vector (list of numbers)
        embedding = model.encode(chunk["text"]).tolist()

        # Store in ChromaDB with metadata
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"]
            }]
        )

    print(f"\nSuccessfully embedded and stored {len(all_chunks)} chunks!")
    print(f"ChromaDB saved to: {CHROMA_DB_PATH}")
    return collection


# --- Run this file directly to build the vector store ---
if __name__ == "__main__":
    collection = build_vector_store()

    # Quick verification: count stored chunks
    count = collection.count()
    print(f"\nVerification: ChromaDB collection contains {count} chunks")

    # Show a sample stored entry
    sample = collection.peek(1)
    print(f"\n--- SAMPLE STORED CHUNK ---")
    print(f"ID: {sample['ids'][0]}")
    print(f"Source: {sample['metadatas'][0]['source']}")
    print(f"Text preview: {sample['documents'][0][:200]}")