import os

DOCUMENTS_FOLDER = "documents"

def load_documents():
    """
    Loads all .txt files from the documents/ folder.
    Returns a list of dictionaries, each with:
      - 'filename': the name of the .txt file
      - 'text': the full cleaned text content
    """
    documents = []

    # Loop through every file in the documents folder
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCUMENTS_FOLDER, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # Clean the text
            cleaned_text = clean_text(raw_text)

            documents.append({
                "filename": filename,
                "text": cleaned_text
            })

            print(f"Loaded: {filename} ({len(cleaned_text)} characters)")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def clean_text(text):
    """
    Cleans raw text from a .txt file.
    - Strips leading/trailing whitespace
    - Removes excessive blank lines (3+ in a row become 2)
    - Keeps all review content intact
    """
    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace 3 or more consecutive newlines with just 2
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


# --- Run this file directly to test it ---
if __name__ == "__main__":
    docs = load_documents()

    print("\n--- SAMPLE: First document preview ---")
    if docs:
        print(f"Filename: {docs[0]['filename']}")
        print(f"Preview:\n{docs[0]['text'][:500]}")