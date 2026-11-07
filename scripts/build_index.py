import argparse

from src.config import DATA_PATH, VECTOR_STORE_PATH, COLLECTION_NAME
from src.preprocessing import load_requirements, create_documents
from src.vector_store import build_vector_store

def main(argv=None):
    parser = argparse.ArgumentParser(description="Preview or rebuild the configured NASA collection.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the configured collection.")
    args = parser.parse_args(argv)
    df = load_requirements(DATA_PATH)
    documents = create_documents(df)
    if (len(documents) != 130 or len({doc.id for doc in documents}) != 130
            or len({doc.metadata["clause"] for doc in documents}) != 130):
        raise ValueError("Expected exactly 130 unique NASA requirements and clauses.")
    print(f"Source: {DATA_PATH}; records: {len(documents)}")
    print(f"Target: {VECTOR_STORE_PATH}; collection: {COLLECTION_NAME}")
    if not args.rebuild:
        print("Preview only: no index opened. Use --rebuild to replace the configured collection.")
        return
    build_vector_store(documents, rebuild=True)
    print(f"Verified {len(documents)} indexed records, including exact text and metadata.")

if __name__ == "__main__":
    main()
