from src.config import DATA_PATH
from src.preprocessing import load_requirements, create_documents
from src.vector_store import build_vector_store

def main():
    print("Loading NASA requirements...")
    df = load_requirements(DATA_PATH)

    print("Creating structured chunks...")
    documents = create_documents(df)

    print("Building vector index...")
    vector_store = build_vector_store(documents)

    print(f"Indexed {len(documents)} requirements.")

if __name__ == "__main__":
    main()
