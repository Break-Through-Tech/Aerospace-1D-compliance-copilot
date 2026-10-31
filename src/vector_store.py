from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import EMBEDDING_MODEL, OLLAMA_HOST, VECTOR_STORE_PATH, COLLECTION_NAME


def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)


def build_vector_store(documents):
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_PATH),
        ids=[doc.id for doc in documents]
    )


def load_vector_store():
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}. Run the indexing pipeline first.")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME, 
        embedding_function=get_embeddings(),
        persist_directory=str(VECTOR_STORE_PATH),
        create_collection_if_not_exists=False
    )

    if not vector_store.get(include=[])["ids"]:
        raise ValueError("Vector store is empty. Rebuild the index.")

    return vector_store
