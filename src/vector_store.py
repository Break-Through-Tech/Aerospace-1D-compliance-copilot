from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import EMBEDDING_MODEL, OLLAMA_HOST, VECTOR_STORE_PATH, COLLECTION_NAME


def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)


def build_vector_store(documents, *, rebuild=False):
    """Build the configured collection; replacing existing records is explicit.

    Rebuilding deletes only this collection. Stop readers until verification
    succeeds; a failed rebuild must be rerun before the index is used.
    """
    documents = list(documents)
    ids = [doc.id for doc in documents]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Documents must have nonempty, unique IDs.")
    store = Chroma(
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_PATH),
    )
    if rebuild:
        store.reset_collection()
    elif store.get(include=[])["ids"]:
        raise ValueError("Collection already contains records; use an explicit rebuild.")
    store.add_documents(documents, ids=ids)
    stored = store.get(include=["documents", "metadatas"])
    expected = {doc.id: (doc.page_content, doc.metadata) for doc in documents}
    actual = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))
    if len(stored["ids"]) != len(documents) or actual != expected:
        raise ValueError("Index verification failed: stored IDs, text or metadata differ from source.")
    return store


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
