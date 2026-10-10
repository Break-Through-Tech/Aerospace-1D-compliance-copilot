from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "npr-7150-2d-requirements.csv"

VECTOR_STORE_PATH = PROJECT_ROOT / "chroma_db"

OLLAMA_HOST = "http://localhost:11434"

EMBEDDING_MODEL = "nomic-embed-text"

COLLECTION_NAME = "nasa_requirements"
