import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_PATH = PROJECT_ROOT / "data" / "npr-7150-2d-requirements_fixed.csv"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "npr7150_ground_truth_benchmark.csv"
VECTOR_STORE_PATH = PROJECT_ROOT / os.environ["VECTOR_STORE_PATH"]

OLLAMA_HOST = os.environ["OLLAMA_HOST"]
GENERATION_MODEL = os.environ["GENERATION_MODEL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
COLLECTION_NAME = os.environ["COLLECTION_NAME"]
