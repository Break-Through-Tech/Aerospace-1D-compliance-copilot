"""Run with: python -m scripts.audit_requirement 'The software shall ...' --model llama3.2"""

import argparse

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_HOST, VECTOR_STORE_PATH
from src.rag import audit_requirement


def main():
    parser = argparse.ArgumentParser(description="Audit a requirement against retrieved NASA clauses")
    parser.add_argument("requirement")
    parser.add_argument("--model", required=True, help="An installed Ollama chat model")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_PATH),
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST),
        create_collection_if_not_exists=False,
    )
    result = audit_requirement(args.requirement, store, model=args.model, k=args.k)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
