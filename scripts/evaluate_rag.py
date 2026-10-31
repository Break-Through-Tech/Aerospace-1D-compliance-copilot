"""Evaluate the canonical RAG using the existing Chroma index and Ollama service."""

import argparse

import pandas as pd
from httpx import RequestError
from ollama import ResponseError

from src.config import GENERATION_MODEL, PROJECT_ROOT
from src.evaluation import evaluate
from src.pipeline import run_audits
from src.rag import create_audit_client
from src.vector_store import load_vector_store


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N benchmark requirements")
    parser.add_argument("--k", type=int, default=5, help="Number of clauses to retrieve (default: 5)")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.k < 1:
        parser.error("--k must be positive")

    benchmark = pd.read_csv(
        PROJECT_ROOT / "data" / "npr7150_ground_truth_benchmark.csv",
        dtype=str,
        keep_default_na=False,
    )
    if args.limit is not None:
        benchmark = benchmark.head(args.limit)

    print("Loading persistent Chroma index...")
    try:
        vector_store = load_vector_store()
        client = create_audit_client()
        requirements = benchmark[["requirement_id", "requirement_text"]].itertuples(index=False, name=None)
        print(f"Auditing {len(benchmark)} requirements...", flush=True)
        records = run_audits(requirements, vector_store, client, k=args.k)
    except (ConnectionError, RequestError, ResponseError) as exc:
        raise SystemExit(f"Evaluation stopped: Ollama service failure: {exc}") from exc

    for index, record in enumerate(records, 1):
        status = record.verdict.verdict if record.verdict else f"FAILED: {record.error}"
        print(f"[{index}/{len(records)}] {record.requirement_id}: {status}")
    _, metrics, _ = evaluate(benchmark, records, k=args.k)
    print(f"\nEvaluated {metrics['n']} requirements (k={args.k}, model={GENERATION_MODEL}).")
    print(f"Verdict accuracy: {metrics['verdict_accuracy']:.2%}; macro F1: {metrics['macro_f1']:.4f}")
    print(f"Retrieval Hit@1: {metrics['retrieval']['hit_at_1']:.2%}; "
          f"Hit@{args.k}: {metrics['retrieval'][f'hit_at_{args.k}']:.2%}")
    print(f"Schema valid: {metrics['schema_valid_rate']:.2%}; "
          f"failed audits: {sum(record.verdict is None for record in records)}")
    membership = metrics["citation_groundedness"]
    membership_text = f"{membership:.2%}" if membership is not None else "N/A"
    print(f"Citation membership: {membership_text} "
          f"({metrics['citation_groundedness_n']} valid verdicts); "
          f"cited gold clause: {metrics['cited_gold_clause_rate']:.2%}")


if __name__ == "__main__":
    main()
