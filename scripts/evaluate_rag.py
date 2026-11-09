"""Evaluate the canonical RAG using the existing Chroma index and Ollama service."""

import argparse

from httpx import RequestError
from ollama import ResponseError

from src.cache import validate_cache_path
from src.config import BENCHMARK_PATH, GENERATION_MODEL
from src.evaluation import evaluate, load_benchmark
from src.rag import create_audit_client, run_audits
from src.reporting import save_results, validate_output_directory
from src.vector_store import load_vector_store


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N benchmark requirements")
    parser.add_argument("--k", type=int, default=5, help="Number of clauses to retrieve (default: 5)")
    parser.add_argument("--cache", help="JSONL cache path for resuming completed audits")
    parser.add_argument("--out", help="Directory for evaluation artifacts (existing files are never overwritten)")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.k < 1:
        parser.error("--k must be positive")
    try:
        if args.cache is not None:
            validate_cache_path(args.cache)
        if args.out is not None:
            validate_output_directory(args.out, input_paths=[args.cache] if args.cache else [])
    except ValueError as exc:
        parser.error(str(exc))

    benchmark, requirements = load_benchmark(BENCHMARK_PATH)
    if args.limit is not None:
        benchmark = benchmark.head(args.limit)
        requirements = requirements[:args.limit]

    print("Loading persistent Chroma index...")
    try:
        vector_store = load_vector_store()
        client = create_audit_client()
        print(f"Auditing {len(benchmark)} requirements...", flush=True)
        records = run_audits(requirements, vector_store, client, k=args.k, cache_path=args.cache)
    except (ConnectionError, RequestError, ResponseError) as exc:
        raise SystemExit(f"Evaluation stopped: Ollama service failure: {exc}") from exc

    rows, metrics, matrix = evaluate(benchmark, records, k=args.k)
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
    if args.out is not None:
        out = save_results(args.out, records, rows, metrics, matrix, model=GENERATION_MODEL)
        print(f"Saved evaluation results to {out}")


if __name__ == "__main__":
    main()
