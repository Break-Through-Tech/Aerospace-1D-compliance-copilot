"""Command-line configuration and orchestration for the baseline."""
import argparse
import json
from pathlib import Path
from .data import load_clauses, load_benchmark
from .embeddings import TfidfEmbedder, SbertEmbedder, OllamaEmbedder
from .retrieval import ClauseIndex
from .judges import ClaudeJudge, GeminiJudge, OllamaJudge, HeuristicJudge
from .pipeline import run_pipeline
from .evaluation import evaluate
from .reporting import save_results

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clauses", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--embedder", choices=["sbert", "ollama", "tfidf"], default="sbert")
    ap.add_argument("--sbert-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ollama-embed-model", default=None, help="default: $OLLAMA_EMBED_MODEL or nomic-embed-text")
    ap.add_argument("--judge", choices=["claude", "gemini", "ollama", "heuristic"], default="claude")
    ap.add_argument("--model", default=None, help="judge model id (default: $CLAUDE_MODEL / $GEMINI_MODEL / $OLLAMA_MODEL=qwen3:4b)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="only first N benchmark rows (cheap dry run)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache", default=None, help="jsonl cache for resumable runs (default: <out>/cache.jsonl)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to pause between API calls (free-tier Gemini: try 6)")
    a = ap.parse_args(argv)

    if a.k < 1 or (a.limit is not None and a.limit < 1) or a.sleep < 0:
        ap.error("--k and --limit must be positive; --sleep must be nonnegative")

    clauses = load_clauses(a.clauses)
    bench = load_benchmark(a.benchmark)
    if a.limit:
        bench = bench.head(a.limit)
    missing = set(bench.clause_id) - {c.clause_id for c in clauses}
    if missing:
        print(f"WARNING: benchmark clause_ids not in clause corpus: {sorted(missing)}")

    embedder = {"sbert": lambda: SbertEmbedder(a.sbert_model),
                "ollama": lambda: OllamaEmbedder(a.ollama_embed_model),
                "tfidf": lambda: TfidfEmbedder([c.text for c in clauses])}[a.embedder]()
    index = ClauseIndex(clauses, embedder)
    judge = {"claude": lambda: ClaudeJudge(a.model), "gemini": lambda: GeminiJudge(a.model), "ollama": lambda: OllamaJudge(a.model),
             "heuristic": HeuristicJudge}[a.judge]()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    records = run_pipeline(bench, index, judge, a.k, a.cache or str(Path(a.out) / "cache.jsonl"), a.sleep)
    df, metrics, cm = evaluate(bench, records, a.k)

    out = Path(a.out)
    save_results(out, records, df, metrics, cm, dict(embedder=a.embedder, judge=a.judge))

    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("confusion_matrix", "per_class")}, indent=2))
    print(f"\nSaved to {out.resolve()}/ (predictions.csv, metrics.json, report.md, audit_records.json)")
    return df, metrics


def main_cli():
    """Console entry point; do not return the programmatic DataFrame to sys.exit."""
    main()
