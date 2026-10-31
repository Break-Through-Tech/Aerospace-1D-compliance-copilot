"""Local Python equivalent of the October Ollama notebook.

VS Code: select your .venv interpreter, then run this file or use its CLI.
Install dependencies and Ollama first; see README.md. This script does not install
system software, start background servers, or require Google Colab.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compliance_rag.cli import main as run_baseline
from src.compliance_rag.data import discover_csvs, combine_benchmarks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--clauses", type=Path, help="Override auto-detected clause corpus")
    parser.add_argument("--benchmark", type=Path, nargs="+", help="Override auto-detected benchmark CSVs")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3:4b"))
    parser.add_argument("--embedding-model", default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=5, help="Notebook dry-run default; use 0 for all rows")
    parser.add_argument("--out", type=Path, default=ROOT / "results/runs/october-local")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--pull-models", action="store_true", help="Pull both models using the Ollama CLI")
    parser.add_argument("--archive", action="store_true", help="Create a zip of the output directory")
    args = parser.parse_args(argv)
    if args.k < 1 or args.limit < 0:
        parser.error("--k must be positive and --limit must be nonnegative")
    if args.clauses and args.benchmark:
        clauses, benchmarks = args.clauses, args.benchmark
    else:
        detected_clauses, detected_benchmarks = discover_csvs(args.data_dir)
        clauses = args.clauses or detected_clauses
        benchmarks = args.benchmark or detected_benchmarks
    if not clauses or not benchmarks:
        parser.error("No clause corpus or benchmark CSV found; specify --clauses and --benchmark")

    os.environ["OLLAMA_HOST"] = args.host
    from ollama import Client
    try:
        Client(host=args.host).list()
    except Exception as exc:
        parser.error(f"Ollama is unavailable at {args.host}. Start `ollama serve` first. {exc}")
    if args.pull_models:
        for model in (args.embedding_model, args.model):
            subprocess.run(["ollama", "pull", model], check=True)

    combined = combine_benchmarks(benchmarks)
    args.out.mkdir(parents=True, exist_ok=True)
    combined_path = args.out / "combined_benchmark.csv"
    if combined_path.resolve() in {Path(p).resolve() for p in [clauses, *benchmarks]}:
        parser.error("Output combined_benchmark.csv would overwrite an input")
    combined.to_csv(combined_path, index=False)
    print(f"Clause corpus: {clauses}; combined benchmark rows: {len(combined)}")
    baseline_args = ["--clauses", str(clauses), "--benchmark", str(combined_path),
                     "--embedder", "ollama", "--ollama-embed-model", args.embedding_model,
                     "--judge", "ollama", "--model", args.model, "--k", str(args.k), "--out", str(args.out)]
    if args.limit:
        baseline_args += ["--limit", str(args.limit)]
    if args.cache:
        baseline_args += ["--cache", str(args.cache)]
    df, metrics = run_baseline(baseline_args)
    print((args.out / "report.md").read_text(encoding="utf-8"))
    if args.archive:
        archive = shutil.make_archive(str(args.out.resolve()), "zip", args.out)
        print(f"Archived outputs: {archive}")
    return df, metrics


if __name__ == "__main__":
    main()
