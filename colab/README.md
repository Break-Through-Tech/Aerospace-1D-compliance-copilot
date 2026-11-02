# Preserved October baseline

This is the October Colab RAG implementation, retained separately from the
canonical Chroma implementation in `src/`. Algorithms and model defaults are
unchanged.

- `compliance_rag/` contains the shared pipeline, embeddings, FAISS/NumPy retrieval,
  prompts, judges, validation/retries, resume cache, evaluation, and reporting.
- `scripts/compliance_rag_baseline.py` exposes those modules through the general CLI.
- `scripts/compliance_copilot_october_ollama.py` runs the notebook workflow locally:
  CSV discovery, benchmark combination, Ollama checks, and optional ZIP export.
- `notebooks/compliance_copilot_october_ollama.ipynb` runs that baseline in Colab.
- `scripts/evaluate_saved_predictions.py` evaluates saved October predictions
  without models; its retrieval analysis expects exactly five retrieved clauses.

## Local execution

Use Python 3.10+ and run from the repository root:

```bash
python -m pip install -r requirements-october.txt
python colab/scripts/compliance_rag_baseline.py \
  --clauses data/npr-7150-2d-requirements.csv \
  --benchmark data/npr7150_ground_truth_benchmark.csv \
  --embedder tfidf --judge heuristic --k 5 --limit 5 \
  --out results/runs/smoke
```

This smoke run requires no model downloads or API keys. For the Ollama notebook
workflow, install Ollama separately, start `ollama serve`, then explicitly pull
models and run:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b
python colab/scripts/compliance_copilot_october_ollama.py
```

Defaults are `nomic-embed-text`, `qwen3:4b`, top-k 5, a five-row trial, shared
repository `data/`, and `results/runs/october-local/`. The local script resolves
these default paths from its own location, including when run from another working
directory. Use `--limit 0` for all rows, `--archive` for a ZIP, or `--pull-models`
to explicitly request downloads. See `--help` for input, output, host, and model
options. The general baseline CLI instead requires a positive `--limit` or its
omission for all rows.

Successful rows resume from `<out>/cache.jsonl`; `--cache` selects another path.
Use fresh output/cache paths when changing inputs or experiment settings. Reports
include `predictions.csv`, `metrics.json`, `report.md`, and `audit_records.json`.
FAISS is optional with a NumPy fallback. SBERT, Claude, and Gemini require their
optional provider dependencies. Direct imports use `colab.compliance_rag`, for
example `from colab.compliance_rag.pipeline import run_pipeline`.

Evaluate published predictions without rerunning models:

```bash
python colab/scripts/evaluate_saved_predictions.py
```

Its default output remains `results/issue13_saved_evaluation/`.

## Colab execution

Open `colab/notebooks/compliance_copilot_october_ollama.ipynb` in Google Colab and
run its cells in order. The setup cells install dependencies and Ollama and pull
models. The repository cell clones `integration/rag-pipeline`; that remote branch
must contain this relocation before a fresh clone can use it. Alternatively, set
`REPO` to an existing checkout containing `colab/`. The notebook adds the repository
root to Python's import path and imports `colab.compliance_rag`.

The notebook reads the checkout's shared `data/` directory and keeps its original
`/content/out_<RUN_NAME>` outputs, optional Drive cache, and downloadable ZIP paths.
Shared source inputs remain in repository `data/`. Published artifacts remain in
`results/october/`; write new experiments elsewhere.
