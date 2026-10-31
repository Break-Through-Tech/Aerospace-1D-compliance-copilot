# Aerospace 1D Compliance Copilot

October milestone: a non-agentic retrieval-augmented generation (RAG) baseline
that compares software requirements with NPR 7150.2D clauses. It retrieves
clauses, requests a structured verdict (`Meets`, `Partial`, or `Gap`), checks
citation grounding, and evaluates against a human-labeled benchmark.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/compliance_rag_baseline.py` | Compatible CLI entry point for the modular baseline |
| `scripts/compliance_copilot_october_ollama.py` | Local Python conversion of the Ollama notebook for VS Code |
| `notebooks/compliance_copilot_october_ollama.ipynb` | Colab workflow with Ollama setup and file uploads |
| `results/october/` | Original v2 archive and extracted published outputs |
| `data/` | NASA source documents, 130 clauses, and ground-truth benchmark |
| `src/compliance_rag/` | Schemas, data, embeddings, retrieval, judges, caching, pipeline, evaluation, reporting, CLI |
| `src/*.py` | Earlier preprocessing and persistent vector-index utilities |
| `scripts/build_index.py` | Earlier persistent-index builder |
| `notebooks/01_preprocessing.ipynb`, `02_vector_indexing.ipynb` | Earlier data and indexing notebooks |
| `docs/coach/` | Coach events and office-hours notes |

## Run the October baseline locally

Use Python 3.10 or newer. Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-october.txt
```

Start Ollama in a separate terminal (`ollama serve`), then pull the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b
python scripts/compliance_rag_baseline.py \
  --clauses data/npr-7150-2d-requirements.csv \
  --benchmark data/npr7150_ground_truth_benchmark.csv \
  --embedder ollama --judge ollama --k 5 \
  --out results/runs/october-local
```

Add `--limit 5` for a short trial. Successful rows are cached in the output
directory; rerunning with the same configuration resumes the run. Use a new
output directory for a different experiment or changed input data.

For a smoke check without a model download or API key:

```bash
python scripts/compliance_rag_baseline.py \
  --clauses data/npr-7150-2d-requirements.csv \
  --benchmark data/npr7150_ground_truth_benchmark.csv \
  --embedder tfidf --judge heuristic --k 5 \
  --out results/runs/smoke
```

The heuristic check verifies the pipeline; its metrics do not reproduce the
published Ollama experiment. FAISS is optional (`pip install faiss-cpu`); the
pipeline falls back to NumPy search. Optional SBERT, Claude, and Gemini providers
require `sentence-transformers`, `anthropic`, and `google-genai`, respectively.
See the CLI's `--help` for provider and model options. The earlier persistent
indexing utilities use `requirements.txt` / `pyproject.toml`.

## Run the converted notebook in VS Code

Open this repository folder in VS Code and select `.venv/bin/python` with
**Python: Select Interpreter**. Install `requirements-october.txt` as above,
install Ollama separately, and run `ollama serve` in a terminal. Then run:

```bash
python scripts/compliance_copilot_october_ollama.py --pull-models
```

This defaults to `nomic-embed-text`, `qwen3:4b`, top-k 5, and a five-row trial.
It discovers CSVs in the repository's `data/` directory, combines benchmark
files, removes duplicate requirement/clause pairs, assigns sequential IDs, and
writes `combined_benchmark.csv` with the normal output artifacts to
`results/runs/october-local/`. Defaults resolve from the script location, so it
also runs using VS Code's Run Python File action. It uses an existing Ollama
server; `--pull-models` downloads models only when explicitly requested.

For the full benchmark with a ZIP archive:

```bash
python scripts/compliance_copilot_october_ollama.py --limit 0 --archive \
  --out results/runs/october-full
```

Use `--clauses PATH --benchmark PATH [PATH ...]` to select inputs explicitly,
`--data-dir` to discover another folder, `--host` for another Ollama server,
and `--model` / `--embedding-model` to change models. `--cache` selects a
persistent cache location. Use a fresh output directory and cache when changing
input data, embedding models, or experiment settings. Unlike the general
baseline CLI, this notebook conversion uses `--limit 0` for all rows.
Reports are local files; no Colab uploads, Drive mounts, or notebook magics
are required. Open `report.md` with VS Code's Markdown preview.

## Working on the modules

The baseline CLI composes these components in `src/compliance_rag/`:

| Module | Responsibility |
| --- | --- |
| `schemas.py` | Clause, audit verdict, record, and validation rules |
| `data.py` | CSV loading, notebook-style discovery and deduplication |
| `embeddings.py`, `retrieval.py` | Embedding providers and in-memory clause search |
| `prompts.py`, `judges/` | Shared prompt and separate Claude, Gemini, Ollama, heuristic judges |
| `cache.py`, `pipeline.py` | JSONL resume cache and retrieve/judge/grounding workflow |
| `evaluation.py`, `reporting.py` | Metrics and JSON/CSV/Markdown artifacts |
| `cli.py` | Argument parsing, provider selection, and orchestration |

Import components directly, for example:

```python
from src.compliance_rag.data import load_clauses
from src.compliance_rag.retrieval import ClauseIndex
from src.compliance_rag.pipeline import run_pipeline
```

The original `scripts/compliance_rag_baseline.py` command and its Python exports
remain available. The new RAG package does not import `src.config` or require its `.env`
variables. Provider SDKs are imported when their provider is selected.

For an installed package, `python -m pip install -e '.[october]'` installs the
existing indexing dependencies plus the baseline dependencies and adds the
`compliance-rag` command. Optional extras are `sbert`, `claude`, `gemini`, and
`faiss`. The lean local workflow only needs `requirements-october.txt`.

Run checks with:

```bash
python -m pip install pytest
python -m pytest
```

For November, a tool can call `ClauseIndex.search` for retrieval, a judge's
`judge(requirement_text, hits)` for checking, and consume `AuditRecord` for gap
logging and reports. This refactor retains the October deterministic pipeline;
the agent loop and dedicated November tools remain upcoming work. Existing
embedding batching, provider retry behavior, cache resume, and report formats
are retained.

## Run in Colab

The [October notebook](notebooks/compliance_copilot_october_ollama.ipynb) remains
an experiment workflow. Its repository-loading cell now loads the modularization
branch, including `src`, rather than uploading a standalone baseline file.
The Python script above is the recommended team workflow for VS Code.

## Published October results

The original Ollama v2 experiment has 50 benchmark rows and top-k = 5:

| Metric | Value |
| --- | --- |
| Retrieval hit@1 / hit@5 | 76% / 92% |
| Verdict accuracy | 62% |
| Gap precision / recall / F1 | 71.43% / 71.43% / 71.43% |
| Citation groundedness | 96% |
| Schema-valid outputs | 96% |

Read the [full report](results/october/report.md),
[metrics](results/october/metrics.json), and
[result provenance](results/october/README.md).

## Branch transition

`main-backup-2026-10-24` preserves the previous `main` at
`0a12fa54e0a888c48f99c68856534941abda294b`, including the structured RAG
implementation from PR #14. The organized `venya-october-rag-baseline+copilot`
branch supplies the new main contents. The previous implementation's
`src/rag.py`, `scripts/audit_requirement.py`, `tests/test_rag.py`, and `RAG.md`
remain available on the backup branch. Git history is retained.

Project context: [challenge overview](Challenge-Project-Overview.md),
[getting started](Getting-Started-for-Fellows.md), and [data documentation](data/README.md).
