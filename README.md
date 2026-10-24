# Aerospace 1D Compliance Copilot

October milestone: a non-agentic retrieval-augmented generation (RAG) baseline
that compares software requirements with NPR 7150.2D clauses. It retrieves
clauses, requests a structured verdict (`Meets`, `Partial`, or `Gap`), checks
citation grounding, and evaluates against a human-labeled benchmark.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/compliance_rag_baseline.py` | October baseline CLI and evaluation pipeline |
| `notebooks/compliance_copilot_october_ollama.ipynb` | Colab workflow with Ollama setup and file uploads |
| `results/october/` | Original v2 archive and extracted published outputs |
| `data/` | NASA source documents, 130 clauses, and ground-truth benchmark |
| `src/` | Earlier preprocessing and persistent vector-index utilities |
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

## Run in Colab

Open [the October notebook](notebooks/compliance_copilot_october_ollama.ipynb)
in Colab and run its cells in order. At step 3, upload:

- `scripts/compliance_rag_baseline.py`
- `data/npr-7150-2d-requirements.csv`
- `data/npr7150_ground_truth_benchmark.csv`

The notebook copies the uploaded script into Colab's import directory. The
notebook's `/content` paths refer to the Colab runtime, not this repository.

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
