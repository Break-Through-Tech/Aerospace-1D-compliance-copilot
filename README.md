# Aerospace 1D Compliance Copilot

Compare software requirements with NASA NPR 7150.2D clauses using retrieval,
structured verdicts (`Meets`, `Partial`, `Gap`), and benchmark evaluation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Canonical RAG: preprocessing, persistent Chroma retrieval, Ollama audits, schemas, and evaluation |
| `scripts/build_index.py`, `scripts/evaluate_rag.py` | Canonical index building and benchmark evaluation |
| `notebooks/01_preprocessing.ipynb`, `02_vector_indexing.ipynb`, `03_retrieval.ipynb` | Canonical exploration notebooks |
| `colab/` | Preserved October baseline: modules, local scripts, and Colab notebook |
| `data/` | Shared source documents, clause corpus, and ground-truth benchmark |
| `results/october/` | Published October artifacts; preserved unchanged |
| `tests/` | Canonical and October offline checks |

## Canonical Chroma workflow

Use Python 3.10 or newer and run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Configure `.env` with your Ollama host, generation and embedding models, Chroma
collection, and vector-store path. Start Ollama and explicitly install the models
you selected, then run:

```bash
python -m scripts.build_index
python -m scripts.evaluate_rag --limit 5
```

Index building and evaluation require Ollama; evaluation uses the existing Chroma
index. Omit `--limit` for the full benchmark. The canonical implementation is
independent of the October package. To save completed audits and resume a later run:

```bash
python -m scripts.evaluate_rag --cache results/audit_cache.jsonl
```

The optional JSONL cache reuses only successful, grounded audits with compatible
requirement text, model settings, prompts, validation, and index contents. Each
success is saved immediately; failed requirements are retried next time. A summary
reports reused and newly audited records. Without `--cache`, every requirement is
audited normally. Use one writer per cache file and keep the index fixed during a
batch. Use fixed model versions: replacing a model under the same Ollama tag is
not detected. Cache checking reads the stored index once per batch, including its
vectors; it does not rebuild the index. Reporting remains future work.

## Preserved October baseline

See [colab/README.md](colab/README.md) for local and Colab instructions. This
baseline uses FAISS/NumPy retrieval and retains its original prompts, model
defaults, judges, validation, caching, evaluation, and reports. Its lean dependency
file remains at `requirements-october.txt`; `requirements.txt` serves the canonical
workflow. Installing `python -m pip install -e '.[october]'` includes both workflows
and provides the October `compliance-rag` console command. Optional October extras
are `sbert`, `claude`, `gemini`, and `faiss`.

The published October Ollama v2 run has 50 benchmark rows and top-k 5. See its
[report](results/october/report.md), [metrics](results/october/metrics.json), and
[provenance](results/october/README.md).

## Tests

```bash
python -m pytest -q
```

Project context: [challenge overview](Challenge-Project-Overview.md),
[getting started](Getting-Started-for-Fellows.md), and [data documentation](data/README.md).
