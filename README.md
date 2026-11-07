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
python -m scripts.build_index --rebuild
python -m scripts.evaluate_rag --limit 5
```

The first command previews the source and target; `--rebuild` replaces only the
configured collection. See the [NASA corpus repair](docs/nasa-corpus-repair.md)
for verification, recovery, and the separate retry-safety dependency required
before production judgments or full benchmark evaluations of the repaired index.

Rebuilding and evaluation require Ollama; evaluation uses the existing Chroma
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
vectors; it does not rebuild the index.

Use a dedicated `.jsonl` file for the cache. Cache loading may repair an interrupted
final write, so do not point it at an unrelated file. Cache destinations under
`results/october/` and `data/` are rejected, including symlink aliases.

To save an evaluation report, add `--out` with a run directory:

```bash
python -m scripts.evaluate_rag --out results/runs/first-run
python -m scripts.evaluate_rag --cache results/audit_cache.jsonl --out results/runs/cached-run
```

Each directory contains complete canonical records (`audit_records.json`), benchmark
rows with predictions and audit details (`predictions.csv`), the unchanged evaluator
metrics (`metrics.json`), a human-row/model-column matrix including `INVALID`
(`confusion_matrix.csv`), and a readable `report.md`. Failed audits remain in the
records and CSV with their errors and an `INVALID` prediction. List fields in the
CSV use JSON, alongside the evaluator's existing comma-separated `retrieved` column.

Reporting reuses evaluation scores and ID alignment; it does not run another audit
or change caching. Citation groundedness measures clause-ID membership, not evidence
quality or overall compliance. The report distinguishes accepted-verdict and
all-record denominators. Without `--out`, evaluation keeps its terminal-only output.
Existing artifact files are never overwritten: choose a new run name for each report,
including a resumed cached evaluation. Destinations under `results/october/` and
`data/` are protected, and report files cannot collide with the cache path.

## Known classification limitation

The latest full canonical benchmark with `qwen3:4b` (50 requirements, top-k 5)
achieved 30% verdict accuracy, 92% retrieval Hit@5, 100% accepted verdicts, and
0% Meets recall. It predicted `Partial` for 40 requirements and `Gap` for 10,
including cases where added specificity was incorrectly treated as missing
compliance. The published October baseline achieved 62% verdict accuracy.

Classification quality remains a separate follow-up from architectural integration.
An accepted verdict passes schema, citation, and evidence checks; those checks do
not establish that its compliance judgment is correct. The canonical workflow
should not be treated as a validated compliance decision system. Full-run outputs
in `results/runs/` are local, ignored artifacts; the figures above record the known
limitation for reviewers without requiring those files.

## Preserved October baseline

See [colab/README.md](colab/README.md) for local and Colab instructions. This
baseline uses FAISS/NumPy retrieval and retains its original prompts, model
defaults, judges, verdict validation, evaluation, and reports. The module extraction
added checks for empty/malformed inputs and invalid numeric arguments, and limits
cache persistence and reuse to grounded verdicts. These safeguards preserve the
valid-input algorithms; the relocation was not entirely mechanical. Its lean
dependency file remains at `requirements-october.txt`; `requirements.txt` serves the canonical
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
