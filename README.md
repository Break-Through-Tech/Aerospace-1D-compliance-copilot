# Aerospace 1D Compliance Copilot

Compare software requirements with NASA NPR 7150.2D clauses using retrieval,
structured verdicts (`Meets`, `Partial`, `Gap`), and benchmark evaluation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Canonical RAG: preprocessing, persistent Chroma retrieval, Ollama audits, schemas, and evaluation |
| `scripts/setup.sh` | Create/reuse the virtual environment, install Python dependencies, and install missing Ollama/system prerequisites |
| `scripts/build_index.py`, `scripts/evaluate_rag.py` | Canonical index building and benchmark evaluation |
| `notebooks/01_preprocessing.ipynb`, `02_vector_indexing.ipynb`, `03_retrieval.ipynb` | Canonical exploration notebooks |
| `colab/` | Preserved October baseline: modules, local scripts, and Colab notebook |
| `data/` | Shared source documents, clause corpus, and ground-truth benchmark |
| `results/october/` | Published October artifacts; preserved unchanged |
| `tests/` | Canonical and October offline checks |

## Canonical Chroma workflow

Use Python 3.10 or newer on Linux/WSL and run the existing setup script from the
repository root:

```bash
bash scripts/setup.sh
source .venv/bin/activate
test -f .env || cp .env.example .env
```

The script creates `.venv` if needed, installs the project dependencies, and
installs `zstd` and Ollama when missing. System installation steps require the
appropriate permissions. Activating the environment applies to your current shell;
the script cannot activate it in the parent shell.

Review `.env` before proceeding. The example uses `http://localhost:11434`,
`qwen3:4b`, `nomic-embed-text`, collection `nasa_requirements`, and database
directory `chroma_db` relative to the repository root. Existing shell environment
variables take precedence over `.env`. Start `ollama serve` in another terminal
if it is not already running, then install the
configured models. These commands match the example; substitute your model names
if your configuration differs:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b
```

For a first build, when the configured collection is absent or empty:

```bash
python -m scripts.build_index
```

For an existing collection, use the [verification snippet](docs/nasa-corpus-repair.md#verify-an-existing-collection-without-rebuilding)
instead of rebuilding. Only replace a collection when verification or an intended
source or embedding-model change establishes that replacement is needed; see the
[rebuild procedure](docs/nasa-corpus-repair.md#build-or-replace-the-configured-collection).
Without `--rebuild`, a populated collection is rejected. Rebuilding is destructive
and non-atomic; stop readers until verification succeeds.

Once the collection is verified, run a small evaluation:

```bash
python -m scripts.evaluate_rag --limit 5
```

See the [NASA corpus repair](docs/nasa-corpus-repair.md) for source provenance,
recovery, and audit limits, and [WSL/Ollama troubleshooting](docs/nasa-corpus-repair.md#wslollama-memory-troubleshooting)
for the memory issue observed during evaluation. Audits use the configured generation
model and retry invalid responses once with fixed feedback, preserving the original
source without appending failed answers or error dumps. Ollama manages context
length; the application does not independently detect silent context truncation.

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

## Verified benchmark and limitations

The November 7, 2026 evaluation used `qwen3:4b`, `nomic-embed-text`, top-k 5,
the corrected 130-clause NASA corpus, and all 50 benchmark requirements without
cached predictions. The directional prompt and rationale-first verdict schema
achieved the following results, compared with 36% accuracy for the prior baseline:

| Metric | Value |
| --- | --- |
| Verdict accuracy | 74% (37/50) |
| Macro F1 | 0.7005 |
| Retrieval Hit@1 | 62% |
| Retrieval Hit@5 | 98% |
| Meets recall | 90.91% |
| Partial recall | 35.71% |
| Gap recall | 85.71% |
| Accepted verdicts | 49/50 |
| Invalid audits | 1 |

Five `Meets` predictions were incorrect against the benchmark: requirements
5, 9, 13, 16, and 23. These are five of 25 approvals (20%). The model still
struggles with partially satisfied obligations and selecting the intended NASA
clause. Accepted verdicts pass schema, citation, and evidence checks; those checks
do not establish that the compliance judgment is correct.

Each verdict evaluates one model-selected clause. `Meets` means the model judges
that clause's obligations satisfied; it does not establish coverage of every
applicable NASA clause or overall project compliance. The benchmark uses an intended
clause, so a different selection can answer a narrower question. For example,
measuring coverage during tests may address clause 4.5.10 while omitting the
tracking, recording, and reporting required by clause 4.5.9.

The 50 requirements have been reused during development, so these results do not
demonstrate generalization; independently reviewed, unseen requirements are needed.
This is an experimental compliance assistant, not a validated automated compliance
authority. Full-run outputs remain local, ignored artifacts under `results/runs/`.

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
