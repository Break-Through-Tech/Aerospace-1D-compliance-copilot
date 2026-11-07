# NASA NPR 7150.2D corpus repair

The original extraction stopped too early around SWE markers, omitting attached
lists, nested items, follow-up conditions, source Notes, and a traceability table.
The reviewed repair restores clause-local text in **62 of 130 records**; **68 are
unchanged**. SWE IDs, clause IDs, record order, original main sentences, and source
metadata are preserved. The original CSV and JSON remain unchanged.

Canonical input is `data/npr-7150-2d-requirements_fixed.csv`; `_fixed.json` contains
equivalent records. Existing preprocessing still creates one document per SWE,
with complete requirement text and the same metadata structure.

The source review compared all 62 repaired passages against the
[official NASA NPR 7150.2D PDF](https://explorers.larc.nasa.gov/APSMEX26/SMEX/pdf_files/NASA21_NPR_7150_2D.pdf),
also retained as `data/npr-7150-2d.pdf` (SHA-256
`1a5b7f9b0b0141e88374adf7a21a4d6ced0b29c221ca89a0a514f495d6532802`).
Checks covered clause boundaries, SWE markers, wording, and rendered complex
passages, including the traceability table on PDF page 31. The canonical pipeline
uses the reviewed data; `tests/test_corpus_integration.py` pins its bytes and checks
complete text through loading and document creation. Source accuracy does not
establish applicability: Notes and “should” recommendations are not automatically
mandatory obligations,
and global applicability rules, appendices, and referenced standards remain
outside this 130-record corpus.

## Example: a missing traceability table

Clause **3.12.1 (SWE-052)** originally ended at:

> The project manager shall perform, record, and maintain bi-directional traceability between the following software elements:

The extraction omitted the table that says which elements and software classes
apply. The repaired record retains that sentence and appends the complete table
and following note from PDF page 31. Two restored table rows illustrate the change
(the repaired file contains the full table):

| Bi-directional traceability | Class A, B, and C | Class D | Class F |
| --- | --- | --- | --- |
| Higher-level requirements to the software requirements | X | | X |
| Software requirements to the system hazards | X | X | |

The following note addresses software-related hazards and controls. This is one
longer record, not a new clause.

Without the table, the model could not see the actual traceability scope. Restoring
it supplies the missing evidence; determining the project's software class and
applicability still requires project context. The benchmark's existing labels
were not revised as part of the repair.

## Repair inventory

These are all 62 records with changed requirement text, identified by comparing
the preserved original JSON with the repaired JSON. The page column locates the
bracketed SWE marker in the pinned 89-page PDF; attached text may continue on the
next page. These are source navigation anchors, not claims that every appended
note is mandatory or that each passage fits on one page.

| PDF page | Repaired clause (SWE ID) |
| --- | --- |
| 11 | 2.1.1.2 (SWE-004) |
| 12 | 2.1.5.2 (SWE-003); 2.1.5.4 (SWE-140) |
| 13 | 2.1.5.6 (SWE-006); 2.1.5.7 (SWE-091); 2.1.5.10 (SWE-142) |
| 14 | 2.1.5.12 (SWE-153); 2.1.5.15 (SWE-217); 2.1.5.16 (SWE-214) |
| 16 | 2.1.8.2 (SWE-126) |
| 20 | 3.1.2 (SWE-033); 3.1.3 (SWE-013) |
| 21 | 3.1.4 (SWE-024); 3.1.6 (SWE-036); 3.1.8 (SWE-039); 3.1.10 (SWE-042) |
| 22 | 3.1.13 (SWE-125); 3.1.14 (SWE-027) |
| 23 | 3.2.1 (SWE-015); 3.2.2 (SWE-151) |
| 24 | 3.3.1 (SWE-016); 3.4.1 (SWE-017); 3.5.1 (SWE-020) |
| 25 | 3.6.1 (SWE-022); 3.6.2 (SWE-141); 3.6.3 (SWE-131); 3.6.4 (SWE-178) |
| 26 | 3.7.3 (SWE-134) |
| 27 | 3.7.4 (SWE-219); 3.7.5 (SWE-220); 3.8.1 (SWE-146) |
| 28 | 3.8.2 (SWE-206); 3.9.2 (SWE-032) |
| 29 | 3.10.2 (SWE-148) |
| 30 | 3.11.3 (SWE-154); 3.11.5 (SWE-159) |
| 31 | 3.11.7 (SWE-185); 3.11.8 (SWE-210); 3.12.1 (SWE-052) |
| 32 | 4.1.2 (SWE-050) |
| 33 | 4.2.3 (SWE-057); 4.2.4 (SWE-143) |
| 34 | 4.4.4 (SWE-135); 4.4.5 (SWE-062) |
| 35 | 4.4.8 (SWE-136); 4.5.2 (SWE-065); 4.5.3 (SWE-066); 4.5.4 (SWE-187) |
| 36 | 4.5.6 (SWE-070); 4.5.8 (SWE-073); 4.5.9 (SWE-189) |
| 37 | 4.5.10 (SWE-190); 4.5.13 (SWE-193) |
| 39 | 5.1.4 (SWE-081); 5.1.5 (SWE-082) |
| 40 | 5.2 (SWE-086); 5.3.2 (SWE-087) |
| 41 | 5.3.3 (SWE-088); 5.4.2 (SWE-090) |
| 42 | 5.4.5 (SWE-199); 5.5.2 (SWE-202); 5.5.3 (SWE-203) |

## Verify an existing collection without rebuilding

Activate the virtual environment and review `.env`, particularly the database path
and collection. Run this from the repository root. It reads the existing collection
through the normal loader and compares count, IDs, complete text, and metadata;
it does not add/delete records, compute embeddings, or call Ollama:

```bash
python - <<'PY'
from src.config import DATA_PATH
from src.preprocessing import create_documents, load_requirements
from src.vector_store import load_vector_store

documents = create_documents(load_requirements(DATA_PATH))
expected = {doc.id: (doc.page_content, doc.metadata) for doc in documents}
stored = load_vector_store().get(include=["documents", "metadatas"])
actual = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))
if len(documents) != 130 or len(expected) != 130 or len(stored["ids"]) != 130 or actual != expected:
    raise SystemExit("Index differs from the repaired source; investigate before rebuilding.")
print("Verified 130 indexed records, including exact text and metadata.")
PY
```

An absent/empty collection fails rather than being populated. A successful check
establishes source-record equality, not embedding-model provenance or retrieval
quality. Keep the embedding model used to build the index fixed; model/tag changes
need separate review. Corpus integrity tests can also be run without Ollama using
`python -m pytest tests/test_corpus_integration.py -q`; their disposable test indexes
do not verify your configured database.

## Build or replace the configured collection

For a first build only, start Ollama with the configured embedding model and run:

```bash
python -m scripts.build_index
```

To deliberately replace an outdated collection, back it up if rollback is needed,
stop all readers, review `.env` for the target, and run the alternative command:

```bash
python -m scripts.build_index --rebuild
```

The default command builds an absent or empty collection and rejects a populated
collection without modifying its records. `--rebuild` explicitly allows destructive
replacement of the configured collection using `reset_collection()`. Check `.env`
for the intended database path and collection before rebuilding. Both commands
validate 130 unique SWE/clause IDs, use `add_documents()` to create fresh embeddings,
then verify stored count, IDs, complete text, and metadata against the source.
Require exit status zero and
`Verified 130 indexed records, including exact text and metadata.` Repeated
rebuilds remove stale records without duplicates and preserve unrelated collections.

Replacement is **not atomic**: failure can leave an empty or partial collection.
Keep readers stopped, fix the failure, and rerun the complete rebuild until
verification succeeds. Recreation uses pipeline defaults and discards custom
collection settings; back up the database first if rollback is needed.

## Validation retries and context limits

Audits use the configured `GENERATION_MODEL` and request a structured verdict.
Schema, retrieved-clause citation, and exact-evidence checks must pass before a
verdict is accepted. A validation failure triggers one retry with the original
requirement and complete retrieved clauses plus short, fixed corrective feedback.
Failed answers and exception dumps are never appended. After two invalid attempts,
the batch records an error with no verdict; service, configuration, and unexpected
errors propagate immediately. Cache fingerprints include the retry policy.

Ollama manages context length; the application no longer sets `num_ctx`, counts
tokens, or independently detects silent context truncation. Configure Ollama's
context capacity for the selected model and input size. The 600-token output limit
remains and can still cause invalid JSON. Source text is preserved in the submitted
messages, but fitting it into the server context is not guaranteed. Schema and
citation validation establish format and source membership, not logical correctness.

## WSL/Ollama memory troubleshooting

During local WSL2 evaluations with about 7.7 GiB RAM and 2 GiB swap, Linux killed
Ollama runners after host-memory prompt-cache usage grew beyond 5 GiB. Swap was
exhausted while GPU memory remained available. Kernel OOM records established
memory exhaustion; sleep/resume was not established as the cause.

The environment-specific fix verified with Ollama **0.40.2** was
`LLAMA_ARG_CACHE_RAM=0` in the Ollama systemd service environment. This setting
belongs to the service, not this project's `.env`. After the user configured it
and restarted the service, both runners logged **"prompt cache is disabled"** and
the 10-row stability check and full 50-row evaluation completed with stable memory.
This documents the tested setup, not a requirement for every Ollama version or host.

To reproduce that specific systemd configuration when diagnosing the same issue,
stop active audits, run `sudo systemctl edit ollama`, and add these lines to the
service override while preserving any other settings:

```ini
[Service]
Environment="LLAMA_ARG_CACHE_RAM=0"
```

Then run `sudo systemctl daemon-reload` and `sudo systemctl restart ollama`.
Restarting interrupts in-flight requests. This service-level setting is not applied
by the Python setup or `.env.example`; verify runner behavior before relying on it.

To inspect an existing systemd setup without changing it:

```bash
systemctl show ollama --property=ActiveState,Environment
journalctl -u ollama --since today --no-pager
journalctl -k --since today --no-pager
free -h
nvidia-smi  # if an NVIDIA GPU is available
```

The environment confirms configuration; runner startup logs confirm whether it is
honored. Search the service log for `prompt cache is disabled` and the kernel log
for `oom-kill` or `Out of memory`. A responsive API alone does not establish stable
generation. After a runtime change, use a small run first and inspect RAM, swap,
and logs before attempting all 50 requirements. Stop after a runtime failure and
preserve diagnostics instead of repeatedly restarting full evaluations.

Ollama's internal prompt cache retains model state in host memory. It is separate
from the application's optional JSONL **prediction cache**, which stores accepted
audit results. Disabling the former does not control reuse of the latter; omit
`--cache` for fresh predictions. The model still needs working context memory even
when the host prompt cache is disabled.
