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
passages, including the traceability table on PDF page 31. This branch copies
the reviewed data unchanged; tests pin its bytes and check complete text through
loading and document creation. Source accuracy does not establish applicability:
Notes and “should” recommendations are not automatically mandatory obligations,
and global applicability rules, appendices, and referenced standards remain
outside this 130-record corpus.

## Rebuild and verify the configured collection

After review and merge, activate the virtual environment, configure `.env`, and
start Ollama with the configured embedding model. Stop all readers before writing.
From the repository root:

```bash
python -m scripts.build_index            # first build: collection absent or empty
python -m scripts.build_index --rebuild  # replace an existing index when needed
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

## Retry safety and context limits

Longer clauses could overflow the judge's context window when retries retained
failed answers and exception text. Retries now keep the complete original
requirement and retrieved clauses and append only fixed corrective feedback.
A tokenizer-based preflight reserves the source, retry feedback, verdict schema,
64 tokens of chat framing, and the output allowance before either attempt.
Oversized requests raise `ContextBudgetError` without inference or source trimming.
Validation remains unchanged, with at most two attempts; service and unexpected
errors propagate. Cache fingerprints include the retry and budgeting policy.

Context budgeting supports only the reviewed standard `qwen3:4b` tokenizer and
chat template. Defaults remain 4,096 context tokens and 600 output tokens.
Although the generation model is configurable, other models and unknown or
unbounded client limits are explicitly rejected. The guard cannot detect a
custom model/template substituted under the same tag; retain the reviewed
configuration. Longer inputs or larger retrieval/output settings may be rejected;
the output limit can still cause invalid JSON. Budgets are checked offline,
not against live server token telemetry, and do not establish verdict quality.

The bundled tokenizer is an unmodified, gzip-compressed copy from
[Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer.json).
Its uncompressed SHA-256 is
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`,
checked on load. The Apache-2.0 license is included in
`src/assets/Qwen3-LICENSE.txt`; packaging includes both files. The direct
`tokenizers` dependency enables accurate offline counting without runtime downloads.
