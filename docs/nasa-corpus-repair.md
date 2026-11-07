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

## Separate retry-safety dependency

**Integrate the separate retry-safety PR before running production judgments or
full benchmark evaluations against the repaired index.** The longer clauses can
overflow the original judge's 4,096-token context window during retries, risking
silent source truncation. This corpus-only PR does not resolve that issue:
`src/rag.py`, retrieval, prompts, and verdict schemas remain unchanged, and no
tokenizer or retry protection is included. A successful index rebuild verifies
storage integrity; it does not establish judge-context safety or verdict quality.
