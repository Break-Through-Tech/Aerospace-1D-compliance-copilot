# Published October baseline (v2)

These are the original results committed by Venya in
`3ee21124d8303428702404c4660cbb96b9fcb212`. The archive was moved from the
repository root without changing its bytes. Its five members are also extracted
here, byte for byte, so they can be reviewed directly on GitHub:

- `report.md`: summary, confusion matrix, and error analysis.
- `metrics.json`: machine-readable metrics.
- `predictions.csv`: per-requirement predictions and retrieval results.
- `audit_records.json`: structured verdict records and errors.
- `cache.jsonl`: original resumable-run cache.

Configuration: Ollama `nomic-embed-text` embeddings, Ollama `qwen3:4b` judge,
top-k 5, 50 benchmark rows. These are saved results, not a new model evaluation.

Write new runs to `results/runs/<experiment>/`, which is ignored by Git, to
preserve these published artifacts. Do not seed a different experiment with the
published cache.
