# Structured RAG baseline

This adds only the retrieval and structured judgment pipeline. It reuses
`src/vector_store.py` and does not change indexing or add benchmark evaluation.
The supplied repo currently uses a persistent Chroma collection; the pipeline
also accepts an in-memory Chroma store directly.

From the repository root, install dependencies with `pip install -e .`.
Start Ollama, install `nomic-embed-text` and a chat model (for example,
`ollama pull llama3.2`), then build the existing index:

```sh
python -m scripts.build_index
python -m scripts.audit_requirement "The software requirements shall be documented in an SRS." --model llama3.2 --k 3
```

To reuse a store already built in Python:

```python
from src.rag import audit_requirement

result = audit_requirement("The software requirements shall be documented in an SRS.",
                           vector_store, model="llama3.2", k=3)
print(result.model_dump_json(indent=2))
```

The fixed flow retrieves top-k clauses by vector distance, orders returned ties
by identifier, asks Ollama for schema-constrained JSON, and validates the response.
Each retrieved clause must receive exactly one `Meets`, `Partial`, or `Gap`
verdict, a rationale, and a citation containing `swe_id`, `clause`, `source`, and
an exact `quote`. Invalid schemas, unknown citations, fabricated quotes,
duplicates, missing verdicts, and empty retrieval raise errors rather than
returning a report. There are no retries or agent loops.

Temperature is zero and seed is 42. Reproducibility requires a fixed corpus,
index, embedding model, chat model, and runtime; these settings do not guarantee
bit-identical LLM output or tie membership at the top-k boundary. Citation
validation proves that the cited text was retrieved, not that the judgment is
correct. Results assess candidate clauses individually; they are not an overall
compliance verdict and do not determine whether a clause applies to a project.

Run the isolated tests without an Ollama server:

```sh
python -m unittest discover -s tests -v
```
