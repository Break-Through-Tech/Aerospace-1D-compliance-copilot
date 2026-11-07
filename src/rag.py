"""Retrieve NASA clauses and audit individual requirements or ordered batches."""

import inspect
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.exceptions import OutputParserException
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.config import OLLAMA_HOST, GENERATION_MODEL
from src.prompts import SYSTEM_PROMPT, RETRY_FEEDBACK, build_user_prompt
from src.schemas import AuditRecord, AuditVerdict, Clause
from src.cache import append_record, cache_key, client_settings, index_fingerprint, load_cache, successful


class CitationValidationError(ValueError):
    """The verdict cites an unretrieved clause or evidence outside its text."""


def create_audit_client(model: str = GENERATION_MODEL) -> ChatOllama:
    """Create an Ollama client with the shared compliance audit settings."""
    return ChatOllama(
        model=model,
        base_url=OLLAMA_HOST,
        temperature=0,
        num_predict=600,
        reasoning=False,
    )


def retrieve_clauses(
    requirement_text: str, vector_store: Chroma, *, k: int = 5
) -> list[Clause]:
    """Retrieve clauses in Chroma's relevance order without generation.

    Pass a store returned by src.vector_store.load_vector_store(). The store
    handles embeddings; this function does not build an index or start Ollama.
    """
    if not isinstance(requirement_text, str) or not requirement_text.strip():
        raise ValueError("requirement_text must be a nonempty string")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    documents = vector_store.similarity_search(requirement_text, k=k)
    return [Clause.model_validate(document.metadata) for document in documents]


def audit_requirement(
    requirement_text: str,
    clauses: list[Clause],
    *,
    model: str = GENERATION_MODEL,
    client: ChatOllama | None = None,
) -> AuditVerdict:
    """Audit already retrieved clauses; return one grounded verdict.

    Retry schema and citation failures once, with corrective feedback. After two
    invalid outputs, raise the last validation error. Service, configuration, and
    unexpected errors propagate immediately.
    Grounding proves evidence provenance, not that the judgment is correct.
    An injected ChatOllama instance supplies its own model and generation settings;
    model configures the default instance only. Ollama manages context length;
    the application does not independently detect source truncation.
    """
    prompt = build_user_prompt(requirement_text, clauses)
    if client is None:
        client = create_audit_client(model=model)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    retry_messages = [*messages, {"role": "user", "content": RETRY_FEEDBACK}]
    structured_client = client.with_structured_output(
        AuditVerdict, method="json_schema"
    )
    for attempt in range(2):
        try:
            verdict = structured_client.invoke(messages)
        except (ValidationError, OutputParserException) as exc:
            error = exc
        else:
            selected_clauses = [clause for clause in clauses if clause.clause == verdict.clause_id]
            if not selected_clauses:
                error = CitationValidationError("Cited clause is not in the retrieved set")
            elif not any(verdict.evidence in clause.requirement for clause in selected_clauses):
                error = CitationValidationError("Evidence must be an exact excerpt of the selected clause's requirement")
            else:
                return verdict

        if attempt == 1:
            raise error
        # Never echo model output or exception text: either may be arbitrarily
        # large. The original exception still propagates after the second failure.
        messages = retry_messages


def run_audits(
    requirements: Iterable[tuple[str, str]],
    vector_store: Chroma,
    client: ChatOllama,
    *,
    k: int,
    cache_path: str | Path | None = None,
) -> list[AuditRecord]:
    """Audit ordered ``(requirement_id, requirement_text)`` pairs.

    Reuse the supplied store and client and return records in input order.
    Model output and citation failures affect one record. Retrieval failures
    (including invalid stored metadata), service failures, and unexpected errors
    propagate to the caller. No benchmark labels or evaluation data are needed.
    An optional single-writer JSONL cache saves grounded successes immediately.
    Keep the index and model versions fixed while a batch is running.
    """
    records = []
    reused = 0
    if cache_path is not None:
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        # Source fingerprints cover context formatting and Python validators,
        # which JSON schemas alone cannot describe. Compute once per batch.
        configuration = cache_key({
            "format_version": 1,
            "generation": client_settings(client),
            "embedding": client_settings(vector_store.embeddings),
            "k": k,
            "system_prompt": SYSTEM_PROMPT,
            "retry_feedback": RETRY_FEEDBACK,
            "schemas": [schema.model_json_schema() for schema in (Clause, AuditVerdict, AuditRecord)],
            "libraries": {package: version(package) for package in (
                "pydantic", "pydantic-core", "langchain-ollama", "langchain-chroma", "chromadb",
            )},
            "implementation": [inspect.getsource(obj) for obj in (
                build_user_prompt, audit_requirement, retrieve_clauses,
                Clause, AuditVerdict, AuditRecord,
            )],
            "index": index_fingerprint(vector_store),
        })
        cached = load_cache(cache_path)
    for requirement_id, requirement_text in requirements:
        if cache_path is not None:
            key = cache_key({"configuration": configuration, "requirement": requirement_text})
            previous = cached.get(key)
            if previous is not None and previous.requirement_text == requirement_text:
                # IDs belong to this run, not to the wording's previous use.
                records.append(previous.model_copy(update={"requirement_id": str(requirement_id)}, deep=True))
                reused += 1
                continue
        clauses = retrieve_clauses(requirement_text, vector_store, k=k)
        record = AuditRecord(
            requirement_id=str(requirement_id),
            requirement_text=requirement_text,
            retrieved_clause_ids=[clause.clause for clause in clauses],
        )
        if not clauses:
            record.error = "Retrieval returned no clauses"
        else:
            try:
                verdict = audit_requirement(requirement_text, clauses, client=client)
            except (ValidationError, OutputParserException) as exc:
                record.error = f"Invalid audit output: {exc}"
            except CitationValidationError as exc:
                record.error = f"Citation validation failed: {exc}"
            else:
                record.verdict = verdict
                record.grounded = verdict.clause_id in record.retrieved_clause_ids
        records.append(record)
        if cache_path is not None and successful(record):
            append_record(cache_path, key, record)
            cached[key] = record
    if cache_path is not None:
        print(f"Cache: {reused} reused; {len(records) - reused} newly audited (including failed attempts).")
    return records
