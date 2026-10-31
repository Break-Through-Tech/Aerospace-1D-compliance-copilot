"""Batch auditing with the canonical retrieval and audit functions."""

from collections.abc import Iterable

from langchain_chroma import Chroma
from langchain_core.exceptions import OutputParserException
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.rag import audit_requirement, retrieve_clauses
from src.schemas import AuditRecord


def run_audits(
    requirements: Iterable[tuple[str, str]],
    vector_store: Chroma,
    client: ChatOllama,
    *,
    k: int,
) -> list[AuditRecord]:
    """Audit ordered ``(requirement_id, requirement_text)`` pairs.

    Reuse the supplied store and client and return records in input order.
    Model output and citation failures affect one record. Retrieval failures
    (including invalid stored metadata), service failures, and unexpected errors
    propagate to the caller. No benchmark labels or evaluation data are needed.
    """
    records = []
    for requirement_id, requirement_text in requirements:
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
            except ValueError as exc:
                # The current audit API uses ValueError for these two citation checks.
                # Other ValueErrors may indicate configuration or programming errors.
                if str(exc) not in {
                    "Cited clause is not in the retrieved set",
                    "Evidence must be an exact excerpt of the selected clause's requirement",
                }:
                    raise
                record.error = f"Citation validation failed: {exc}"
            else:
                record.verdict = verdict
                record.grounded = verdict.clause_id in record.retrieved_clause_ids
        records.append(record)
    return records
