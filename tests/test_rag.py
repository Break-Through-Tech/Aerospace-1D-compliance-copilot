"""Current retrieval and audit contracts using injected stores and clients."""

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from src import rag
from src.schemas import AuditVerdict, Clause


def test_retrieve_uses_injected_store_and_preserves_order(clause_data):
    store = Mock()
    second = {**clause_data, "clause": "4.1.3", "swe_id": "SWE-051"}
    store.similarity_search.return_value = [
        Document(page_content="Search text", metadata=data)
        for data in [second, clause_data]
    ]
    result = rag.retrieve_clauses("Maintain requirements", store, k=2)
    store.similarity_search.assert_called_once_with("Maintain requirements", k=2)
    assert [clause.model_dump() for clause in result] == [second, clause_data]


def test_retrieve_rejects_invalid_metadata(clause_data):
    store = Mock()
    store.similarity_search.return_value = [
        Document(page_content="Search text", metadata={**clause_data, "requirement": " "})
    ]
    with pytest.raises(ValidationError):
        rag.retrieve_clauses("Maintain requirements", store)


@pytest.mark.parametrize("requirement,k", [
    (" \t", 5), (None, 5),
    ("Requirement", 0), ("Requirement", True), ("Requirement", 1.5),
])
def test_retrieve_rejects_invalid_inputs_before_search(requirement, k):
    store = Mock()
    with pytest.raises(ValueError):
        rag.retrieve_clauses(requirement, store, k=k)
    store.similarity_search.assert_not_called()


def test_audit_returns_grounded_verdict(clause_data, verdict_data):
    client = Mock()
    verdict = AuditVerdict.model_validate(verdict_data)
    client.with_structured_output.return_value.invoke.return_value = verdict
    clause = Clause.model_validate(clause_data)
    result = rag.audit_requirement("Maintain requirements", [clause], client=client)
    assert result == verdict


def test_audit_creates_default_client_with_requested_model(monkeypatch, clause_data, verdict_data):
    client = Mock()
    verdict = AuditVerdict.model_validate(verdict_data)
    client.with_structured_output.return_value.invoke.return_value = verdict
    make_client = Mock(return_value=client)
    monkeypatch.setattr(rag, "create_audit_client", make_client)

    result = rag.audit_requirement(
        "Maintain requirements", [Clause.model_validate(clause_data)], model="custom-model",
    )

    make_client.assert_called_once_with(model="custom-model")
    assert result == verdict


@pytest.mark.parametrize("clause_id,evidence,error", [
    ("999.999", "document", "Cited clause"),
    ("4.1.2", "invented quotation", "exact excerpt"),
    ("4.1.2", "Evidence from another clause", "exact excerpt"),
])
def test_audit_rejects_ungrounded_verdicts(
    clause_data, verdict_data, clause_id, evidence, error,
):
    clauses = [Clause.model_validate(clause_data), Clause.model_validate({
        **clause_data, "clause": "4.1.3", "requirement": "Evidence from another clause",
    })]
    client = Mock()
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict.model_validate({
        **verdict_data, "clause_id": clause_id, "evidence": evidence,
    })
    with pytest.raises(ValueError, match=error):
        rag.audit_requirement("Maintain requirements", clauses, client=client)
