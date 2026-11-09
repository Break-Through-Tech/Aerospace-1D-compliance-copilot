"""Reusable batch auditing with real RAG functions and mocked external services."""

from unittest.mock import Mock, call

import pytest
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from ollama import ResponseError
from pydantic import ValidationError

from src.rag import run_audits
from src.schemas import AuditRecord, AuditVerdict


@pytest.fixture
def requirements():
    return [("r3", "First requirement"), ("r1", "Second requirement"), ("r2", "Third requirement")]


@pytest.mark.parametrize("invalid_fields,error", [
    ({"clause_id": "999.999"}, "Cited clause is not in the retrieved set"),
    ({"evidence": "Invented quotation"},
     "Evidence must be an exact excerpt of the selected clause's requirement"),
])
def test_records_preserve_order_failures_and_shared_resources(
    requirements, clause_data, verdict_data, invalid_fields, error, capsys,
):
    store = Mock()
    store.similarity_search.return_value = [Document(page_content="Search text", metadata=clause_data)]
    valid_verdict = AuditVerdict.model_validate(verdict_data)
    invalid_citation = AuditVerdict.model_validate({**verdict_data, **invalid_fields})
    client = Mock()
    client.with_structured_output.return_value.invoke.side_effect = [
        valid_verdict, invalid_citation, invalid_citation, valid_verdict,
    ]

    records = run_audits(iter(requirements), store, client, k=2)

    assert all(isinstance(record, AuditRecord) for record in records)
    assert [record.requirement_id for record in records] == ["r3", "r1", "r2"]
    assert store.similarity_search.call_args_list == [
        call(text, k=2) for _, text in requirements
    ]
    assert client.with_structured_output.call_args_list == [
        call(AuditVerdict, method="json_schema")
    ] * 3
    assert records[0].verdict == valid_verdict
    assert records[0].grounded and records[0].error is None
    assert records[1].verdict is None
    assert records[1].retrieved_clause_ids == ["4.1.2"]
    assert not records[1].grounded
    assert records[1].error == f"Citation validation failed: {error}"
    assert records[2].verdict == valid_verdict
    assert client.with_structured_output.return_value.invoke.call_count == 4
    assert capsys.readouterr().out.splitlines() == [
        "[1] r3: Meets", "[2] r1: FAILED", "[3] r2: Meets",
    ]


@pytest.mark.parametrize("failure_kind", ["json", "schema"])
def test_empty_retrieval_and_invalid_output_become_failed_records(requirements, clause_data, failure_kind):
    store = Mock()
    store.similarity_search.side_effect = [[], [Document(page_content="Search text", metadata=clause_data)]]
    client = Mock()
    if failure_kind == "json":
        failure = OutputParserException("Malformed verdict JSON")
    else:
        with pytest.raises(ValidationError) as caught:
            AuditVerdict.model_validate({})
        failure = caught.value
    client.with_structured_output.return_value.invoke.side_effect = failure

    records = run_audits(requirements[:2], store, client, k=1)

    assert len(records) == 2
    assert records[0].retrieved_clause_ids == []
    assert records[0].verdict is None and "no clauses" in records[0].error
    assert records[1].verdict is None
    assert not records[1].grounded
    assert records[1].error.startswith("Invalid audit output:")
    assert str(failure) in records[1].error
    assert client.with_structured_output.return_value.invoke.call_count == 2


@pytest.mark.parametrize("failure", [
    ResponseError("Model unavailable", status_code=404),
    ValueError("Unexpected configuration or programming error"),
    ValueError("Cited clause is not in the retrieved set"),
    ValueError("Evidence must be an exact excerpt of the selected clause's requirement"),
    RuntimeError("Unexpected programming error"),
])
def test_service_and_unexpected_errors_stop_processing(requirements, clause_data, failure):
    store = Mock()
    store.similarity_search.return_value = [Document(page_content="Search text", metadata=clause_data)]
    client = Mock()
    client.with_structured_output.return_value.invoke.side_effect = failure

    with pytest.raises(type(failure)) as caught:
        run_audits(requirements, store, client, k=2)
    assert caught.value is failure
    store.similarity_search.assert_called_once()
    client.with_structured_output.return_value.invoke.assert_called_once()
