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
    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    verdict = AuditVerdict.model_validate(verdict_data)
    client.with_structured_output.return_value.invoke.return_value = verdict
    clause = Clause.model_validate(clause_data)
    result = rag.audit_requirement("Maintain requirements", [clause], client=client)
    assert result == verdict


def test_audit_creates_default_client_with_requested_model(monkeypatch, clause_data, verdict_data):
    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
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
    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    client.with_structured_output.return_value.invoke.return_value = AuditVerdict.model_validate({
        **verdict_data, "clause_id": clause_id, "evidence": evidence,
    })
    with pytest.raises(rag.CitationValidationError, match=error):
        rag.audit_requirement("Maintain requirements", clauses, client=client)
    assert client.with_structured_output.return_value.invoke.call_count == 2


@pytest.fixture(params=["json", "schema", "validation", "citation", "evidence"])
def invalid_output(request, verdict_data):
    """Use the actual LangChain parser for malformed JSON and schema failures."""
    from langchain_core.output_parsers import PydanticOutputParser

    if request.param == "validation":
        with pytest.raises(ValidationError) as caught:
            AuditVerdict.model_validate({})
        return caught.value
    if request.param in {"json", "schema"}:
        from langchain_core.exceptions import OutputParserException

        parser = PydanticOutputParser(pydantic_object=AuditVerdict)
        try:
            parser.parse("not JSON" if request.param == "json" else "{}")
        except OutputParserException as exc:
            return exc
        pytest.fail("The parser should reject invalid output")
    if request.param == "citation":
        return AuditVerdict.model_validate({**verdict_data, "clause_id": "999.999"})
    return AuditVerdict.model_validate({**verdict_data, "evidence": "invented quotation"})


def test_invalid_output_is_corrected_with_feedback(clause_data, verdict_data, invalid_output):
    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    invoke = client.with_structured_output.return_value.invoke
    corrected = AuditVerdict.model_validate(verdict_data)
    invoke.side_effect = [invalid_output, corrected]

    result = rag.audit_requirement("Maintain requirements", [Clause(**clause_data)], client=client)

    assert result == corrected
    assert invoke.call_count == 2
    client.with_structured_output.assert_called_once_with(AuditVerdict, method="json_schema")
    original, retry = [call.args[0] for call in invoke.call_args_list]
    assert len(original) == 2
    assert retry[:2] == original
    feedback = retry[-1]["content"]
    assert retry[-1]["role"] == "user"
    assert "previous response was invalid" in feedback
    assert feedback == rag.RETRY_FEEDBACK
    assert len(retry) == 3
    assert all(message["role"] != "assistant" for message in retry)
    assert "4.1.2" in retry[1]["content"]
    if isinstance(invalid_output, Exception):
        assert str(invalid_output) not in feedback
    else:
        assert invalid_output.model_dump_json() not in feedback


def test_repeated_invalid_output_raises_last_error_after_two_attempts(clause_data, invalid_output):
    from langchain_core.exceptions import OutputParserException

    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    invoke = client.with_structured_output.return_value.invoke
    invoke.side_effect = [invalid_output, invalid_output]

    with pytest.raises((ValidationError, OutputParserException, rag.CitationValidationError)) as caught:
        rag.audit_requirement("Maintain requirements", [Clause(**clause_data)], client=client)

    assert invoke.call_count == 2
    if isinstance(invalid_output, Exception):
        assert caught.value is invalid_output


@pytest.mark.parametrize("failure", [
    ConnectionError("Ollama unavailable"),
    ValueError("Invalid model configuration"),
    RuntimeError("Unexpected programming error"),
    TypeError("Unexpected response handling error"),
])
def test_non_output_errors_propagate_without_retry(clause_data, failure):
    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    invoke = client.with_structured_output.return_value.invoke
    invoke.side_effect = failure

    with pytest.raises(type(failure)) as caught:
        rag.audit_requirement("Maintain requirements", [Clause(**clause_data)], client=client)

    assert caught.value is failure
    invoke.assert_called_once()


def test_service_failure_during_correction_stops_immediately(clause_data):
    from langchain_core.exceptions import OutputParserException
    from ollama import ResponseError

    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    invoke = client.with_structured_output.return_value.invoke
    failure = ResponseError("Model unavailable", status_code=404)
    invoke.side_effect = [OutputParserException("Invalid JSON"), failure]

    with pytest.raises(ResponseError) as caught:
        rag.audit_requirement("Maintain requirements", [Clause(**clause_data)], client=client)

    assert caught.value is failure
    assert invoke.call_count == 2


def test_structured_client_configuration_error_propagates(clause_data):
    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    failure = ValueError("Unsupported structured output configuration")
    client.with_structured_output.side_effect = failure

    with pytest.raises(ValueError) as caught:
        rag.audit_requirement("Maintain requirements", [Clause(**clause_data)], client=client)

    assert caught.value is failure
    client.with_structured_output.assert_called_once()
