"""Source-resolved evidence keeps strict verdict and retrieved-clause checks."""

import json
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ValidationError

from src import agent, rag
from src.prompts import RETRY_FEEDBACK
from src.schemas import AuditDecision, AuditVerdict, Clause


def decision(verdict_data, **overrides):
    return AuditDecision(**{key: value for key, value in {**verdict_data, **overrides}.items()
                            if key != "evidence"})


def test_decision_removes_only_evidence_from_advertised_fields():
    canonical = AuditVerdict.model_json_schema()
    resolved = AuditDecision.model_json_schema()
    assert resolved["properties"] == {
        key: value for key, value in canonical["properties"].items() if key != "evidence"
    }
    assert resolved["required"] == [key for key in canonical["required"] if key != "evidence"]
    assert resolved["additionalProperties"] is False


def test_model_cannot_supply_evidence_in_resolved_mode(verdict_data):
    with pytest.raises(ValidationError, match="evidence"):
        AuditDecision(**verdict_data)


@pytest.mark.parametrize("label", ["Meets", "Partial", "Gap"])
def test_copies_complete_text_only_from_selected_retrieved_clause(clause_data, verdict_data, label):
    source = 'Exact “NASA” text.\nSecond line, punctuation preserved: A/B & C.'
    clauses = [Clause(**clause_data), Clause(**{
        **clause_data, "clause": "4.1.3", "swe_id": "SWE-051", "requirement": source,
    })]
    output = decision(verdict_data, clause_id="4.1.3", verdict=label,
                      missing_elements=[] if label == "Meets" else ["Maintenance"])
    client = Mock()
    client.with_structured_output.return_value.invoke.return_value = output
    result = rag.audit_requirement("Original requirement", clauses, client=client)
    assert isinstance(result, AuditVerdict)
    assert result.model_dump() == {**output.model_dump(), "evidence": source}
    client.with_structured_output.assert_called_once_with(AuditDecision, method="json_schema")
    messages = client.with_structured_output.return_value.invoke.call_args.args[0]
    assert "Do not return evidence" in messages[0]["content"]
    assert "evidence must be an exact" not in messages[0]["content"]
    assert json.loads(messages[1]["content"])["requirement_under_audit"] == "Original requirement"


@pytest.mark.parametrize("ambiguous", [False, True])
def test_unknown_or_ambiguous_clause_ids_fail_after_one_retry(clause_data, verdict_data, ambiguous):
    clauses = [Clause(**clause_data)]
    if ambiguous:
        clauses.append(Clause(**{**clause_data, "requirement": "Different source with the same ID."}))
    output = decision(verdict_data, clause_id="4.1.2" if ambiguous else "999.999")
    client = Mock()
    invoke = client.with_structured_output.return_value.invoke
    invoke.return_value = output
    with pytest.raises(rag.CitationValidationError, match="exactly one" if ambiguous else "retrieved set"):
        rag.audit_requirement("Original requirement", clauses, client=client)
    assert invoke.call_count == 2


@pytest.mark.parametrize("label,missing", [("Meets", ["Maintenance"]), ("Partial", []), ("Gap", [])])
def test_completed_verdict_still_checks_missing_elements_and_can_retry(
    clause_data, verdict_data, label, missing,
):
    with pytest.raises(ValidationError) as caught:
        decision(verdict_data, verdict=label, missing_elements=missing)
    invalid = caught.value
    valid = decision(verdict_data)
    client = Mock()
    invoke = client.with_structured_output.return_value.invoke
    invoke.side_effect = [invalid, valid]
    result = rag.audit_requirement(
        "Original requirement", [Clause(**clause_data)], client=client,
    )
    assert result.verdict == "Meets" and result.missing_elements == []
    assert result.evidence == clause_data["requirement"]
    assert invoke.call_count == 2
    original, retry = [call.args[0] for call in invoke.call_args_list]
    assert retry[:2] == original
    assert retry[-1]["content"] == RETRY_FEEDBACK
    assert "Copy evidence" not in retry[-1]["content"]
    assert str(invalid) not in retry[-1]["content"]


def test_repeated_inconsistent_decisions_never_become_verdicts(clause_data, verdict_data):
    client = Mock()
    invoke = client.with_structured_output.return_value.invoke
    with pytest.raises(ValidationError) as caught:
        decision(verdict_data, verdict="Partial", missing_elements=[])
    invoke.side_effect = caught.value
    with pytest.raises(ValidationError, match="missing element"):
        rag.audit_requirement(
            "Original requirement", [Clause(**clause_data)], client=client,
        )
    assert invoke.call_count == 2


@pytest.mark.parametrize("label", ["Meets", "Partial", "Gap"])
def test_real_agent_and_structured_parser_use_resolved_evidence(
    clause_data, verdict_data, label, monkeypatch,
):
    """Use the real graph and ChatOllama schema parser; replace only generation."""
    def tool_call(name, arguments, call_id):
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": arguments, "id": call_id, "type": "tool_call"},
        ])

    responses = [tool_call("retrieve_clause", {"query": "Original requirement"}, "search"),
                 tool_call("check_requirement", {"retrieval_id": "retrieval_1"}, "check")]
    if label != "Meets":
        responses.append(tool_call("log_gap", {}, "log"))
    responses.append(AIMessage(content="Finished"))
    responses = iter(responses)
    output = decision(verdict_data, verdict=label,
                      missing_elements=[] if label == "Meets" else ["Maintenance"])
    formats = []

    def generate(client, messages, stop=None, **kwargs):
        if "format" in kwargs:
            formats.append(kwargs["format"])
            assert client.num_predict == 600 and client.reasoning is False
            message = AIMessage(content=output.model_dump_json())
        else:
            assert client.num_predict == 2048 and client.reasoning is False
            message = next(responses)
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(agent.ChatOllama, "_generate", generate)
    store = Mock()
    store.similarity_search.return_value = [Document(page_content="Search", metadata=clause_data)]
    result = agent.run_agent(
        "r1", "Original requirement", store, client=rag.create_audit_client(),
        reasoning=False, agent_num_predict=2048,
    )
    assert result.error is None and result.grounded
    assert result.verdict.evidence == clause_data["requirement"]
    assert result.verdict.verdict == label
    assert bool(result.finding_id) == (label != "Meets")
    assert formats == [AuditDecision.model_json_schema()]
