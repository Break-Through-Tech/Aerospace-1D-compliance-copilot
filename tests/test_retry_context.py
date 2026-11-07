"""Offline context safety using repaired clauses, without model generation."""

import json
from unittest.mock import Mock

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src import rag
from src.config import DATA_PATH
from src.preprocessing import create_documents, load_requirements
from src.schemas import AuditVerdict, Clause


@pytest.fixture(scope="module")
def largest_clauses():
    tokenizer = rag._audit_tokenizer()
    clauses = [Clause.model_validate(doc.metadata) for doc in create_documents(load_requirements(DATA_PATH))]
    return sorted(clauses, key=lambda c: len(tokenizer.encode(
        rag.build_user_prompt("x", [c]), add_special_tokens=False,
    ).ids), reverse=True)[:5]


def retry_messages(text, clauses):
    return [
        {"role": "system", "content": rag.SYSTEM_PROMPT},
        {"role": "user", "content": rag.build_user_prompt(text, clauses)},
        {"role": "user", "content": rag.RETRY_FEEDBACK},
    ]


def test_largest_repaired_clauses_fit_with_schema_framing_output_and_retry(largest_clauses):
    client = rag.create_audit_client()
    assert (client.num_ctx, client.num_predict) == (4096, 600)
    assert [c.clause for c in largest_clauses] == ["2.1.5.16", "2.1.8.2", "3.9.2", "3.10.2", "3.1.14"]
    messages = retry_messages("The project shall maintain software requirements.", largest_clauses)
    required = rag.check_audit_context(messages, client)
    assert 3800 < required <= 4096
    tokenizer = rag._audit_tokenizer()
    assert tokenizer.truncation is None
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    framed = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    ) + "<|im_start|>assistant\n<think>\n</think>\n\n"
    assert count(framed) + count(json.dumps(AuditVerdict.model_json_schema(), ensure_ascii=False)) + 600 <= required


@pytest.mark.parametrize("failure_kind", ["parser", "schema", "citation"])
def test_retry_keeps_complete_source_and_excludes_unbounded_failure(largest_clauses, failure_kind):
    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    first = largest_clauses[0]
    valid = AuditVerdict(
        clause_id=first.clause, evidence=first.requirement[-80:],
        missing_elements=["Not assessed"], rationale="Synthetic citation test.", verdict="Partial",
    )
    unbounded = "FAILED_OUTPUT_SENTINEL " * 10000
    if failure_kind == "parser":
        failure = OutputParserException(unbounded, llm_output=unbounded)
    elif failure_kind == "schema":
        with pytest.raises(ValidationError) as caught:
            AuditVerdict.model_validate({"verdict": unbounded})
        failure = caught.value
    else:
        failure = valid.model_copy(update={"evidence": unbounded, "rationale": unbounded})
    invoke = client.with_structured_output.return_value.invoke
    invoke.side_effect = [failure, valid]
    assert rag.audit_requirement("Maintain requirements.", largest_clauses, client=client) == valid
    original, retry = [call.args[0] for call in invoke.call_args_list]
    assert retry == retry_messages("Maintain requirements.", largest_clauses)
    assert retry[:2] == original
    assert "FAILED_OUTPUT_SENTINEL" not in json.dumps(retry)
    payload = json.loads(retry[1]["content"])
    assert [c["requirement"] for c in payload["retrieved_clauses"]] == [c.requirement for c in largest_clauses]


@pytest.mark.parametrize("text", ["oversized requirement " * 5000, "🚀" * 5000])
def test_oversized_requirement_fails_before_inference(largest_clauses, text):
    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    with pytest.raises(rag.ContextBudgetError, match="No source text was truncated"):
        rag.audit_requirement(text, largest_clauses, client=client)
    client.with_structured_output.assert_not_called()


def test_guard_reserves_retry_and_rejects_oversized_source(largest_clauses):
    client = Mock(model="qwen3:4b", num_ctx=4096, num_predict=600)
    messages = retry_messages("Requirement", largest_clauses)
    client.num_ctx = rag.check_audit_context(messages, client) - 1
    assert rag.check_audit_context(messages[:2], client) <= client.num_ctx
    with pytest.raises(rag.ContextBudgetError):
        rag.audit_requirement("Requirement", largest_clauses, client=client)
    client.num_ctx = 4096
    oversized = largest_clauses[0].model_copy(update={"requirement": "Complete source " * 5000})
    with pytest.raises(rag.ContextBudgetError):
        rag.audit_requirement("Requirement", [oversized], client=client)
    client.with_structured_output.assert_not_called()


@pytest.mark.parametrize("settings", [
    {"model": "another-model"}, {"num_ctx": None}, {"num_ctx": True},
    {"num_ctx": 1024}, {"num_predict": None}, {"num_predict": -1},
    {"num_predict": 4000},
])
def test_unknown_or_insufficient_client_limits_fail_explicitly(largest_clauses, settings):
    client = Mock(**{"model": "qwen3:4b", "num_ctx": 4096, "num_predict": 600, **settings})
    with pytest.raises(rag.ContextBudgetError):
        rag.audit_requirement("Requirement", largest_clauses, client=client)
    client.with_structured_output.assert_not_called()
