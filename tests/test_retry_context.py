"""Retries preserve repaired source text without carrying failed output."""

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
def repaired_clauses():
    documents = create_documents(load_requirements(DATA_PATH))
    return [Clause.model_validate(doc.metadata) for doc in documents
            if doc.metadata["clause"] in {"2.1.5.16", "3.12.1", "3.7.3", "4.5.9"}]


def retry_messages(text, clauses):
    return [
        {"role": "system", "content": rag.SYSTEM_PROMPT},
        {"role": "user", "content": rag.build_user_prompt(text, clauses)},
        {"role": "user", "content": rag.RETRY_FEEDBACK},
    ]


@pytest.mark.parametrize("failure_kind", ["parser", "schema", "citation"])
def test_retry_keeps_complete_source_and_excludes_unbounded_failure(repaired_clauses, failure_kind):
    client = Mock()
    first = repaired_clauses[0]
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
    assert rag.audit_requirement("Maintain requirements.", repaired_clauses, client=client) == valid
    original, retry = [call.args[0] for call in invoke.call_args_list]
    assert retry == retry_messages("Maintain requirements.", repaired_clauses)
    assert retry[:2] == original
    assert "FAILED_OUTPUT_SENTINEL" not in json.dumps(retry)
    payload = json.loads(retry[1]["content"])
    assert [c["requirement"] for c in payload["retrieved_clauses"]] == [c.requirement for c in repaired_clauses]
