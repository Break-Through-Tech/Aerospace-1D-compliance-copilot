"""Application validation contracts for the current RAG schemas."""

import pytest
from pydantic import ValidationError

from src.schemas import AuditVerdict, Clause


def test_clause_ignores_unrelated_document_metadata(clause_data):
    clause = Clause.model_validate({**clause_data, "chunk_index": 3})
    assert clause.model_dump() == clause_data


def test_clause_rejects_missing_or_invalid_required_data(clause_data):
    missing_source = clause_data.copy()
    del missing_source["source"]
    with pytest.raises(ValidationError):
        Clause.model_validate(missing_source)

    with pytest.raises(ValidationError):
        Clause.model_validate({**clause_data, "requirement": " \t\n"})

    with pytest.raises(ValidationError):
        Clause.model_validate({**clause_data, "clause": 412})


@pytest.mark.parametrize("verdict,missing", [
    ("Meets", []), ("Partial", ["Maintenance"]), ("Gap", ["Documentation"]),
])
def test_verdict_accepts_consistent_missing_elements(verdict_data, verdict, missing):
    result = AuditVerdict.model_validate({
        **verdict_data, "verdict": verdict, "missing_elements": missing,
    })
    assert result.verdict == verdict
    assert result.missing_elements == missing


@pytest.mark.parametrize("verdict,missing", [
    ("Meets", ["Maintenance"]), ("Partial", []), ("Gap", []),
])
def test_verdict_rejects_inconsistent_missing_elements(verdict_data, verdict, missing):
    with pytest.raises(ValidationError, match="missing element"):
        AuditVerdict.model_validate({
            **verdict_data, "verdict": verdict, "missing_elements": missing,
        })


@pytest.mark.parametrize("overrides", [
    {"extra_field": True},
    {"verdict": "Unknown"},
    {"evidence": " \n"},
    {"rationale": "Too short"},
    {"verdict": "Partial", "missing_elements": [" \t"]},
])
def test_verdict_rejects_unexpected_or_invalid_fields(verdict_data, overrides):
    with pytest.raises(ValidationError):
        AuditVerdict.model_validate({**verdict_data, **overrides})


def test_verdict_schema_preserves_reasoning_before_decision():
    schema = AuditVerdict.model_json_schema()
    fields = ["clause_id", "evidence", "missing_elements", "rationale", "verdict"]
    assert list(schema["properties"]) == fields
    assert schema["required"] == fields
