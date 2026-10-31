"""Focused checks for evaluation of canonical RAG audit records."""

import pandas as pd
import pytest

from src.evaluation import evaluate
from src.schemas import AuditRecord, AuditVerdict


def make_benchmark():
    return pd.DataFrame([
        {"requirement_id": "r1", "requirement_text": "Req one", "clause_id": "1.1", "label": "Meets"},
        {"requirement_id": "r2", "requirement_text": "Req two", "clause_id": "2.1", "label": "Partial"},
        {"requirement_id": "r3", "requirement_text": "Req three", "clause_id": "3.1", "label": "Gap"},
    ])


def verdict(label, clause_id):
    return AuditVerdict(
        clause_id=clause_id,
        evidence="required evidence",
        missing_elements=[] if label == "Meets" else ["an element"],
        rationale="A sufficiently detailed rationale.",
        verdict=label,
    )


def record(req_id, text, label=None, cited_clause=None, retrieved=None):
    return AuditRecord(
        requirement_id=req_id,
        requirement_text=text,
        retrieved_clause_ids=retrieved or [],
        verdict=verdict(label, cited_clause) if label else None,
        grounded=bool(label and cited_clause in (retrieved or [])),
        error="invalid model output" if label is None else None,
    )


def test_known_predictions_and_id_matching():
    benchmark = make_benchmark()
    records = [
        record("r3", "Req three", "Gap", "3.1", ["3.1", "4.1"]),
        record("r1", "Req one", "Meets", "1.1", ["9.1", "1.1"]),
        record("r2", "Req two", "Meets", "2.1", ["8.1", "2.1"]),
    ]
    _, metrics, _ = evaluate(benchmark, records, k=2)
    assert metrics["verdict_accuracy"] == pytest.approx(0.6667)
    assert metrics["per_class"]["Partial"]["fn"] == 1
    assert metrics["per_class"]["Partial"]["f1"] == 0
    assert metrics["macro_f1"] == pytest.approx(0.5556)
    assert metrics["confusion_matrix"]["Partial"]["Meets"] == 1
    assert metrics["gap_detection"]["f1"] == 1
    assert metrics["noncompliance_detection_gap_or_partial"]["fn"] == 1


def test_missing_verdict_is_invalid_and_a_false_negative():
    records = [
        record("r1", "Req one", "Meets", "1.1", ["1.1"]),
        record("r2", "Req two", retrieved=["2.1"]),
        record("r3", "Req three", retrieved=["3.1"]),
    ]
    _, metrics, _ = evaluate(make_benchmark(), records, k=1)
    assert metrics["confusion_matrix"]["Partial"]["INVALID"] == 1
    assert metrics["confusion_matrix"]["Gap"]["INVALID"] == 1
    assert metrics["per_class"]["Partial"]["fn"] == 1
    assert metrics["per_class"]["Gap"]["fn"] == 1
    assert metrics["schema_valid_rate"] == pytest.approx(0.3333)


def test_retrieval_and_citation_metrics_have_explicit_denominators():
    records = [
        record("r1", "Req one", "Meets", "1.1", ["x", "1.1"]),
        record("r2", "Req two", "Partial", "wrong-citation", ["2.1"]),
        record("r3", "Req three", retrieved=[]),
    ]
    _, metrics, _ = evaluate(make_benchmark(), records, k=2)
    assert metrics["retrieval"] == {"hit_at_1": 0.3333, "hit_at_2": 0.6667}
    # Membership is measured for the two valid verdicts; missing verdicts are excluded.
    assert metrics["citation_groundedness"] == 0.5
    assert metrics["citation_groundedness_n"] == 2
    # Citing the benchmark's gold clause is measured across all three records.
    assert metrics["cited_gold_clause_rate"] == pytest.approx(0.3333)


def test_duplicate_mismatched_ids_and_text_are_rejected():
    benchmark = make_benchmark()
    records = [record("r1", "Req one"), record("r2", "Req two"), record("r3", "Req three")]
    with pytest.raises(ValueError, match="IDs must match"):
        evaluate(benchmark, records[:2], k=3)
    with pytest.raises(ValueError, match="unique"):
        evaluate(benchmark, records + [records[0]], k=3)
    with pytest.raises(ValueError, match="text mismatch"):
        evaluate(benchmark, [records[0], records[1], record("r3", "different")], k=3)
