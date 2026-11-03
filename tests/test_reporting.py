"""Offline reporting checks using the real evaluator and canonical schemas."""

import json

import pandas as pd
import pytest

from src import reporting
from src.evaluation import evaluate
from src.schemas import AuditRecord, AuditVerdict


@pytest.fixture
def evaluated(verdict_data):
    benchmark = pd.DataFrame([
        {"requirement_id": "002", "requirement_text": "Second", "clause_id": "4.1.2", "label": "Partial"},
        {"requirement_id": "001", "requirement_text": "First", "clause_id": "4.1.2", "label": "Gap"},
    ])
    records = [
        AuditRecord(requirement_id="001", requirement_text="First", retrieved_clause_ids=["4.1.2"],
                    error="Invalid output: café, retry exhausted"),
        AuditRecord(requirement_id="002", requirement_text="Second", retrieved_clause_ids=["other", "4.1.2"],
                    verdict=AuditVerdict.model_validate({
                        **verdict_data, "verdict": "Partial", "missing_elements": ["review, approval", "traceability"],
                        "rationale": "Review is missing.\nApproval must also be specified.",
                    }), grounded=True),
    ]
    rows, metrics, matrix = evaluate(benchmark, records, k=2)
    return records, rows, metrics, matrix


def test_all_artifacts_preserve_evaluation_and_failed_audits(tmp_path, evaluated):
    records, rows, metrics, matrix = evaluated
    original = rows.copy(deep=True)
    out = reporting.save_results(tmp_path / "nested" / "run", *evaluated, model="test-model")
    assert {p.name for p in out.iterdir()} == set(reporting.ARTIFACT_NAMES)
    restored = [AuditRecord.model_validate(value) for value in json.loads((out / "audit_records.json").read_text())]
    assert restored == list(reversed(records))
    predictions = pd.read_csv(out / "predictions.csv", dtype={"requirement_id": str}, keep_default_na=False)
    assert predictions.requirement_id.tolist() == ["002", "001"]
    assert predictions.pred_verdict.tolist() == ["Partial", "INVALID"]
    assert predictions.label.tolist() == ["Partial", "Gap"]
    assert predictions.pred_clause.tolist() == ["4.1.2", ""]
    assert predictions.correct.tolist() == [True, False]
    assert predictions.hit_at_1.tolist() == [False, True]
    assert predictions.hit_at_k.tolist() == [True, True]
    assert json.loads(predictions.iloc[0].retrieved_clause_ids) == ["other", "4.1.2"]
    assert json.loads(predictions.iloc[0].missing_elements) == ["review, approval", "traceability"]
    assert predictions.iloc[0].evidence == restored[0].verdict.evidence
    assert predictions.iloc[0].rationale == restored[0].verdict.rationale
    assert predictions.iloc[1].audit_error == records[0].error
    assert predictions.iloc[1].evidence == ""
    assert json.loads((out / "metrics.json").read_text()) == metrics
    saved_matrix = pd.read_csv(out / "confusion_matrix.csv", index_col="human_verdict")
    assert saved_matrix.columns.tolist() == ["Meets", "Partial", "Gap", "INVALID"]
    assert saved_matrix.index.tolist() == ["Meets", "Partial", "Gap"]
    assert saved_matrix.loc["Gap", "INVALID"] == 1
    assert saved_matrix.values.tolist() == matrix.values.tolist()
    pd.testing.assert_frame_equal(rows, original)
    report = (out / "report.md").read_text()
    for expected in (
        "Evaluated requirements: 2", "Model: test-model", "Retrieval top-K: 2", "Failed audits: 1",
        "Verdict accuracy | 50.00%", "Macro F1 | 0.3333", "Retrieval Hit@1 | 50.00%",
        "Retrieval Hit@2 | 100.00%", "Schema-valid rate", "50.00%",
        "Citation membership / accepted verdicts (n=1) | 100.00%",
        "Citation membership / all records (n=2) | 50.00%", "Cited gold clause",
        "Partial | 1.0000 | 1.0000 | 1.0000", "Gap detection | 0.0000 | 0.0000 | 0.0000",
        "Noncompliance detection (Gap + Partial) | 1.0000 | 0.5000 | 0.6667",
        "rows = human verdicts; columns = model verdicts", "INVALID", "evidence quality", "overall compliance",
    ):
        assert expected in report


def test_all_failed_report_has_undefined_accepted_membership(tmp_path, evaluated):
    records, rows, _, _ = evaluated
    failed = [r.model_copy(update={"verdict": None, "error": "failed", "grounded": False}) for r in records]
    rows, metrics, matrix = evaluate(rows, failed, k=2)
    reporting.save_results(tmp_path, failed, rows, metrics, matrix, model="offline")
    report = (tmp_path / "report.md").read_text()
    assert "accepted verdicts (n=0) | N/A" in report
    assert "all records (n=2) | 0.00%" in report
    assert "Failed audits: 2" in report


@pytest.mark.parametrize("artifact", reporting.ARTIFACT_NAMES)
def test_existing_artifacts_are_never_overwritten(tmp_path, evaluated, artifact):
    existing = tmp_path / artifact
    existing.write_text("input to preserve")
    with pytest.raises(ValueError, match="overwrite"):
        reporting.save_results(tmp_path, *evaluated, model="offline")
    assert existing.read_text() == "input to preserve"
    assert list(tmp_path.iterdir()) == [existing]


@pytest.mark.parametrize("relative", ["results/october", "results/october/new-run", "data", "data/new-run"])
def test_historical_and_data_directories_are_protected(tmp_path, monkeypatch, evaluated, relative):
    monkeypatch.setattr(reporting, "PROJECT_ROOT", tmp_path)
    out = tmp_path / relative
    with pytest.raises(ValueError, match="protected"):
        reporting.save_results(out, *evaluated, model="offline")
    assert not out.exists()


def test_symlinks_cannot_bypass_protection(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "PROJECT_ROOT", tmp_path)
    history = tmp_path / "results/october"
    history.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(history, target_is_directory=True)
    with pytest.raises(ValueError, match="protected"):
        reporting.validate_output_directory(alias / "new")
    dangling = tmp_path / "run"
    dangling.mkdir()
    (dangling / "report.md").symlink_to(tmp_path / "input.md")
    with pytest.raises(ValueError, match="overwrite"):
        reporting.validate_output_directory(dangling)
    assert not (tmp_path / "input.md").exists()
