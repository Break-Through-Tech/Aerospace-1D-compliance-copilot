"""Save canonical evaluation outputs without recalculating their metrics."""

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.evaluation import _prepare_evaluation
from src.schemas import AuditRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = (
    "audit_records.json", "predictions.csv", "metrics.json",
    "confusion_matrix.csv", "report.md",
)


def validate_output_directory(out: str | Path, *, input_paths: Sequence[str | Path] = ()) -> Path:
    """Reject protected destinations and collisions before any report is written.

    Existing directories are allowed, but existing artifacts are never replaced.
    Resolve symlinks so aliases cannot bypass historical/input protection.
    """
    path = Path(out).resolve()
    for protected in (PROJECT_ROOT / "results" / "october", PROJECT_ROOT / "data"):
        protected = protected.resolve()
        if path == protected or protected in path.parents:
            raise ValueError(f"Reporting destination is protected: {path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Reporting destination must be a directory: {path}")
    inputs = [Path(item).resolve() for item in input_paths]
    for name in ARTIFACT_NAMES:
        target = path / name
        if target.exists() or target.is_symlink():
            raise ValueError(f"Refusing to overwrite existing artifact: {target}")
        if any(target == item or target in item.parents for item in inputs):
            raise ValueError(f"Reporting artifact conflicts with an input: {target}")
    if any(path == item or item in path.parents for item in inputs):
        raise ValueError(f"Reporting destination conflicts with an input: {path}")
    return path


def _markdown_table(headers, rows):
    """Render simple tables without the optional pandas tabulate dependency."""
    def line(cells):
        return "| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def _report(metrics: dict, matrix: pd.DataFrame, model: str, failed: int) -> str:
    k = metrics["k"]
    membership = metrics["citation_groundedness"]
    membership_text = "N/A" if membership is None else f"{membership:.2%}"
    rows = [
        ("Verdict accuracy", f"{metrics['verdict_accuracy']:.2%}"),
        ("Macro F1", f"{metrics['macro_f1']:.4f}"),
        ("Retrieval Hit@1", f"{metrics['retrieval']['hit_at_1']:.2%}"),
        (f"Retrieval Hit@{k}", f"{metrics['retrieval'][f'hit_at_{k}']:.2%}"),
        ("Schema-valid rate (accepted verdicts / all requirements)", f"{metrics['schema_valid_rate']:.2%}"),
        (f"Citation membership / accepted verdicts (n={metrics['citation_groundedness_n']})", membership_text),
        (f"Citation membership / all records (n={metrics['n']})", f"{metrics['citation_groundedness_all_records']:.2%}"),
        ("Cited gold clause / all records", f"{metrics['cited_gold_clause_rate']:.2%}"),
    ]
    scores = list(metrics["per_class"].items()) + [
        ("Gap detection", metrics["gap_detection"]),
        ("Noncompliance detection (Gap + Partial)", metrics["noncompliance_detection_gap_or_partial"]),
    ]
    return "\n\n".join([
        "# Canonical RAG evaluation",
        f"Evaluated requirements: {metrics['n']} | Model: {model} | Retrieval top-K: {k}",
        f"Failed audits: {failed}. Failed audits are INVALID predictions in the confusion matrix.",
        "## Metrics",
        _markdown_table(["Metric", "Value"], rows),
        "## Per-class and detection scores",
        _markdown_table(["Class / detection", "Precision", "Recall", "F1"], [
            [label, *(f"{score[key]:.4f}" for key in ("precision", "recall", "f1"))]
            for label, score in scores
        ]),
        "## Confusion matrix (rows = human verdicts; columns = model verdicts)",
        _markdown_table(["Human / model", *matrix.columns], [
            [label, *values] for label, values in matrix.iterrows()
        ]),
        "Citation groundedness measures only whether the cited clause ID belongs to the retrieved set. "
        "It does not measure evidence quality or establish overall compliance. The accepted-verdict "
        "denominator excludes failed audits (N/A if none are accepted); the all-record denominator "
        "includes failures as ungrounded. Schema-valid rate is the existing accepted-verdict rate, "
        "not a separate measurement of JSON/schema validity.",
    ]) + "\n"


def save_results(
    out: str | Path,
    records: Sequence[AuditRecord],
    rows: pd.DataFrame,
    metrics: dict,
    matrix: pd.DataFrame,
    *,
    model: str,
) -> Path:
    """Save the outputs of evaluate() and complete records in benchmark order.

    Reuse the evaluator's alignment/validation helper; never align by list position
    or independently implement benchmark matching. Existing score columns stay intact.
    """
    path = validate_output_directory(out)
    predictions, ordered = _prepare_evaluation(rows, records, metrics["k"])
    predictions["evidence"] = [r.verdict.evidence if r.verdict else "" for r in ordered]
    predictions["missing_elements"] = [
        json.dumps(r.verdict.missing_elements if r.verdict else [], ensure_ascii=False) for r in ordered
    ]
    predictions["rationale"] = [r.verdict.rationale if r.verdict else "" for r in ordered]
    predictions["audit_error"] = [r.error for r in ordered]
    # JSON lists preserve clause boundaries even when identifiers contain commas.
    predictions["retrieved_clause_ids"] = [json.dumps(r.retrieved_clause_ids) for r in ordered]
    audit_json = json.dumps([r.model_dump(mode="json") for r in ordered], indent=2, ensure_ascii=False)
    metrics_json = json.dumps(metrics, indent=2, ensure_ascii=False)
    report = _report(metrics, matrix, model, sum(r.verdict is None for r in ordered))
    path.mkdir(parents=True, exist_ok=True)
    # Exclusive creation provides a second collision check at write time.
    with (path / "audit_records.json").open("x", encoding="utf-8") as handle:
        handle.write(audit_json + "\n")
    predictions.to_csv(path / "predictions.csv", index=False, mode="x", encoding="utf-8")
    with (path / "metrics.json").open("x", encoding="utf-8") as handle:
        handle.write(metrics_json + "\n")
    matrix.to_csv(path / "confusion_matrix.csv", mode="x", encoding="utf-8")
    with (path / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(report)
    return path
