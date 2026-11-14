"""Evaluate completed canonical audit records against a labeled benchmark."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.schemas import AuditRecord


LABELS = ("Meets", "Partial", "Gap")
PREDICTION_LABELS = LABELS + ("INVALID",)
BENCHMARK_COLUMNS = {"requirement_id", "requirement_text", "clause_id", "label"}


def load_benchmark(path: str | Path) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Return labeled rows and reusable (ID, text) pairs in the same row order.

    Preserve IDs and empty fields as strings. Requirements exclude gold clauses
    and labels so they can be passed directly to run_audits().
    """
    benchmark = pd.read_csv(path, dtype=str, keep_default_na=False)
    requirements = list(
        benchmark[["requirement_id", "requirement_text"]].itertuples(index=False, name=None)
    )
    return benchmark, requirements


def prf(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    """Calculate precision, recall, and F1, using zero for undefined scores."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def _prepare_evaluation(
    benchmark: pd.DataFrame,
    records: Sequence[AuditRecord],
    k: int,
) -> tuple[pd.DataFrame, list[AuditRecord]]:
    """Validate inputs and align records to a copy of the benchmark in row order."""
    if not isinstance(benchmark, pd.DataFrame):
        raise TypeError("benchmark must be a pandas DataFrame")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if benchmark.empty:
        raise ValueError("Benchmark must contain at least one row")
    missing_columns = BENCHMARK_COLUMNS - set(benchmark.columns)
    if missing_columns:
        raise ValueError(f"Benchmark is missing columns: {sorted(missing_columns)}")
    if any(not isinstance(record, AuditRecord) for record in records):
        raise TypeError("records must contain canonical AuditRecord objects")

    benchmark_by_id = {}
    for row in benchmark.to_dict(orient="records"):
        req_id = str(row["requirement_id"])
        if not req_id.strip() or req_id in benchmark_by_id:
            raise ValueError("Benchmark requirement IDs must be nonempty and unique")
        if row["label"] not in LABELS:
            raise ValueError(f"Unknown benchmark label for {req_id}: {row['label']}")
        benchmark_by_id[req_id] = row

    record_by_id = {}
    for record in records:
        req_id = record.requirement_id
        if not req_id.strip() or req_id in record_by_id:
            raise ValueError("Audit record requirement IDs must be nonempty and unique")
        record_by_id[req_id] = record
    if set(record_by_id) != set(benchmark_by_id):
        raise ValueError("Benchmark and audit record requirement IDs must match exactly")

    # Preserve benchmark order while matching each prediction by ID.
    df = benchmark.copy().reset_index(drop=True)
    df["requirement_id"] = df["requirement_id"].astype(str)
    ordered_records = [record_by_id[req_id] for req_id in df["requirement_id"]]
    for record, text in zip(ordered_records, df["requirement_text"]):
        if record.requirement_text != text:
            raise ValueError(f"Requirement text mismatch: {record.requirement_id}")

    return df, ordered_records


def evaluate(
    benchmark: pd.DataFrame,
    records: Sequence[AuditRecord],
    k: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Align by requirement ID, score predictions, and return rows/metrics/matrix.

    Accuracy and class scores include all benchmark rows. Missing verdicts are
    INVALID predictions and false negatives for their actual class. Retrieval
    Hit@K checks only the first K clauses and handles empty retrieval.

    Citation groundedness checks clause-ID membership, not evidence quality:
    ``citation_groundedness`` uses accepted verdicts as its denominator (given by
    ``citation_groundedness_n``), or None when there are none;
    ``citation_groundedness_all_records`` uses all rows, matching October when
    saved grounding flags reflect membership. Failed audits count as ungrounded.

    ``schema_valid_rate`` retains October's name and formula: the fraction of
    records with an accepted verdict. It does not measure JSON/schema validity
    separately from citation/evidence validation performed during auditing.

    Existing aliases, four-decimal rounding, and matrix orientation are retained:
    ``accuracy`` aliases ``verdict_accuracy``, ``gap_detection`` duplicates the Gap
    class scores, and macro F1 averages the rounded per-class F1 scores. Matrix
    rows (and outer JSON keys) are human labels; columns are predicted labels.
    """
    df, ordered_records = _prepare_evaluation(benchmark, records, k)

    df["pred_verdict"] = [r.verdict.verdict if r.verdict else "INVALID" for r in ordered_records]
    df["pred_clause"] = [r.verdict.clause_id if r.verdict else "" for r in ordered_records]
    retrieved_ids = [r.retrieved_clause_ids for r in ordered_records]
    gold_clauses = df["clause_id"].astype(str)
    df["retrieved"] = [",".join(ids) for ids in retrieved_ids]
    df["hit_at_1"] = [bool(ids and ids[0] == gold) for ids, gold in zip(retrieved_ids, gold_clauses)]
    df["hit_at_k"] = [gold in ids[:k] for ids, gold in zip(retrieved_ids, gold_clauses)]
    df["citation_member"] = [
        bool(r.verdict and r.verdict.clause_id in r.retrieved_clause_ids) for r in ordered_records
    ]
    df["correct"] = df["pred_verdict"] == df["label"]

    matrix = pd.crosstab(df["label"], df["pred_verdict"])
    matrix = matrix.reindex(index=LABELS, columns=PREDICTION_LABELS, fill_value=0)
    matrix.index.name = "human_verdict"
    matrix.columns.name = "model_verdict"

    per_class = {}
    for label in LABELS:
        tp = int(matrix.at[label, label])
        fp = int(matrix.loc[:, label].sum()) - tp
        fn = int(matrix.loc[label, :].sum()) - tp
        per_class[label] = prf(tp, fp, fn)

    gold_noncompliance = df["label"].isin(("Partial", "Gap"))
    pred_noncompliance = df["pred_verdict"].isin(("Partial", "Gap"))
    noncompliance_detection = prf(
        int((gold_noncompliance & pred_noncompliance).sum()),
        int((~gold_noncompliance & pred_noncompliance).sum()),
        int((gold_noncompliance & ~pred_noncompliance).sum()),
    )

    n = len(df)
    valid = df["pred_verdict"] != "INVALID"
    valid_count = int(valid.sum())
    accuracy = round(float(df["correct"].mean()), 4)
    metrics = {
        "n": n,
        "k": k,
        "retrieval": {
            "hit_at_1": round(float(df["hit_at_1"].mean()), 4),
            f"hit_at_{k}": round(float(df["hit_at_k"].mean()), 4),
        },
        "verdict_accuracy": accuracy,
        "accuracy": accuracy,
        "gap_detection": per_class["Gap"].copy(),
        "noncompliance_detection_gap_or_partial": noncompliance_detection,
        "per_class": per_class,
        "macro_f1": round(sum(score["f1"] for score in per_class.values()) / len(LABELS), 4),
        "citation_groundedness": round(float(df.loc[valid, "citation_member"].mean()), 4)
        if valid_count else None,
        "citation_groundedness_n": valid_count,
        "citation_groundedness_all_records": round(float(df["citation_member"].mean()), 4),
        "cited_gold_clause_rate": round(float((df["pred_clause"] == gold_clauses).mean()), 4),
        "schema_valid_rate": round(valid_count / n, 4),
        "confusion_matrix": matrix.to_dict(orient="index"),
    }
    return df, metrics, matrix
