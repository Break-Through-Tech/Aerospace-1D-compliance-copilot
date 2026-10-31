"""Evaluate completed canonical audit records against a labeled benchmark."""

from collections.abc import Sequence

import pandas as pd

from src.schemas import AuditRecord


LABELS = ("Meets", "Partial", "Gap")
PREDICTION_LABELS = LABELS + ("INVALID",)
BENCHMARK_COLUMNS = {"requirement_id", "requirement_text", "clause_id", "label"}


def prf(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    """Calculate precision, recall, and F1, using zero for undefined scores."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def evaluate(
    benchmark: pd.DataFrame,
    records: Sequence[AuditRecord],
    k: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Compare audit records to benchmark rows by requirement ID.

    The confusion matrix includes an ``INVALID`` prediction column. Records with
    no valid verdict lower accuracy and count as false negatives for their actual
    class. Citation groundedness is clause membership among valid verdicts; it
    does not establish that the quoted evidence is exact or sufficient.
    """
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

    # Align once by ID, then build each scored column in benchmark order.
    df = benchmark.copy().reset_index(drop=True)
    df["requirement_id"] = df["requirement_id"].astype(str)
    ordered_records = [record_by_id[req_id] for req_id in df["requirement_id"]]
    for record, text in zip(ordered_records, df["requirement_text"]):
        if record.requirement_text != text:
            raise ValueError(f"Requirement text mismatch: {record.requirement_id}")

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
        "cited_gold_clause_rate": round(float((df["pred_clause"] == gold_clauses).mean()), 4),
        "schema_valid_rate": round(valid_count / n, 4),
        "confusion_matrix": matrix.to_dict(orient="index"),
    }
    return df, metrics, matrix
