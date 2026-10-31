"""Benchmark metrics independent of model providers."""
from typing import List
import pandas as pd
from .schemas import AuditRecord

def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4), tp=tp, fp=fp, fn=fn)


def evaluate(bench: pd.DataFrame, records: List[AuditRecord], k: int):
    df = bench.copy().reset_index(drop=True)
    df["pred_verdict"] = [r.verdict.verdict.value if r.verdict else "INVALID" for r in records]
    df["pred_clause"] = [r.verdict.clause_id if r.verdict else "" for r in records]
    df["pred_rationale"] = [r.verdict.rationale if r.verdict else (r.error or "") for r in records]
    df["pred_missing"] = [" | ".join(r.verdict.missing_elements) if r.verdict else "" for r in records]
    df["retrieved"] = [",".join(r.retrieved_clause_ids) for r in records]
    df["grounded"] = [r.grounded for r in records]
    df["hit_at_k"] = [g in r.retrieved_clause_ids for g, r in zip(df.clause_id, records)]
    df["hit_at_1"] = [r.retrieved_clause_ids[0] == g for g, r in zip(df.clause_id, records)]
    df["clause_match"] = df.clause_id == df.pred_clause
    df["correct"] = df.label == df.pred_verdict

    gold_gap, pred_gap = df.label == "Gap", df.pred_verdict == "Gap"
    gold_nc, pred_nc = df.label != "Meets", ~df.pred_verdict.isin(["Meets"]) & (df.pred_verdict != "INVALID")

    labels = ["Meets", "Partial", "Gap"]
    cm = pd.crosstab(df.label, df.pred_verdict).reindex(index=labels, columns=labels + ["INVALID"], fill_value=0)
    per_class = {c: prf(int(((df.label == c) & (df.pred_verdict == c)).sum()),
                        int(((df.label != c) & (df.pred_verdict == c)).sum()),
                        int(((df.label == c) & (df.pred_verdict != c)).sum())) for c in labels}

    n = len(df)
    metrics = dict(
        n=n, k=k,
        retrieval=dict(hit_at_1=round(df.hit_at_1.mean(), 4), **{f"hit_at_{k}": round(df.hit_at_k.mean(), 4)}),
        verdict_accuracy=round(df.correct.mean(), 4),
        gap_detection=prf(int((gold_gap & pred_gap).sum()), int((~gold_gap & pred_gap).sum()),
                          int((gold_gap & ~pred_gap).sum())),
        noncompliance_detection_gap_or_partial=prf(int((gold_nc & pred_nc).sum()), int((~gold_nc & pred_nc).sum()),
                                                   int((gold_nc & ~pred_nc).sum())),
        per_class=per_class,
        citation_groundedness=round(df.grounded.mean(), 4),
        cited_gold_clause_rate=round(df.clause_match.mean(), 4),
        schema_valid_rate=round((df.pred_verdict != "INVALID").mean(), 4),
        confusion_matrix=cm.to_dict(),
    )
    return df, metrics, cm
