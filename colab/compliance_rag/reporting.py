"""Write human-readable evaluation reports and machine-readable artifacts."""
import json
from pathlib import Path
import pandas as pd

def write_report(path: Path, metrics: dict, cm: pd.DataFrame, df: pd.DataFrame, cfg: dict):
    g, nc, r = metrics["gap_detection"], metrics["noncompliance_detection_gap_or_partial"], metrics["retrieval"]
    kk = metrics["k"]
    lines = [
        "# October Milestone - Non-Agentic RAG Baseline Results", "",
        f"Embedder: `{cfg['embedder']}` | Judge: `{cfg['judge']}` | top-k: {kk} | n = {metrics['n']}", "",
        "## Headline metrics", "",
        "| Metric | Value |", "|---|---|",
        f"| Retrieval hit@1 | {r['hit_at_1']:.2%} |",
        f"| Retrieval hit@{kk} | {r[f'hit_at_{kk}']:.2%} |",
        f"| 3-way verdict accuracy | {metrics['verdict_accuracy']:.2%} |",
        f"| **Gap precision** | {g['precision']:.2%} |",
        f"| **Gap recall** | {g['recall']:.2%} |",
        f"| Gap F1 | {g['f1']:.2%} |",
        f"| Non-compliance (Gap+Partial) precision / recall | {nc['precision']:.2%} / {nc['recall']:.2%} |",
        f"| Citation groundedness | {metrics['citation_groundedness']:.2%} |",
        f"| Cited the gold clause | {metrics['cited_gold_clause_rate']:.2%} |",
        f"| Schema-valid outputs | {metrics['schema_valid_rate']:.2%} |", "",
        "## Confusion matrix (rows = human label, cols = predicted)", "", cm.to_markdown(), "",
        "## Errors (for the error-analysis section)", "",
        "| req | gold | pred | gold clause | cited | retrieved? |", "|---|---|---|---|---|---|",
    ]
    for x in df[~df.correct].itertuples():
        lines.append(f"| {x.requirement_id} | {x.label} | {x.pred_verdict} | {x.clause_id} | "
                     f"{x.pred_clause or '-'} | {'yes' if x.hit_at_k else 'NO'} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_results(out, records, df, metrics, cm, cfg):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    write_report(out / "report.md", metrics, cm, df, cfg)
    (out / "audit_records.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in records], indent=2), encoding="utf-8")
