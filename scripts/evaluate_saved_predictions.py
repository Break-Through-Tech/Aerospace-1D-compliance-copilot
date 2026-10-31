"""Evaluate saved verdicts without loading models. Run from the repository root."""
import argparse
import csv
import json
from pathlib import Path

LABELS = ("Meets", "Partial", "Gap")
PRED_LABELS = LABELS + ("INVALID",)


def read_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing required columns in {path}: {required}")
        rows = list(reader)
    indexed = {row["requirement_id"]: row for row in rows}
    if not rows or len(indexed) != len(rows) or "" in indexed:
        raise ValueError(f"Empty data, duplicate IDs, or blank IDs in {path}")
    return indexed


def evaluate(benchmark, predictions):
    if benchmark.keys() != predictions.keys():
        raise ValueError("Benchmark and prediction IDs must match exactly")
    matrix = {label: dict.fromkeys(PRED_LABELS, 0) for label in LABELS}
    errors = []
    hit1 = hit5 = 0
    for req_id, gold in benchmark.items():
        pred = predictions[req_id]
        if gold["requirement_text"] != pred["requirement_text"]:
            raise ValueError(f"Requirement text mismatch: {req_id}")
        actual, predicted = gold["label"], pred["pred_verdict"]
        if actual not in LABELS or predicted not in PRED_LABELS:
            raise ValueError(f"Unknown verdict for {req_id}: {actual}, {predicted}")
        retrieved = [c.strip() for c in pred["retrieved"].split(",") if c.strip()]
        if len(retrieved) != 5 or len(set(retrieved)) != 5:
            raise ValueError(f"Expected five distinct retrieved clauses: {req_id}")
        found = gold["clause_id"] in retrieved
        hit1 += gold["clause_id"] == retrieved[0]
        hit5 += found
        matrix[actual][predicted] += 1
        if actual != predicted:
            errors.append(dict(
                requirement_id=req_id, requirement_text=gold["requirement_text"],
                human_verdict=actual, model_verdict=predicted,
                gold_clause=gold["clause_id"], model_clause=pred["pred_clause"],
                retrieved=pred["retrieved"], retrieval_success=found,
                clause_match=gold["clause_id"] == pred["pred_clause"],
                human_rationale=gold["rationale"], model_rationale=pred["pred_rationale"],
                error_category=("invalid_output" if predicted == "INVALID" else
                                "retrieval_miss" if not found else "verdict_error_with_gold_retrieved")))
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = support - tp  # INVALID outputs are false negatives for their human class.
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[label] = dict(tp=tp, fp=fp, fn=fn, support=support,
                                precision=precision, recall=recall, f1=f1)
    n = len(benchmark)
    metrics = dict(n=n, confusion_matrix=matrix, per_class=per_class,
                   macro_f1=sum(v["f1"] for v in per_class.values()) / len(LABELS),
                   accuracy=sum(matrix[c][c] for c in LABELS) / n,
                   invalid_count=sum(matrix[c]["INVALID"] for c in LABELS),
                   retrieval_hit_at_1=hit1 / n, retrieval_hit_at_5=hit5 / n,
                   error_count=len(errors),
                   errors_with_gold_retrieved=sum(e["retrieval_success"] for e in errors))
    return metrics, errors


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/npr7150_ground_truth_benchmark.csv")
    parser.add_argument("--predictions", default="results/october/predictions.csv")
    parser.add_argument("--out", default="results/issue13_saved_evaluation")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    published = Path(__file__).resolve().parents[1] / "results" / "october"
    if out == published or published in out.parents:
        parser.error("Outputs must be outside published results/october")
    inputs = [Path(args.benchmark).resolve(), Path(args.predictions).resolve()]
    output_names = ("metrics.json", "confusion_matrix.csv", "per_class_metrics.csv",
                    "error_analysis.csv", "report.md")
    if any(out / name in inputs for name in output_names):
        parser.error("An output would overwrite an input")
    benchmark = read_rows(args.benchmark, {"requirement_id", "requirement_text", "clause_id", "label", "rationale"})
    predictions = read_rows(args.predictions, {"requirement_id", "requirement_text", "pred_verdict",
                                             "pred_clause", "retrieved", "pred_rationale"})
    metrics, errors = evaluate(benchmark, predictions)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_csv(out / "confusion_matrix.csv", ["human_verdict", *PRED_LABELS],
              [dict(human_verdict=c, **metrics["confusion_matrix"][c]) for c in LABELS])
    write_csv(out / "per_class_metrics.csv", ["class", "tp", "fp", "fn", "support", "precision", "recall", "f1"],
              [dict({"class": c}, **metrics["per_class"][c]) for c in LABELS])
    error_fields = ["requirement_id", "requirement_text", "human_verdict", "model_verdict",
                    "gold_clause", "model_clause", "retrieved", "retrieval_success", "clause_match",
                    "human_rationale", "model_rationale", "error_category"]
    write_csv(out / "error_analysis.csv", error_fields, errors)
    lines = ["# Issue #13: Saved October prediction evaluation", "",
             f"Inputs: `{args.benchmark}` and `{args.predictions}`. No model was rerun.", "",
             f"Evaluated {metrics['n']} examples, matched by requirement ID and checked for identical text.",
             "Published October configuration: nomic-embed-text embeddings, qwen3:4b judge, top-k 5.", "",
             "## Confusion matrix", "", "Rows are human verdicts; columns are model verdicts.", "",
             "| Human / Model | Meets | Partial | Gap | INVALID |", "|---|---:|---:|---:|---:|"]
    for c in LABELS:
        lines.append(f"| {c} | " + " | ".join(str(metrics['confusion_matrix'][c][p]) for p in PRED_LABELS) + " |")
    lines += ["", "## Per-class scores", "", "| Class | Precision | Recall | F1 | Support |",
              "|---|---:|---:|---:|---:|"]
    for c, v in metrics["per_class"].items():
        lines.append(f"| {c} | {v['precision']:.2%} | {v['recall']:.2%} | {v['f1']:.2%} | {v['support']} |")
    lines += ["", f"Macro F1: **{metrics['macro_f1']:.2%}**. Accuracy: **{metrics['accuracy']:.2%}**.",
              "Macro F1 is the unweighted average of the three class F1 scores. Undefined scores use zero.",
              f"INVALID is a prediction outcome, not a human class; its {metrics['invalid_count']} outputs count as false negatives for their actual classes.",
              "", "## Error analysis", "",
              f"There are {len(errors)} verdict errors; {metrics['errors_with_gold_retrieved']} occurred despite retrieving the reference clause.",
              "The error CSV contains every incorrect verdict, human/model explanations, clause choices, and retrieval success.",
              "Categories prioritize invalid output, then retrieval miss, then a verdict error with the reference retrieved; they do not prove root cause.",
              "", f"Retrieval Hit@1: {metrics['retrieval_hit_at_1']:.2%}; Hit@5: {metrics['retrieval_hit_at_5']:.2%}.",
              "Only one reference clause is labeled per example. These are reference-hit rates, not exhaustive retrieval precision/recall.",
              "These scores describe this saved benchmark, not performance on unseen requirements.",
              "", "## Reproduce", "", "```bash",
              f'python scripts/evaluate_saved_predictions.py --benchmark "{args.benchmark}" --predictions "{args.predictions}" --out "{args.out}"',
              "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Evaluated {metrics['n']} examples; macro F1={metrics['macro_f1']:.6f}; outputs: {out}")


if __name__ == "__main__":
    main()
