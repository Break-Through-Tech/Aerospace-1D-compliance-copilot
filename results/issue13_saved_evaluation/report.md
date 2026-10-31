# Issue #13: Saved October prediction evaluation

Inputs: `data/npr7150_ground_truth_benchmark.csv` and `results/october/predictions.csv`. No model was rerun.

Evaluated 50 examples, matched by requirement ID and checked for identical text.
Published October configuration: nomic-embed-text embeddings, qwen3:4b judge, top-k 5.

## Confusion matrix

Rows are human verdicts; columns are model verdicts.

| Human / Model | Meets | Partial | Gap | INVALID |
|---|---:|---:|---:|---:|
| Meets | 18 | 4 | 0 | 0 |
| Partial | 7 | 3 | 4 | 0 |
| Gap | 2 | 0 | 10 | 2 |

## Per-class scores

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Meets | 66.67% | 81.82% | 73.47% | 22 |
| Partial | 42.86% | 21.43% | 28.57% | 14 |
| Gap | 71.43% | 71.43% | 71.43% | 14 |

Macro F1: **57.82%**. Accuracy: **62.00%**.
Macro F1 is the unweighted average of the three class F1 scores. Undefined scores use zero.
INVALID is a prediction outcome, not a human class; its 2 outputs count as false negatives for their actual classes.

## Error analysis

There are 19 verdict errors; 16 occurred despite retrieving the reference clause.
The error CSV contains every incorrect verdict, human/model explanations, clause choices, and retrieval success.
Categories prioritize invalid output, then retrieval miss, then a verdict error with the reference retrieved; they do not prove root cause.

Retrieval Hit@1: 76.00%; Hit@5: 92.00%.
Only one reference clause is labeled per example. These are reference-hit rates, not exhaustive retrieval precision/recall.
These scores describe this saved benchmark, not performance on unseen requirements.

## Reproduce

```bash
python scripts/evaluate_saved_predictions.py --benchmark "data/npr7150_ground_truth_benchmark.csv" --predictions "results/october/predictions.csv" --out "results/issue13_saved_evaluation"
```
