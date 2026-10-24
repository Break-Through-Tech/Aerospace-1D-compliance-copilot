# October Milestone - Non-Agentic RAG Baseline Results

Embedder: `ollama` | Judge: `ollama` | top-k: 5 | n = 50

## Headline metrics

| Metric | Value |
|---|---|
| Retrieval hit@1 | 76.00% |
| Retrieval hit@5 | 92.00% |
| 3-way verdict accuracy | 62.00% |
| **Gap precision** | 71.43% |
| **Gap recall** | 71.43% |
| Gap F1 | 71.43% |
| Non-compliance (Gap+Partial) precision / recall | 80.95% / 60.71% |
| Citation groundedness | 96.00% |
| Cited the gold clause | 72.00% |
| Schema-valid outputs | 96.00% |

## Confusion matrix (rows = human label, cols = predicted)

| label   |   Meets |   Partial |   Gap |   INVALID |
|:--------|--------:|----------:|------:|----------:|
| Meets   |      18 |         4 |     0 |         0 |
| Partial |       7 |         3 |     4 |         0 |
| Gap     |       2 |         0 |    10 |         2 |

## Errors (for the error-analysis section)

| req | gold | pred | gold clause | cited | retrieved? |
|---|---|---|---|---|---|
| 2 | Partial | Meets | 4.1.3 | 4.1.3 | yes |
| 3 | Partial | Gap | 4.1.5 | 4.1.5 | yes |
| 4 | Meets | Partial | 4.1.7 | 4.1.7 | yes |
| 5 | Partial | Meets | 4.2.3 | 4.2.3 | yes |
| 9 | Partial | Meets | 4.4.4 | 4.4.4 | yes |
| 13 | Gap | Meets | 4.5.3 | 4.5.8 | yes |
| 15 | Meets | Partial | 4.5.5 | 4.5.5 | yes |
| 16 | Partial | Meets | 4.5.9 | 4.5.9 | yes |
| 18 | Partial | Meets | 4.5.11 | 4.5.11 | yes |
| 20 | Partial | Gap | 4.5.13 | 2.1.1.3 | NO |
| 23 | Partial | Meets | 3.11.3 | 3.11.3 | yes |
| 26 | Meets | Partial | 3.11.6 | 3.11.7 | yes |
| 27 | Gap | INVALID | 3.11.7 | - | yes |
| 28 | Partial | Gap | 3.11.8 | 4.5.11 | NO |
| 30 | Gap | INVALID | 3.7.4 | - | yes |
| 37 | Partial | Gap | 5.5.1 | 5.5.1 | yes |
| 39 | Meets | Partial | 5.2 | 5.2 | yes |
| 40 | Partial | Meets | 5.4.2 | 3.2.3 | NO |
| 44 | Gap | Meets | 3.1.5 | 3.1.5 | yes |