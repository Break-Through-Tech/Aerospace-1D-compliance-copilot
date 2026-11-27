# Compliance agent

The agent uses `langchain.agents.create_agent` to choose tools over the canonical
RAG pipeline. The same configured Ollama model handles tool selection and audit
judgments. Each requirement has its own session and validated result.

## Run one requirement

Install `requirements.txt`, configure `.env` from `.env.example`, start Ollama,
and verify the Chroma index using the root README. The recommended generation
model is `qwen3.5:4b`; existing `.env` files must also set
`GENERATION_MODEL=qwen3.5:4b`. Pull the model with `ollama pull qwen3.5:4b`.
Restart notebook kernels after changing configuration or code.

```python
from src.agent import run_agent
from src.vector_store import load_vector_store

result = run_agent(
    "example-1",
    "The project shall document software requirements without formal approval.",
    load_vector_store(),
)
if result.error:
    print("FAILED:", result.error)
else:
    print(result.verdict.model_dump())
    print("Finding:", result.finding_id)
```

The configured client uses temperature 0, thinking off, and a 600-token audit
budget. Agent turns use a shallow client copy with a 2,048-token output budget;
the original audit client is unchanged. Pass `client=...` to inject one shared
client, or override the agent budget with `agent_num_predict=...`. Passing `None`
preserves the injected client's budget. `reasoning=True` enables thinking only
for agent turns; omit it to retain the shared client's default of `False`.

`k` fixes retrieval size for every search in a run (default 5, allowed 1–10).
The model can reformulate its query and choose a stored retrieval, but cannot
supply a different retrieval size or replacement source records.

## Execution flow

1. `run_agent()` creates a fresh `AuditTools` instance with the original
   requirement. Its tools share only that requirement's working state.
2. `create_agent()` advertises all three tools, invokes the model, executes
   requested tools, and feeds their results back into the conversation.
3. `retrieve_clause` searches Chroma and stores actual `Clause` objects under a
   retrieval ID. `check_requirement` audits the original requirement against one
   chosen stored retrieval using `rag.audit_requirement()`.
4. The audit model returns `AuditDecision`: clause ID, rationale, missing
   elements, and verdict. Python verifies that the ID identifies exactly one
   retrieved clause and copies its full original requirement text into
   `AuditVerdict.evidence`. The model never writes the citation text.
5. For `Partial` or `Gap`, the agent must call `log_gap`. Python records a
   deterministic finding ID against the stored assessment. `Meets` cannot be
   logged, and repeated logging returns the same finding ID.
6. When the model finishes, Python returns a copy of the stored `AgentResult`.
   Final model prose is never parsed as a verdict and cannot replace it.

`AuditDecision` defines the judgment fields and their consistency checks.
`AuditVerdict` extends that schema with source evidence, preserving the existing
result and reporting format. Both canonical RAG and the agent use this audit
path. Source text establishes provenance; it does not establish that the verdict
or selected clause is correct.

## Tools and safeguards

| Tool | Arguments | Operation |
| --- | --- | --- |
| `retrieve_clause` | `query`: nonblank string, at most 20,000 characters | Retrieves the configured number of clauses; allows at most two successful searches before assessment |
| `check_requirement` | `retrieval_id`: nonblank string | Performs one assessment attempt against stored clauses; the canonical audit may retry invalid output once |
| `log_gap` | None | Logs the stored Partial or Gap result without accepting model-supplied finding content |

Strict Pydantic schemas reject wrong types and unexpected tool arguments. The
small `StrictStructuredTool` subclass preserves `additionalProperties: false`
and validates no-argument calls: the tested LangChain version otherwise drops
that advertised restriction and silently discards unexpected `log_gap` arguments.

The runtime uses `max_concurrency=1` because the tools modify the same session
state. This makes tool batches execute in order, avoiding retrieval, assessment,
and logging races. A fresh session per requirement prevents earlier results from
being reused by later requirements. Model-selected retrieval IDs must exist in
that session; source text and verdicts cannot be supplied through tool arguments.

`Meets` requires no missing elements; `Partial` and `Gap` require at least one.
Unknown or ambiguous clause IDs fail, and Python never substitutes another clause.
The audit retries schema, parsing, and citation failures once with fixed feedback,
without appending failed model output or exception dumps to the prompt. After an
assessment attempt, additional searches and assessments are rejected.

Built-in model/tool call-limit middleware caps model turns and total tool calls
separately at `max_steps` (default 8). Oversized tool-call batches fail before
execution. A small `after_model` hook rejects malformed/truncated responses and
missing or duplicate call IDs. Limits bound requests, not wall-clock time.

Tool errors return as error `ToolMessage` objects so the model can correct a
request within the budget. Handler exceptions have bounded JSON error messages;
LangChain also produces its own validation and unknown-tool error messages.
Direct `AuditTools.tools[name].invoke(...)` calls raise their errors.

A successful `AgentResult` has a validated verdict, `grounded=True`, and no error;
Partial or Gap also requires a finding ID. A failed run has no verdict or finding
and `grounded=False`. Missing assessment/logging, service failures, exhausted
budgets, or invalid responses fail explicitly. A truncated final generation still
fails even if earlier tools succeeded. Check `error` before consuming results.

## Inspect messages and evaluate

Pass `verbose=True` to enable LangChain's built-in graph debug output, including
returned messages, tool calls/results, and token/stop metadata. This prints state
updates after calls complete; it is not token streaming.

```bash
python -m scripts.evaluate_rag --agent --limit 5
python -m scripts.evaluate_rag --agent --out results/runs/agent-run
python -m scripts.evaluate_rag --agent --limit 5 --k 3 --out results/runs/agent-top3
```

[Notebook 05](../notebooks/05_agent_evaluation_benchmark.ipynb) uses this same
runtime, starts with five requirements, and keeps its execution results in memory.
Set `limit=None` for all 50 requirements. Execution outputs are cleared so the
notebook can be run against the current local environment.

The evaluator shares one vector store/client but creates a fresh agent session
for every requirement. Labels, expected clauses, and reference rationales stay
in scoring data and are not passed to the agent. Failures count as `INVALID`.
Retrieval metrics use the retrieval chosen for assessment, not a union of searches.
The reporter preserves errors, citations, and finding IDs in the existing output
formats. Agent caching/resumability is not implemented; `--agent --cache ...` is
rejected before services initialize. Canonical RAG still supports its existing
cache, with schema/prompt/implementation fingerprints preventing reuse of
incompatible older audit results.

## Measured source-evidence experiment

The configuration selected for integration used Qwen3.5 4B in both roles, thinking
off, top-k 5, a 2,048-token agent budget, and a 600-token audit budget. Its full
50-requirement development-benchmark run measured:

| Measure | Result |
| --- | --- |
| Completed audits | 50/50 |
| Tool errors | 0 |
| Verdict matches | 37/50 (74%) |
| Macro F1 | 0.7200 |
| Expected clause retrieved in top 5 | 49/50 (98%) |
| Incorrect Meets predictions | 5 (IDs 8, 13, 18, 20, 28) |

All 50 evidence fields were verified against the selected retrieved source text.
Detailed artifacts remain local under
`results/runs/agent-qwen35-source-evidence-20261127T001247/`. These are measured
experiment results, not a guarantee that every rerun produces identical outputs.
The benchmark has been reused during development and is not an independent holdout.

Valid output, exact source text, and completed tool use do not prove compliance
accuracy. Each verdict evaluates one selected clause, rather than establishing
coverage of all applicable NASA obligations. The five incorrect Meets predictions
remain judgment/selection errors requiring review. Treat source text as untrusted
data; the prompt and dispatch checks do not guarantee prompt-injection immunity.

## Verify offline

Install canonical and October dependencies with
`python -m pip install -e '.[october]'`, configure `.env`, and run:

```bash
python -m pytest -q
python -m pip check
```

Tests exercise real LangChain tools/messages and the `create_agent` graph with
mocked inference. Real-client tests replace generation while retaining schema
binding and parsing. They cover original-input preservation, source-resolved
evidence, unknown/ambiguous IDs, consistent verdicts, required findings, per-run
isolation, sequential batches, errors/retries, budgets, evaluation/reporting, and
cache invalidation. Offline tests require no Ollama server or model download.
