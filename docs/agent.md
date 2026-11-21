# Compliance agent

`src/agent.py` contains the LangChain tools, per-requirement state, and synchronous
tool-calling runtime. Agent instructions live alongside the audit prompts in
`src/prompts.py`; tool argument schemas and `AgentResult` live with the canonical
models in `src/schemas.py`. The runtime uses the existing `src.rag` functions;
the October baseline remains historical and independent.

## Run one requirement

Use the canonical setup in the root README: install `requirements.txt`, copy
`.env.example` to `.env`, start Ollama, install the configured models, and build
or verify the Chroma index. The generation model must support tool calling and
structured output. The existing example uses `qwen3:4b`.

From the repository root:

```python
from src.agent import run_agent
from src.vector_store import load_vector_store

result = run_agent(
    "example-1",
    "The project shall document software requirements without formal approval.",
    load_vector_store(),
)
if result.error is not None:
    print("FAILED:", result.error)
else:
    print(result.verdict.model_dump())
    print("Finding:", result.finding_id)
```

`AgentResult` extends the existing `AuditRecord` with `finding_id`. Success has a
validated `verdict`, `grounded=True`, and `error=None`; `Partial` or `Gap` also
requires a finding ID. A failed run has an error, no verdict, no finding ID, and
`grounded=False`. Invalid caller inputs (blank ID/text, invalid `max_steps`, or
`k` outside the integer range 1–10)
raise `ValueError` before any model calls. Service and agent failures become
failed records. Check `error` before consuming results.

By default, `rag.create_audit_client()` uses the shared generation model,
Ollama host, temperature, output budget, and reasoning settings. Pass
`client=...` to inject an existing `ChatOllama` client or an offline test client.
The same unbound client performs the canonical audit; binding tools returns a
separate LangChain runnable and does not change the audit's structured output.

Pass `k=...` to `run_agent()` to set retrieval size (default 5). Every search in
that run uses the configured size. The model supplies only query text to the
retrieval tool, so it can reformulate a query and choose between two retrievals
without changing the size. Direct `AuditTools` callers configure the same `k`
keyword when constructing the session.

## Evaluate a benchmark

After the canonical setup, use the existing command:

```bash
python -m scripts.evaluate_rag --agent --limit 5
python -m scripts.evaluate_rag --agent --out results/runs/agent-first-run
python -m scripts.evaluate_rag --agent --limit 5 --k 3 --out results/runs/agent-top3
```

The benchmark loader returns labeled rows for scoring and separate ID/text pairs
for execution. After applying `--limit`, the command loads one Chroma store and
creates one configured Ollama client. Each pair goes to a new `run_agent()` call
with the shared store/client and configured `k`; labels, gold clause IDs, and
reference rationales never go to the agent. The command prints progress after
each completed requirement, including failures.

The existing evaluator aligns results by requirement ID and counts failures as
`INVALID` predictions. Retrieval Hit@K uses the same configured `k` and the
retrieval chosen for assessment, rather than merging searches. Runs that never
select an assessment retrieval have empty recorded clause IDs and miss retrieval
metrics. The existing reporter saves the same five artifacts, including errors
and agent finding IDs in complete records. The terminal and `report.md` identify
Agent versus Canonical RAG mode; metric formulas and file formats are shared.

Without `--agent`, evaluation retains the RAG batch runner and existing cache
behavior, including support for positive `k` values above 10. Agent mode does not
support caching or resumability: `--agent --cache ...` is rejected before reading
the benchmark or initializing services. Tool selection and conversation completion
add model calls beyond the canonical audit, so expect additional work per
requirement. No agent benchmark scores have been measured here; improvement over
RAG must be established by running and reviewing evaluations.

## Tools

LangChain `StructuredTool` generates tool definitions and validates arguments
against strict Pydantic schemas. Unknown fields and wrong types are rejected.
The local `StrictStructuredTool` subclass preserves the original Pydantic schema
when advertising tools to the model and validates empty argument schemas before
LangChain's no-argument shortcut. Without these two overrides, the tested
LangChain version drops `additionalProperties: false` from advertised schemas
and silently discards unexpected arguments to `log_gap`. These tools have no
injected runtime arguments, so their complete schemas can be advertised directly.

| Tool | Arguments | Work and result |
| --- | --- | --- |
| `retrieve_clause` | `query`: nonblank string, at most 20,000 characters | Calls `rag.retrieve_clauses(query, vector_store, k=configured_k)`; stores actual `Clause` objects and returns a retrieval ID and canonical clause metadata |
| `check_requirement` | `retrieval_id`: nonblank string | Calls `rag.audit_requirement(original_text, stored_clauses, client=client)` and stores/returns the validated assessment |
| `log_gap` | None | Marks the one stored `Partial` or `Gap` assessment as logged and returns `finding_id` and `already_logged` |

Successful responses contain `ok: true`. In the runtime, unknown tools,
validation errors, retrieval errors and audit errors become LangChain
`ToolMessage` responses with `status="error"` and JSON content containing
`ok: false` and an error description (at most 1,000 characters). Errors never
include a verdict. The model may correct arguments within the request budget.
Calling an individual tool through `AuditTools.tools[name].invoke(...)` raises
its exception; the runtime provides error-to-message conversion.

## Why keep this state?

- Each `run_agent()` creates a fresh `AuditTools` instance and message list. No
  requirement can read another run's retrievals, assessment, or finding.
- Retrieval IDs distinguish up to two successful searches. The assessment uses
  the original requirement and the chosen stored clauses, never model-supplied
  evidence or replacement clause objects. IDs are local to a run.
- One assessment attempt is allowed. The canonical audit already retries invalid
  output once, so agent-level assessment retries would multiply model invocations.
  Invalid retrieval IDs do not consume that attempt.
- After an assessment attempt, further searches and assessments are rejected.
- There is only one assessment and one possible finding, so assessment IDs,
  requirement IDs in tool arguments, assessment/finding dictionaries, manual
  JSON tool schemas, and the custom dispatcher are unnecessary.
- Logging attaches one deterministic finding ID to the stored assessment;
  repeated logging returns the same ID. `Meets` cannot be logged. Returned data
  is detached from private state, so changing a response cannot change the
  assessment. Logging is in memory only; retain the returned result to keep
  the assessment and finding together.

The canonical audit owns schema validation, the relationship between verdict
and missing elements, retrieved-clause membership, exact evidence excerpts, and
one corrective retry. The agent does not duplicate those checks or implement a
second compliance judge.

## How the Python code fits together

1. **`run_agent()` creates the session.** The caller supplies the requirement
   and an already loaded Chroma store. Instructions go into a `SystemMessage`;
   the requirement ID and text go into a JSON `HumanMessage` as data.
2. **`bind_tools()` tells the model what it can call.** Each `StructuredTool`
   combines a Python handler, description, and argument schema. Binding advertises
   the tools; it does not run them automatically.
3. **The model returns an `AIMessage`.** Its `tool_calls` contain names,
   arguments and call IDs. The runtime invokes the corresponding LangChain
   tools in order. LangChain validates arguments and returns `ToolMessage`
   objects linked to those call IDs.
4. **Retrieval and auditing happen in the existing pipeline.** Chroma handles
   embeddings/search through `retrieve_clauses()`. `check_requirement` hands
   stored clauses and original text to `audit_requirement()`, which uses
   Ollama structured output and validates the result.
5. **Messages let the model continue.** Assistant tool requests and all tool
   responses are appended to the conversation. The next model invocation can
   choose another tool, log a gap, or finish. Multiple calls in one response
   execute sequentially and all receive matching tool responses.
6. **Python chooses the returned assessment.** When the model stops calling
   tools, `result()` returns a copy of the stored validated record. Final model
   text is never parsed as a verdict and cannot override the audit.

The runtime uses `ChatOllama.bind_tools`, `StructuredTool.invoke`, `AIMessage`,
and `ToolMessage` from the existing LangChain packages. A short bounded loop
connects these APIs because `langchain-core` provides tool/message primitives,
not a complete agent executor. No LangGraph or additional framework is required.

## Termination and limits

`max_steps` defaults to 8 and caps both model turns and total tool calls. An
oversized tool-call batch is rejected before executing any of it. Malformed
responses, missing/duplicate call IDs, model-service exceptions, and exhausted
budgets fail explicitly. Finishing without an assessment or required finding
also fails. Reaching a limit fails even if an assessment was produced but the
model did not finish.

These limits bound requests, not wall-clock time; service calls depend on the
client's timeout settings. Context length remains managed by Ollama. The agent
does not implement context-truncation detection, persistent conversations,
long-term memory, or agent caching/resumability.
Existing README benchmark numbers describe canonical RAG, not this agent.

Grounding verifies evidence provenance, not judgment accuracy. Each assessment
still evaluates one selected clause and does not establish complete NASA
compliance. Source text is explicitly treated as data in instructions; this
does not guarantee immunity to prompt injection. Programmatic safeguards prevent
substituting agent-authored verdicts for canonical validated results.

## Verify offline

Install both the canonical and October dependencies in a virtual environment
with `python -m pip install -e '.[october]'`. Configure `.env` from `.env.example`
(back up an existing `.env` first). From that environment, run:

```bash
python -m pytest tests/test_agent.py tests/test_evaluate_rag.py \
  tests/test_evaluation.py tests/test_reporting.py tests/test_rag.py -q
python -m pytest -q
```

Tests use real LangChain messages and structured tools with mocked Chroma search
and model responses. Agent tests exercise actual canonical retrieval and audit
functions, including evidence rejection and the canonical retry. No Ollama
server, embedding model, or model download is needed. Other repository tests
also exercise disposable Chroma stores and the preserved October baseline.

Verified on November 21, 2026, using Python 3.12.3 and the example environment:

| Check | Result |
| --- | --- |
| Focused command above (agent, evaluation, reporting, RAG) | 142 passed in 3.17 seconds |
| `python -m pytest -q` | 243 passed in 10.54 seconds |
| `python -m pip check` | No broken requirements found |

Installed versions were `pytest==9.1.1`, `langchain-core==1.6.9`,
`langchain-chroma==1.1.0`, `langchain-ollama==1.1.0`, `pydantic==2.14.0`,
`chromadb==1.5.9`, and `ollama==0.6.3`. These record the verified environment;
the dependency files remain unpinned.

The imported implementation initially had three failing tests, all caused by the
tool schema/empty-argument behavior described above. After fixing those failures,
four additional regression cases verify that invalid `log_gap` arguments cannot
create a finding after a valid assessment, through either direct invocation or a
model tool call. A subsequent valid call still logs successfully.

Benchmark integration tests run the real agent, loader, evaluator, and reporter
with mocked services. They cover shared resources with isolated conversations,
gold-data separation, limits, fixed retrieval size across reformulations, progress,
failed predictions and findings in reports, mode labels, and rejection of agent
caching before initialization. Existing RAG cache tests also run with `k=11` to
check that the agent's upper bound does not restrict RAG-only evaluation.

A real Ollama smoke test remains unrun. Offline tests verify the tool protocol,
validation, and local integration, but do not measure the configured model's
ability to choose tools, its compliance judgment accuracy, or service timeouts.
