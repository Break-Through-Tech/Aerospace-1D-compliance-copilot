# Compliance agent

`src/agent.py` contains the instructions, LangChain tools, per-requirement state,
and synchronous tool-calling runtime. It uses the canonical `src.rag` functions
and `src.schemas` models; the October baseline remains historical and independent.

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
`grounded=False`. Invalid caller inputs (blank ID/text or invalid `max_steps`)
raise `ValueError` before any model calls. Service and agent failures become
failed records. Check `error` before consuming results.

By default, `rag.create_audit_client()` uses the shared generation model,
Ollama host, temperature, output budget, and reasoning settings. Pass
`client=...` to inject an existing `ChatOllama` client or an offline test client.
The same unbound client performs the canonical audit; binding tools returns a
separate LangChain runnable and does not change the audit's structured output.

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
| `retrieve_clause` | `query`: nonblank string, at most 20,000 characters; `k`: integer 1–10, default 5 | Calls `rag.retrieve_clauses(query, vector_store, k=k)`; stores actual `Clause` objects and returns a retrieval ID and canonical clause metadata |
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
batch CLI integration, long-term memory, or baseline-versus-agent evaluation.
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
python -m pytest tests/test_agent.py tests/test_rag.py -q
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
| `python -m pytest tests/test_agent.py tests/test_rag.py -q` | 83 passed in 1.26 seconds |
| `python -m pytest -q` | 221 passed in 10.98 seconds |
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

A real Ollama smoke test remains unrun. Offline tests verify the tool protocol,
validation, and local integration, but do not measure the configured model's
ability to choose tools, its compliance judgment accuracy, or service timeouts.
