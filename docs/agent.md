# Issue #25: Define tools the agent will use

This change defines three callable tools, connects them to the October baseline,
and provides agent instructions. A model-driven tool loop, CLI runner, report
generation, and baseline-versus-agent evaluation belong to later work.

## Definitions and connections

`src/agent_tools.py` exports `TOOL_SCHEMAS` (JSON function definitions) and
`AuditTools` (handlers and state for one requirement). Input schemas are generated
from strict Pydantic models; unknown arguments and wrong types are rejected.

| Tool | Inputs | Successful result fields (in addition to `ok: true`) | Connection |
|---|---|---|---|
| `retrieve_clause` | `query`: nonblank string, max 20,000 characters; `k`: integer 1-10, default 5 | `retrieval_id`; `clauses`: list of `{clause_id, swe_id, text, score}` | `ClauseIndex.search(query, k)` |
| `check_requirement` | `requirement_id`, `retrieval_id`: nonempty strings | `assessment_id`, `retrieval_id`, `requirement_id`, `requirement_text`, `retrieved_clause_ids`, `grounded`, `verdict` | Existing judge `.judge(original_requirement_text, stored_hits)`, validated with `AuditVerdict` |
| `log_gap` | `assessment_id`: nonempty string | `finding_id`, `already_logged` | New in-memory wrapper over the validated assessment; no existing standalone logging function |

The `verdict` object uses the existing `AuditVerdict` schema: `clause_id`,
`evidence`, `missing_elements`, `rationale`, and `verdict` (`Meets`, `Partial`, `Gap`).
Errors return `{ "ok": false, "error": "description" }` without a verdict.

## Tool behavior

- Use `dispatch(name, arguments)` as the validated entry point.
- Create a separate `AuditTools` instance for each requirement. IDs are local to
  that instance; they are not global database IDs.
- Retrieval results are stored by ID. The compliance tool uses the original
  requirement and stored clause objects, never agent-supplied replacements.
- Wrong requirement IDs, unknown retrieval/assessment IDs, and citations outside
  the retrieved set are rejected.
- Two successful retrievals and two judge invocations are allowed per instance.
  After one valid assessment, further assessment calls are rejected. The existing
  judge retains its own validation/retry behavior.
- `log_gap` copies only validated `Partial` and `Gap` assessments, including
  evidence, rationale, and missing elements. Repeated logging returns the same
  finding ID. `Meets` cannot be logged as a gap.
- `snapshot()` returns independent copies of stored assessments and findings.
  Logging is in memory only; persistence and reports are not implemented here.
- Citation validation checks membership in retrieved clauses; it does not prove
  that the model's interpretation or evidence is correct.

## Agent instructions

`src/agent_instructions.py` exports `AGENT_INSTRUCTIONS`, ready to use as a
future agent's system message. It specifies tool order, use of original
requirement text, conditional finding logging, evidence restrictions, and
explicit failure handling. The future caller must supply only requirement ID
and text, keeping benchmark labels and human rationales out of model input.
The instructions alone do not execute tools or create an agent runtime.

## Manual tool example

Install `requirements-october.txt` using a working Python 3.10+ environment.
From the repository root, the following Python code exercises the wrappers
without a language model or API key. The heuristic judge is only a plumbing
check, not a real compliance assessment.

```python
from scripts.compliance_rag_baseline import (
    ClauseIndex, HeuristicJudge, TfidfEmbedder, load_clauses,
)
from src.agent_tools import AuditTools

clauses = load_clauses("data/npr-7150-2d-requirements.csv")
index = ClauseIndex(clauses, TfidfEmbedder([c.text for c in clauses]))
text = "The project will document software requirements without formal approval."
tools = AuditTools("example-1", text, index, HeuristicJudge())

retrieval = tools.dispatch("retrieve_clause", {"query": text, "k": 5})
if not retrieval["ok"]:
    raise RuntimeError(retrieval["error"])
assessment = tools.dispatch("check_requirement", {
    "requirement_id": "example-1", "retrieval_id": retrieval["retrieval_id"],
})
if not assessment["ok"]:
    raise RuntimeError(assessment["error"])
if assessment["verdict"]["verdict"] in ("Partial", "Gap"):
    finding = tools.dispatch("log_gap", {"assessment_id": assessment["assessment_id"]})
    if not finding["ok"]:
        raise RuntimeError(finding["error"])
print(tools.snapshot())
```

For actual assessments, supply an existing `OllamaJudge`, `ClaudeJudge`, or
`GeminiJudge` with its required model/service configuration. No provider is
started or selected by the tool definitions themselves.

## Verify

```bash
python -m unittest discover -s tests -v
```

Focused tests cover schemas, function connections, original requirement use,
unknown IDs, grounding, logging, tool errors, and isolated state. Integration
uses the real TF-IDF index and heuristic judge, without language-model calls.
