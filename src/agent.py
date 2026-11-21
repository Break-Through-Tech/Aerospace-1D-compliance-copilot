"""One-requirement LangChain tool loop over the canonical RAG pipeline."""

import json

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src import rag
from src.prompts import AGENT_INSTRUCTIONS
from src.schemas import AgentResult, CheckArguments, RetrieveArguments, ToolArguments


MAX_RETRIEVAL_K = 10


class StrictStructuredTool(StructuredTool):
    """Preserve our explicit schemas, including validation of no-argument tools.

    These tools have no injected arguments to filter. LangChain's generated
    subset schema loses extra="forbid", and its empty-schema shortcut otherwise
    discards unexpected arguments before Pydantic can reject them.
    """

    @property
    def tool_call_schema(self) -> type[BaseModel]:
        return self.get_input_schema()

    def _to_args_and_kwargs(self, tool_input, tool_call_id):
        schema = self.get_input_schema()
        if not schema.model_fields:
            schema.model_validate(tool_input)
        return super()._to_args_and_kwargs(tool_input, tool_call_id)


class AuditTools:
    """Tools and private state for exactly one requirement; create anew per run.

    Only retrievals need IDs: the model may choose between two searches. There
    is only one immutable assessment and at most one finding, so neither needs
    a model-supplied ID. LangChain validates arguments and formats tool results.
    """

    def __init__(self, requirement_id: str, requirement_text: str,
                 vector_store: Chroma, client: ChatOllama, *, k: int = 5):
        if type(k) is not int or not 1 <= k <= MAX_RETRIEVAL_K:
            raise ValueError(f"k must be an integer between 1 and {MAX_RETRIEVAL_K}")
        self._record = AgentResult(
            requirement_id=requirement_id, requirement_text=requirement_text,
            retrieved_clause_ids=[],
        )
        self._vector_store = vector_store
        self._client = client
        self._k = k
        self._retrievals = {}
        self._audit_attempted = False
        self.tools = {
            name: StrictStructuredTool.from_function(handler, name=name, args_schema=schema)
            for name, handler, schema in (
                ("retrieve_clause", self._retrieve_clause, RetrieveArguments),
                ("check_requirement", self._check_requirement, CheckArguments),
                ("log_gap", self._log_gap, ToolArguments),
            )
        }

    def _retrieve_clause(self, query: str) -> dict:
        """Retrieve NASA clauses; at most two successful searches per requirement."""
        if self._audit_attempted:
            raise ValueError("Retrieval must precede assessment")
        if len(self._retrievals) >= 2:
            raise ValueError("Retrieval limit reached")
        clauses = rag.retrieve_clauses(query, self._vector_store, k=self._k)
        if not clauses:
            raise ValueError("No clauses retrieved")
        retrieval_id = f"retrieval_{len(self._retrievals) + 1}"
        self._retrievals[retrieval_id] = clauses
        return {"ok": True, "retrieval_id": retrieval_id,
                "clauses": [clause.model_dump(mode="json") for clause in clauses]}

    def _check_requirement(self, retrieval_id: str) -> dict:
        """Audit the original requirement against one stored retrieval."""
        if retrieval_id not in self._retrievals:
            raise ValueError("Unknown retrieval ID")
        if self._audit_attempted:
            raise ValueError("Assessment already attempted; use its result or report failure")
        self._audit_attempted = True
        clauses = self._retrievals[retrieval_id]
        self._record.retrieved_clause_ids = [clause.clause for clause in clauses]
        # The canonical function owns schema validation, evidence checks and retry.
        verdict = rag.audit_requirement(self._record.requirement_text, clauses, client=self._client)
        self._record.verdict = verdict.model_copy(deep=True)
        self._record.grounded = True
        return {"ok": True, **self._record.model_dump(mode="json")}

    def _log_gap(self) -> dict:
        """Record the stored Partial or Gap assessment once, without modifying it."""
        if self._record.verdict is None:
            raise ValueError("No validated assessment to log")
        if self._record.verdict.verdict == "Meets":
            raise ValueError("Only Partial and Gap assessments may be logged")
        already_logged = self._record.finding_id is not None
        self._record.finding_id = f"{self._record.requirement_id}:finding_1"
        return {"ok": True, "finding_id": self._record.finding_id,
                "already_logged": already_logged}

    def result(self, error: str | None = None) -> AgentResult:
        """Return a detached result; incomplete runs never expose a success verdict."""
        record = self._record.model_copy(deep=True)
        if error is None:
            if record.verdict is None:
                error = "Agent finished without a validated assessment"
            elif record.verdict.verdict != "Meets" and record.finding_id is None:
                error = "Agent finished without logging the required gap finding"
        if error is not None:
            record.error = error
            record.verdict = None
            record.grounded = False
            record.finding_id = None
        return record


def run_agent(
    requirement_id: str,
    requirement_text: str,
    vector_store: Chroma,
    *,
    client: ChatOllama | None = None,
    k: int = 5,
    max_steps: int = 8,
) -> AgentResult:
    """Return a stored, validated assessment or an explicit failed record.

    Each call gets fresh tools and messages. Both model turns and total tool
    calls are capped by max_steps. This bounds requests, not elapsed service time.
    Model-written final text is never parsed as an assessment. Inject a client
    for offline tests; otherwise use the canonical Ollama settings and model.
    k fixes retrieval size for every search (1–10, default 5); the model chooses
    query text and which retrieval to audit, but cannot change that size.
    """
    if not isinstance(requirement_id, str) or not requirement_id.strip():
        raise ValueError("requirement_id must be a nonempty string")
    if not isinstance(requirement_text, str) or not requirement_text.strip():
        raise ValueError("requirement_text must be a nonempty string")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if type(k) is not int or not 1 <= k <= MAX_RETRIEVAL_K:
        raise ValueError(f"k must be an integer between 1 and {MAX_RETRIEVAL_K}")

    # Keep configuration and service failures explicit, including client creation.
    try:
        if client is None:
            client = rag.create_audit_client()
        session = AuditTools(requirement_id, requirement_text, vector_store, client, k=k)
        model = client.bind_tools(list(session.tools.values()))
        messages = [
            SystemMessage(content=AGENT_INSTRUCTIONS + f"\nRetrieval size is fixed at k={k}."),
            HumanMessage(content=json.dumps({
                "requirement_id": requirement_id, "requirement_text": requirement_text,
            })),
        ]
        call_ids = set()
        for _ in range(max_steps):
            response = model.invoke(messages)
            if not isinstance(response, AIMessage) or response.invalid_tool_calls:
                raise ValueError("Model returned an invalid tool-call response")
            messages.append(response)
            if not response.tool_calls:
                return session.result()
            ids = [call.get("id") for call in response.tool_calls]
            if any(not call_id or call_id in call_ids for call_id in ids) or len(set(ids)) != len(ids):
                raise ValueError("Tool calls require unique nonempty IDs")
            if len(call_ids) + len(ids) > max_steps:
                return session.result("Agent tool-call limit reached")
            call_ids.update(ids)
            # Execute in the requested order; later calls may use earlier results.
            for call in response.tool_calls:
                try:
                    if call["name"] not in session.tools:
                        raise ValueError(f"Unknown tool: {call['name']}")
                    result = session.tools[call["name"]].invoke(call)
                except Exception as exc:
                    result = ToolMessage(
                        content=json.dumps({"ok": False, "error": str(exc)[:1000]}),
                        tool_call_id=call["id"], status="error",
                    )
                messages.append(result)
        return session.result("Agent model-turn limit reached")
    except Exception as exc:
        return AgentResult(
            requirement_id=requirement_id, requirement_text=requirement_text,
            retrieved_clause_ids=[], error=f"Agent failed: {str(exc)[:1000]}",
        )
