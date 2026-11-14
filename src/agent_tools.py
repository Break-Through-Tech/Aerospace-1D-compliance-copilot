"""Validated tools backed by the October baseline; state is isolated per requirement."""
import copy
from pydantic import BaseModel, ConfigDict, Field

from scripts.compliance_rag_baseline import AuditVerdict


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RetrieveArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=20000)
    k: int = Field(default=5, ge=1, le=10)


class CheckArguments(ToolArguments):
    requirement_id: str = Field(min_length=1)
    retrieval_id: str = Field(min_length=1)


class LogArguments(ToolArguments):
    assessment_id: str = Field(min_length=1)


ARGUMENTS = {"retrieve_clause": RetrieveArguments, "check_requirement": CheckArguments,
             "log_gap": LogArguments}
DESCRIPTIONS = {
    "retrieve_clause": "Retrieve NASA clauses for this requirement. At most two searches per requirement.",
    "check_requirement": "Judge the original requirement against a stored retrieval; returns a validated, grounded assessment.",
    "log_gap": "Record a stored Partial or Gap assessment once; cannot change its verdict or evidence.",
}
TOOL_SCHEMAS = [{"type": "function", "function": {
    "name": name, "description": DESCRIPTIONS[name], "parameters": schema.model_json_schema()
}} for name, schema in ARGUMENTS.items()]


class AuditTools:
    def __init__(self, requirement_id, requirement_text, index, judge):
        self.requirement_id = requirement_id
        self.requirement_text = requirement_text
        self.index, self.judge = index, judge
        self._retrievals = {}
        self._assessments = {}
        self._findings = {}
        self._check_attempts = 0

    def dispatch(self, name, arguments):
        """The only dispatch entry point. Errors are tool results, never verdicts."""
        try:
            if name not in ARGUMENTS:
                raise ValueError(f"Unknown tool: {name}")
            args = ARGUMENTS[name].model_validate(arguments)
            result = getattr(self, name)(**args.model_dump())
            response = {"ok": True, **result}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)[:1000]}
        return response

    def retrieve_clause(self, query, k=5):
        if not query.strip():
            raise ValueError("Query must not be blank")
        if len(self._retrievals) >= 2:
            raise ValueError("Retrieval limit reached")
        hits = self.index.search(query, k)
        if not hits:
            raise ValueError("No clauses retrieved")
        retrieval_id = f"retrieval_{len(self._retrievals) + 1}"
        self._retrievals[retrieval_id] = hits
        return {"retrieval_id": retrieval_id, "clauses": [
            {"clause_id": c.clause_id, "swe_id": c.swe_id, "text": c.text, "score": float(score)}
            for c, score in hits]}

    def check_requirement(self, requirement_id, retrieval_id):
        if requirement_id != self.requirement_id:
            raise ValueError("Requirement ID does not match this audit")
        if retrieval_id not in self._retrievals:
            raise ValueError("Unknown retrieval ID")
        if self._assessments:
            raise ValueError("A validated assessment already exists; use it")
        if self._check_attempts >= 2:
            raise ValueError("Assessment attempt limit reached")
        self._check_attempts += 1
        hits = self._retrievals[retrieval_id]
        verdict = AuditVerdict.model_validate(self.judge.judge(self.requirement_text, hits))
        if verdict.clause_id not in {c.clause_id for c, _ in hits}:
            raise ValueError("Assessment cites a clause outside the retrieved set")
        assessment_id = "assessment_1"
        record = {"assessment_id": assessment_id, "retrieval_id": retrieval_id,
                  "requirement_id": self.requirement_id, "requirement_text": self.requirement_text,
                  "retrieved_clause_ids": [c.clause_id for c, _ in hits], "grounded": True,
                  "verdict": verdict.model_dump(mode="json")}
        self._assessments[assessment_id] = record
        return copy.deepcopy(record)

    def log_gap(self, assessment_id):
        if assessment_id not in self._assessments:
            raise ValueError("Unknown assessment ID")
        record = self._assessments[assessment_id]
        if record["verdict"]["verdict"] not in ("Partial", "Gap"):
            raise ValueError("Only Partial and Gap assessments may be logged")
        if assessment_id in self._findings:
            return {"finding_id": self._findings[assessment_id]["finding_id"], "already_logged": True}
        finding = {"finding_id": f"{self.requirement_id}:finding_1", **copy.deepcopy(record)}
        self._findings[assessment_id] = finding
        return {"finding_id": finding["finding_id"], "already_logged": False}

    def snapshot(self):
        """Return independent copies of this tool session's assessments and findings."""
        return copy.deepcopy({"assessments": list(self._assessments.values()),
                              "findings": list(self._findings.values())})
