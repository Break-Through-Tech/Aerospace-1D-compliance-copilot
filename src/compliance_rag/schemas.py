"""Validated clauses, verdicts, and audit records."""
import re
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

class Verdict(str, Enum):
    MEETS = "Meets"
    PARTIAL = "Partial"
    GAP = "Gap"


class Clause(BaseModel):
    clause_id: str
    swe_id: str
    text: str

    def as_context(self) -> str:
        return f"Clause ID: {self.clause_id}\nSWE ID: {self.swe_id}\nText: {self.text}"


class AuditVerdict(BaseModel):
    """The ONLY output shape the judge is allowed to produce.
    Field ORDER matters: structured decoding fills fields top to bottom, so the model must write the
    clause choice, evidence and missing elements BEFORE it commits to a verdict (reason first, decide last)."""
    clause_id: str = Field(description="Clause number ONLY of the single retrieved clause this verdict is about, "
                                       "e.g. '4.1.2'. No SWE ID, no other text.")
    evidence: str = Field(description="Short phrase copied from the CLAUSE text naming the obligation being checked.")
    missing_elements: List[str] = Field(
        default_factory=list,
        description="Each element the CLAUSE requires that the requirement does NOT address, or addresses only "
                    "vaguely. Never list things the requirement adds beyond the clause. Empty ONLY if every "
                    "element of the clause is covered.")
    rationale: str = Field(min_length=15, description="1-2 sentences comparing the requirement to the clause wording, "
                                                      "justifying the missing_elements and the verdict.")
    verdict: Verdict = Field(description="Decide LAST, from missing_elements: Meets = nothing missing; "
                                         "Partial = the clause's intent is addressed but something it requires is "
                                         "missing or vague; Gap = the clause's intent is not met.")

    @field_validator("clause_id")
    @classmethod
    def _norm_clause(cls, v: str) -> str:
        v = re.sub(r"^(clause( id)?|cl\.?)[:\s]*", "", v.strip(), flags=re.I)
        m = re.match(r"\d+(?:\.\d+)*", v)          # tolerate '4.1.2 | SWE-050' -> '4.1.2' (formatting only)
        return m.group(0) if m else v

    @model_validator(mode="after")
    def _meets_has_no_gaps(self):
        if self.verdict == Verdict.MEETS and self.missing_elements:
            raise ValueError("verdict is Meets but missing_elements is non-empty")
        if self.verdict != Verdict.MEETS and not self.missing_elements:
            raise ValueError("Partial/Gap verdicts must list at least one missing element")
        return self


class AuditRecord(BaseModel):
    requirement_id: str
    requirement_text: str
    retrieved_clause_ids: List[str]
    verdict: Optional[AuditVerdict] = None
    grounded: bool = False          # cited clause_id is in retrieved set
    error: Optional[str] = None
