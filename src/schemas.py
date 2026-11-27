"""Canonical clause, verdict, record, and agent tool argument schemas."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


NonEmpty = Annotated[str, Field(min_length=1, pattern=r"\S")]


class Clause(BaseModel):
    """NASA requirement and provenance; unrelated Document metadata is ignored."""

    model_config = ConfigDict(extra="ignore", strict=True)

    swe_id: NonEmpty
    clause: NonEmpty
    requirement: NonEmpty
    chapter_title: NonEmpty
    section_title: NonEmpty
    source: NonEmpty
    source_url: NonEmpty


class AuditDecision(BaseModel):
    """Model-owned judgment fields, validated before adding source evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    clause_id: NonEmpty = Field(
        description="Exact clause identifier of the selected retrieved clause, e.g. '4.1.2'. No SWE ID."
    )
    rationale: str = Field(
        min_length=15,
        pattern=r"\S",
        description="1-2 sentences comparing the requirement to the clause and explaining missing elements.",
    )
    missing_elements: list[NonEmpty] = Field(
        description="Required clause elements the requirement omits or covers vaguely. "
        "Never list extra obligations added by the requirement. Empty only if every element is covered.",
    )
    verdict: Literal["Meets", "Partial", "Gap"] = Field(
        description="Decide LAST: Meets = nothing missing; Partial = intent addressed but required elements "
        "missing or vague; Gap = clause intent not met."
    )

    @model_validator(mode="after")
    def validate_missing_elements(self):
        if self.verdict == "Meets" and self.missing_elements:
            raise ValueError("Meets must have no missing elements")
        if self.verdict != "Meets" and not self.missing_elements:
            raise ValueError("Partial/Gap must list at least one missing element")
        return self


class AuditVerdict(AuditDecision):
    """Validated judgment plus the full text of its selected retrieved clause."""

    evidence: NonEmpty = Field(description="Original requirement text copied from the selected retrieved clause.")


class AuditRecord(BaseModel):
    """Result of processing one requirement through retrieval and auditing.

    ``grounded`` records whether the cited clause ID belongs to the retrieved
    clauses. ``audit_requirement`` separately resolves the full original source
    text for evidence. Neither check establishes judgment accuracy.
    """

    requirement_id: str
    requirement_text: str
    retrieved_clause_ids: list[str]
    verdict: AuditVerdict | None = None
    grounded: bool = False
    error: str | None = None


class AgentResult(AuditRecord):
    """Canonical record plus the optional in-memory gap finding identifier."""

    finding_id: str | None = None


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RetrieveArguments(ToolArguments):
    query: NonEmpty = Field(max_length=20000)


class CheckArguments(ToolArguments):
    retrieval_id: NonEmpty
