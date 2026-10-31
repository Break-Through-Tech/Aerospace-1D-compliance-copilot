"""Canonical clause and verdict schemas for the current RAG implementation."""

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


class AuditVerdict(BaseModel):
    """Reason about a single selected clause before deciding its verdict."""

    model_config = ConfigDict(extra="forbid", strict=True)

    clause_id: NonEmpty = Field(
        description="Exact clause identifier of the selected retrieved clause, e.g. '4.1.2'. No SWE ID."
    )
    evidence: NonEmpty = Field(
        description="Exact nonempty phrase copied from the selected clause's requirement text."
    )
    missing_elements: list[NonEmpty] = Field(
        description="Required clause elements the requirement omits or covers vaguely. "
        "Never list extra obligations added by the requirement. Empty only if every element is covered.",
    )
    rationale: str = Field(
        min_length=15,
        pattern=r"\S",
        description="1-2 sentences comparing the requirement to the clause and explaining missing elements.",
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
