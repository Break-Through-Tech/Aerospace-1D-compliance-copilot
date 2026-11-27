"""Representative inputs for the current RAG schemas."""

import pytest


@pytest.fixture
def clause_data():
    return dict(
        swe_id="SWE-050",
        clause="4.1.2",
        requirement="The project shall document and maintain software requirements.",
        chapter_title="Software Requirements",
        section_title="Requirements Management",
        source="NASA NPR 7150.2D",
        source_url="https://example.com/standard",
    )


@pytest.fixture
def verdict_data(clause_data):
    return dict(
        clause_id="4.1.2",
        evidence=clause_data["requirement"],
        missing_elements=[],
        rationale="The requirement covers documenting and maintaining requirements.",
        verdict="Meets",
    )


@pytest.fixture
def decision_data(verdict_data):
    return {key: value for key, value in verdict_data.items() if key != "evidence"}
