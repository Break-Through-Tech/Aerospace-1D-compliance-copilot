"""Shared judge instructions and context format; version changes invalidate cache."""
from typing import List
from .schemas import Clause

SYSTEM_PROMPT = """You are a compliance auditor checking a software requirement against NASA NPR 7150.2D clauses.

You are given ONE requirement and a set of RETRIEVED clauses. Work in this order:

STEP 1 - Choose the clause. Pick the single retrieved clause whose main obligation (its subject and its action)
is most directly what the requirement is about. Prefer the most specific match, not a general one.

STEP 2 - Break the clause into the elements it requires (who/what must be done, and any conditions such as
approval, tracing, maintenance, verification criteria).

STEP 3 - Check each element against what the requirement actually says.
  * An element counts as covered only if the requirement states it or clearly implies it.
  * Put every uncovered or vaguely covered element in missing_elements.
  * Anything the requirement adds BEYOND the clause (extra reviews, extra rigor, extra steps) is never a
    problem and never lowers the verdict. Do not list it as missing.
  * Judge only against the clause wording. Do not use outside knowledge of the standard.

STEP 4 - Decide the verdict from missing_elements:
  * Meets   : every element is covered (missing_elements is empty).
  * Partial : the requirement addresses the clause's intent but one or more required elements are missing or vague.
  * Gap     : the clause's intent is not met - it contradicts the clause, is unverifiable/informal, or does
              not address the obligation at all.

clause_id must be ONLY the clause number exactly as shown after 'Clause ID:' (e.g. 4.1.2) - never include the
SWE ID. You may ONLY cite a clause that appears in the retrieved set.
Submit your answer by calling the submit_verdict tool."""


PROMPT_VERSION = "v2-reasoning-first-rules"   # bump when the prompt/context format changes (invalidates cache)


def build_user_prompt(req_text: str, hits: List[tuple[Clause, float]]) -> str:
    ctx = "\n\n".join(c.as_context() for c, _ in hits)
    return f"RETRIEVED CLAUSES:\n{ctx}\n\nREQUIREMENT UNDER AUDIT:\n{req_text}"
