"""HeuristicJudge provider."""
import re
from ..schemas import AuditVerdict, Verdict


class HeuristicJudge:
    """Offline pipeline smoke-test ONLY (not a real baseline): top-1 clause, verdict from similarity + vague words.
    Lets you verify retrieval, schema, grounding and metrics code without an API key."""
    name = "heuristic"
    VAGUE = re.compile(r"\b(manual|manually|spreadsheet|no formal|without|ad hoc|informal|as needed|"
                       r"best effort|when possible|optional)\b", re.I)

    def judge(self, req_text, hits):
        clause, score = hits[0]
        if self.VAGUE.search(req_text) or score < 0.15:
            return AuditVerdict(clause_id=clause.clause_id, verdict=Verdict.GAP,
                                rationale="Requirement is informal or weakly related to the clause wording.",
                                evidence=clause.text[:60], missing_elements=["formal compliance with clause"])
        if score < 0.35:
            return AuditVerdict(clause_id=clause.clause_id, verdict=Verdict.PARTIAL,
                                rationale="Requirement overlaps only part of the clause wording.",
                                evidence=clause.text[:60], missing_elements=["remaining clause elements"])
        return AuditVerdict(clause_id=clause.clause_id, verdict=Verdict.MEETS,
                            rationale="Requirement closely matches the clause wording.",
                            evidence=clause.text[:60], missing_elements=[])
