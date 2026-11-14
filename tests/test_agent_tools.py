"""Issue #25 tool contracts and baseline connections; no agent or model calls."""
from pathlib import Path
import unittest
from unittest.mock import Mock

from scripts.compliance_rag_baseline import (
    AuditVerdict, Clause, ClauseIndex, HeuristicJudge, TfidfEmbedder, load_clauses,
)
from src.agent_tools import AuditTools, TOOL_SCHEMAS

ROOT = Path(__file__).resolve().parents[1]


class AgentToolTests(unittest.TestCase):
    def setUp(self):
        self.hits = [(Clause(clause_id="4.1.2", swe_id="SWE-050",
                            text="Document and approve all requirements."), .9)]
        self.index = Mock()
        self.index.search.return_value = self.hits
        self.judge = Mock()
        self.judge.judge.return_value = self.verdict()
        self.tools = AuditTools("r1", "Document requirements", self.index, self.judge)

    def verdict(self, label="Partial", clause="4.1.2"):
        return AuditVerdict(clause_id=clause, evidence="approve all requirements",
                            missing_elements=[] if label == "Meets" else ["Approval"],
                            rationale="The requirement is compared with the clause obligations.", verdict=label)

    def retrieve(self):
        return self.tools.dispatch("retrieve_clause", {"query": "focused query"})["retrieval_id"]

    def assess(self):
        return self.tools.dispatch("check_requirement", {
            "requirement_id": "r1", "retrieval_id": self.retrieve()})

    def test_tool_schemas_and_invalid_arguments(self):
        self.assertEqual({t["function"]["name"] for t in TOOL_SCHEMAS},
                         {"retrieve_clause", "check_requirement", "log_gap"})
        for name, args in [("unknown", {}), ("retrieve_clause", {"query": "x", "k": 0}),
                           ("retrieve_clause", {"query": "x", "k": "5"}),
                           ("retrieve_clause", {"query": "x", "extra": True}),
                           ("retrieve_clause", {"query": " "}), ("retrieve_clause", None),
                           ("check_requirement", {"requirement_id": "r1", "retrieval_id": "invented"}),
                           ("log_gap", {"assessment_id": "invented"})]:
            with self.subTest(name=name, args=args):
                self.assertFalse(self.tools.dispatch(name, args)["ok"])
        self.index.search.assert_not_called()
        self.judge.judge.assert_not_called()

    def test_existing_functions_receive_expected_inputs(self):
        result = self.assess()
        self.assertTrue(result["ok"])
        self.index.search.assert_called_once_with("focused query", 5)
        self.judge.judge.assert_called_once_with("Document requirements", self.hits)
        self.assertTrue(result["grounded"])
        self.assertEqual(result["verdict"]["verdict"], "Partial")

    def test_wrong_requirement_and_ungrounded_citation_rejected(self):
        rid = self.retrieve()
        self.assertFalse(self.tools.dispatch("check_requirement", {
            "requirement_id": "other", "retrieval_id": rid})["ok"])
        self.judge.judge.assert_not_called()
        self.judge.judge.return_value = self.verdict(clause="9.9")
        result = self.tools.dispatch("check_requirement", {"requirement_id": "r1", "retrieval_id": rid})
        self.assertFalse(result["ok"])
        self.assertEqual(self.tools.snapshot()["assessments"], [])

    def test_logging_partial_and_gap_preserves_assessment_and_avoids_duplicates(self):
        for label in ("Partial", "Gap"):
            with self.subTest(label=label):
                self.tools = AuditTools("r1", "Document requirements", self.index, self.judge)
                self.judge.judge.return_value = self.verdict(label)
                assessment = self.assess()
                assessment["verdict"]["verdict"] = "Meets"
                first = self.tools.dispatch("log_gap", {"assessment_id": "assessment_1"})
                second = self.tools.dispatch("log_gap", {"assessment_id": "assessment_1"})
                self.assertTrue(first["ok"])
                self.assertEqual(first["finding_id"], second["finding_id"])
                self.assertTrue(second["already_logged"])
                saved = self.tools.snapshot()
                self.assertEqual(len(saved["findings"]), 1)
                self.assertEqual(saved["findings"][0]["verdict"]["verdict"], label)
                saved["findings"].clear()
                self.assertEqual(len(self.tools.snapshot()["findings"]), 1)

    def test_meets_cannot_be_logged(self):
        self.judge.judge.return_value = self.verdict("Meets")
        self.assess()
        self.assertFalse(self.tools.dispatch("log_gap", {"assessment_id": "assessment_1"})["ok"])
        self.assertEqual(self.tools.snapshot()["findings"], [])

    def test_tool_failures_remain_errors(self):
        self.index.search.side_effect = RuntimeError("Search failed")
        self.assertFalse(self.tools.dispatch("retrieve_clause", {"query": "x"})["ok"])
        self.index.search.side_effect = None
        rid = self.retrieve()
        self.judge.judge.side_effect = RuntimeError("Judge failed")
        result = self.tools.dispatch("check_requirement", {"requirement_id": "r1", "retrieval_id": rid})
        self.assertFalse(result["ok"])
        self.assertIn("Judge failed", result["error"])
        self.assertEqual(self.tools.snapshot()["assessments"], [])

    def test_tool_limits_and_reassessment(self):
        rid = self.retrieve()
        self.retrieve()
        self.assertFalse(self.tools.dispatch("retrieve_clause", {"query": "third"})["ok"])
        self.judge.judge.return_value = self.verdict(clause="9.9")
        for _ in range(3):
            result = self.tools.dispatch("check_requirement", {"requirement_id": "r1", "retrieval_id": rid})
        self.assertIn("attempt limit", result["error"])
        self.assertEqual(self.judge.judge.call_count, 2)
        self.tools = AuditTools("r1", "Document requirements", self.index, self.judge)
        self.judge.judge.return_value = self.verdict()
        result = self.assess()
        self.assertTrue(result["ok"])
        self.assertFalse(self.tools.dispatch("check_requirement", {
            "requirement_id": "r1", "retrieval_id": result["retrieval_id"]})["ok"])

    def test_sessions_do_not_share_ids_or_findings(self):
        self.assess()
        self.tools.dispatch("log_gap", {"assessment_id": "assessment_1"})
        other = AuditTools("r2", "Another requirement", self.index, self.judge)
        self.assertFalse(other.dispatch("log_gap", {"assessment_id": "assessment_1"})["ok"])
        self.assertEqual(other.snapshot(), {"assessments": [], "findings": []})

    def test_real_baseline_retrieval_and_judge_integration(self):
        clauses = load_clauses(ROOT / "data/npr-7150-2d-requirements.csv")
        index = ClauseIndex(clauses, TfidfEmbedder([c.text for c in clauses]))
        tools = AuditTools("r1", clauses[0].text, index, HeuristicJudge())
        retrieval = tools.dispatch("retrieve_clause", {"query": clauses[0].text})
        self.assertTrue(retrieval["ok"])
        result = tools.dispatch("check_requirement", {
            "requirement_id": "r1", "retrieval_id": retrieval["retrieval_id"]})
        self.assertTrue(result["ok"], result)
        self.assertIn(result["verdict"]["clause_id"], result["retrieved_clause_ids"])


if __name__ == "__main__":
    unittest.main()
