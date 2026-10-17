import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import ValidationError

from src.rag import AuditResult, audit_requirement


def hit(swe_id="SWE-050", clause="4.1.2", distance=0.1):
    return SimpleNamespace(
        page_content="The project shall document and approve software requirements.",
        metadata={"swe_id": swe_id, "clause": clause, "source": "NASA NPR 7150.2D"},
    ), distance


def verdict(swe_id="SWE-050", clause="4.1.2"):
    return {
        "verdict": "Partial", "rationale": "Approval is not specified.",
        "citation": {
            "swe_id": swe_id, "clause": clause, "source": "NASA NPR 7150.2D",
            "quote": "document and approve software requirements",
        },
    }


class RagTests(unittest.TestCase):
    def run_audit(self, verdicts, hits=None):
        self.store = Mock()
        self.store.similarity_search_with_score.return_value = [hit()] if hits is None else hits
        self.client = Mock()
        self.client.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({"verdicts": verdicts})),
        )
        return audit_requirement("We document requirements.", self.store,
                                 model="test-model", client=self.client)

    def test_schema_and_deterministic_settings(self):
        result = self.run_audit([verdict()])
        self.assertEqual(result.verdicts[0].verdict, "Partial")
        self.store.similarity_search_with_score.assert_called_once_with("We document requirements.", k=3)
        args = self.client.chat.call_args.kwargs
        self.assertEqual(args["format"], AuditResult.model_json_schema())
        self.assertEqual(args["options"], {"temperature": 0, "seed": 42})
        self.assertEqual(AuditResult.model_validate_json(result.model_dump_json()), result)

    def test_invalid_schema(self):
        for change in ({"verdict": "Maybe"}, {"rationale": " "}, {"unexpected": True}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                self.run_audit([{**verdict(), **change}])
        with self.assertRaises(ValidationError):
            self.run_audit([])

    def test_citation_must_match_retrieved_evidence(self):
        for field, value in (("swe_id", "SWE-999"), ("clause", "9.9"),
                             ("source", "Another standard"), ("quote", "invented quote")):
            output = verdict()
            output["citation"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_audit([output])

    def test_missing_or_duplicate_verdict(self):
        with self.assertRaises(ValueError):
            self.run_audit([verdict(), verdict()])
        with self.assertRaises(ValueError):
            self.run_audit([verdict()], [hit(), hit("SWE-051", "4.1.3")])

    def test_order_is_stable(self):
        result = self.run_audit(
            [verdict("SWE-051", "4.1.3"), verdict()],
            [hit("SWE-051", "4.1.3"), hit()],
        )
        self.assertEqual([v.citation.swe_id for v in result.verdicts], ["SWE-050", "SWE-051"])

    def test_no_evidence_does_not_call_model(self):
        with self.assertRaises(ValueError):
            self.run_audit([], [])
        self.client.chat.assert_not_called()

    def test_invalid_inputs_do_not_retrieve(self):
        store = Mock()
        for text, k in ((" ", 3), ("Requirement", 0), ("Requirement", True)):
            with self.subTest(text=text, k=k), self.assertRaises(ValueError):
                audit_requirement(text, store, model="test", k=k)
        store.similarity_search_with_score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
