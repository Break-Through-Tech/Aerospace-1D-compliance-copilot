"""Checks for saved-prediction evaluation; no models or third-party packages."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("saved_eval", ROOT / "scripts/evaluate_saved_predictions.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class SavedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.gold = {}
        self.pred = {}
        for i, label in enumerate(evaluation.LABELS):
            key = str(i)
            self.gold[key] = dict(requirement_id=key, requirement_text=key,
                                  clause_id="1", label=label, rationale="human")
            self.pred[key] = dict(requirement_text=key, pred_verdict=label,
                                  pred_clause="1", retrieved="1,2,3,4,5", pred_rationale="model")

    def test_perfect_and_reordered_predictions(self):
        metrics, errors = evaluation.evaluate(self.gold, dict(reversed(list(self.pred.items()))))
        self.assertEqual(metrics["macro_f1"], 1)
        self.assertEqual(errors, [])

    def test_invalid_counts_as_false_negative_and_absent_prediction_class(self):
        self.pred["2"]["pred_verdict"] = "INVALID"
        metrics, errors = evaluation.evaluate(self.gold, self.pred)
        self.assertEqual(metrics["per_class"]["Gap"]["fn"], 1)
        self.assertEqual(metrics["per_class"]["Gap"]["precision"], 0)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"]["Gap"]["INVALID"], 1)
        self.assertEqual(errors[0]["error_category"], "invalid_output")

    def test_missing_ids_and_text_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluation.evaluate(self.gold, {"0": self.pred["0"]})
        self.pred["0"]["requirement_text"] = "different"
        with self.assertRaises(ValueError):
            evaluation.evaluate(self.gold, self.pred)

    def test_published_metrics_and_error_totals(self):
        gold = evaluation.read_rows(ROOT / "data/npr7150_ground_truth_benchmark.csv", {"requirement_id"})
        pred = evaluation.read_rows(ROOT / "results/october/predictions.csv", {"requirement_id"})
        metrics, errors = evaluation.evaluate(gold, pred)
        self.assertEqual(metrics["confusion_matrix"], {
            "Meets": {"Meets": 18, "Partial": 4, "Gap": 0, "INVALID": 0},
            "Partial": {"Meets": 7, "Partial": 3, "Gap": 4, "INVALID": 0},
            "Gap": {"Meets": 2, "Partial": 0, "Gap": 10, "INVALID": 2}})
        self.assertEqual(len(errors), 19)
        self.assertEqual(metrics["errors_with_gold_retrieved"], 16)
        self.assertEqual(metrics["retrieval_hit_at_5"], .92)
        self.assertAlmostEqual(metrics["macro_f1"], (36/49 + 2/7 + 5/7) / 3)


if __name__ == "__main__":
    unittest.main()
