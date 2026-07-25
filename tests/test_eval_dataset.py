import unittest

from evals.evaluator import (
    EvalObservation,
    evaluate_observation,
    load_eval_cases,
)


class EvalDatasetTests(unittest.TestCase):
    def test_dataset_has_balanced_quality_levels(self):
        cases = load_eval_cases()
        counts = {
            level: sum(case.level == level for case in cases)
            for level in ("low", "mid", "high")
        }
        self.assertEqual(len(cases), 12)
        self.assertEqual(counts, {"low": 4, "mid": 4, "high": 4})

    def test_expected_ranges_are_ordered_by_level(self):
        cases = load_eval_cases()
        by_level = {
            level: [case for case in cases if case.level == level]
            for level in ("low", "mid", "high")
        }
        self.assertLess(
            max(case.expected_score_max for case in by_level["low"]),
            min(case.expected_score_min for case in by_level["high"]),
        )

    def test_observation_passes_inside_range_with_required_categories(self):
        case = next(case for case in load_eval_cases() if case.id == "low_place_01")
        outcome = evaluate_observation(
            case,
            EvalObservation(
                score=5.0,
                issue_categories=["时态", "主谓一致", "句子结构"],
            ),
        )
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.failures, [])

    def test_observation_reports_score_and_recall_failures(self):
        case = next(case for case in load_eval_cases() if case.id == "low_place_01")
        outcome = evaluate_observation(
            case,
            EvalObservation(score=8.0, issue_categories=[]),
        )
        self.assertFalse(outcome.passed)
        self.assertEqual(len(outcome.failures), 2)


if __name__ == "__main__":
    unittest.main()
