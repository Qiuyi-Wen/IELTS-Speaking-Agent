import unittest

from evals.evaluator import (
    EvalOutcome,
    EvalObservation,
    evaluate_observation,
    load_eval_cases,
    run_cases_resilient,
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
                issue_categories=["时态", "主谓一致", "搭配"],
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
        self.assertEqual(len(outcome.failures), 3)

    def test_alternative_issue_category_group_accepts_any_member(self):
        case = next(
            case for case in load_eval_cases()
            if case.id == "low_technology_01"
        )
        outcome = evaluate_observation(
            case,
            EvalObservation(
                score=4.5,
                issue_categories=["句子结构", "介词"],
            ),
        )
        self.assertTrue(outcome.passed)

    def test_alternative_issue_category_group_reports_when_none_match(self):
        case = next(
            case for case in load_eval_cases()
            if case.id == "low_technology_01"
        )
        outcome = evaluate_observation(
            case,
            EvalObservation(
                score=4.5,
                issue_categories=["句子结构"],
            ),
        )
        self.assertEqual(
            outcome.failures,
            ["missing required issue category group: 冠词 | 介词"],
        )

    def test_case_failure_does_not_stop_later_cases(self):
        cases = load_eval_cases()[:3]

        def runner(case):
            if case.id == cases[1].id:
                raise ValueError("invalid model structure")
            return EvalOutcome(
                case_id=case.id,
                passed=True,
                observation=EvalObservation(
                    score=case.expected_score_min,
                    issue_categories=case.required_issue_categories,
                ),
            )

        outcomes = list(run_cases_resilient(cases, runner))

        self.assertEqual([outcome.case_id for outcome in outcomes], [
            case.id for case in cases
        ])
        self.assertTrue(outcomes[0].passed)
        self.assertFalse(outcomes[1].passed)
        self.assertIn("invalid model structure", outcomes[1].error)
        self.assertTrue(outcomes[2].passed)


if __name__ == "__main__":
    unittest.main()
