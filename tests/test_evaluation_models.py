import unittest

from pydantic import ValidationError

from evaluation_models import (
    CoachEvaluation,
    FeedbackIssue,
    GrammarEvaluation,
    ReviewPlan,
    VocabularyEvaluation,
    VocabularyUpgrade,
    collect_weaknesses,
    find_ungrounded_quotes,
)


def grammar_evaluation(**overrides):
    data = {
        "score": 6.5,
        "accuracy_score": 6.0,
        "range_score": 7.0,
        "strengths": ["Uses a relative clause."],
        "issues": [],
        "summary": "Generally clear with one agreement error.",
    }
    data.update(overrides)
    return GrammarEvaluation(**data)


def vocabulary_evaluation(**overrides):
    data = {
        "score": 6.5,
        "diversity_score": 6.5,
        "accuracy_score": 6.5,
        "strengths": ["Uses topic vocabulary."],
        "issues": [],
        "summary": "Vocabulary is adequate for the topic.",
        "upgrades": [],
        "topic_vocabulary": ["tranquil"],
    }
    data.update(overrides)
    return VocabularyEvaluation(**data)


class EvaluationModelTests(unittest.TestCase):
    def test_scores_must_use_half_band_increments(self):
        with self.assertRaises(ValidationError):
            grammar_evaluation(score=6.3)

    def test_quotes_must_be_exact_answer_substrings(self):
        answer = "The tree are very green."
        grammar = grammar_evaluation(
            issues=[
                FeedbackIssue(
                    category="主谓一致",
                    severity="major",
                    quote="The trees are very green.",
                    suggestion="The trees are very green.",
                    explanation="The quote was silently corrected.",
                )
            ]
        )
        self.assertEqual(
            find_ungrounded_quotes(answer, grammar),
            ["The trees are very green."],
        )

    def test_valid_issue_and_upgrade_quotes_are_grounded(self):
        answer = "The tree are very green and the place is quiet."
        grammar = grammar_evaluation(
            issues=[
                FeedbackIssue(
                    category="主谓一致",
                    severity="major",
                    quote="The tree are very green",
                    suggestion="The trees are very green",
                    explanation="Plural subject requires a plural noun and verb.",
                )
            ]
        )
        vocabulary = vocabulary_evaluation(
            upgrades=[
                VocabularyUpgrade(
                    source_quote="quiet",
                    replacement="tranquil",
                    level="中级",
                    explanation="A more precise description of a calm place.",
                )
            ]
        )
        self.assertEqual(find_ungrounded_quotes(answer, grammar, vocabulary), [])

    def test_weaknesses_come_only_from_explicit_issues(self):
        grammar = grammar_evaluation(
            summary="Pronunciation cannot be assessed from text.",
            issues=[
                FeedbackIssue(
                    category="主谓一致",
                    severity="major",
                    quote="The tree are",
                    suggestion="The trees are",
                    explanation="Agreement error.",
                )
            ],
        )
        vocabulary = vocabulary_evaluation(
            summary="Fluency cannot be assessed from text.",
            issues=[],
        )
        self.assertEqual(collect_weaknesses([grammar, vocabulary]), ["主谓一致"])

    def test_coach_score_must_match_validated_dimension_average(self):
        with self.assertRaises(ValidationError):
            CoachEvaluation(
                grammar_score=6.0,
                vocabulary_score=6.5,
                coherence_score=6.5,
                text_based_overall_score=8.0,
                evidence_summary=["The component scores do not support 8.0."],
                history_comparison="No history.",
                review_plan=ReviewPlan(),
                encouragement="Keep practising.",
            )


if __name__ == "__main__":
    unittest.main()
