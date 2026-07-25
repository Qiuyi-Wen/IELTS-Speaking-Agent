import unittest

from pydantic import ValidationError

from evaluation_models import (
    CoachEvaluation,
    EvidenceItem,
    FeedbackIssue,
    GrammarEvaluation,
    Recommendation,
    ReviewPlan,
    VocabularyEvaluation,
    VocabularyUpgrade,
    collect_weaknesses,
    find_coach_alignment_errors,
    find_ungrounded_quotes,
    half_band_mean,
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

    def test_half_band_ties_round_up(self):
        self.assertEqual(half_band_mean(6.0, 6.5), 6.5)

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

    def test_specialist_scores_are_calibrated_from_dimensions(self):
        grammar = grammar_evaluation(
            score=9.0,
            accuracy_score=6.0,
            range_score=7.0,
        )
        vocabulary = vocabulary_evaluation(
            score=9.0,
            diversity_score=6.0,
            accuracy_score=7.0,
        )
        self.assertEqual(grammar.score, 6.5)
        self.assertEqual(vocabulary.score, 6.5)

    def test_coach_score_is_calibrated_from_dimensions(self):
        coach = CoachEvaluation(
            grammar_score=6.0,
            vocabulary_score=6.5,
            coherence_score=6.5,
            text_based_overall_score=8.0,
            evidence_summary=[
                EvidenceItem(claim="The component scores support 6.5.")
            ],
            history_comparison="No history.",
            review_plan=ReviewPlan(),
            encouragement="Keep practising.",
        )
        self.assertEqual(coach.text_based_overall_score, 6.5)

    def test_issue_recommendation_requires_category_link(self):
        with self.assertRaises(ValidationError):
            Recommendation(
                action="Review agreement.",
                basis="issue",
                related_issue_categories=[],
            )

    def test_coach_cannot_reference_unobserved_issue(self):
        grammar = grammar_evaluation(issues=[])
        vocabulary = vocabulary_evaluation(issues=[])
        coach = CoachEvaluation(
            grammar_score=grammar.score,
            vocabulary_score=vocabulary.score,
            coherence_score=6.5,
            text_based_overall_score=6.5,
            evidence_summary=[
                EvidenceItem(
                    claim="There is a tense problem.",
                    related_issue_categories=["时态"],
                )
            ],
            history_comparison="No history.",
            review_plan=ReviewPlan(),
            encouragement="Keep practising.",
        )
        self.assertEqual(
            find_coach_alignment_errors(coach, grammar, vocabulary),
            ["evidence references unobserved issue category: 时态"],
        )

    def test_coach_must_cover_observed_issue_categories(self):
        grammar = grammar_evaluation(
            issues=[
                FeedbackIssue(
                    category="主谓一致",
                    severity="major",
                    quote="The tree are",
                    suggestion="The trees are",
                    explanation="Agreement error.",
                )
            ]
        )
        vocabulary = vocabulary_evaluation(issues=[])
        coach = CoachEvaluation(
            grammar_score=grammar.score,
            vocabulary_score=vocabulary.score,
            coherence_score=6.5,
            text_based_overall_score=6.5,
            evidence_summary=[EvidenceItem(claim="The response is understandable.")],
            history_comparison="No history.",
            review_plan=ReviewPlan(
                short_term=[
                    Recommendation(
                        action="Build on the clear organization.",
                        basis="strength",
                    )
                ]
            ),
            encouragement="Keep practising.",
        )
        self.assertEqual(
            find_coach_alignment_errors(coach, grammar, vocabulary),
            [
                "specialist issues missing from coach evidence/recommendations: "
                "主谓一致"
            ],
        )


if __name__ == "__main__":
    unittest.main()
