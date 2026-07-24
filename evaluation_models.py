"""Validated data contracts for text-based IELTS feedback.

These models deliberately exclude pronunciation and spoken fluency: neither can
be assessed from a typed answer.  They also keep every reported issue grounded
in an exact quote from the learner's answer.
"""

from __future__ import annotations

from typing import Iterable, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FeedbackIssue(BaseModel):
    category: str = Field(min_length=1)
    severity: Literal["minor", "major"]
    quote: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class VocabularyUpgrade(BaseModel):
    source_quote: str = Field(min_length=1)
    replacement: str = Field(min_length=1)
    level: Literal["基础", "中级", "高级"]
    explanation: str = Field(min_length=1)


class ScoredEvaluation(BaseModel):
    score: float = Field(ge=0, le=9)
    strengths: list[str] = Field(default_factory=list)
    issues: list[FeedbackIssue] = Field(default_factory=list)
    summary: str = Field(min_length=1)

    @field_validator("score")
    @classmethod
    def require_half_band_score(cls, value: float) -> float:
        doubled = value * 2
        if abs(doubled - round(doubled)) > 1e-9:
            raise ValueError("IELTS scores must use 0.5-band increments")
        return value


class GrammarEvaluation(ScoredEvaluation):
    accuracy_score: float = Field(ge=0, le=9)
    range_score: float = Field(ge=0, le=9)

    @field_validator("accuracy_score", "range_score")
    @classmethod
    def require_half_band_dimension_score(cls, value: float) -> float:
        return ScoredEvaluation.require_half_band_score(value)


class VocabularyEvaluation(ScoredEvaluation):
    diversity_score: float = Field(ge=0, le=9)
    accuracy_score: float = Field(ge=0, le=9)
    upgrades: list[VocabularyUpgrade] = Field(default_factory=list)
    topic_vocabulary: list[str] = Field(default_factory=list)

    @field_validator("diversity_score", "accuracy_score")
    @classmethod
    def require_half_band_dimension_score(cls, value: float) -> float:
        return ScoredEvaluation.require_half_band_score(value)


class ReviewPlan(BaseModel):
    short_term: list[str] = Field(default_factory=list)
    medium_term: list[str] = Field(default_factory=list)
    long_term: list[str] = Field(default_factory=list)


class CoachEvaluation(BaseModel):
    grammar_score: float = Field(ge=0, le=9)
    vocabulary_score: float = Field(ge=0, le=9)
    coherence_score: float = Field(ge=0, le=9)
    text_based_overall_score: float = Field(ge=0, le=9)
    evidence_summary: list[str] = Field(min_length=1)
    history_comparison: str
    review_plan: ReviewPlan
    encouragement: str = Field(min_length=1)
    unsupported_dimensions: list[str] = Field(
        default_factory=lambda: ["pronunciation", "spoken_fluency"]
    )

    @field_validator(
        "grammar_score",
        "vocabulary_score",
        "coherence_score",
        "text_based_overall_score",
    )
    @classmethod
    def require_half_band_score(cls, value: float) -> float:
        return ScoredEvaluation.require_half_band_score(value)

    @model_validator(mode="after")
    def validate_text_score_and_boundaries(self):
        expected = round(
            (
                self.grammar_score
                + self.vocabulary_score
                + self.coherence_score
            )
            / 3
            * 2
        ) / 2
        if self.text_based_overall_score != expected:
            raise ValueError(
                "text_based_overall_score must be the half-band rounded mean "
                "of grammar, vocabulary, and coherence scores"
            )
        self.unsupported_dimensions = ["pronunciation", "spoken_fluency"]
        return self


def find_ungrounded_quotes(
    answer: str,
    *evaluations: GrammarEvaluation | VocabularyEvaluation,
) -> list[str]:
    """Return model-produced source quotes that are not exact answer substrings."""
    candidates: list[str] = []
    for evaluation in evaluations:
        candidates.extend(issue.quote for issue in evaluation.issues)
        if isinstance(evaluation, VocabularyEvaluation):
            candidates.extend(upgrade.source_quote for upgrade in evaluation.upgrades)
    return [quote for quote in candidates if quote not in answer]


def collect_weaknesses(
    evaluations: Iterable[GrammarEvaluation | VocabularyEvaluation],
) -> list[str]:
    """Build the learner profile from explicit issues, never report keywords."""
    weaknesses: list[str] = []
    for evaluation in evaluations:
        for issue in evaluation.issues:
            if issue.category not in weaknesses:
                weaknesses.append(issue.category)
    return weaknesses
