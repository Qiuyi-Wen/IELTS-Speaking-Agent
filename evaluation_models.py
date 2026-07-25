"""Validated data contracts for text-based IELTS feedback.

These models deliberately exclude pronunciation and spoken fluency: neither can
be assessed from a typed answer.  They also keep every reported issue grounded
in an exact quote from the learner's answer.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator


IssueCategory: TypeAlias = Literal[
    "时态",
    "主谓一致",
    "冠词",
    "介词",
    "句子结构",
    "从句",
    "语态",
    "代词指代",
    "拼写",
    "词汇准确性",
    "搭配",
    "词汇多样性",
    "同义改写",
    "话题词汇",
]


class FeedbackIssue(BaseModel):
    category: IssueCategory
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

    @model_validator(mode="after")
    def calibrate_score_from_dimensions(self):
        self.score = half_band_mean(self.accuracy_score, self.range_score)
        return self


class VocabularyEvaluation(ScoredEvaluation):
    diversity_score: float = Field(ge=0, le=9)
    accuracy_score: float = Field(ge=0, le=9)
    upgrades: list[VocabularyUpgrade] = Field(default_factory=list)
    topic_vocabulary: list[str] = Field(default_factory=list)

    @field_validator("diversity_score", "accuracy_score")
    @classmethod
    def require_half_band_dimension_score(cls, value: float) -> float:
        return ScoredEvaluation.require_half_band_score(value)

    @model_validator(mode="after")
    def calibrate_score_from_dimensions(self):
        self.score = half_band_mean(self.diversity_score, self.accuracy_score)
        return self


class EvidenceItem(BaseModel):
    claim: str = Field(min_length=1)
    related_issue_categories: list[IssueCategory] = Field(default_factory=list)


class Recommendation(BaseModel):
    action: str = Field(min_length=1)
    basis: Literal["issue", "strength"]
    related_issue_categories: list[IssueCategory] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_basis_from_links(cls, data):
        """Treat an explicit issue link as authoritative over the model's label."""
        if isinstance(data, dict) and data.get("related_issue_categories"):
            normalized = dict(data)
            normalized["basis"] = "issue"
            return normalized
        return data

    @model_validator(mode="after")
    def validate_basis_links(self):
        if self.basis == "issue" and not self.related_issue_categories:
            raise ValueError(
                "issue-based recommendations must reference an issue category"
            )
        return self


class ReviewPlan(BaseModel):
    short_term: list[Recommendation] = Field(default_factory=list)
    medium_term: list[Recommendation] = Field(default_factory=list)
    long_term: list[Recommendation] = Field(default_factory=list)

    def all_recommendations(self) -> list[Recommendation]:
        return self.short_term + self.medium_term + self.long_term


class CoachEvaluation(BaseModel):
    grammar_score: float = Field(ge=0, le=9)
    vocabulary_score: float = Field(ge=0, le=9)
    coherence_score: float = Field(ge=0, le=9)
    text_based_overall_score: float = Field(ge=0, le=9)
    evidence_summary: list[EvidenceItem] = Field(min_length=1)
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
        self.text_based_overall_score = half_band_mean(
            self.grammar_score,
            self.vocabulary_score,
            self.coherence_score,
        )
        self.unsupported_dimensions = ["pronunciation", "spoken_fluency"]
        return self


def half_band_mean(*scores: float) -> float:
    """Return an arithmetic mean rounded to the nearest IELTS half band."""
    if not scores:
        raise ValueError("at least one score is required")
    doubled_mean = sum(scores) / len(scores) * 2
    return math.floor(doubled_mean + 0.5) / 2


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


def find_coach_alignment_errors(
    coach: CoachEvaluation,
    grammar: GrammarEvaluation,
    vocabulary: VocabularyEvaluation,
) -> list[str]:
    """Find coach claims or actions that are not backed by specialist issues."""
    errors: list[str] = []
    if coach.grammar_score != grammar.score:
        errors.append(
            f"coach grammar score {coach.grammar_score} != specialist {grammar.score}"
        )
    if coach.vocabulary_score != vocabulary.score:
        errors.append(
            "coach vocabulary score "
            f"{coach.vocabulary_score} != specialist {vocabulary.score}"
        )

    observed_categories = {
        issue.category for issue in grammar.issues + vocabulary.issues
    }
    linked_categories: set[str] = set()

    for evidence in coach.evidence_summary:
        for category in evidence.related_issue_categories:
            linked_categories.add(category)
            if category not in observed_categories:
                errors.append(
                    f"evidence references unobserved issue category: {category}"
                )

    for recommendation in coach.review_plan.all_recommendations():
        for category in recommendation.related_issue_categories:
            linked_categories.add(category)
            if category not in observed_categories:
                errors.append(
                    f"recommendation references unobserved issue category: {category}"
                )

    missing_categories = observed_categories - linked_categories
    if missing_categories:
        errors.append(
            "specialist issues missing from coach evidence/recommendations: "
            + ", ".join(sorted(missing_categories))
        )

    return errors
