"""Validated request and response contracts for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from evaluation_models import (
    CoachEvaluation,
    GrammarEvaluation,
    VocabularyEvaluation,
)


class EvaluateRequest(BaseModel):
    question: str = Field(min_length=10, max_length=1000)
    answer: str = Field(min_length=1, max_length=6000)

    @field_validator("question", "answer", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value


class EvaluationResponse(BaseModel):
    question: str
    answer_word_count: int = Field(ge=0)
    score_type: Literal["text_based"] = "text_based"
    grammar: GrammarEvaluation
    vocabulary: VocabularyEvaluation
    coach: CoachEvaluation
    text_based_overall_score: float = Field(ge=0, le=9)
    weaknesses: list[str] = Field(default_factory=list)
    unsupported_dimensions: list[str] = Field(
        default_factory=lambda: ["pronunciation", "spoken_fluency"]
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_configured: bool
