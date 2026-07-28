"""Contracts and deterministic checks for model-evaluation observations."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator

from pydantic import BaseModel, Field, model_validator

from evaluation_models import IssueCategory


class EvalCase(BaseModel):
    id: str = Field(min_length=1)
    level: str = Field(pattern=r"^(low|mid|high)$")
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    expected_score_min: float = Field(ge=0, le=9)
    expected_score_max: float = Field(ge=0, le=9)
    required_issue_categories: list[IssueCategory] = Field(default_factory=list)
    required_issue_category_groups: list[list[IssueCategory]] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_score_range(self):
        if self.expected_score_min > self.expected_score_max:
            raise ValueError("expected_score_min cannot exceed expected_score_max")
        if any(not group for group in self.required_issue_category_groups):
            raise ValueError("required issue category groups cannot be empty")
        return self


class EvalObservation(BaseModel):
    score: float = Field(ge=0, le=9)
    issue_categories: list[IssueCategory] = Field(default_factory=list)


class EvalOutcome(BaseModel):
    case_id: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    observation: EvalObservation | None = None
    error: str | None = None


def load_eval_cases(path: Path | None = None) -> list[EvalCase]:
    cases_path = path or Path(__file__).with_name("cases.json")
    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = [EvalCase.model_validate(case) for case in raw_cases]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case IDs must be unique")
    return cases


def evaluate_observation(
    case: EvalCase,
    observation: EvalObservation,
) -> EvalOutcome:
    failures: list[str] = []
    if not case.expected_score_min <= observation.score <= case.expected_score_max:
        failures.append(
            f"score {observation.score} outside expected range "
            f"{case.expected_score_min}-{case.expected_score_max}"
        )

    missing_categories = set(case.required_issue_categories) - set(
        observation.issue_categories
    )
    if missing_categories:
        failures.append(
            "missing required issue categories: "
            + ", ".join(sorted(missing_categories))
        )

    observed_categories = set(observation.issue_categories)
    for group in case.required_issue_category_groups:
        if not observed_categories.intersection(group):
            failures.append(
                "missing required issue category group: "
                + " | ".join(group)
            )

    return EvalOutcome(
        case_id=case.id,
        passed=not failures,
        failures=failures,
        observation=observation,
    )


def run_cases_resilient(
    cases: Iterable[EvalCase],
    runner: Callable[[EvalCase], EvalOutcome],
) -> Iterator[EvalOutcome]:
    """Run every case and turn one case's exception into a failed outcome."""
    for case in cases:
        try:
            yield runner(case)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            yield EvalOutcome(
                case_id=case.id,
                passed=False,
                failures=[f"execution error: {message}"],
                error=message,
            )
