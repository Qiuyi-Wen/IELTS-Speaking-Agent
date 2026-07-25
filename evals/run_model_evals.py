"""Run selected evaluation cases against DashScope.

This script makes paid model calls.  It is intentionally separate from CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AILab_AgentIELTSTestPreparation import (  # noqa: E402
    grammar_judge_node,
    head_coach_node,
    vocab_judge_node,
)
from evaluation_models import (  # noqa: E402
    CoachEvaluation,
    GrammarEvaluation,
    VocabularyEvaluation,
    collect_weaknesses,
)
from evals.evaluator import (  # noqa: E402
    EvalObservation,
    evaluate_observation,
    load_eval_cases,
)


def run_case(case):
    state = {
        "current_question": case.question,
        "user_answer": case.answer,
        "timer_result": "离线评测：不使用交互耗时进行评分",
        "profile_path": str(REPO_ROOT / "evals" / ".empty_profile.json"),
    }
    state.update(grammar_judge_node(state))
    state.update(vocab_judge_node(state))
    state.update(head_coach_node(state))

    grammar = GrammarEvaluation.model_validate(state["grammar_result"])
    vocabulary = VocabularyEvaluation.model_validate(state["vocab_result"])
    coach = CoachEvaluation.model_validate(state["coach_result"])
    categories = collect_weaknesses([grammar, vocabulary])
    return evaluate_observation(
        case,
        EvalObservation(
            score=coach.text_based_overall_score,
            issue_categories=categories,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    cases = load_eval_cases()
    if args.case_id:
        selected = [case for case in cases if case.id in set(args.case_id)]
        missing = set(args.case_id) - {case.id for case in selected}
        if missing:
            raise SystemExit(f"unknown case IDs: {sorted(missing)}")
    else:
        selected = cases[: args.limit]

    outcomes = [run_case(case) for case in selected]
    payload = [outcome.model_dump() for outcome in outcomes]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    raise SystemExit(0 if all(outcome.passed for outcome in outcomes) else 1)


if __name__ == "__main__":
    main()
