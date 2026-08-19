"""Reusable, stateless orchestration for CLI, API, and future frontends."""

from collections.abc import Callable

from AILab_AgentIELTSTestPreparation import (
    grammar_judge_node,
    head_coach_node,
    vocab_judge_node,
)
from api_models import EvaluateRequest, EvaluationResponse
from evaluation_models import (
    CoachEvaluation,
    GrammarEvaluation,
    VocabularyEvaluation,
    collect_weaknesses,
)


AgentNode = Callable[[dict], dict]


class IELTSEvaluationService:
    """Run the three evaluation agents without CLI input or profile writes."""

    def __init__(
        self,
        grammar_node: AgentNode = grammar_judge_node,
        vocabulary_node: AgentNode = vocab_judge_node,
        coach_node: AgentNode = head_coach_node,
    ):
        self._grammar_node = grammar_node
        self._vocabulary_node = vocabulary_node
        self._coach_node = coach_node

    def evaluate(self, request: EvaluateRequest) -> EvaluationResponse:
        state = {
            "current_question": request.question,
            "user_answer": request.answer,
            "timer_result": "Web API: interaction time is not used for scoring.",
            "profile_path": "",
        }
        state.update(self._grammar_node(state))
        state.update(self._vocabulary_node(state))
        state.update(self._coach_node(state))

        grammar = GrammarEvaluation.model_validate(state["grammar_result"])
        vocabulary = VocabularyEvaluation.model_validate(state["vocab_result"])
        coach = CoachEvaluation.model_validate(state["coach_result"])

        return EvaluationResponse(
            question=request.question,
            answer_word_count=len(request.answer.split()),
            grammar=grammar,
            vocabulary=vocabulary,
            coach=coach,
            text_based_overall_score=coach.text_based_overall_score,
            weaknesses=collect_weaknesses([grammar, vocabulary]),
            unsupported_dimensions=coach.unsupported_dimensions,
        )
