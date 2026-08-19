import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from AILab_AgentIELTSTestPreparation import MissingAPIKeyError
from api import app, get_evaluation_service
from api_models import EvaluateRequest
from evaluation_service import IELTSEvaluationService


QUESTION = "Describe a quiet place where you like to relax."
ANSWER = "I relax beside a lake because the natural setting feels peaceful."


def grammar_result():
    return {
        "score": 6.5,
        "accuracy_score": 6.0,
        "range_score": 7.0,
        "strengths": ["The response is understandable."],
        "issues": [],
        "summary": "Clear grammar overall.",
    }


def vocabulary_result():
    return {
        "score": 6.5,
        "diversity_score": 6.5,
        "accuracy_score": 6.5,
        "strengths": ["Uses relevant nature vocabulary."],
        "issues": [],
        "summary": "Appropriate vocabulary overall.",
        "upgrades": [],
        "topic_vocabulary": ["tranquil"],
    }


def coach_result():
    return {
        "grammar_score": 6.5,
        "vocabulary_score": 6.5,
        "coherence_score": 6.5,
        "text_based_overall_score": 6.5,
        "evidence_summary": [
            {
                "claim": "The response is clear and relevant.",
                "related_issue_categories": [],
            }
        ],
        "history_comparison": "Stateless web evaluation.",
        "review_plan": {
            "short_term": [],
            "medium_term": [],
            "long_term": [],
        },
        "encouragement": "Keep developing the answer.",
    }


def fake_service(captured_states=None):
    captured_states = captured_states if captured_states is not None else []

    def grammar_node(state):
        captured_states.append(dict(state))
        return {"grammar_result": grammar_result()}

    def vocabulary_node(state):
        return {"vocab_result": vocabulary_result()}

    def coach_node(state):
        return {"coach_result": coach_result()}

    return IELTSEvaluationService(
        grammar_node=grammar_node,
        vocabulary_node=vocabulary_node,
        coach_node=coach_node,
    )


class RaisingService:
    def __init__(self, error):
        self.error = error

    def evaluate(self, request):
        raise self.error


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        app.dependency_overrides.clear()

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_health_does_not_require_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "model_configured": False},
        )

    def test_evaluate_returns_validated_structured_result(self):
        captured_states = []
        app.dependency_overrides[get_evaluation_service] = lambda: fake_service(
            captured_states
        )

        response = self.client.post(
            "/api/v1/evaluate",
            json={"question": QUESTION, "answer": ANSWER},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["text_based_overall_score"], 6.5)
        self.assertEqual(payload["answer_word_count"], len(ANSWER.split()))
        self.assertEqual(payload["score_type"], "text_based")
        self.assertEqual(
            payload["unsupported_dimensions"],
            ["pronunciation", "spoken_fluency"],
        )
        self.assertEqual(captured_states[0]["profile_path"], "")

    def test_empty_answer_returns_422(self):
        response = self.client.post(
            "/api/v1/evaluate",
            json={"question": QUESTION, "answer": "   "},
        )
        self.assertEqual(response.status_code, 422)

    def test_oversized_answer_returns_422(self):
        response = self.client.post(
            "/api/v1/evaluate",
            json={"question": QUESTION, "answer": "a" * 6001},
        )
        self.assertEqual(response.status_code, 422)

    def test_missing_api_key_returns_503_without_exposing_details(self):
        app.dependency_overrides[get_evaluation_service] = lambda: RaisingService(
            MissingAPIKeyError("secret configuration detail")
        )

        response = self.client.post(
            "/api/v1/evaluate",
            json={"question": QUESTION, "answer": ANSWER},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"]["code"],
            "model_not_configured",
        )
        self.assertNotIn("secret configuration detail", response.text)

    def test_model_failure_returns_sanitized_502(self):
        app.dependency_overrides[get_evaluation_service] = lambda: RaisingService(
            RuntimeError("provider response contained sensitive details")
        )

        response = self.client.post(
            "/api/v1/evaluate",
            json={"question": QUESTION, "answer": ANSWER},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"]["code"],
            "evaluation_failed",
        )
        self.assertNotIn("sensitive details", response.text)


class EvaluationServiceTests(unittest.TestCase):
    def test_service_accepts_validated_request_and_remains_stateless(self):
        captured_states = []
        service = fake_service(captured_states)

        result = service.evaluate(
            EvaluateRequest(question=QUESTION, answer=ANSWER)
        )

        self.assertEqual(result.grammar.score, 6.5)
        self.assertEqual(result.vocabulary.score, 6.5)
        self.assertEqual(result.coach.text_based_overall_score, 6.5)
        self.assertEqual(captured_states[0]["profile_path"], "")


if __name__ == "__main__":
    unittest.main()
