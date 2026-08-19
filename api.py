"""FastAPI entrypoint for the stateless IELTS text-evaluation service."""

import logging

from fastapi import Depends, FastAPI, HTTPException, status

from AILab_AgentIELTSTestPreparation import (
    MissingAPIKeyError,
    model_is_configured,
)
from api_models import EvaluateRequest, EvaluationResponse, HealthResponse
from evaluation_service import IELTSEvaluationService


logger = logging.getLogger(__name__)

app = FastAPI(
    title="IELTS Speaking Agent API",
    description=(
        "Text-only grammar, vocabulary, and coherence evaluation. "
        "Pronunciation and spoken fluency require audio evidence."
    ),
    version="0.1.0",
)


def get_evaluation_service() -> IELTSEvaluationService:
    return IELTSEvaluationService()


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(model_configured=model_is_configured())


@app.post(
    "/api/v1/evaluate",
    response_model=EvaluationResponse,
    tags=["evaluation"],
)
def evaluate(
    request: EvaluateRequest,
    service: IELTSEvaluationService = Depends(get_evaluation_service),
) -> EvaluationResponse:
    try:
        return service.evaluate(request)
    except MissingAPIKeyError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "model_not_configured",
                "message": "The server model API key is not configured.",
            },
        ) from error
    except Exception as error:
        logger.error(
            "IELTS evaluation failed with %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "evaluation_failed",
                "message": "The model evaluation could not be completed.",
            },
        ) from error
