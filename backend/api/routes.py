from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from backend.api.auth import ApiKeyContext, require_api_key
from backend.api.schemas import AnalyzeRequest, AnalyzeResponse, ErrorResponse
from backend.orchestration.errors import PipelineError
from backend.orchestration.metrics import mark_analyze_error
from backend.orchestration.pipeline import run_pipeline


router = APIRouter()


@router.get("/health")
def health_check(_api_key: ApiKeyContext = Depends(require_api_key)) -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def analyze(
    request: Request,
    payload: AnalyzeRequest,
    _api_key: ApiKeyContext = Depends(require_api_key),
) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", None)
    try:
        result = run_pipeline(payload, trace_id=trace_id)
        return JSONResponse(status_code=200, content=result.model_dump())
    except PipelineError as exc:
        mark_analyze_error(exc.error_code)
        error = ErrorResponse(
            error_code=exc.error_code, message=exc.message, trace_id=exc.trace_id
        )
        return JSONResponse(status_code=400, content=error.model_dump())
    except Exception:
        mark_analyze_error("INTERNAL_ERROR")
        error = ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="Internal server error.",
            trace_id=trace_id or "",
        )
        return JSONResponse(status_code=500, content=error.model_dump())
