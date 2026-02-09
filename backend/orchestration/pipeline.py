import logging
import os
import uuid
from dataclasses import dataclass
from pydantic import BaseModel
from backend.api.schemas import AnalyzeRequest, AnalyzeResponse, Scenarios
from backend.data.assembly import assemble_data
from backend.data.schemas import DataBundle
from backend.models.llm import (
    OllamaClient,
    build_prompt,
    deterministic_fallback,
    parse_llm_response,
)
from backend.orchestration.compliance import build_disclaimer, sanitize_summary
from backend.orchestration.errors import PipelineError
from backend.orchestration.intent_bias import IntentBiasResult, detect_bias
from backend.orchestration.risk import RiskMetrics, compute_risk_metrics


class ValidationOutput(BaseModel):
    normalized_question: str
    normalized_ticker: str
    time_horizon: str


class DataRetrievalOutput(BaseModel):
    bundle: DataBundle


class LlmOutput(BaseModel):
    response: AnalyzeResponse


class RiskOutput(BaseModel):
    risk_flags: list[str]
    expected_return: float
    confidence_interval: list[float]
    scenarios: dict[str, float]
    probability_positive: float


class ComplianceOutput(BaseModel):
    disclaimer: str
    summary_override: str


class PipelineResponse(BaseModel):
    summary: str
    expected_return: float
    confidence_interval: list[float]
    probability_positive: float
    scenarios: dict[str, float]
    risk_flags: list[str]
    bias_notice: str | None
    sources: list[str]
    disclaimer: str


@dataclass
class PipelineContext:
    request: AnalyzeRequest
    trace_id: str
    sources: list[str]
    validate: ValidationOutput | None = None
    intent_bias: IntentBiasResult | None = None
    data: DataRetrievalOutput | None = None
    llm: LlmOutput | None = None
    risk: RiskOutput | None = None
    compliance: ComplianceOutput | None = None


logger = logging.getLogger("misalignment")


def _validate_request(context: PipelineContext) -> ValidationOutput:
    req = context.request
    if not req.ticker or not req.question or not req.time_horizon:
        raise PipelineError(
            "INVALID_REQUEST", "Missing required fields", context.trace_id
        )
    return ValidationOutput(
        normalized_question=req.question.strip(),
        normalized_ticker=req.ticker.strip().upper(),
        time_horizon=req.time_horizon.strip(),
    )


def _detect_intent_bias(context: PipelineContext) -> IntentBiasResult:
    if not context.validate:
        raise PipelineError("PIPELINE_STATE", "Validation missing", context.trace_id)
    logger.debug("intent_bias_start", extra={"trace_id": context.trace_id})
    return detect_bias(context.validate.normalized_question)


def _retrieve_data(context: PipelineContext) -> DataRetrievalOutput:
    if not context.validate:
        raise PipelineError("PIPELINE_STATE", "Validation missing", context.trace_id)
    logger.debug(
        "data_retrieval_start",
        extra={"trace_id": context.trace_id, "ticker": context.validate.normalized_ticker},
    )
    bundle = assemble_data(context.validate.normalized_ticker, context.trace_id)
    return DataRetrievalOutput(bundle=bundle)


def _llm_inference(context: PipelineContext) -> LlmOutput:
    if not context.intent_bias:
        raise PipelineError("PIPELINE_STATE", "Intent/bias missing", context.trace_id)
    if not context.data:
        raise PipelineError("PIPELINE_STATE", "Data missing", context.trace_id)

    use_llm = os.getenv("USE_LLM", "true").lower() != "false"
    if context.intent_bias and context.intent_bias.neutralized_question:
        question = context.intent_bias.neutralized_question
    else:
        question = context.validate.normalized_question if context.validate else ""

    if not use_llm or os.getenv("ENV") == "test":
        response = deterministic_fallback(context.data.bundle, context.intent_bias)
        return LlmOutput(response=response)

    prompt = build_prompt(context.data.bundle, question, context.intent_bias)
    client = OllamaClient()
    raw_text = client.generate(prompt)
    response = parse_llm_response(raw_text, context.trace_id)
    return LlmOutput(response=response)


def _risk_estimation(context: PipelineContext) -> RiskOutput:
    if not context.data:
        raise PipelineError("PIPELINE_STATE", "Data missing", context.trace_id)
    bundle = context.data.bundle
    risk_flags: list[str] = []
    if bundle.data_gaps:
        risk_flags.append("data_gaps_present")

    if not bundle.price_history or not bundle.price_history.points:
        raise PipelineError("RISK_INPUT_INVALID", "Missing price history", context.trace_id)

    points = bundle.price_history.points
    if len(points) < 2:
        raise PipelineError("RISK_INPUT_INVALID", "Insufficient price history", context.trace_id)

    metrics: RiskMetrics
    try:
        metrics = compute_risk_metrics(bundle.price_history)
    except ValueError:
        metrics = RiskMetrics(
            expected_return=0.0,
            confidence_interval=[0.0, 0.0],
            scenarios={"bull": 0.0, "base": 0.0, "bear": 0.0},
            probability_positive=0.0,
            volatility=0.0,
            max_drawdown=0.0,
        )
        risk_flags.append("volatility_unavailable")

    if metrics.volatility >= 0.05:
        risk_flags.append("high_volatility")
    if metrics.max_drawdown >= 0.2:
        risk_flags.append("high_drawdown")

    return RiskOutput(
        risk_flags=risk_flags,
        expected_return=metrics.expected_return,
        confidence_interval=metrics.confidence_interval,
        scenarios=metrics.scenarios,
        probability_positive=metrics.probability_positive,
    )


def _compliance_filter(context: PipelineContext) -> ComplianceOutput:
    if not context.llm:
        raise PipelineError("PIPELINE_STATE", "Model output missing", context.trace_id)
    summary = sanitize_summary(context.llm.response.summary)
    return ComplianceOutput(disclaimer=build_disclaimer(), summary_override=summary)


def _assemble_response(context: PipelineContext) -> AnalyzeResponse:
    if not (context.data and context.llm and context.risk and context.compliance):
        raise PipelineError("PIPELINE_STATE", "Missing pipeline outputs", context.trace_id)
    response = context.llm.response.model_copy(deep=True)

    if not response.sources:
        raise PipelineError(
            "MODEL_OUTPUT_INVALID", "Missing sources in model output", context.trace_id
        )

    required_sources = set(context.data.bundle.sources)
    output_sources = set(response.sources)
    if not required_sources.issubset(output_sources):
        raise PipelineError(
            "MODEL_OUTPUT_INVALID", "Model output missing required sources", context.trace_id
        )

    response.expected_return = context.risk.expected_return
    response.confidence_interval = context.risk.confidence_interval
    response.scenarios = Scenarios.model_validate(context.risk.scenarios)
    response.probability_positive = context.risk.probability_positive
    response.risk_flags = list({*response.risk_flags, *context.risk.risk_flags})
    response.disclaimer = context.compliance.disclaimer
    response.summary = context.compliance.summary_override
    response.bias_notice = (
        context.intent_bias.bias_notice
        if context.intent_bias and context.intent_bias.bias_notice
        else "No notable prompt framing detected."
    )

    try:
        return AnalyzeResponse.model_validate(response.model_dump())
    except Exception as exc:
        raise PipelineError(
            "MODEL_OUTPUT_INVALID", f"Final output schema validation failed: {exc}", context.trace_id
        ) from exc


def run_pipeline(request: AnalyzeRequest, trace_id: str | None = None) -> AnalyzeResponse:
    trace = trace_id or str(uuid.uuid4())
    context = PipelineContext(request=request, trace_id=trace, sources=[])

    logger.debug("pipeline_validate", extra={"trace_id": trace})
    context.validate = _validate_request(context)
    logger.debug("pipeline_intent_bias", extra={"trace_id": trace})
    context.intent_bias = _detect_intent_bias(context)
    logger.debug("pipeline_data_retrieval", extra={"trace_id": trace})
    context.data = _retrieve_data(context)
    logger.debug("pipeline_llm_inference", extra={"trace_id": trace})
    context.llm = _llm_inference(context)
    logger.debug("pipeline_risk", extra={"trace_id": trace})
    context.risk = _risk_estimation(context)
    logger.debug("pipeline_compliance", extra={"trace_id": trace})
    context.compliance = _compliance_filter(context)
    context.sources = context.data.bundle.sources
    return _assemble_response(context)
