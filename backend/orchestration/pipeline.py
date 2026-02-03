import uuid
from dataclasses import dataclass
from pydantic import BaseModel
import os
from backend.api.schemas import AnalyzeRequest, AnalyzeResponse
from backend.data.assembly import assemble_data
from backend.data.schemas import DataBundle
from backend.models.llm import (
    OllamaClient,
    build_prompt,
    deterministic_fallback,
    parse_llm_response,
)
from backend.orchestration.errors import PipelineError
from backend.orchestration.intent_bias import IntentBiasResult, detect_bias


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


class ComplianceOutput(BaseModel):
    disclaimer: str


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
    return detect_bias(context.validate.normalized_question)


def _retrieve_data(context: PipelineContext) -> DataRetrievalOutput:
    if not context.validate:
        raise PipelineError("PIPELINE_STATE", "Validation missing", context.trace_id)
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
    return RiskOutput(risk_flags=[])


def _compliance_filter(context: PipelineContext) -> ComplianceOutput:
    return ComplianceOutput(disclaimer="This output is probabilistic and not advice.")


def _assemble_response(context: PipelineContext) -> AnalyzeResponse:
    if not (context.data and context.llm and context.risk and context.compliance):
        raise PipelineError("PIPELINE_STATE", "Missing pipeline outputs", context.trace_id)
    response = context.llm.response

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

    if context.intent_bias and context.intent_bias.bias_notice:
        if response.bias_notice is None:
            raise PipelineError(
                "MODEL_OUTPUT_INVALID", "Missing bias_notice in model output", context.trace_id
            )

    return response


def run_pipeline(request: AnalyzeRequest, trace_id: str | None = None) -> AnalyzeResponse:
    trace = trace_id or str(uuid.uuid4())
    context = PipelineContext(request=request, trace_id=trace, sources=[])

    context.validate = _validate_request(context)
    context.intent_bias = _detect_intent_bias(context)
    context.data = _retrieve_data(context)
    context.llm = _llm_inference(context)
    context.risk = _risk_estimation(context)
    context.compliance = _compliance_filter(context)
    context.sources = context.data.bundle.sources
    return _assemble_response(context)
