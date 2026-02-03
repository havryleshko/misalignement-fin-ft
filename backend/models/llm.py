import json
from datetime import datetime
import httpx
from backend.api.schemas import AnalyzeResponse
from backend.config import load_config
from backend.data.schemas import DataBundle
from backend.orchestration.errors import PipelineError
from backend.orchestration.intent_bias import IntentBiasResult


class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self) -> None:
        self.config = load_config()

    def generate(self, prompt: str) -> str:
        url = f"{self.config.ollama_host}/api/generate"
        payload = {
            "model": self.config.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.llm_temperature,
                "num_predict": self.config.llm_max_tokens,
            },
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("response", "")


def build_prompt(
    bundle: DataBundle, question: str, intent_bias: IntentBiasResult | None
) -> str:
    latest_price = None
    if bundle.price_history and bundle.price_history.points:
        latest_price = bundle.price_history.points[0]

    filings_lines: list[str] = []
    if bundle.filings:
        for filing in bundle.filings.filings:
            filings_lines.append(f"- {filing.type} {filing.period} {filing.url}")

    sources_lines = "\n".join(f"- {s}" for s in bundle.sources)
    bias_notice = intent_bias.bias_notice if intent_bias else None

    schema_hint = (
        "{"
        "\"summary\": string, "
        "\"expected_return\": number, "
        "\"confidence_interval\": [number, number], "
        "\"probability_positive\": number, "
        "\"scenarios\": {\"bull\": number, \"base\": number, \"bear\": number}, "
        "\"risk_flags\": [string], "
        "\"bias_notice\": string|null, "
        "\"sources\": [string], "
        "\"disclaimer\": string"
        "}"
    )

    prompt = f"""
You are a finance analysis model. Use ONLY the facts below. Do NOT add outside knowledge.
Return ONLY valid JSON that matches this schema exactly:
{schema_hint}

Question:
{question}

Bias notice (if any):
{bias_notice}

Latest price point:
{latest_price}

Filings:
{chr(10).join(filings_lines) if filings_lines else "None"}

Sources (must be included in output as-is):
{sources_lines}
""".strip()
    return prompt


def parse_llm_response(text: str, trace_id: str) -> AnalyzeResponse:
    try:
        raw = json.loads(text)
    except Exception as exc:
        raise PipelineError(
            "MODEL_OUTPUT_INVALID", f"Invalid JSON: {exc}", trace_id
        ) from exc
    try:
        return AnalyzeResponse.model_validate(raw)
    except Exception as exc:
        raise PipelineError(
            "MODEL_OUTPUT_INVALID", f"Schema validation failed: {exc}", trace_id
        ) from exc


def deterministic_fallback(
    bundle: DataBundle, intent_bias: IntentBiasResult | None
) -> AnalyzeResponse:
    return AnalyzeResponse(
        summary="Deterministic fallback response based on retrieved data.",
        expected_return=0.0,
        confidence_interval=[0.0, 0.0],
        probability_positive=0.0,
        scenarios={"bull": 0.0, "base": 0.0, "bear": 0.0},
        risk_flags=[],
        bias_notice=intent_bias.bias_notice if intent_bias else None,
        sources=bundle.sources,
        disclaimer="This output is probabilistic and not advice.",
    )
