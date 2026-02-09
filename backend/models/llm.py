import json
from datetime import datetime
from typing import Any
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

    analyst_consensus = None
    if bundle.analyst_consensus:
        analyst_consensus = {
            "rating": bundle.analyst_consensus.rating,
            "mean_target": bundle.analyst_consensus.mean_target,
            "as_of": bundle.analyst_consensus.as_of.isoformat(),
            "source": bundle.analyst_consensus.source,
        }

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
        "\"bias_notice\": string, "
        "\"sources\": [string], "
        "\"disclaimer\": string"
        "}"
    )

    prompt = f"""
You are a finance analysis model. Use ONLY the facts below. Do NOT add outside knowledge.
Return ONLY valid JSON that matches this schema exactly. Do not omit keys. Do not add extra keys.
{schema_hint}

Question:
{question}

Bias notice (if any):
{bias_notice}

Facts:
- Latest price point: {latest_price}
- Filings: {chr(10).join(filings_lines) if filings_lines else "None"}
- Analyst consensus: {json.dumps(analyst_consensus) if analyst_consensus else "None"}
- Data gaps: {", ".join(bundle.data_gaps) if bundle.data_gaps else "None"}

Sources (must be included in output as-is, at least one):
{sources_lines}
""".strip()
    return prompt


def _extract_json_payload(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    start_idx = cleaned.find("{")
    if start_idx == -1:
        raise ValueError("No JSON object found")

    depth = 0
    end_idx = None
    for idx in range(start_idx, len(cleaned)):
        ch = cleaned[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break

    if end_idx is None:
        raise ValueError("Unterminated JSON object")

    payload = cleaned[start_idx : end_idx + 1]
    return json.loads(payload)


def parse_llm_response(text: str, trace_id: str) -> AnalyzeResponse:
    try:
        raw = _extract_json_payload(text)
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
        bias_notice=(
            intent_bias.bias_notice
            if intent_bias and intent_bias.bias_notice
            else "No notable prompt framing detected."
        ),
        sources=bundle.sources,
        disclaimer="This output is probabilistic and not advice.",
    )
