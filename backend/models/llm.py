import json
from datetime import datetime
from typing import Any
import httpx
from backend.api.schemas import AnalyzeResponse
from backend.config import Config, load_config
from backend.data.schemas import DataBundle
from backend.orchestration.errors import PipelineError
from backend.orchestration.intent_bias import IntentBiasResult


class LLMClient:
    def generate(self, prompt: str, model_override: str | None = None) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()

    def generate(self, prompt: str, model_override: str | None = None) -> str:
        url = f"{self.config.ollama_host}/api/generate"
        payload = {
            "model": model_override or self.config.llm_model,
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


class TogetherClient(LLMClient):
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        if not self.config.together_api_key:
            raise RuntimeError("TOGETHER_API_KEY is required when LLM_PROVIDER=together")

    def generate(self, prompt: str, model_override: str | None = None) -> str:
        model = model_override or self.config.together_model_lora
        url = f"{self.config.together_base_url.rstrip('/')}/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.llm_max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config.together_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Together response did not include choices")
        text = choices[0].get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Together response text is empty")
        return text


class OpenRouterClient(LLMClient):
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        if not self.config.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")

    def generate(self, prompt: str, model_override: str | None = None) -> str:
        model = model_override or self.config.openrouter_model_lora
        url = f"{self.config.openrouter_base_url.rstrip('/')}/chat/completions"
        response_schema = AnalyzeResponse.model_json_schema()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.llm_max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "AnalyzeResponse",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenRouter response did not include choices")
        message = choices[0].get("message", {})
        text = message.get("content", "")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("OpenRouter response content is empty")
        return text


class HFEndpointClient(LLMClient):
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        if not self.config.hf_api_token:
            raise RuntimeError("HF_API_TOKEN is required when LLM_PROVIDER=hf_endpoint")

    def generate(self, prompt: str, model_override: str | None = None) -> str:
        endpoint_url = model_override or self.config.hf_lora_endpoint_url
        if not endpoint_url:
            raise RuntimeError("HF endpoint URL is required when LLM_PROVIDER=hf_endpoint")

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.config.llm_max_tokens,
                "temperature": self.config.llm_temperature,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        headers = {
            "Authorization": f"Bearer {self.config.hf_api_token}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(endpoint_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return _extract_hf_generated_text(data)


def _extract_hf_generated_text(payload: Any) -> str:
    if isinstance(payload, str) and payload.strip():
        return payload
    if isinstance(payload, dict):
        generated_text = payload.get("generated_text")
        if isinstance(generated_text, str) and generated_text.strip():
            return generated_text
    if isinstance(payload, list) and payload:
        first_item = payload[0]
        if isinstance(first_item, dict):
            generated_text = first_item.get("generated_text")
            if isinstance(generated_text, str) and generated_text.strip():
                return generated_text
        if isinstance(first_item, str) and first_item.strip():
            return first_item
    raise RuntimeError("HF endpoint response did not include generated_text")


def get_llm_client(config: Config | None = None) -> LLMClient:
    resolved = config or load_config()
    provider = resolved.llm_provider.strip().lower()
    if provider == "ollama":
        return OllamaClient(resolved)
    if provider == "together":
        return TogetherClient(resolved)
    if provider == "openrouter":
        return OpenRouterClient(resolved)
    if provider == "hf_endpoint":
        return HFEndpointClient(resolved)
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {resolved.llm_provider}")


def build_prompt(
    bundle: DataBundle, question: str, intent_bias: IntentBiasResult | None
) -> str:
    latest_price = None
    if bundle.price_history and bundle.price_history.points:
        latest_price = bundle.price_history.points[0]

    analyst_consensus = None
    if bundle.analyst_consensus:
        analyst_consensus = {
            "rating": bundle.analyst_consensus.rating,
            "mean_target": bundle.analyst_consensus.mean_target,
            "as_of": bundle.analyst_consensus.as_of.isoformat(),
            "source": bundle.analyst_consensus.source,
        }

    filings_lines: list[str] = []
    if bundle.filings:
        for filing in bundle.filings.filings:
            filings_lines.append(f"- {filing.type} {filing.period} {filing.url}")

    retrieved_market_data = {
        "latest_price_point": latest_price.model_dump(mode="json") if latest_price else None,
        "analyst_consensus": analyst_consensus,
        "data_gaps": bundle.data_gaps,
        "required_sources": bundle.sources,
        "bias_notice": intent_bias.bias_notice if intent_bias else None,
    }
    retrieved_sec_filings = "\n".join(filings_lines) if filings_lines else "None"

    prompt = f"""
SYSTEM:
You are a finance risk analysis engine.
Rules:
- Use only the provided data
- Never hallucinate facts
- Express uncertainty explicitly
- Never provide guarantees
- Output ONLY valid JSON matching the schema

USER:
<context>
retrieved_market_data:
{json.dumps(retrieved_market_data, ensure_ascii=True)}
retrieved_sec_filings:
{retrieved_sec_filings}
</context>

Question:
{question}
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
        disclaimer="This output is probabilistic and not investment advice.",
    )
