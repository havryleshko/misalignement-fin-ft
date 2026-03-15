from datetime import date, datetime

import pytest

from backend.config import Config
from backend.data.schemas import (
    AnalystConsensus,
    DataBundle,
    Filing,
    FilingsBundle,
    PriceHistory,
    PricePoint,
)
from backend.models.llm import HFEndpointClient, build_prompt, get_llm_client, parse_llm_response
from backend.orchestration.errors import PipelineError
from backend.orchestration.intent_bias import IntentBiasResult


def _sample_bundle() -> DataBundle:
    price_history = PriceHistory(
        ticker="AAPL",
        points=[
            PricePoint(
                date=date(2024, 1, 3),
                open=1.0,
                high=1.2,
                low=0.9,
                close=1.1,
                volume=1000,
            )
        ],
        as_of=datetime(2024, 1, 3),
    )
    filings = FilingsBundle(
        ticker="AAPL",
        filings=[Filing(type="10-K", period="2023", url="https://example.com/10k")],
        as_of=datetime(2024, 1, 3),
    )
    consensus = AnalystConsensus(
        rating="buy",
        mean_target=150.0,
        as_of=datetime(2024, 1, 3),
        source="consensus_provider",
    )
    return DataBundle(
        price_history=price_history,
        filings=filings,
        analyst_consensus=consensus,
        sources=["alpha_vantage", "sec_edgar"],
        data_gaps=["analyst_consensus_recent"],
    )


def test_build_prompt_includes_schema_and_sources():
    bundle = _sample_bundle()
    intent_bias = IntentBiasResult(
        bias_flags=["bullish_framing"],
        bias_notice="User prompt included potential bias signals: bullish_framing.",
        neutralized_question="Provide an objective, data-grounded assessment.",
    )
    prompt = build_prompt(bundle, "Is this a good investment?", intent_bias)

    assert "SYSTEM:" in prompt
    assert "You are a finance risk analysis engine." in prompt
    assert "Rules:" in prompt
    assert "- Use only the provided data" in prompt
    assert "- Never hallucinate facts" in prompt
    assert "- Express uncertainty explicitly" in prompt
    assert "- Never provide guarantees" in prompt
    assert "- Output ONLY valid JSON matching the schema" in prompt
    assert "USER:" in prompt
    assert "<context>" in prompt
    assert "retrieved_market_data:" in prompt
    assert "retrieved_sec_filings:" in prompt
    assert "required_sources" in prompt
    assert "alpha_vantage" in prompt
    assert "sec_edgar" in prompt
    assert "Question:" in prompt


def test_parse_llm_response_rejects_invalid_json():
    with pytest.raises(PipelineError) as exc:
        parse_llm_response("not json", "trace-test")

    assert exc.value.error_code == "MODEL_OUTPUT_INVALID"
    assert "Invalid JSON" in exc.value.message


def test_parse_llm_response_rejects_schema_mismatch():
    invalid_payload = '{"summary": "only summary"}'
    with pytest.raises(PipelineError) as exc:
        parse_llm_response(invalid_payload, "trace-test")

    assert exc.value.error_code == "MODEL_OUTPUT_INVALID"
    assert "Schema validation failed" in exc.value.message


def test_parse_llm_response_rejects_null_bias_notice():
    invalid_payload = """
{
  "summary": "Example summary.",
  "expected_return": 0.05,
  "confidence_interval": [0.02, 0.08],
  "probability_positive": 0.6,
  "scenarios": {"bull": 0.12, "base": 0.05, "bear": -0.03},
  "risk_flags": ["high_volatility"],
  "bias_notice": null,
  "sources": ["alpha_vantage"],
  "disclaimer": "This output is probabilistic and not investment advice."
}
""".strip()
    with pytest.raises(PipelineError) as exc:
        parse_llm_response(invalid_payload, "trace-test")

    assert exc.value.error_code == "MODEL_OUTPUT_INVALID"
    assert "Schema validation failed" in exc.value.message


def test_parse_llm_response_rejects_disclaimer_mismatch():
    invalid_payload = """
{
  "summary": "Example summary.",
  "expected_return": 0.05,
  "confidence_interval": [0.02, 0.08],
  "probability_positive": 0.6,
  "scenarios": {"bull": 0.12, "base": 0.05, "bear": -0.03},
  "risk_flags": ["high_volatility"],
  "bias_notice": "No notable prompt framing detected.",
  "sources": ["alpha_vantage"],
  "disclaimer": "This output is probabilistic and not advice."
}
""".strip()
    with pytest.raises(PipelineError) as exc:
        parse_llm_response(invalid_payload, "trace-test")

    assert exc.value.error_code == "MODEL_OUTPUT_INVALID"
    assert "Schema validation failed" in exc.value.message


def test_parse_llm_response_accepts_code_fenced_json():
    payload = """
```json
{
  "summary": "Example summary.",
  "expected_return": 0.05,
  "confidence_interval": [0.02, 0.08],
  "probability_positive": 0.6,
  "scenarios": {"bull": 0.12, "base": 0.05, "bear": -0.03},
  "risk_flags": ["high_volatility"],
  "bias_notice": "No notable prompt framing detected.",
  "sources": ["alpha_vantage"],
  "disclaimer": "This output is probabilistic and not investment advice."
}
```
""".strip()
    response = parse_llm_response(payload, "trace-test")
    assert response.summary == "Example summary."


def _sample_config(**overrides) -> Config:
    values = {
        "alphavantage_api_key": "test-key",
        "database_url": "postgresql://test:test@localhost:5432/test",
        "redis_url": "redis://localhost:6379/0",
        "api_key_hash_salt": "test-salt",
        "llm_provider": "hf_endpoint",
        "hf_api_token": "hf-test-token",
        "hf_base_endpoint_url": "https://base.example.com",
        "hf_lora_endpoint_url": "https://ft.example.com",
    }
    values.update(overrides)
    return Config(**values)


def test_get_llm_client_supports_hf_endpoint():
    client = get_llm_client(_sample_config())
    assert isinstance(client, HFEndpointClient)


def test_hf_endpoint_client_uses_endpoint_override_and_extracts_generated_text(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [{"generated_text": "{\"summary\":\"ok\"}"}]

    class FakeHttpxClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict[str, object], headers: dict[str, str]):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr("backend.models.llm.httpx.Client", FakeHttpxClient)

    client = HFEndpointClient(_sample_config())
    response_text = client.generate(
        "Prompt text", model_override="https://override.example.com"
    )

    assert response_text == "{\"summary\":\"ok\"}"
    assert captured["url"] == "https://override.example.com"
    assert captured["timeout"] == 120.0
    assert captured["headers"] == {
        "Authorization": "Bearer hf-test-token",
        "Content-Type": "application/json",
    }
    assert captured["json"] == {
        "inputs": "Prompt text",
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.2,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
