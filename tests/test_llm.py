from datetime import date, datetime

import pytest

from backend.data.schemas import (
    AnalystConsensus,
    DataBundle,
    Filing,
    FilingsBundle,
    PriceHistory,
    PricePoint,
)
from backend.models.llm import build_prompt, parse_llm_response
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

    assert "Return ONLY valid JSON" in prompt
    assert "\"summary\": string" in prompt
    assert "Sources (must be included in output as-is" in prompt
    assert "- alpha_vantage" in prompt
    assert "- sec_edgar" in prompt
    assert "analyst_consensus" in prompt
    assert "Data gaps" in prompt


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
  "disclaimer": "This output is probabilistic and not advice."
}
```
""".strip()
    response = parse_llm_response(payload, "trace-test")
    assert response.summary == "Example summary."
