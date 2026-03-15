from datetime import date, datetime

import pytest

from backend.api.schemas import AnalyzeRequest, AnalyzeResponse
from backend.data.schemas import DataBundle, Filing, FilingsBundle, PriceHistory, PricePoint
from backend.orchestration.errors import PipelineError
from backend.orchestration.pipeline import LlmOutput, run_pipeline
from backend.orchestration.risk import RiskMetrics


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
            ),
            PricePoint(
                date=date(2024, 1, 2),
                open=1.0,
                high=1.1,
                low=0.8,
                close=0.9,
                volume=1200,
            ),
            PricePoint(
                date=date(2024, 1, 1),
                open=1.0,
                high=1.0,
                low=0.9,
                close=1.0,
                volume=1100,
            ),
        ],
        as_of=datetime(2024, 1, 3),
    )
    filings = FilingsBundle(
        ticker="AAPL",
        filings=[Filing(type="10-K", period="2023", url="https://example.com/10k")],
        as_of=datetime(2024, 1, 3),
    )
    return DataBundle(
        price_history=price_history,
        filings=filings,
        analyst_consensus=None,
        sources=["alpha_vantage", "sec_edgar"],
        data_gaps=[],
    )


def _sample_prediction() -> AnalyzeResponse:
    return AnalyzeResponse.model_validate(
        {
            "summary": "Neutral probabilistic view.",
            "expected_return": 0.01,
            "confidence_interval": [-0.03, 0.05],
            "probability_positive": 0.52,
            "scenarios": {"bull": 0.08, "base": 0.01, "bear": -0.05},
            "risk_flags": [],
            "bias_notice": "No notable prompt framing detected.",
            "sources": ["alpha_vantage", "sec_edgar"],
            "disclaimer": "This output is probabilistic and not investment advice.",
        }
    )


def _set_required_env(monkeypatch) -> None:
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("API_KEY_HASH_SALT", "test-salt")
    monkeypatch.setenv("USE_LLM", "true")
    monkeypatch.setenv("ENV", "local")


def test_pipeline_success(monkeypatch):
    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    monkeypatch.setenv("ENV", "test")
    monkeypatch.setattr(
        "backend.orchestration.pipeline.assemble_data", fake_assemble_data
    )

    response = run_pipeline(
        AnalyzeRequest(
            ticker="AAPL",
            question="Is this a good investment over the next 12 months?",
            time_horizon="12m",
        )
    )

    assert response.disclaimer == "This output is probabilistic and not investment advice."
    assert response.sources
    assert response.bias_notice


def test_pipeline_data_unavailable(monkeypatch):
    def fake_assemble_data(_ticker: str, trace_id: str) -> DataBundle:
        raise PipelineError("DATA_UNAVAILABLE", "Data gaps detected", trace_id)

    monkeypatch.setenv("ENV", "test")
    monkeypatch.setattr(
        "backend.orchestration.pipeline.assemble_data", fake_assemble_data
    )

    with pytest.raises(PipelineError) as exc:
        run_pipeline(
            AnalyzeRequest(
                ticker="AAPL",
                question="Is this a good investment over the next 12 months?",
                time_horizon="12m",
            ),
            trace_id="trace-test",
        )

    assert exc.value.error_code == "DATA_UNAVAILABLE"


def test_pipeline_rejects_invalid_final_contract(monkeypatch):
    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    monkeypatch.setenv("ENV", "test")
    monkeypatch.setattr(
        "backend.orchestration.pipeline.assemble_data", fake_assemble_data
    )
    monkeypatch.setattr(
        "backend.orchestration.pipeline.compute_risk_metrics",
        lambda _history: RiskMetrics(
            expected_return=0.01,
            confidence_interval=[0.05, -0.05],
            scenarios={"bull": 0.1, "base": 0.02, "bear": -0.04},
            probability_positive=1.2,
            volatility=0.0,
            max_drawdown=0.0,
        ),
    )

    with pytest.raises(PipelineError) as exc:
        run_pipeline(
            AnalyzeRequest(
                ticker="AAPL",
                question="Is this a good investment over the next 12 months?",
                time_horizon="12m",
            ),
            trace_id="trace-test",
        )

    assert exc.value.error_code == "MODEL_OUTPUT_INVALID"
    assert "Final output schema validation failed" in exc.value.message


def test_pipeline_raises_when_risk_estimation_fails(monkeypatch):
    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    monkeypatch.setenv("ENV", "test")
    monkeypatch.setattr(
        "backend.orchestration.pipeline.assemble_data", fake_assemble_data
    )
    monkeypatch.setattr(
        "backend.orchestration.pipeline.compute_risk_metrics",
        lambda _history: (_ for _ in ()).throw(ValueError("bad risk input")),
    )

    with pytest.raises(PipelineError) as exc:
        run_pipeline(
            AnalyzeRequest(
                ticker="AAPL",
                question="Is this a good investment over the next 12 months?",
                time_horizon="12m",
            ),
            trace_id="trace-test",
        )

    assert exc.value.error_code == "RISK_INPUT_INVALID"


def test_pipeline_routes_together_lora_model_when_enabled(monkeypatch):
    class FakeClient:
        def __init__(self) -> None:
            self.model_override = None

        def generate(self, _prompt: str, model_override: str | None = None) -> str:
            self.model_override = model_override
            return "{}"

    fake_client = FakeClient()

    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    _set_required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "together")
    monkeypatch.setenv("LORA_ENABLED", "true")
    monkeypatch.setenv("TOGETHER_API_KEY", "together-test")
    monkeypatch.setenv("TOGETHER_MODEL_LORA", "llama3-8b-fin-lora-v3")
    monkeypatch.setattr("backend.orchestration.pipeline.assemble_data", fake_assemble_data)
    monkeypatch.setattr("backend.orchestration.pipeline.get_llm_client", lambda _cfg: fake_client)
    monkeypatch.setattr(
        "backend.orchestration.pipeline.parse_llm_response",
        lambda _text, _trace_id: _sample_prediction(),
    )

    run_pipeline(
        AnalyzeRequest(
            ticker="AAPL",
            question="Is this a good investment over the next 12 months?",
            time_horizon="12m",
        )
    )
    assert fake_client.model_override == "llama3-8b-fin-lora-v3"


def test_pipeline_rollback_routes_together_base_model_when_lora_disabled(monkeypatch):
    class FakeClient:
        def __init__(self) -> None:
            self.model_override = None

        def generate(self, _prompt: str, model_override: str | None = None) -> str:
            self.model_override = model_override
            return "{}"

    fake_client = FakeClient()

    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    _set_required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "together")
    monkeypatch.setenv("LORA_ENABLED", "false")
    monkeypatch.setenv("TOGETHER_API_KEY", "together-test")
    monkeypatch.setenv("TOGETHER_MODEL_BASE", "meta-llama/Meta-Llama-3-8B-Instruct")
    monkeypatch.setattr("backend.orchestration.pipeline.assemble_data", fake_assemble_data)
    monkeypatch.setattr("backend.orchestration.pipeline.get_llm_client", lambda _cfg: fake_client)
    monkeypatch.setattr(
        "backend.orchestration.pipeline.parse_llm_response",
        lambda _text, _trace_id: _sample_prediction(),
    )

    run_pipeline(
        AnalyzeRequest(
            ticker="AAPL",
            question="Is this a good investment over the next 12 months?",
            time_horizon="12m",
        )
    )
    assert fake_client.model_override == "meta-llama/Meta-Llama-3-8B-Instruct"


def test_pipeline_routes_openrouter_lora_model_when_enabled(monkeypatch):
    class FakeClient:
        def __init__(self) -> None:
            self.model_override = None

        def generate(self, _prompt: str, model_override: str | None = None) -> str:
            self.model_override = model_override
            return "{}"

    fake_client = FakeClient()

    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    _set_required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LORA_ENABLED", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test")
    monkeypatch.setenv("OPENROUTER_MODEL_LORA", "llama3-8b-fin-lora-v3")
    monkeypatch.setattr("backend.orchestration.pipeline.assemble_data", fake_assemble_data)
    monkeypatch.setattr("backend.orchestration.pipeline.get_llm_client", lambda _cfg: fake_client)
    monkeypatch.setattr(
        "backend.orchestration.pipeline.parse_llm_response",
        lambda _text, _trace_id: _sample_prediction(),
    )

    run_pipeline(
        AnalyzeRequest(
            ticker="AAPL",
            question="Is this a good investment over the next 12 months?",
            time_horizon="12m",
        )
    )
    assert fake_client.model_override == "llama3-8b-fin-lora-v3"


def test_pipeline_rollback_routes_openrouter_base_model_when_lora_disabled(monkeypatch):
    class FakeClient:
        def __init__(self) -> None:
            self.model_override = None

        def generate(self, _prompt: str, model_override: str | None = None) -> str:
            self.model_override = model_override
            return "{}"

    fake_client = FakeClient()

    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    _set_required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LORA_ENABLED", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test")
    monkeypatch.setenv("OPENROUTER_MODEL_BASE", "meta-llama/Meta-Llama-3-8B-Instruct")
    monkeypatch.setattr("backend.orchestration.pipeline.assemble_data", fake_assemble_data)
    monkeypatch.setattr("backend.orchestration.pipeline.get_llm_client", lambda _cfg: fake_client)
    monkeypatch.setattr(
        "backend.orchestration.pipeline.parse_llm_response",
        lambda _text, _trace_id: _sample_prediction(),
    )

    run_pipeline(
        AnalyzeRequest(
            ticker="AAPL",
            question="Is this a good investment over the next 12 months?",
            time_horizon="12m",
        )
    )
    assert fake_client.model_override == "meta-llama/Meta-Llama-3-8B-Instruct"


def test_pipeline_routes_hf_endpoint_lora_url_when_enabled(monkeypatch):
    class FakeClient:
        def __init__(self) -> None:
            self.model_override = None

        def generate(self, _prompt: str, model_override: str | None = None) -> str:
            self.model_override = model_override
            return "{}"

    fake_client = FakeClient()

    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    _set_required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "hf_endpoint")
    monkeypatch.setenv("LORA_ENABLED", "true")
    monkeypatch.setenv(
        "HF_LORA_ENDPOINT_URL",
        "https://ft.us-east-1.aws.endpoints.huggingface.cloud",
    )
    monkeypatch.setattr("backend.orchestration.pipeline.assemble_data", fake_assemble_data)
    monkeypatch.setattr("backend.orchestration.pipeline.get_llm_client", lambda _cfg: fake_client)
    monkeypatch.setattr(
        "backend.orchestration.pipeline.parse_llm_response",
        lambda _text, _trace_id: _sample_prediction(),
    )

    run_pipeline(
        AnalyzeRequest(
            ticker="AAPL",
            question="Is this a good investment over the next 12 months?",
            time_horizon="12m",
        )
    )

    assert (
        fake_client.model_override
        == "https://ft.us-east-1.aws.endpoints.huggingface.cloud"
    )


def test_pipeline_rollback_routes_hf_endpoint_base_url_when_lora_disabled(monkeypatch):
    class FakeClient:
        def __init__(self) -> None:
            self.model_override = None

        def generate(self, _prompt: str, model_override: str | None = None) -> str:
            self.model_override = model_override
            return "{}"

    fake_client = FakeClient()

    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    _set_required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "hf_endpoint")
    monkeypatch.setenv("LORA_ENABLED", "false")
    monkeypatch.setenv(
        "HF_BASE_ENDPOINT_URL",
        "https://base.us-east-1.aws.endpoints.huggingface.cloud",
    )
    monkeypatch.setattr("backend.orchestration.pipeline.assemble_data", fake_assemble_data)
    monkeypatch.setattr("backend.orchestration.pipeline.get_llm_client", lambda _cfg: fake_client)
    monkeypatch.setattr(
        "backend.orchestration.pipeline.parse_llm_response",
        lambda _text, _trace_id: _sample_prediction(),
    )

    run_pipeline(
        AnalyzeRequest(
            ticker="AAPL",
            question="Is this a good investment over the next 12 months?",
            time_horizon="12m",
        )
    )

    assert (
        fake_client.model_override
        == "https://base.us-east-1.aws.endpoints.huggingface.cloud"
    )


def test_pipeline_maps_provider_failures_to_model_provider_error(monkeypatch):
    class FakeClient:
        def generate(self, _prompt: str, model_override: str | None = None) -> str:
            raise RuntimeError(f"upstream failed for {model_override}")

    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    _set_required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "hf_endpoint")
    monkeypatch.setenv("LORA_ENABLED", "true")
    monkeypatch.setenv(
        "HF_LORA_ENDPOINT_URL",
        "https://ft.us-east-1.aws.endpoints.huggingface.cloud",
    )
    monkeypatch.setattr("backend.orchestration.pipeline.assemble_data", fake_assemble_data)
    monkeypatch.setattr(
        "backend.orchestration.pipeline.get_llm_client", lambda _cfg: FakeClient()
    )

    with pytest.raises(PipelineError) as exc:
        run_pipeline(
            AnalyzeRequest(
                ticker="AAPL",
                question="Is this a good investment over the next 12 months?",
                time_horizon="12m",
            ),
            trace_id="trace-test",
        )

    assert exc.value.error_code == "MODEL_PROVIDER_ERROR"
    assert "upstream failed" in exc.value.message


def test_pipeline_rejects_model_output_missing_required_sources(monkeypatch):
    def fake_assemble_data(_ticker: str, _trace_id: str) -> DataBundle:
        return _sample_bundle()

    invalid_prediction = _sample_prediction().model_copy(
        update={"sources": ["alpha_vantage"]}
    )

    _set_required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "hf_endpoint")
    monkeypatch.setenv("LORA_ENABLED", "true")
    monkeypatch.setenv(
        "HF_LORA_ENDPOINT_URL",
        "https://ft.us-east-1.aws.endpoints.huggingface.cloud",
    )
    monkeypatch.setattr("backend.orchestration.pipeline.assemble_data", fake_assemble_data)
    monkeypatch.setattr(
        "backend.orchestration.pipeline.get_llm_client", lambda _cfg: object()
    )
    monkeypatch.setattr(
        "backend.orchestration.pipeline.parse_llm_response",
        lambda _text, _trace_id: invalid_prediction,
    )
    monkeypatch.setattr(
        "backend.orchestration.pipeline._llm_inference",
        lambda _context: LlmOutput(response=invalid_prediction),
    )

    with pytest.raises(PipelineError) as exc:
        run_pipeline(
            AnalyzeRequest(
                ticker="AAPL",
                question="Is this a good investment over the next 12 months?",
                time_horizon="12m",
            ),
            trace_id="trace-test",
        )

    assert exc.value.error_code == "MODEL_OUTPUT_INVALID"
    assert "missing required sources" in exc.value.message.lower()
