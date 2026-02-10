from datetime import date, datetime

import pytest

from backend.api.schemas import AnalyzeRequest
from backend.data.schemas import DataBundle, Filing, FilingsBundle, PriceHistory, PricePoint
from backend.orchestration.errors import PipelineError
from backend.orchestration.pipeline import run_pipeline
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
