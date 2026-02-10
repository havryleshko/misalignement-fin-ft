from datetime import date, datetime

import pytest

from backend.data.schemas import PriceHistory, PricePoint
from backend.orchestration.compliance import build_disclaimer, sanitize_summary
from backend.orchestration.risk import compute_risk_metrics


def _price_history() -> PriceHistory:
    return PriceHistory(
        ticker="AAPL",
        points=[
            PricePoint(
                date=date(2024, 1, 4),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.21,
                volume=100,
            ),
            PricePoint(
                date=date(2024, 1, 3),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.1,
                volume=100,
            ),
            PricePoint(
                date=date(2024, 1, 2),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=100,
            ),
        ],
        as_of=datetime(2024, 1, 4),
    )


def test_compute_risk_metrics_outputs_deterministic_values():
    metrics = compute_risk_metrics(_price_history())
    assert metrics.confidence_interval[0] <= metrics.confidence_interval[1]
    assert metrics.scenarios["bull"] >= metrics.scenarios["base"] >= metrics.scenarios["bear"]
    assert 0.0 <= metrics.probability_positive <= 1.0
    assert metrics.volatility >= 0.0


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ("This will definitely go up.", "This may go up."),
        ("Guaranteed returns ahead.", "may returns ahead."),
    ],
)
def test_sanitize_summary_removes_guarantees(summary, expected):
    assert sanitize_summary(summary) == expected


def test_build_disclaimer_is_machine_readable():
    assert build_disclaimer() == "This output is probabilistic and not investment advice."


def test_sanitize_summary_appends_uncertainty_when_missing():
    summary = "The company has steady revenue growth."
    assert sanitize_summary(summary).endswith("Outcomes are uncertain.")


def test_monte_carlo_risk_metrics_are_reproducible_in_test_env(monkeypatch):
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("RISK_MONTE_CARLO_SIMS", "500")
    first = compute_risk_metrics(_price_history())
    second = compute_risk_metrics(_price_history())
    assert first.confidence_interval == second.confidence_interval
    assert first.scenarios == second.scenarios
