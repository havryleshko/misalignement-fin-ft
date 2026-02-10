from backend.api.schemas import AnalyzeResponse, Scenarios
from backend.orchestration.metrics import (
    mark_analyze_attempt,
    mark_analyze_error,
    mark_analyze_success,
    record_external_signal,
    record_response_quality,
    reset_for_tests,
    snapshot,
)


def test_metrics_capture_success_and_time_to_first_success():
    reset_for_tests()
    mark_analyze_attempt(api_key_id=7, now_epoch=100.0)
    mark_analyze_success(api_key_id=7, now_epoch=104.5)
    state = snapshot()
    assert state["analyze_requests"] == 1
    assert state["analyze_successes"] == 1
    assert state["time_to_first_successful_analyze_seconds"] == [4.5]


def test_metrics_capture_quality_and_external_signals():
    reset_for_tests()
    response = AnalyzeResponse(
        summary="Outcomes are uncertain and may vary by market regime.",
        expected_return=0.01,
        confidence_interval=[-0.05, 0.08],
        probability_positive=0.55,
        scenarios=Scenarios(bull=0.12, base=0.01, bear=-0.07),
        risk_flags=["high_volatility"],
        bias_notice="No notable prompt framing detected.",
        sources=["sec", "alpha"],
        disclaimer="This output is probabilistic and not investment advice.",
    )
    record_response_quality(response, required_sources={"sec", "alpha"})
    mark_analyze_error("DATA_UNAVAILABLE")
    record_external_signal("integration_refusal_rate", 0.2)
    state = snapshot()
    assert state["uncertainty_conformant_responses"] == 1
    assert state["source_coverage_conformant_responses"] == 1
    assert state["error_codes"]["DATA_UNAVAILABLE"] == 1
    assert state["external_signals"]["integration_refusal_rate"] == [0.2]
