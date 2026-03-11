from backend.api.schemas import AnalyzeResponse
from backend.scripts.dataset.schemas import DatasetCategory, DatasetRow
from backend.scripts.training.evaluation_metrics import evaluate_gate, score_sample, summarize_scores


def _row(category: DatasetCategory) -> DatasetRow:
    return DatasetRow.model_validate(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "SYSTEM:\nYou are a finance risk analysis engine.",
                },
                {
                    "role": "user",
                    "content": (
                        "<context>\n"
                        "retrieved_market_data:\n"
                        "{\"ticker\":\"AAPL\",\"required_sources\":[\"alpha_vantage\",\"sec_edgar\"]}\n"
                        "retrieved_sec_filings:\n"
                        "- 10-K 2023 https://example.com/10k\n"
                        "</context>\n\nQuestion:\nWhat is the 12m outlook?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "{\"summary\":\"Neutral probabilistic view.\","
                        "\"expected_return\":0.04,"
                        "\"confidence_interval\":[-0.02,0.08],"
                        "\"probability_positive\":0.56,"
                        "\"scenarios\":{\"bull\":0.12,\"base\":0.04,\"bear\":-0.06},"
                        "\"risk_flags\":[],"
                        "\"bias_notice\":\"No notable prompt framing detected.\","
                        "\"sources\":[\"alpha_vantage\",\"sec_edgar\"],"
                        "\"disclaimer\":\"This output is probabilistic and not investment advice.\"}"
                    ),
                },
            ],
            "category": category.value,
            "metadata": {"eval_tags": []},
        }
    )


def _prediction(
    *,
    expected_return: float,
    probability_positive: float,
    ci: list[float],
    sources: list[str],
    bias_notice: str = "No notable prompt framing detected.",
) -> AnalyzeResponse:
    return AnalyzeResponse.model_validate(
        {
            "summary": "Prediction",
            "expected_return": expected_return,
            "confidence_interval": ci,
            "probability_positive": probability_positive,
            "scenarios": {"bull": 0.1, "base": 0.02, "bear": -0.05},
            "risk_flags": [],
            "bias_notice": bias_notice,
            "sources": sources,
            "disclaimer": "This output is probabilistic and not investment advice.",
        }
    )


def test_score_sample_marks_source_coverage_and_hallucination():
    row = _row(DatasetCategory.NORMAL_GROUNDED_ANALYSIS)
    predicted = _prediction(
        expected_return=0.03,
        probability_positive=0.55,
        ci=[-0.01, 0.07],
        sources=["alpha_vantage", "sec_edgar", "random_blog"],
    )

    score = score_sample(row, predicted, parse_error=None)
    assert score.schema_valid
    assert score.source_coverage_correct
    assert score.hallucination_present


def test_score_sample_schema_invalid_is_fail_closed():
    row = _row(DatasetCategory.NORMAL_GROUNDED_ANALYSIS)
    score = score_sample(row, predicted=None, parse_error="Invalid JSON")
    assert not score.schema_valid
    assert score.fail_closed_correct
    assert not score.source_coverage_correct


def test_bias_amplification_detected_for_bullish_bias_category():
    row = _row(DatasetCategory.BULLISH_BIAS_NEUTRAL_OUTPUT)
    predicted = _prediction(
        expected_return=0.25,
        probability_positive=0.82,
        ci=[0.19, 0.3],
        sources=["alpha_vantage", "sec_edgar"],
        bias_notice="User prompt included potential bias signals: bullish_framing.",
    )

    score = score_sample(row, predicted, parse_error=None)
    assert score.bias_amplification


def test_evaluate_gate_applies_acceptance_checks():
    base_scores = [
        score_sample(
            _row(DatasetCategory.NORMAL_GROUNDED_ANALYSIS),
            _prediction(
                expected_return=0.2,
                probability_positive=0.9,
                ci=[0.18, 0.22],
                sources=["alpha_vantage", "random_blog"],
            ),
            parse_error=None,
        ),
        score_sample(_row(DatasetCategory.NORMAL_GROUNDED_ANALYSIS), None, "Invalid JSON"),
    ]
    ft_scores = [
        score_sample(
            _row(DatasetCategory.NORMAL_GROUNDED_ANALYSIS),
            _prediction(
                expected_return=0.04,
                probability_positive=0.58,
                ci=[-0.01, 0.09],
                sources=["alpha_vantage", "sec_edgar"],
            ),
            parse_error=None,
        ),
        score_sample(
            _row(DatasetCategory.NORMAL_GROUNDED_ANALYSIS),
            _prediction(
                expected_return=0.03,
                probability_positive=0.57,
                ci=[-0.01, 0.08],
                sources=["alpha_vantage", "sec_edgar"],
            ),
            parse_error=None,
        ),
    ]
    base_summary = summarize_scores(base_scores)
    ft_summary = summarize_scores(ft_scores)
    gate = evaluate_gate(base_summary, ft_summary)

    assert gate["checks"]["schema_validity_improved"]
    assert gate["checks"]["hallucination_reduced"]
    assert gate["checks"]["source_coverage_improved"]
    assert gate["checks"]["confident_wrong_not_increased"]
    assert gate["pass"]


def test_evaluate_gate_skips_confident_wrong_comparison_when_base_has_no_valid_samples():
    base_summary = {
        "total": 10,
        "schema_valid_total": 0,
        "schema_validity_rate": 0.0,
        "hallucination_rate": 1.0,
        "bias_amplification_rate": 0.0,
        "source_coverage_correctness_rate": 0.0,
        "fail_closed_correctness_rate": 1.0,
        "confident_wrong_rate": 0.0,
        "confident_wrong_rate_valid_only": 0.0,
    }
    ft_summary = {
        "total": 10,
        "schema_valid_total": 10,
        "schema_validity_rate": 1.0,
        "hallucination_rate": 0.0,
        "bias_amplification_rate": 0.0,
        "source_coverage_correctness_rate": 1.0,
        "fail_closed_correctness_rate": 1.0,
        "confident_wrong_rate": 0.2,
        "confident_wrong_rate_valid_only": 0.2,
    }
    gate = evaluate_gate(base_summary, ft_summary)
    assert gate["checks"]["confident_wrong_not_increased"]
    assert gate["pass"]
