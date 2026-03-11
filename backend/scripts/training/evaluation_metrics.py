import json
from dataclasses import dataclass
from statistics import mean
from typing import Any
from backend.api.schemas import AnalyzeResponse
from backend.scripts.dataset.schemas import DatasetCategory, DatasetRow


@dataclass(frozen=True)
class SampleScore:
    schema_valid: bool
    hallucination_present: bool
    bias_amplification: bool
    source_coverage_correct: bool
    fail_closed_correct: bool
    confident_wrong: bool
    parse_error: str | None = None


def _extract_required_sources(user_content: str) -> set[str]:
    marker = "retrieved_market_data:"
    marker_idx = user_content.find(marker)
    if marker_idx == -1:
        return set()

    open_idx = user_content.find("{", marker_idx)
    if open_idx == -1:
        return set()

    depth = 0
    close_idx = None
    for idx in range(open_idx, len(user_content)):
        ch = user_content[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close_idx = idx
                break

    if close_idx is None:
        return set()

    payload = user_content[open_idx : close_idx + 1]
    try:
        market_data = json.loads(payload)
    except json.JSONDecodeError:
        return set()

    required_sources = market_data.get("required_sources", [])
    if not isinstance(required_sources, list):
        return set()
    return {str(source) for source in required_sources}


def _source_coverage_correct(predicted: AnalyzeResponse, required_sources: set[str]) -> bool:
    if not required_sources:
        return True
    predicted_sources = {source for source in predicted.sources}
    return required_sources.issubset(predicted_sources)


def _hallucination_present(predicted: AnalyzeResponse, required_sources: set[str]) -> bool:
    if not required_sources:
        return False
    predicted_sources = {source for source in predicted.sources}
    return len(predicted_sources - required_sources) > 0


def _bias_amplification(
    predicted: AnalyzeResponse,
    expected: AnalyzeResponse,
    category: DatasetCategory,
) -> bool:
    if category == DatasetCategory.BULLISH_BIAS_NEUTRAL_OUTPUT:
        return (
            predicted.expected_return > expected.expected_return
            and predicted.probability_positive >= max(0.7, expected.probability_positive + 0.1)
        )
    if category == DatasetCategory.BEARISH_BIAS_NEUTRAL_OUTPUT:
        return (
            predicted.expected_return < expected.expected_return
            and predicted.probability_positive <= min(0.3, expected.probability_positive - 0.1)
        )
    return False


def _confident_wrong(predicted: AnalyzeResponse, expected: AnalyzeResponse) -> bool:
    ci_width = predicted.confidence_interval[1] - predicted.confidence_interval[0]
    high_confidence = (
        ci_width <= 0.08
        or predicted.probability_positive >= 0.75
        or predicted.probability_positive <= 0.25
    )
    wrong_direction = (predicted.expected_return >= 0) != (expected.expected_return >= 0)
    large_error = abs(predicted.expected_return - expected.expected_return) >= 0.15
    return high_confidence and (wrong_direction or large_error)


def score_sample(
    row: DatasetRow,
    predicted: AnalyzeResponse | None,
    parse_error: str | None,
) -> SampleScore:
    schema_valid = predicted is not None and parse_error is None
    required_sources = _extract_required_sources(row.messages[1].content)

    if not schema_valid or predicted is None:
        return SampleScore(
            schema_valid=False,
            hallucination_present=True,
            bias_amplification=False,
            source_coverage_correct=False,
            fail_closed_correct=parse_error is not None,
            confident_wrong=False,
            parse_error=parse_error,
        )

    expected = AnalyzeResponse.model_validate_json(row.messages[2].content)
    return SampleScore(
        schema_valid=True,
        hallucination_present=_hallucination_present(predicted, required_sources),
        bias_amplification=_bias_amplification(predicted, expected, row.category),
        source_coverage_correct=_source_coverage_correct(predicted, required_sources),
        fail_closed_correct=True,
        confident_wrong=_confident_wrong(predicted, expected),
        parse_error=None,
    )


def summarize_scores(scores: list[SampleScore]) -> dict[str, Any]:
    total = len(scores)
    if total == 0:
        return {
            "total": 0,
            "schema_valid_total": 0,
            "schema_validity_rate": 0.0,
            "hallucination_rate": 0.0,
            "bias_amplification_rate": 0.0,
            "source_coverage_correctness_rate": 0.0,
            "fail_closed_correctness_rate": 0.0,
            "confident_wrong_rate": 0.0,
            "confident_wrong_rate_valid_only": 0.0,
        }

    schema_valid_total = sum(1 for s in scores if s.schema_valid)
    valid_scores = [s for s in scores if s.schema_valid]
    return {
        "total": total,
        "schema_valid_total": schema_valid_total,
        "schema_validity_rate": round(mean(1.0 if s.schema_valid else 0.0 for s in scores), 6),
        "hallucination_rate": round(
            mean(1.0 if s.hallucination_present else 0.0 for s in scores), 6
        ),
        "bias_amplification_rate": round(
            mean(1.0 if s.bias_amplification else 0.0 for s in scores), 6
        ),
        "source_coverage_correctness_rate": round(
            mean(1.0 if s.source_coverage_correct else 0.0 for s in scores), 6
        ),
        "fail_closed_correctness_rate": round(
            mean(1.0 if s.fail_closed_correct else 0.0 for s in scores), 6
        ),
        "confident_wrong_rate": round(
            mean(1.0 if s.confident_wrong else 0.0 for s in scores), 6
        ),
        "confident_wrong_rate_valid_only": round(
            mean(1.0 if s.confident_wrong else 0.0 for s in valid_scores), 6
        )
        if valid_scores
        else 0.0,
    }


def evaluate_gate(base_summary: dict[str, Any], ft_summary: dict[str, Any]) -> dict[str, Any]:
    base_valid_total = int(base_summary.get("schema_valid_total", 0))
    ft_confident_wrong_rate = float(
        ft_summary.get(
            "confident_wrong_rate_valid_only",
            ft_summary.get("confident_wrong_rate", 0.0),
        )
    )
    if base_valid_total == 0:
        confident_wrong_not_increased = True
    else:
        base_confident_wrong_rate = float(
            base_summary.get(
                "confident_wrong_rate_valid_only",
                base_summary.get("confident_wrong_rate", 0.0),
            )
        )
        confident_wrong_not_increased = ft_confident_wrong_rate <= base_confident_wrong_rate

    checks = {
        "schema_validity_improved": ft_summary["schema_validity_rate"]
        > base_summary["schema_validity_rate"],
        "hallucination_reduced": ft_summary["hallucination_rate"]
        < base_summary["hallucination_rate"],
        "source_coverage_improved": ft_summary["source_coverage_correctness_rate"]
        > base_summary["source_coverage_correctness_rate"],
        "confident_wrong_not_increased": confident_wrong_not_increased,
    }
    return {"pass": all(checks.values()), "checks": checks}
