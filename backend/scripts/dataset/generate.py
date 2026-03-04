import argparse
import json
import random
from typing import Any
from backend.api.schemas import AnalyzeResponse, Scenarios
from backend.scripts.dataset.constants import (
    DISCLAIMER_TEXT,
    EVAL_TAG_ADVERSARIAL,
    EVAL_TAG_COMPLIANCE_EDGE,
    EVAL_TAG_SCHEMA_STRESS,
    FROZEN_SYSTEM_PROMPT,
)
from backend.scripts.dataset.io import write_rows_jsonl
from backend.scripts.dataset.schemas import (
    ChatMessage,
    ChatRole,
    DatasetCategory,
    DatasetRow,
)


def _build_user_content(
    ticker: str,
    question: str,
    category: DatasetCategory,
    idx: int,
) -> str:
    market_data: dict[str, Any] = {
        "ticker": ticker,
        "latest_price_point": {
            "date": "2024-01-03",
            "open": 100.0 + idx,
            "high": 103.0 + idx,
            "low": 98.0 + idx,
            "close": 101.0 + idx,
            "volume": 1000000 + (idx * 100),
        },
        "analyst_consensus": {
            "rating": "hold",
            "mean_target": 108.0 + idx,
            "as_of": "2024-01-03T00:00:00Z",
            "source": "consensus_provider",
        },
        "data_gaps": [],
        "required_sources": ["alpha_vantage", "sec_edgar"],
    }
    if category == DatasetCategory.MISSING_PARTIAL_DATA:
        market_data["data_gaps"] = ["analyst_consensus_recent"]
        market_data["analyst_consensus"] = None
    if category == DatasetCategory.CONFLICTING_DATA_UNCERTAINTY_ESCALATION:
        market_data["conflict_notice"] = "filings cautionary tone vs positive price momentum"
    if category == DatasetCategory.HIGH_UNCERTAINTY_BIAS_NEUTRAL_OUTPUT:
        market_data["bias_notice"] = "prompt_contains_both_bullish_and_bearish_framing"
    if category == DatasetCategory.PROCESS_RISK_MANAGEMENT:
        market_data["process_constraints"] = {
            "must_manage_downside": True,
            "avoid_overconfidence": True,
            "escalate_when_data_is_weak": True,
        }

    sec_filings = "- 10-K 2023 https://example.com/10k\n- 10-Q 2024Q1 https://example.com/10q"
    return (
        "<context>\n"
        "retrieved_market_data:\n"
        f"{json.dumps(market_data, ensure_ascii=True)}\n"
        "retrieved_sec_filings:\n"
        f"{sec_filings}\n"
        "</context>\n\n"
        "Question:\n"
        f"{question}"
    )


def _build_assistant_response(
    category: DatasetCategory,
    idx: int,
    rng: random.Random,
) -> AnalyzeResponse:
    base_expected = rng.uniform(-0.03, 0.08)
    lower = base_expected - rng.uniform(0.02, 0.06)
    upper = base_expected + rng.uniform(0.02, 0.06)
    prob_positive = min(max(0.5 + (base_expected * 2), 0.05), 0.95)
    risk_flags: list[str] = []
    bias_notice = "No notable prompt framing detected."
    summary = "Outlook is uncertain and should be treated as probabilistic."

    if category == DatasetCategory.HIGH_UNCERTAINTY_SCENARIO:
        lower -= 0.05
        upper += 0.05
        prob_positive = 0.5
        risk_flags.append("high_uncertainty")
        summary = "High uncertainty: outcomes vary by regime and confidence is limited."
    elif category == DatasetCategory.MISSING_PARTIAL_DATA:
        risk_flags.extend(["data_gaps_present", "high_uncertainty"])
        summary = "Partial data is available, so uncertainty is elevated."
    elif category == DatasetCategory.BULLISH_BIAS_NEUTRAL_OUTPUT:
        bias_notice = "User prompt included potential bias signals: bullish_framing."
        summary = "Despite bullish framing, this is a neutral probabilistic assessment."
    elif category == DatasetCategory.BEARISH_BIAS_NEUTRAL_OUTPUT:
        bias_notice = "User prompt included potential bias signals: bearish_framing."
        summary = "Despite bearish framing, this is a neutral probabilistic assessment."
    elif category == DatasetCategory.CONFLICTING_DATA_UNCERTAINTY_ESCALATION:
        risk_flags.append("conflicting_signals")
        prob_positive = 0.5
        summary = "Conflicting signals increase uncertainty; no deterministic outlook is supported."
    elif category == DatasetCategory.HIGH_UNCERTAINTY_BIAS_NEUTRAL_OUTPUT:
        # Calibrate this slice to avoid high-confidence directional claims.
        base_expected = max(min(base_expected, 0.03), -0.03)
        lower = base_expected - 0.12
        upper = base_expected + 0.12
        prob_positive = 0.5
        risk_flags.extend(["high_uncertainty", "bias_calibrated"])
        summary = "Mixed prompt framing and uncertain evidence require a neutral, low-confidence outlook."
    elif category == DatasetCategory.PROCESS_RISK_MANAGEMENT:
        # Force risk-managed labels: wider interval and balanced probability.
        base_expected = max(min(base_expected, 0.03), -0.03)
        lower = base_expected - 0.12
        upper = base_expected + 0.12
        prob_positive = 0.5
        risk_flags.extend(["high_uncertainty", "process_risk_management"])
        summary = "Risk-managed process favors conservative confidence under uncertainty."

    return AnalyzeResponse(
        summary=summary,
        expected_return=round(base_expected, 4),
        confidence_interval=[round(lower, 4), round(upper, 4)],
        probability_positive=round(prob_positive, 4),
        scenarios=Scenarios(
            bull=round(upper + 0.04, 4),
            base=round(base_expected, 4),
            bear=round(lower - 0.04, 4),
        ),
        risk_flags=sorted(set(risk_flags)),
        bias_notice=bias_notice,
        sources=["alpha_vantage", "sec_edgar"],
        disclaimer=DISCLAIMER_TEXT,
    )


def _default_question(category: DatasetCategory, ticker: str) -> str:
    if category == DatasetCategory.BULLISH_BIAS_NEUTRAL_OUTPUT:
        return f"Everyone says {ticker} will skyrocket. Should I go all in?"
    if category == DatasetCategory.BEARISH_BIAS_NEUTRAL_OUTPUT:
        return f"{ticker} looks terrible. Is it guaranteed to collapse?"
    if category == DatasetCategory.HIGH_UNCERTAINTY_SCENARIO:
        return f"What is the 12m outlook for {ticker} with explicit uncertainty?"
    if category == DatasetCategory.MISSING_PARTIAL_DATA:
        return f"Given incomplete data, what is the cautious forecast for {ticker}?"
    if category == DatasetCategory.CONFLICTING_DATA_UNCERTAINTY_ESCALATION:
        return f"How should I interpret conflicting signals for {ticker} over 12m?"
    if category == DatasetCategory.HIGH_UNCERTAINTY_BIAS_NEUTRAL_OUTPUT:
        return (
            f"I see both very bullish and very bearish takes on {ticker}; "
            "give a neutral high-uncertainty forecast with no directional bias."
        )
    if category == DatasetCategory.PROCESS_RISK_MANAGEMENT:
        return (
            f"For {ticker}, provide a risk-managed 12m process: key scenarios, "
            "guardrails, and explicit uncertainty limits."
        )
    return f"What is a grounded 12m outlook for {ticker}?"


def generate_synthetic_rows(
    count_per_category: int = 50,
    seed: int = 1337,
) -> list[DatasetRow]:
    rng = random.Random(seed)
    categories = list(DatasetCategory)
    rows: list[DatasetRow] = []

    for category in categories:
        for idx in range(count_per_category):
            ticker = "AAPL" if idx % 2 == 0 else "MSFT"
            question = _default_question(category, ticker)
            user_content = _build_user_content(ticker, question, category, idx)
            assistant = _build_assistant_response(category, idx, rng)
            eval_tags: list[str] = []
            if idx % 10 == 0:
                eval_tags.append(EVAL_TAG_ADVERSARIAL)
            if idx % 10 == 1:
                eval_tags.append(EVAL_TAG_COMPLIANCE_EDGE)
            if idx % 10 == 2:
                eval_tags.append(EVAL_TAG_SCHEMA_STRESS)

            row = DatasetRow(
                messages=[
                    ChatMessage(role=ChatRole.SYSTEM, content=FROZEN_SYSTEM_PROMPT),
                    ChatMessage(role=ChatRole.USER, content=user_content),
                    ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content=assistant.model_dump_json(),
                    ),
                ],
                category=category,
                metadata={
                    "source_type": "synthetic",
                    "seed": seed,
                    "sample_index": idx,
                    "ticker": ticker,
                    "eval_tags": eval_tags,
                },
            )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning rows")
    parser.add_argument(
        "--count-per-category",
        type=int,
        default=50,
        help="Number of synthetic rows per required category",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for deterministic output",
    )
    parser.add_argument(
        "--output",
        default="data/ft/synthetic.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    rows = generate_synthetic_rows(
        count_per_category=args.count_per_category,
        seed=args.seed,
    )
    write_rows_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} synthetic rows to {args.output}")


if __name__ == "__main__":
    main()
