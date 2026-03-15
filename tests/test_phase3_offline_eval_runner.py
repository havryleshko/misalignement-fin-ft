import json

from backend.scripts.dataset.schemas import DatasetCategory, DatasetRow
from backend.scripts.training.evaluate_phase3_gate_offline import run_phase3_gate_offline


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


def test_run_phase3_gate_offline_writes_report_and_samples(tmp_path, monkeypatch):
    eval_path = tmp_path / "eval.jsonl"
    row = _row(DatasetCategory.NORMAL_GROUNDED_ANALYSIS)
    eval_path.write_text(json.dumps(row.model_dump(mode="json")) + "\n", encoding="utf-8")

    def fake_call(endpoint_url, hf_api_token, prompt, max_new_tokens, temperature, timeout):
        if "base" in endpoint_url:
            return (
                json.dumps(
                    {
                        "summary": "Base response.",
                        "expected_return": -0.2,
                        "confidence_interval": [-0.22, -0.18],
                        "probability_positive": 0.1,
                        "scenarios": {"bull": 0.01, "base": -0.2, "bear": -0.3},
                        "risk_flags": [],
                        "bias_notice": "No notable prompt framing detected.",
                        "sources": ["alpha_vantage", "random_blog"],
                        "disclaimer": "This output is probabilistic and not investment advice.",
                    }
                ),
                None,
            )
        return (
            json.dumps(
                {
                    "summary": "FT response.",
                    "expected_return": 0.04,
                    "confidence_interval": [-0.02, 0.08],
                    "probability_positive": 0.56,
                    "scenarios": {"bull": 0.12, "base": 0.04, "bear": -0.06},
                    "risk_flags": [],
                    "bias_notice": "No notable prompt framing detected.",
                    "sources": ["alpha_vantage", "sec_edgar"],
                    "disclaimer": "This output is probabilistic and not investment advice.",
                }
            ),
            None,
        )

    monkeypatch.setattr(
        "backend.scripts.training.evaluate_phase3_gate_offline._call_hf_endpoint",
        fake_call,
    )

    output_dir = tmp_path / "artifacts"
    report = run_phase3_gate_offline(
        eval_path=eval_path,
        output_dir=output_dir,
        base_endpoint_url="https://base.endpoint",
        ft_endpoint_url="https://ft.endpoint",
        hf_api_token="hf-test",
        max_new_tokens=128,
        temperature=0.0,
        timeout=30.0,
    )

    assert report["deltas"]["schema_validity_rate_delta"] == 0.0
    assert report["deltas"]["source_coverage_correctness_rate_delta"] == 1.0
    assert report["deltas"]["hallucination_rate_delta"] == -1.0
    assert not report["gate"]["pass"]
    assert (output_dir / "phase3_offline_report.json").exists()
    assert (output_dir / "phase3_offline_samples.jsonl").exists()
