import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from backend.api.schemas import AnalyzeResponse
from backend.scripts.dataset.io import load_rows_jsonl
from backend.scripts.training.evaluation_metrics import evaluate_gate, score_sample, summarize_scores


def _build_prompt(system_content: str, user_content: str) -> str:
    return (
        f"{system_content}\n\n"
        f"{user_content}\n\n"
        "Return ONLY valid JSON matching the required schema."
    )


def _extract_hf_generated_text(payload: Any) -> str:
    if isinstance(payload, str) and payload.strip():
        return payload
    if isinstance(payload, dict):
        generated_text = payload.get("generated_text")
        if isinstance(generated_text, str) and generated_text.strip():
            return generated_text
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            generated_text = first.get("generated_text")
            if isinstance(generated_text, str) and generated_text.strip():
                return generated_text
        if isinstance(first, str) and first.strip():
            return first
    raise ValueError("HF response missing generated_text")


def _extract_json_payload(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    start_idx = cleaned.find("{")
    if start_idx == -1:
        raise ValueError("No JSON object found")

    depth = 0
    end_idx = None
    for idx in range(start_idx, len(cleaned)):
        ch = cleaned[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break

    if end_idx is None:
        raise ValueError("Unterminated JSON object")

    payload = cleaned[start_idx : end_idx + 1]
    return json.loads(payload)


def _parse_prediction(text: str) -> tuple[AnalyzeResponse | None, str | None]:
    try:
        payload = _extract_json_payload(text)
        parsed = AnalyzeResponse.model_validate(payload)
        return parsed, None
    except Exception as exc:
        return None, str(exc)


def _call_hf_endpoint(
    endpoint_url: str,
    hf_api_token: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str | None, str | None]:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": False,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    headers = {
        "Authorization": f"Bearer {hf_api_token}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint_url, json=payload, headers=headers)
        if response.status_code >= 400:
            return None, f"http_{response.status_code}:{response.text}"
        text = _extract_hf_generated_text(response.json())
        return text, None
    except Exception as exc:
        return None, str(exc)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def run_phase3_gate_offline(
    eval_path: Path,
    output_dir: Path,
    base_endpoint_url: str,
    ft_endpoint_url: str,
    hf_api_token: str,
    max_new_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    rows = load_rows_jsonl(eval_path)
    if not rows:
        raise ValueError(f"Eval dataset is empty or missing: {eval_path}")

    base_scores = []
    ft_scores = []
    sample_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        prompt = _build_prompt(row.messages[0].content, row.messages[1].content)
        base_raw, base_transport_error = _call_hf_endpoint(
            endpoint_url=base_endpoint_url,
            hf_api_token=hf_api_token,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if base_transport_error is None and base_raw is not None:
            base_pred, base_parse_error = _parse_prediction(base_raw)
            base_error = base_parse_error
        else:
            base_pred, base_error, base_raw = None, base_transport_error, None
        base_score = score_sample(row, base_pred, base_error)
        base_scores.append(base_score)

        ft_raw, ft_transport_error = _call_hf_endpoint(
            endpoint_url=ft_endpoint_url,
            hf_api_token=hf_api_token,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if ft_transport_error is None and ft_raw is not None:
            ft_pred, ft_parse_error = _parse_prediction(ft_raw)
            ft_error = ft_parse_error
        else:
            ft_pred, ft_error, ft_raw = None, ft_transport_error, None
        ft_score = score_sample(row, ft_pred, ft_error)
        ft_scores.append(ft_score)

        sample_rows.append(
            {
                "index": idx,
                "category": row.category.value,
                "metadata": row.metadata,
                "base": {
                    **asdict(base_score),
                    "raw_output": base_raw,
                },
                "fine_tuned": {
                    **asdict(ft_score),
                    "raw_output": ft_raw,
                },
            }
        )

    base_summary = summarize_scores(base_scores)
    ft_summary = summarize_scores(ft_scores)
    gate = evaluate_gate(base_summary, ft_summary)
    deltas = {
        "schema_validity_rate_delta": round(
            ft_summary["schema_validity_rate"] - base_summary["schema_validity_rate"], 6
        ),
        "hallucination_rate_delta": round(
            ft_summary["hallucination_rate"] - base_summary["hallucination_rate"], 6
        ),
        "source_coverage_correctness_rate_delta": round(
            ft_summary["source_coverage_correctness_rate"]
            - base_summary["source_coverage_correctness_rate"],
            6,
        ),
        "confident_wrong_rate_valid_only_delta": round(
            ft_summary["confident_wrong_rate_valid_only"]
            - base_summary["confident_wrong_rate_valid_only"],
            6,
        ),
    }
    report = {
        "eval_path": str(eval_path),
        "base_endpoint_url": base_endpoint_url,
        "ft_endpoint_url": ft_endpoint_url,
        "base_summary": base_summary,
        "ft_summary": ft_summary,
        "deltas": deltas,
        "gate": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "phase3_offline_report.json", report)
    _write_jsonl(output_dir / "phase3_offline_samples.jsonl", sample_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic offline Phase 3 gate against fixed eval contexts via HF endpoints"
    )
    parser.add_argument("--eval", default="data/ft/eval.jsonl")
    parser.add_argument("--output-dir", default="artifacts/phase3-eval-offline")
    parser.add_argument("--base-endpoint-url", required=True)
    parser.add_argument("--ft-endpoint-url", required=True)
    parser.add_argument("--hf-api-token", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    report = run_phase3_gate_offline(
        eval_path=Path(args.eval),
        output_dir=Path(args.output_dir),
        base_endpoint_url=args.base_endpoint_url,
        ft_endpoint_url=args.ft_endpoint_url,
        hf_api_token=args.hf_api_token,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    gate_result = "PASS" if report["gate"]["pass"] else "FAIL"
    print(
        "Phase 3 offline evaluation complete: "
        f"gate={gate_result} output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
