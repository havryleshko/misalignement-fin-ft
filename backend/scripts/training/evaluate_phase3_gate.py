import argparse
import gc
import importlib
import json
from pathlib import Path
from typing import Any

from backend.api.schemas import AnalyzeResponse
from backend.models.llm import parse_llm_response
from backend.scripts.dataset.io import load_rows_jsonl
from backend.scripts.training.evaluation_metrics import evaluate_gate, score_sample, summarize_scores

BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def _require_inference_stack() -> tuple[Any, Any, Any]:
    try:
        peft_mod = importlib.import_module("peft")
        transformers_mod = importlib.import_module("transformers")
        torch_mod = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install project dependencies first "
            "(transformers, peft, torch, accelerate)."
        ) from exc
    return peft_mod, transformers_mod, torch_mod


def _build_prompt(system_content: str, user_content: str, tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|system|>\n{system_content}\n<|user|>\n{user_content}\n<|assistant|>\n"


def _run_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    output_tokens = generated[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(output_tokens, skip_special_tokens=True).strip()


def _parse_prediction(text: str) -> tuple[AnalyzeResponse | None, str | None]:
    try:
        parsed = parse_llm_response(text, trace_id="phase3-eval")
        return parsed, None
    except Exception as exc:
        return None, str(exc)


def _get_runtime_device(torch_mod: Any) -> str:
    if torch_mod.cuda.is_available():
        return "cuda"
    if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(torch_mod: Any, device: str) -> Any:
    return torch_mod.float16 if device in {"cuda", "mps"} else torch_mod.float32


def _load_base_model(
    transformers_mod: Any,
    torch_mod: Any,
    model_id: str,
    device: str,
) -> tuple[Any, Any]:
    tokenizer = transformers_mod.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _dtype_for_device(torch_mod, device)
    model = transformers_mod.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _load_ft_model(
    peft_mod: Any,
    transformers_mod: Any,
    torch_mod: Any,
    model_id: str,
    adapter_id: str,
    device: str,
) -> tuple[Any, Any]:
    tokenizer = transformers_mod.AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _dtype_for_device(torch_mod, device)
    base = transformers_mod.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    base.to(device)
    model = peft_mod.PeftModel.from_pretrained(base, adapter_id)
    model.to(device)
    model.eval()
    return model, tokenizer


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_decision_markdown(
    path: Path,
    gate: dict[str, Any],
    base_summary: dict[str, Any],
    ft_summary: dict[str, Any],
) -> None:
    checks = gate["checks"]
    status = "PASS" if gate["pass"] else "FAIL"
    lines = [
        "# Phase 3 Evaluation Decision",
        "",
        f"- Gate result: `{status}`",
        "",
        "## Acceptance checks",
        f"- schema_validity_improved: `{checks['schema_validity_improved']}`",
        f"- hallucination_reduced: `{checks['hallucination_reduced']}`",
        f"- source_coverage_improved: `{checks['source_coverage_improved']}`",
        f"- confident_wrong_not_increased: `{checks['confident_wrong_not_increased']}`",
        "",
        "## Base summary",
        f"- schema_validity_rate: `{base_summary['schema_validity_rate']}`",
        f"- hallucination_rate: `{base_summary['hallucination_rate']}`",
        f"- source_coverage_correctness_rate: `{base_summary['source_coverage_correctness_rate']}`",
        f"- confident_wrong_rate: `{base_summary['confident_wrong_rate']}`",
        "",
        "## Fine-tuned summary",
        f"- schema_validity_rate: `{ft_summary['schema_validity_rate']}`",
        f"- hallucination_rate: `{ft_summary['hallucination_rate']}`",
        f"- source_coverage_correctness_rate: `{ft_summary['source_coverage_correctness_rate']}`",
        f"- confident_wrong_rate: `{ft_summary['confident_wrong_rate']}`",
        "",
    ]
    if gate["pass"]:
        lines.append("Proceed to Phase 4 deployment planning.")
    else:
        lines.append("Gate failed. Add targeted dataset fixes and retrain before Phase 4.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase3_evaluation(
    eval_path: Path,
    output_dir: Path,
    ft_adapter_id: str,
    base_model_id: str = BASE_MODEL_ID,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    peft_mod, transformers_mod, torch_mod = _require_inference_stack()
    device = _get_runtime_device(torch_mod)

    rows = load_rows_jsonl(eval_path)
    if not rows:
        raise ValueError(f"Eval dataset is empty or missing: {eval_path}")

    base_model, base_tokenizer = _load_base_model(transformers_mod, torch_mod, base_model_id, device)

    base_scores = []
    sample_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt = _build_prompt(row.messages[0].content, row.messages[1].content, base_tokenizer)

        base_raw = _run_generation(base_model, base_tokenizer, prompt, max_new_tokens)
        base_pred, base_error = _parse_prediction(base_raw)
        base_score = score_sample(row, base_pred, base_error)
        base_scores.append(base_score)
        sample_rows.append(
            {
                "index": idx,
                "category": row.category.value,
                "metadata": row.metadata,
                "base": {
                    "schema_valid": base_score.schema_valid,
                    "hallucination_present": base_score.hallucination_present,
                    "bias_amplification": base_score.bias_amplification,
                    "source_coverage_correct": base_score.source_coverage_correct,
                    "fail_closed_correct": base_score.fail_closed_correct,
                    "confident_wrong": base_score.confident_wrong,
                    "parse_error": base_score.parse_error,
                    "raw_output": base_raw,
                },
            }
        )

    del base_model
    gc.collect()
    if device == "mps":
        torch_mod.mps.empty_cache()
    if device == "cuda":
        torch_mod.cuda.empty_cache()

    ft_model, ft_tokenizer = _load_ft_model(
        peft_mod,
        transformers_mod,
        torch_mod,
        base_model_id,
        ft_adapter_id,
        device,
    )
    ft_scores = []
    for idx, row in enumerate(rows):
        prompt = _build_prompt(row.messages[0].content, row.messages[1].content, ft_tokenizer)
        ft_raw = _run_generation(ft_model, ft_tokenizer, prompt, max_new_tokens)
        ft_pred, ft_error = _parse_prediction(ft_raw)
        ft_score = score_sample(row, ft_pred, ft_error)
        ft_scores.append(ft_score)
        sample_rows[idx]["fine_tuned"] = {
            "schema_valid": ft_score.schema_valid,
            "hallucination_present": ft_score.hallucination_present,
            "bias_amplification": ft_score.bias_amplification,
            "source_coverage_correct": ft_score.source_coverage_correct,
            "fail_closed_correct": ft_score.fail_closed_correct,
            "confident_wrong": ft_score.confident_wrong,
            "parse_error": ft_score.parse_error,
            "raw_output": ft_raw,
        }

    base_summary = summarize_scores(base_scores)
    ft_summary = summarize_scores(ft_scores)
    gate = evaluate_gate(base_summary, ft_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "eval_path": str(eval_path),
        "base_model_id": base_model_id,
        "ft_adapter_id": ft_adapter_id,
        "base_summary": base_summary,
        "ft_summary": ft_summary,
        "gate": gate,
    }
    _write_json(output_dir / "phase3_report.json", report)
    _write_jsonl(output_dir / "phase3_samples.jsonl", sample_rows)
    _write_decision_markdown(output_dir / "phase3_decision.md", gate, base_summary, ft_summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 evaluation gate (base vs LoRA)")
    parser.add_argument("--eval", default="data/ft/eval.jsonl", help="Eval split JSONL path")
    parser.add_argument(
        "--output-dir",
        default="artifacts/phase3-eval",
        help="Directory to save phase 3 reports and per-sample audits",
    )
    parser.add_argument(
        "--ft-adapter-id",
        default="havryleshko/ft-lora-v1",
        help="Hugging Face adapter repo id for fine-tuned model",
    )
    parser.add_argument(
        "--base-model-id",
        default=BASE_MODEL_ID,
        help="Base model id used for baseline and adapter loading",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per sample",
    )
    args = parser.parse_args()

    result = run_phase3_evaluation(
        eval_path=Path(args.eval),
        output_dir=Path(args.output_dir),
        ft_adapter_id=args.ft_adapter_id,
        base_model_id=args.base_model_id,
        max_new_tokens=args.max_new_tokens,
    )
    gate_result = "PASS" if result["gate"]["pass"] else "FAIL"
    print(
        "Phase 3 evaluation complete: "
        f"gate={gate_result} output_dir={args.output_dir} "
        f"report={Path(args.output_dir) / 'phase3_report.json'}"
    )


if __name__ == "__main__":
    main()
