import argparse
import importlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any
from backend.scripts.training.config import (
    TrainingConfig,
    get_frozen_lora_config,
    validate_lora_config,
    validate_training_config,
)
from backend.scripts.training.prepare_data import prepare_training_data
from backend.scripts.training.validate_dataset_gate import validate_dataset_gate

BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def _require_training_stack() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        datasets_mod = importlib.import_module("datasets")
        peft_mod = importlib.import_module("peft")
        torch_mod = importlib.import_module("torch")
        transformers_mod = importlib.import_module("transformers")
        trl_mod = importlib.import_module("trl")
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install project dependencies first "
            "(transformers, datasets, peft, trl, accelerate, torch)."
        ) from exc

    Dataset = datasets_mod.Dataset
    LoraConfig = peft_mod.LoraConfig
    AutoModelForCausalLM = transformers_mod.AutoModelForCausalLM
    AutoTokenizer = transformers_mod.AutoTokenizer
    EarlyStoppingCallback = transformers_mod.EarlyStoppingCallback
    SFTTrainer = trl_mod.SFTTrainer
    SFTConfig = getattr(trl_mod, "SFTConfig", transformers_mod.TrainingArguments)

    return (
        Dataset,
        LoraConfig,
        torch_mod,
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        SFTConfig,
        SFTTrainer,
    )


def _assert_dataset_gate(manifest_path: Path, coverage_report_path: Path | None) -> None:
    ok, errors = validate_dataset_gate(manifest_path, coverage_report_path)
    if not ok:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Dataset gate failed. Training aborted.\n{details}")


def _build_training_config(num_train_epochs: int) -> TrainingConfig:
    config = TrainingConfig(
        fp16=True,
        gradient_checkpointing=True,
        num_train_epochs=num_train_epochs,
    )
    validate_training_config(config)
    return config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _supports_param(callable_obj: Any, param_name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return param_name in signature.parameters


def _resolve_hf_token() -> str | None:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.getenv(env_name, "").strip()
        if token:
            return token
    return None


def _assert_torch_supports_current_gpu(torch_mod: Any) -> None:
    if not torch_mod.cuda.is_available():
        return

    capability = torch_mod.cuda.get_device_capability(0)
    supported_arches = set(torch_mod.cuda.get_arch_list())
    if capability[0] < 12:
        return

    blackwell_arches = {"sm_120", "sm_121", "sm_122"}
    if supported_arches & blackwell_arches:
        return

    device_name = torch_mod.cuda.get_device_name(0)
    supported_list = ", ".join(sorted(supported_arches)) or "unknown"
    raise RuntimeError(
        "Installed PyTorch build does not support the current GPU "
        f"{device_name} (capability={capability}). Supported arches: {supported_list}. "
        "For Blackwell GPUs, install a cu128 nightly build first: "
        "pip install torch torchvision torchaudio --index-url "
        "https://download.pytorch.org/whl/nightly/cu128"
    )


def train_lora(
    train_path: Path,
    eval_path: Path,
    manifest_path: Path,
    coverage_report_path: Path | None,
    output_dir: Path,
    num_train_epochs: int = 2,
    max_seq_length: int = 2048,
) -> dict[str, Any]:
    _assert_dataset_gate(manifest_path, coverage_report_path)
    training_config = _build_training_config(num_train_epochs=num_train_epochs)

    (
        Dataset,
        LoraConfig,
        torch_mod,
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        SFTConfig,
        SFTTrainer,
    ) = _require_training_stack()
    _assert_torch_supports_current_gpu(torch_mod)

    train_sft, eval_sft = prepare_training_data(train_path, eval_path, manifest_path)

    frozen_lora_config = get_frozen_lora_config()
    validate_lora_config(frozen_lora_config)
    peft_config = LoraConfig(**frozen_lora_config)

    hf_token = _resolve_hf_token()
    tokenizer_kwargs: dict[str, Any] = {"use_fast": True}
    model_kwargs: dict[str, Any] = {}
    if hf_token:
        tokenizer_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
    model.config.use_cache = False

    # Keep only text so SFTTrainer won't infer chat templating from "messages".
    train_dataset = Dataset.from_list([{"text": row["text"]} for row in train_sft])
    eval_dataset = Dataset.from_list([{"text": row["text"]} for row in eval_sft])

    output_dir.mkdir(parents=True, exist_ok=True)
    args_init = getattr(SFTConfig, "__init__", SFTConfig)
    trainer_init = getattr(SFTTrainer, "__init__", SFTTrainer)

    args_dict = {
        "output_dir": str(output_dir),
        "num_train_epochs": training_config.num_train_epochs,
        "fp16": training_config.fp16,
        "gradient_checkpointing": training_config.gradient_checkpointing,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "report_to": "none",
    }
    if _supports_param(args_init, "eval_strategy"):
        args_dict["eval_strategy"] = "epoch"
    elif _supports_param(args_init, "evaluation_strategy"):
        args_dict["evaluation_strategy"] = "epoch"

    if _supports_param(args_init, "dataset_text_field"):
        args_dict["dataset_text_field"] = "text"
    if _supports_param(args_init, "max_seq_length"):
        args_dict["max_seq_length"] = max_seq_length

    args = SFTConfig(**args_dict)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=1)],
    }
    if _supports_param(trainer_init, "processing_class"):
        trainer_kwargs["processing_class"] = tokenizer
    elif _supports_param(trainer_init, "tokenizer"):
        trainer_kwargs["tokenizer"] = tokenizer
    if _supports_param(trainer_init, "dataset_text_field"):
        trainer_kwargs["dataset_text_field"] = "text"
    if _supports_param(trainer_init, "max_seq_length"):
        trainer_kwargs["max_seq_length"] = max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)
    start = time.time()
    train_result = trainer.train()
    train_seconds = round(time.time() - start, 2)

    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

    _write_json(output_dir / "frozen_lora_config.json", frozen_lora_config)
    _write_json(
        output_dir / "training_config.json",
        {
            "fp16": training_config.fp16,
            "gradient_checkpointing": training_config.gradient_checkpointing,
            "num_train_epochs": training_config.num_train_epochs,
            "max_seq_length": max_seq_length,
        },
    )
    run_metadata = {
        "base_model_id": BASE_MODEL_ID,
        "train_rows": len(train_sft),
        "eval_rows": len(eval_sft),
        "train_runtime_seconds": train_seconds,
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "training_loss": train_result.training_loss,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
    }
    _write_json(output_dir / "run_metadata.json", run_metadata)
    return run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LoRA training after passing dataset gate checks"
    )
    parser.add_argument("--train", default="data/ft/train.jsonl", help="Train split JSONL path")
    parser.add_argument("--eval", default="data/ft/eval.jsonl", help="Eval split JSONL path")
    parser.add_argument(
        "--manifest",
        default="data/ft/manifest.json",
        help="Manifest path used by dataset gate and data prep",
    )
    parser.add_argument(
        "--coverage-report",
        default="data/ft/coverage_report.json",
        help="Coverage report path used by dataset gate",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ft-lora-v1",
        help="Directory to save adapter artifacts and metadata",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (must satisfy frozen constraints)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max sequence length for SFT training",
    )
    args = parser.parse_args()

    result = train_lora(
        train_path=Path(args.train),
        eval_path=Path(args.eval),
        manifest_path=Path(args.manifest),
        coverage_report_path=Path(args.coverage_report) if args.coverage_report else None,
        output_dir=Path(args.output_dir),
        num_train_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
    )
    print(
        "LoRA training complete: "
        f"global_step={result['global_step']} "
        f"best_metric={result['best_metric']} "
        f"output_dir={result['output_dir']}"
    )


if __name__ == "__main__":
    main()

