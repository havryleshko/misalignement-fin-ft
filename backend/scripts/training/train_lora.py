import argparse
import importlib
import json
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


def _require_training_stack() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        datasets_mod = importlib.import_module("datasets")
        peft_mod = importlib.import_module("peft")
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
    TrainingArguments = transformers_mod.TrainingArguments
    SFTTrainer = trl_mod.SFTTrainer
    SFTConfig = trl_mod.SFTConfig

    return (
        Dataset,
        LoraConfig,
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
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        SFTConfig,
        SFTTrainer,
    ) = _require_training_stack()

    train_sft, eval_sft = prepare_training_data(train_path, eval_path, manifest_path)

    frozen_lora_config = get_frozen_lora_config()
    validate_lora_config(frozen_lora_config)
    peft_config = LoraConfig(**frozen_lora_config)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
    model.config.use_cache = False

    train_dataset = Dataset.from_list(train_sft)
    eval_dataset = Dataset.from_list(eval_sft)

    output_dir.mkdir(parents=True, exist_ok=True)
    # We use SFTConfig but we'll be careful about which arguments we pass to it
    # vs which we pass to the trainer, as versions of TRL vary.
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
        "eval_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "report_to": "none",
    }
    
    # In some versions, these MUST be in SFTConfig. In others, they are not allowed there.
    # We'll try to put them in the config, and if that fails, we'll put them in the trainer.
    try:
        args = SFTConfig(**args_dict, dataset_text_field="text", max_seq_length=max_seq_length)
        trainer_kwargs = {}
    except TypeError:
        # Fallback for older SFTConfig or TrainingArguments
        args = SFTConfig(**args_dict)
        trainer_kwargs = {
            "dataset_text_field": "text",
            "max_seq_length": max_seq_length,
        }

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        **trainer_kwargs
    )
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

