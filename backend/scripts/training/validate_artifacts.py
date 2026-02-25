import argparse
import json
import sys
from pathlib import Path
from typing import Any

from backend.scripts.training.config import (
    MAX_TRAIN_EPOCHS,
    TrainingConfig,
    get_frozen_lora_config,
    validate_lora_config,
    validate_training_config,
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path} ({exc})") from exc


def _validate_adapter_files(artifacts_dir: Path) -> list[str]:
    errors: list[str] = []
    adapter_config = artifacts_dir / "adapter_config.json"
    adapter_model_safetensors = artifacts_dir / "adapter_model.safetensors"
    adapter_model_bin = artifacts_dir / "adapter_model.bin"

    if not adapter_config.exists():
        errors.append(f"missing adapter config: {adapter_config}")
    if not adapter_model_safetensors.exists() and not adapter_model_bin.exists():
        errors.append(
            "missing adapter model weights: expected adapter_model.safetensors or adapter_model.bin"
        )
    return errors


def validate_artifacts(artifacts_dir: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    errors.extend(_validate_adapter_files(artifacts_dir))

    frozen_path = artifacts_dir / "frozen_lora_config.json"
    training_path = artifacts_dir / "training_config.json"
    trainer_state_path = artifacts_dir / "trainer_state.json"
    run_metadata_path = artifacts_dir / "run_metadata.json"

    try:
        frozen_lora = _load_json(frozen_path, "frozen lora config")
        expected = get_frozen_lora_config()
        if frozen_lora != expected:
            errors.append("frozen_lora_config.json does not match expected frozen LoRA config")
        validate_lora_config(frozen_lora)
    except Exception as exc:
        errors.append(str(exc))

    try:
        training_payload = _load_json(training_path, "training config")
        config = TrainingConfig(
            fp16=bool(training_payload.get("fp16")),
            gradient_checkpointing=bool(training_payload.get("gradient_checkpointing")),
            num_train_epochs=int(training_payload.get("num_train_epochs")),
        )
        validate_training_config(config)
    except Exception as exc:
        errors.append(f"invalid training config artifact: {exc}")

    try:
        trainer_state = _load_json(trainer_state_path, "trainer state")
        global_step = trainer_state.get("global_step")
        if not isinstance(global_step, int) or global_step <= 0:
            errors.append(
                f"trainer_state.global_step must be positive integer, got {global_step!r}"
            )
        epoch = trainer_state.get("epoch")
        if isinstance(epoch, (int, float)) and epoch > MAX_TRAIN_EPOCHS:
            errors.append(
                f"trainer_state.epoch must be <= {MAX_TRAIN_EPOCHS}, got {epoch}"
            )
        history = trainer_state.get("log_history", [])
        if not isinstance(history, list) or not any(
            isinstance(item, dict) and "eval_loss" in item for item in history
        ):
            errors.append("trainer_state.log_history must include at least one eval_loss record")
    except Exception as exc:
        errors.append(f"invalid trainer state artifact: {exc}")

    try:
        run_metadata = _load_json(run_metadata_path, "run metadata")
        required_keys = {
            "base_model_id",
            "train_rows",
            "eval_rows",
            "train_runtime_seconds",
            "global_step",
            "output_dir",
            "manifest_path",
        }
        missing = sorted(required_keys - set(run_metadata.keys()))
        if missing:
            errors.append(f"run_metadata.json missing keys: {', '.join(missing)}")
    except Exception as exc:
        errors.append(f"invalid run metadata artifact: {exc}")

    return (len(errors) == 0, errors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate LoRA training artifacts against frozen configs"
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/ft-lora-v1",
        help="Path to the LoRA artifacts directory",
    )
    args = parser.parse_args()

    ok, errors = validate_artifacts(Path(args.artifacts_dir))
    if not ok:
        print("Artifact validation FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Artifact validation PASSED")


if __name__ == "__main__":
    main()

