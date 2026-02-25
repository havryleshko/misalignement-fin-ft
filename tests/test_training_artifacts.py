import json

from backend.scripts.training.config import get_frozen_lora_config
from backend.scripts.training.validate_artifacts import validate_artifacts


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_artifacts_passes_for_valid_layout(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "adapter_config.json").write_text("{}", encoding="utf-8")
    (artifacts / "adapter_model.safetensors").write_text("", encoding="utf-8")

    _write_json(artifacts / "frozen_lora_config.json", get_frozen_lora_config())
    _write_json(
        artifacts / "training_config.json",
        {"fp16": True, "gradient_checkpointing": True, "num_train_epochs": 2},
    )
    _write_json(
        artifacts / "trainer_state.json",
        {"global_step": 5, "epoch": 2.0, "log_history": [{"eval_loss": 1.2}]},
    )
    _write_json(
        artifacts / "run_metadata.json",
        {
            "base_model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "train_rows": 100,
            "eval_rows": 20,
            "train_runtime_seconds": 1.23,
            "global_step": 5,
            "output_dir": str(artifacts),
            "manifest_path": "data/ft/manifest.json",
        },
    )

    ok, errors = validate_artifacts(artifacts)
    assert ok
    assert errors == []


def test_validate_artifacts_fails_when_adapter_missing(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    _write_json(artifacts / "frozen_lora_config.json", get_frozen_lora_config())
    _write_json(
        artifacts / "training_config.json",
        {"fp16": True, "gradient_checkpointing": True, "num_train_epochs": 2},
    )
    _write_json(
        artifacts / "trainer_state.json",
        {"global_step": 1, "epoch": 1.0, "log_history": [{"eval_loss": 1.0}]},
    )
    _write_json(
        artifacts / "run_metadata.json",
        {
            "base_model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "train_rows": 100,
            "eval_rows": 20,
            "train_runtime_seconds": 1.23,
            "global_step": 5,
            "output_dir": str(artifacts),
            "manifest_path": "data/ft/manifest.json",
        },
    )

    ok, errors = validate_artifacts(artifacts)
    assert not ok
    assert any("missing adapter" in error for error in errors)

