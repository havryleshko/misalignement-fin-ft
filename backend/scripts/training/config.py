from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping



TARGET_TOTAL_EXAMPLES_MIN = 2000
TARGET_TOTAL_EXAMPLES_MAX = 2600
TARGET_SYNTHETIC_EXAMPLES_MIN = 1200
TARGET_CURATED_EXAMPLES_MIN = 800


FROZEN_LORA_CONFIG: dict[str, Any] = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


REQUIRE_FP16 = True
REQUIRE_GRADIENT_CHECKPOINTING = True
MAX_TRAIN_EPOCHS = 3


@dataclass(frozen=True)
class TrainingConfig:
    fp16: bool = True
    gradient_checkpointing: bool = True
    num_train_epochs: int = 2


def get_frozen_lora_config() -> dict[str, Any]:
    return dict(FROZEN_LORA_CONFIG)


def validate_lora_config(candidate: Mapping[str, Any]) -> None:
    expected_keys = set(FROZEN_LORA_CONFIG.keys())
    candidate_keys = set(candidate.keys())

    if candidate_keys != expected_keys:
        missing = sorted(expected_keys - candidate_keys)
        extra = sorted(candidate_keys - expected_keys)
        raise ValueError(
            "LoRA config keys must match frozen config exactly; "
            f"missing={missing}, extra={extra}"
        )

    for key, expected_value in FROZEN_LORA_CONFIG.items():
        if candidate[key] != expected_value:
            raise ValueError(
                f"LoRA config '{key}' must be {expected_value!r}, got {candidate[key]!r}"
            )


def validate_training_config(config: TrainingConfig) -> None:
    if REQUIRE_FP16 and not config.fp16:
        raise ValueError("Training config must enable fp16.")

    if REQUIRE_GRADIENT_CHECKPOINTING and not config.gradient_checkpointing:
        raise ValueError("Training config must enable gradient checkpointing.")

    if config.num_train_epochs > MAX_TRAIN_EPOCHS:
        raise ValueError(
            f"Training config num_train_epochs must be <= {MAX_TRAIN_EPOCHS}, "
            f"got {config.num_train_epochs}."
        )
