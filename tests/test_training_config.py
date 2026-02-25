import pytest

from backend.scripts.training.config import (
    MAX_TRAIN_EPOCHS,
    TrainingConfig,
    get_frozen_lora_config,
    validate_lora_config,
    validate_training_config,
)


def test_validate_lora_config_accepts_frozen_config():
    validate_lora_config(get_frozen_lora_config())


def test_validate_lora_config_rejects_modified_value():
    candidate = get_frozen_lora_config()
    candidate["r"] = 16
    with pytest.raises(ValueError, match="must be"):
        validate_lora_config(candidate)


def test_validate_training_config_rejects_epoch_over_cap():
    with pytest.raises(ValueError, match="must be <="):
        validate_training_config(
            TrainingConfig(
                fp16=True,
                gradient_checkpointing=True,
                num_train_epochs=MAX_TRAIN_EPOCHS + 1,
            )
        )

