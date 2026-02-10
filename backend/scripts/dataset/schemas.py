import json
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.api.schemas import AnalyzeResponse


class DatasetCategory(StrEnum):
    NORMAL_GROUNDED_ANALYSIS = "normal_grounded_analysis"
    HIGH_UNCERTAINTY_SCENARIO = "high_uncertainty_scenario"
    MISSING_PARTIAL_DATA = "missing_partial_data"
    BULLISH_BIAS_NEUTRAL_OUTPUT = "bullish_bias_neutral_output"
    BEARISH_BIAS_NEUTRAL_OUTPUT = "bearish_bias_neutral_output"
    CONFLICTING_DATA_UNCERTAINTY_ESCALATION = "conflicting_data_uncertainty_escalation"


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str = Field(min_length=1)


class DatasetRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    category: DatasetCategory
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_messages_shape(self) -> "DatasetRow":
        validate_message_triplet(self.messages)
        return self


def validate_message_triplet(messages: list[ChatMessage]) -> None:
    if len(messages) != 3:
        raise ValueError("messages must contain exactly 3 entries")
    expected = [ChatRole.SYSTEM, ChatRole.USER, ChatRole.ASSISTANT]
    for idx, role in enumerate(expected):
        if messages[idx].role != role:
            raise ValueError("messages must be ordered as system, user, assistant")


def parse_assistant_response(content: str) -> AnalyzeResponse:
    try:
        payload = json.loads(content)
    except Exception as exc:
        raise ValueError(f"assistant content is not valid JSON: {exc}") from exc
    return AnalyzeResponse.model_validate(payload)


def validate_dataset_row(row: DatasetRow) -> AnalyzeResponse:
    validate_message_triplet(row.messages)
    return parse_assistant_response(row.messages[2].content)
