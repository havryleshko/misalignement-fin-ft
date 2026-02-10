from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, conlist, model_validator


class AnalyzeRequest(BaseModel):
    ticker: str = Field(
        min_length=1,
        max_length=10,
        pattern=r"^[A-Z][A-Z0-9\.\-]{0,9}$",
        description="Uppercase ticker symbol, e.g. AAPL or BRK.B",
    )
    question: str = Field(min_length=5, max_length=500)
    time_horizon: str = Field(
        pattern=r"^\d+(d|w|m|y)$",
        description="Duration like 30d, 12m, 2y",
    )


class Scenarios(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bull: float
    base: float
    bear: float


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)
    expected_return: float
    confidence_interval: conlist(float, min_length=2, max_length=2)
    probability_positive: float = Field(ge=0.0, le=1.0)
    scenarios: Scenarios
    risk_flags: list[str]
    bias_notice: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)
    disclaimer: Literal["This output is probabilistic and not investment advice."]

    @model_validator(mode="after")
    def _validate_confidence_interval(self) -> "AnalyzeResponse":
        lower, upper = self.confidence_interval
        if lower > upper:
            raise ValueError("confidence_interval must be ordered [min, max]")
        return self


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    trace_id: str
