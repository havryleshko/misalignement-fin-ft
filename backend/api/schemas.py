from pydantic import BaseModel, ConfigDict, Field, conlist


class AnalyzeRequest(BaseModel):
    ticker: str
    question: str
    time_horizon: str


class Scenarios(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bull: float
    base: float
    bear: float


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    expected_return: float
    confidence_interval: conlist(float, min_length=2, max_length=2)
    probability_positive: float
    scenarios: Scenarios
    risk_flags: list[str]
    bias_notice: str | None
    sources: list[str] = Field(min_length=1)
    disclaimer: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    trace_id: str
