from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    ticker: str
    question: str
    time_horizon: str


class AnalyzeResponse(BaseModel):
    summary: str
    expected_return: float
    confidence_interval: list[float]
    probability_positive: float
    scenarios: dict[str, float]
    risk_flags: list[str]
    bias_notice: str | None
    sources: list[str]
    disclaimer: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    trace_id: str
