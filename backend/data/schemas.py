from datetime import date, datetime
from pydantic import BaseModel


class PricePoint(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class PriceHistory(BaseModel):
    ticker: str
    points: list[PricePoint]
    as_of: datetime


class Filing(BaseModel):
    type: str
    period: str
    url: str


class FilingsBundle(BaseModel):
    ticker: str
    filings: list[Filing]
    as_of: datetime


class DataBundle(BaseModel):
    price_history: PriceHistory | None
    filings: FilingsBundle | None
    sources: list[str]
    data_gaps: list[str]
