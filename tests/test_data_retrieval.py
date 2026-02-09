from datetime import UTC, date, datetime

import pytest

from backend.data.assembly import assemble_data
from backend.data.schemas import Filing, FilingsBundle, PriceHistory, PricePoint
from backend.orchestration.errors import PipelineError


def _price_history(latest_date: date) -> PriceHistory:
    return PriceHistory(
        ticker="AAPL",
        points=[
            PricePoint(
                date=latest_date,
                open=1.0,
                high=1.2,
                low=0.9,
                close=1.1,
                volume=1000,
            ),
            PricePoint(
                date=date(2024, 1, 1),
                open=1.0,
                high=1.1,
                low=0.8,
                close=0.9,
                volume=1200,
            ),
        ],
        as_of=datetime.now(UTC),
    )


def _filings(period: str) -> FilingsBundle:
    return FilingsBundle(
        ticker="AAPL",
        filings=[Filing(type="10-K", period=period, url="https://example.com/10k")],
        as_of=datetime.now(UTC),
    )


def test_filings_stale_triggers_error(monkeypatch):
    def fake_price(_ticker: str):
        return _price_history(date.today()), "alpha"

    def fake_filings(_ticker: str):
        return _filings("2020-01-01"), ["sec"]

    monkeypatch.setattr("backend.data.assembly.get_price_history", fake_price)
    monkeypatch.setattr("backend.data.assembly.get_latest_filings", fake_filings)

    with pytest.raises(PipelineError) as exc:
        assemble_data("AAPL", "trace-test")

    assert exc.value.error_code == "DATA_UNAVAILABLE"
    assert "filings_stale" in exc.value.message


def test_sources_missing_triggers_error(monkeypatch):
    def fake_price(_ticker: str):
        return _price_history(date.today()), ""

    def fake_filings(_ticker: str):
        return _filings(date.today().isoformat()), []

    monkeypatch.setattr("backend.data.assembly.get_price_history", fake_price)
    monkeypatch.setattr("backend.data.assembly.get_latest_filings", fake_filings)

    with pytest.raises(PipelineError) as exc:
        assemble_data("AAPL", "trace-test")

    assert "sources_missing" in exc.value.message


def test_assemble_data_success(monkeypatch):
    def fake_price(_ticker: str):
        return _price_history(date.today()), "alpha"

    def fake_filings(_ticker: str):
        return _filings(date.today().isoformat()), ["sec"]

    monkeypatch.setattr("backend.data.assembly.get_price_history", fake_price)
    monkeypatch.setattr("backend.data.assembly.get_latest_filings", fake_filings)

    bundle = assemble_data("AAPL", "trace-test")
    assert bundle.sources == ["alpha", "sec"]
