from datetime import UTC, datetime, timedelta
import os
from typing import Optional
from backend.data.analyst_consensus import get_analyst_consensus
from backend.data.alpha_vantage import get_price_history
from backend.data.sec_edgar import get_latest_filings
from backend.data.schemas import DataBundle
from backend.orchestration.errors import PipelineError


FRESHNESS_DAYS = 5
FILINGS_FRESHNESS_DAYS = 365


def _parse_filing_date(period: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(period, "%Y-%m-%d").date()
    except Exception:
        return None


def assemble_data(ticker: str, trace_id: str) -> DataBundle:
    data_gaps: list[str] = []
    sources: list[str] = []

    price_history = None
    try:
        price_history, price_citation = get_price_history(ticker)
        if not price_history.points:
            data_gaps.append("price_history_missing")
        else:
            latest = price_history.points[0].date
            freshness_cutoff = datetime.now(UTC).date() - timedelta(days=FRESHNESS_DAYS)
            if latest < freshness_cutoff:
                data_gaps.append("price_history_stale")
        sources.append(price_citation)
    except Exception:
        data_gaps.append("price_history_unavailable")

    filings_bundle = None
    try:
        filings_bundle, filings_citations = get_latest_filings(ticker)
        if not filings_bundle.filings:
            data_gaps.append("filings_missing")
        else:
            filing_dates = [
                _parse_filing_date(filing.period) for filing in filings_bundle.filings
            ]
            parsed_dates = [date for date in filing_dates if date is not None]
            if not parsed_dates:
                data_gaps.append("filings_date_unavailable")
            else:
                latest_filing = max(parsed_dates)
                freshness_cutoff = datetime.now(UTC).date() - timedelta(
                    days=FILINGS_FRESHNESS_DAYS
                )
                if latest_filing < freshness_cutoff:
                    data_gaps.append("filings_stale")
        sources.extend(filings_citations)
    except Exception:
        data_gaps.append("filings_unavailable")

    analyst_consensus = None
    if os.getenv("ANALYST_CONSENSUS_ENABLED", "false").lower() == "true":
        consensus, consensus_citation = get_analyst_consensus(ticker)
        if consensus and consensus_citation:
            analyst_consensus = consensus
            sources.append(consensus_citation)

    sources = [source for source in sources if source]
    if not sources:
        data_gaps.append("sources_missing")

    if data_gaps:
        raise PipelineError(
            "DATA_UNAVAILABLE",
            f"Data gaps detected: {', '.join(data_gaps)}",
            trace_id,
        )

    return DataBundle(
        price_history=price_history,
        filings=filings_bundle,
        analyst_consensus=analyst_consensus,
        sources=sources,
        data_gaps=data_gaps,
    )
