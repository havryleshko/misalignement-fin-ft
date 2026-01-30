from datetime import datetime, timedelta

from backend.data.alpha_vantage import get_price_history
from backend.data.sec_edgar import get_latest_filings
from backend.data.schemas import DataBundle
from backend.orchestration.errors import PipelineError


FRESHNESS_DAYS = 5


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
            freshness_cutoff = datetime.utcnow().date() - timedelta(days=FRESHNESS_DAYS)
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
        sources.extend(filings_citations)
    except Exception:
        data_gaps.append("filings_unavailable")

    if data_gaps:
        raise PipelineError(
            "DATA_UNAVAILABLE",
            f"Data gaps detected: {', '.join(data_gaps)}",
            trace_id,
        )

    return DataBundle(
        price_history=price_history,
        filings=filings_bundle,
        sources=sources,
        data_gaps=data_gaps,
    )
