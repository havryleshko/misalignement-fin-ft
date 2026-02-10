import json
import time
from datetime import datetime
from typing import Any
import httpx
import redis
from backend.config import load_config
from backend.data.schemas import Filing, FilingsBundle


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
SEC_CACHE_PREFIX = "edgar"


def _get_redis_client() -> redis.Redis:
    config = load_config()
    return redis.from_url(config.redis_url, decode_responses=True)


def _cache_key(ticker: str) -> str:
    return f"{SEC_CACHE_PREFIX}:{ticker.upper()}"


def _cache_get(ticker: str) -> tuple[FilingsBundle, list[str]] | None:
    client = _get_redis_client()
    cached = client.get(_cache_key(ticker))
    if not cached:
        return None
    payload = json.loads(cached)
    bundle = FilingsBundle.model_validate(payload["filings_bundle"])
    citations = payload["citations"]
    return bundle, citations


def _cache_set(ticker: str, bundle: FilingsBundle, citations: list[str]) -> None:
    config = load_config()
    client = _get_redis_client()
    payload = {
        "filings_bundle": bundle.model_dump(mode="json"),
        "citations": citations,
    }
    client.setex(_cache_key(ticker), config.cache_ttl_seconds, json.dumps(payload))


def _http_client() -> httpx.Client:
    config = load_config()
    timeout = httpx.Timeout(10.0)
    headers = {"User-Agent": config.sec_user_agent}
    return httpx.Client(timeout=timeout, headers=headers)


def _fetch_json(url: str) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with _http_client() as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"SEC request failed: {last_exc}")


def _get_cik_for_ticker(ticker: str) -> str | None:
    mapping = _fetch_json(SEC_TICKER_MAP_URL)
    for item in mapping.values():
        if item.get("ticker", "").upper() == ticker.upper():
            cik_int = int(item.get("cik_str", 0))
            return f"{cik_int:010d}"
    return None


def _build_filing_url(cik: str, accession: str, primary_doc: str) -> str:
    acc_no = accession.replace("-", "")
    return f"{SEC_ARCHIVES_BASE}/{int(cik)}/{acc_no}/{primary_doc}"


def get_latest_filings(ticker: str) -> tuple[FilingsBundle, list[str]]:
    cached = _cache_get(ticker)
    if cached:
        return cached

    cik = _get_cik_for_ticker(ticker)
    filings: list[Filing] = []
    citations: list[str] = []

    if cik:
        submissions = _fetch_json(SEC_SUBMISSIONS_URL.format(cik=cik))
        recent = submissions.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])
        filing_dates = recent.get("filingDate", [])

        targets = {"10-K", "10-Q", "8-K"}
        seen: set[str] = set()

        for idx, form in enumerate(forms):
            if form not in targets or form in seen:
                continue
            accession = accession_numbers[idx]
            primary_doc = primary_docs[idx]
            period = report_dates[idx] or filing_dates[idx]
            url = _build_filing_url(cik, accession, primary_doc)
            filings.append(Filing(type=form, period=period, url=url))
            citations.append(f"SEC EDGAR {form} {period} for {ticker.upper()} ({url})")
            seen.add(form)
            if seen == targets:
                break

    bundle = FilingsBundle(
        ticker=ticker.upper(),
        filings=filings,
        as_of=datetime.utcnow(),
    )
    _cache_set(ticker, bundle, citations)
    return bundle, citations
