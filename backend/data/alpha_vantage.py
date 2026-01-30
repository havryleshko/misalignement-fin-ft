import json
import time
from datetime import datetime
from typing import Any
import httpx
import redis
from backend.config import load_config
from backend.data.schemas import PriceHistory, PricePoint


ALPHA_ENDPOINT = "https://www.alphavantage.co/query"
ALPHA_FUNCTION = "TIME_SERIES_DAILY"
ALPHA_CACHE_PREFIX = "alpha"


def _get_redis_client() -> redis.Redis:
    config = load_config()
    return redis.from_url(config.redis_url, decode_responses=True)


def _cache_key(ticker: str) -> str:
    return f"{ALPHA_CACHE_PREFIX}:{ticker.upper()}"


def _cache_get(ticker: str) -> tuple[PriceHistory, str] | None:
    client = _get_redis_client()
    cached = client.get(_cache_key(ticker))
    if not cached:
        return None
    payload = json.loads(cached)
    history = PriceHistory.model_validate(payload["price_history"])
    citation = payload["citation"]
    return history, citation


def _cache_set(ticker: str, history: PriceHistory, citation: str) -> None:
    config = load_config()
    client = _get_redis_client()
    payload = {
        "price_history": history.model_dump(mode="json"),
        "citation": citation,
    }
    client.setex(_cache_key(ticker), config.cache_ttl_seconds, json.dumps(payload))


def _fetch_alpha_vantage(ticker: str) -> dict[str, Any]:
    config = load_config()
    params = {
        "function": ALPHA_FUNCTION,
        "symbol": ticker,
        "apikey": config.alphavantage_api_key,
    }
    timeout = httpx.Timeout(10.0)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(ALPHA_ENDPOINT, params=params)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Alpha Vantage request failed: {last_exc}")


def _normalize_price_history(ticker: str, payload: dict[str, Any]) -> PriceHistory:
    series = payload.get("Time Series (Daily)", {})
    points: list[PricePoint] = []
    for date_str, values in series.items():
        points.append(
            PricePoint(
                date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                open=float(values.get("1. open", 0.0)),
                high=float(values.get("2. high", 0.0)),
                low=float(values.get("3. low", 0.0)),
                close=float(values.get("4. close", 0.0)),
                volume=int(float(values.get("6. volume", 0.0))),
            )
        )
    points.sort(key=lambda p: p.date, reverse=True)
    return PriceHistory(
        ticker=ticker.upper(),
        points=points,
        as_of=datetime.utcnow(),
    )


def get_price_history(ticker: str) -> tuple[PriceHistory, str]:
    cached = _cache_get(ticker)
    if cached:
        return cached

    payload = _fetch_alpha_vantage(ticker)
    history = _normalize_price_history(ticker, payload)
    citation = (
        f"Alpha Vantage {ALPHA_FUNCTION} for {ticker.upper()} "
        f"(as of {history.as_of.date()})"
    )
    _cache_set(ticker, history, citation)
    return history, citation
