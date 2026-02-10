import os
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from backend.api.auth import ApiKeyContext
from backend.models.entities import ApiKey, Customer, UsageLog


def test_customers_minimal_columns_present():
    columns = set(Customer.__table__.columns.keys())
    assert {"id", "name", "plan"}.issubset(columns)


def test_api_keys_minimal_columns_present():
    columns = set(ApiKey.__table__.columns.keys())
    assert {"id", "key_hash", "customer_id", "rate_limit", "status"}.issubset(columns)


def test_usage_logs_has_design_columns_and_latency_name():
    columns = set(UsageLog.__table__.columns.keys())
    assert {"api_key_id", "endpoint", "tokens_used", "latency", "timestamp"}.issubset(
        columns
    )
    assert "latency_ms" not in columns


def test_usage_logging_failure_does_not_break_request_flow(monkeypatch):
    os.environ.setdefault("ALPHAVANTAGE_API_KEY", "test-key")
    os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/test")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
    os.environ.setdefault("API_KEY_HASH_SALT", "test-salt")

    import backend.main as main

    class DummyRedis:
        async def incr(self, _key: str) -> int:
            return 1

        async def expire(self, _key: str, _seconds: int) -> None:
            return None

    class DummySession:
        def add(self, _obj: object) -> None:
            return None

        async def commit(self) -> None:
            raise RuntimeError("simulated usage log failure")

    class DummySessionMaker:
        def __call__(self) -> "DummySessionMaker":
            return self

        async def __aenter__(self) -> DummySession:
            return DummySession()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    async def fake_authenticate_api_key(request, _session) -> ApiKeyContext:
        context = ApiKeyContext(id=1, rate_limit=60)
        request.state.api_key_context = context
        return context

    monkeypatch.setattr(main, "authenticate_api_key", fake_authenticate_api_key)
    monkeypatch.setattr(main, "async_session_maker", DummySessionMaker())
    main.app.state.redis = DummyRedis()

    client = TestClient(main.app)
    response = client.get("/health", headers={"X-API-Key": "test"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
