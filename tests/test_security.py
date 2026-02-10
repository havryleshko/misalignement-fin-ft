from fastapi.testclient import TestClient

from backend.api.auth import ApiKeyContext


class DummyRedis:
    async def incr(self, _key: str) -> int:
        return 1

    async def expire(self, _key: str, _seconds: int) -> None:
        return None


class DummySession:
    def add(self, _obj: object) -> None:
        return None

    async def commit(self) -> None:
        return None


class DummySessionMaker:
    def __call__(self) -> "DummySessionMaker":
        return self

    async def __aenter__(self) -> DummySession:
        return DummySession()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def test_https_required_in_non_local_env(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("API_KEY_HASH_SALT", "test-salt")
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("REQUIRE_HTTPS", "true")

    import backend.main as main

    async def fake_authenticate_api_key(request, _session) -> ApiKeyContext:
        context = ApiKeyContext(id=1, rate_limit=60)
        request.state.api_key_context = context
        return context

    monkeypatch.setattr(main, "authenticate_api_key", fake_authenticate_api_key)
    monkeypatch.setattr(main, "async_session_maker", DummySessionMaker())

    app = main.create_app()
    app.state.redis = DummyRedis()
    client = TestClient(app)

    blocked = client.get("/health", headers={"X-API-Key": "test"})
    assert blocked.status_code == 400
    assert blocked.json()["detail"] == "HTTPS required"

    allowed = client.get(
        "/health",
        headers={"X-API-Key": "test", "X-Forwarded-Proto": "https"},
    )
    assert allowed.status_code == 200
