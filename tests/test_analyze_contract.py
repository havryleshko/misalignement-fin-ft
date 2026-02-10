from fastapi.testclient import TestClient

from backend.api.auth import ApiKeyContext
from backend.orchestration.errors import PipelineError


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


def _build_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("API_KEY_HASH_SALT", "test-salt")
    monkeypatch.setenv("ENV", "test")

    import backend.main as main

    async def fake_authenticate_api_key(request, _session) -> ApiKeyContext:
        context = ApiKeyContext(id=1, rate_limit=60)
        request.state.api_key_context = context
        return context

    monkeypatch.setattr(main, "authenticate_api_key", fake_authenticate_api_key)
    monkeypatch.setattr(main, "async_session_maker", DummySessionMaker())

    app = main.create_app()
    app.state.redis = DummyRedis()
    return TestClient(app)


def test_analyze_validation_error_uses_contract_shape(monkeypatch):
    client = _build_client(monkeypatch)

    response = client.post("/analyze", json={}, headers={"X-API-Key": "test"})
    assert response.status_code == 400
    payload = response.json()
    assert set(payload.keys()) == {"error_code", "message", "trace_id"}
    assert payload["error_code"] == "validation_error"
    assert payload["message"] == "Invalid request payload."
    assert isinstance(payload["trace_id"], str)
    assert payload["trace_id"]


def test_analyze_pipeline_error_uses_contract_shape(monkeypatch):
    client = _build_client(monkeypatch)

    def fail_pipeline(*_args, **_kwargs):
        raise PipelineError("MODEL_OUTPUT_INVALID", "Invalid output", "trace-from-pipeline")

    monkeypatch.setattr("backend.api.routes.run_pipeline", fail_pipeline)

    response = client.post(
        "/analyze",
        json={
            "ticker": "AAPL",
            "question": "Is this a good investment over the next 12 months?",
            "time_horizon": "12m",
        },
        headers={"X-API-Key": "test"},
    )
    assert response.status_code == 400
    payload = response.json()
    assert set(payload.keys()) == {"error_code", "message", "trace_id"}
    assert payload["error_code"] == "MODEL_OUTPUT_INVALID"
    assert payload["message"] == "Invalid output"
    assert payload["trace_id"] == "trace-from-pipeline"


def test_analyze_internal_error_uses_contract_shape(monkeypatch):
    client = _build_client(monkeypatch)

    def explode(*_args, **_kwargs):
        raise RuntimeError("unexpected crash")

    monkeypatch.setattr("backend.api.routes.run_pipeline", explode)

    response = client.post(
        "/analyze",
        json={
            "ticker": "AAPL",
            "question": "Is this a good investment over the next 12 months?",
            "time_horizon": "12m",
        },
        headers={"X-API-Key": "test"},
    )
    assert response.status_code == 500
    payload = response.json()
    assert set(payload.keys()) == {"error_code", "message", "trace_id"}
    assert payload["error_code"] == "INTERNAL_ERROR"
    assert payload["message"] == "Internal server error."
    assert isinstance(payload["trace_id"], str)
    assert payload["trace_id"]
