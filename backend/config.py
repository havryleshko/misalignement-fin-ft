import os
from dataclasses import dataclass
from typing import Optional


REQUIRED_ENV_VARS = (
    "ALPHAVANTAGE_API_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "API_KEY_HASH_SALT",
)

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_RATE_LIMIT_RPM = 60
DEFAULT_CACHE_TTL_SECONDS = 300
DEFAULT_ENV = "local"
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_LLM_MAX_TOKENS = 800
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


@dataclass(frozen=True)
class Config:
    alphavantage_api_key: str
    database_url: str
    redis_url: str
    api_key_hash_salt: str
    log_level: str = DEFAULT_LOG_LEVEL
    rate_limit_rpm: int = DEFAULT_RATE_LIMIT_RPM
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    env: str = DEFAULT_ENV
    llm_model: str = DEFAULT_LLM_MODEL
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    ollama_host: str = DEFAULT_OLLAMA_HOST


def _load_dotenv_if_local(env: str) -> None:
    if env not in ("local", "development"):
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for env var {name}: {value}") from exc


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for env var {name}: {value}") from exc


def load_config() -> Config:
    env = os.getenv("ENV", DEFAULT_ENV).strip() or DEFAULT_ENV
    _load_dotenv_if_local(env)

    missing: list[str] = []
    for name in REQUIRED_ENV_VARS:
        if not os.getenv(name, "").strip():
            missing.append(name)
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variables: {missing_list}. "
            "See .env.example for required configuration."
        )

    config = Config(
        alphavantage_api_key=_get_required_env("ALPHAVANTAGE_API_KEY"),
        database_url=_get_required_env("DATABASE_URL"),
        redis_url=_get_required_env("REDIS_URL"),
        api_key_hash_salt=_get_required_env("API_KEY_HASH_SALT"),
        log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
        rate_limit_rpm=_get_int_env("RATE_LIMIT_RPM", DEFAULT_RATE_LIMIT_RPM),
        cache_ttl_seconds=_get_int_env(
            "CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS
        ),
        env=env,
        llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        llm_temperature=_get_float_env(
            "LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE
        ),
        llm_max_tokens=_get_int_env("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS),
        ollama_host=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
    )
    return config
