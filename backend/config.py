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
DEFAULT_LLM_PROVIDER = "together"
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_MODEL_VERSION = "llama3-8b-fin-lora-v3"
DEFAULT_LLM_BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_LORA_ENABLED = True
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_LLM_MAX_TOKENS = 800
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_TOGETHER_MODEL_BASE = DEFAULT_LLM_BASE_MODEL_ID
DEFAULT_TOGETHER_MODEL_LORA = DEFAULT_MODEL_VERSION
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL_BASE = DEFAULT_LLM_BASE_MODEL_ID
DEFAULT_OPENROUTER_MODEL_LORA = DEFAULT_MODEL_VERSION
DEFAULT_REQUIRE_HTTPS = True
DEFAULT_SEC_USER_AGENT = "MisalignmentEngine/0.1 (public-api)"


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
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    llm_base_model_id: str = DEFAULT_LLM_BASE_MODEL_ID
    model_version: str = DEFAULT_MODEL_VERSION
    lora_enabled: bool = DEFAULT_LORA_ENABLED
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    ollama_host: str = DEFAULT_OLLAMA_HOST
    together_api_key: Optional[str] = None
    together_base_url: str = DEFAULT_TOGETHER_BASE_URL
    together_model_base: str = DEFAULT_TOGETHER_MODEL_BASE
    together_model_lora: str = DEFAULT_TOGETHER_MODEL_LORA
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    openrouter_model_base: str = DEFAULT_OPENROUTER_MODEL_BASE
    openrouter_model_lora: str = DEFAULT_OPENROUTER_MODEL_LORA
    require_https: bool = DEFAULT_REQUIRE_HTTPS
    sec_user_agent: str = DEFAULT_SEC_USER_AGENT


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean for env var {name}: {value}")


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


def _get_optional_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


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
        llm_provider=os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
        or DEFAULT_LLM_PROVIDER,
        llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        llm_base_model_id=os.getenv("LLM_BASE_MODEL_ID", DEFAULT_LLM_BASE_MODEL_ID),
        model_version=os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION),
        lora_enabled=_get_bool_env("LORA_ENABLED", DEFAULT_LORA_ENABLED),
        llm_temperature=_get_float_env(
            "LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE
        ),
        llm_max_tokens=_get_int_env("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS),
        ollama_host=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        together_api_key=_get_optional_env("TOGETHER_API_KEY"),
        together_base_url=os.getenv("TOGETHER_BASE_URL", DEFAULT_TOGETHER_BASE_URL),
        together_model_base=os.getenv("TOGETHER_MODEL_BASE", DEFAULT_TOGETHER_MODEL_BASE),
        together_model_lora=os.getenv("TOGETHER_MODEL_LORA", DEFAULT_TOGETHER_MODEL_LORA),
        openrouter_api_key=_get_optional_env("OPENROUTER_API_KEY"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
        openrouter_model_base=os.getenv("OPENROUTER_MODEL_BASE", DEFAULT_OPENROUTER_MODEL_BASE),
        openrouter_model_lora=os.getenv("OPENROUTER_MODEL_LORA", DEFAULT_OPENROUTER_MODEL_LORA),
        require_https=_get_bool_env("REQUIRE_HTTPS", DEFAULT_REQUIRE_HTTPS),
        sec_user_agent=os.getenv("SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT).strip()
        or DEFAULT_SEC_USER_AGENT,
    )
    return config
