import hashlib
from dataclasses import dataclass

from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.entities import ApiKey


@dataclass(frozen=True)
class ApiKeyContext:
    id: int
    rate_limit: int | None


def _hash_api_key(raw_key: str, salt: str) -> str:
    digest = hashlib.sha256(f"{raw_key}{salt}".encode("utf-8")).hexdigest()
    return digest


async def authenticate_api_key(
    request: Request, session: AsyncSession
) -> ApiKeyContext:
    raw_key = request.headers.get("X-API-Key")
    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    config = request.app.state.config
    key_hash = _hash_api_key(raw_key, config.api_key_hash_salt)
    stmt = select(ApiKey).where(
        ApiKey.key_hash == key_hash, ApiKey.status == "active"
    )
    result = await session.execute(stmt)
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    api_key_context = ApiKeyContext(id=api_key.id, rate_limit=api_key.rate_limit)
    request.state.api_key_context = api_key_context
    return api_key_context


async def require_api_key(request: Request) -> ApiKeyContext:
    existing = getattr(request.state, "api_key_context", None)
    if existing is not None:
        return existing

    from backend.models.db import async_session_maker

    async with async_session_maker() as session:
        return await authenticate_api_key(request, session)
