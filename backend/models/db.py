import os

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import load_config


class Base(DeclarativeBase):
    pass


def _ensure_async_driver(database_url: str) -> str:
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return database_url


def get_async_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        database_url = load_config().database_url
    return _ensure_async_driver(database_url)


engine = create_async_engine(get_async_database_url(), pool_pre_ping=True)

async_session_maker = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)
