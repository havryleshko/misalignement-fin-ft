import asyncio
import secrets
from sqlalchemy import select
from backend.config import load_config
from backend.models.db import async_session_maker
from backend.models.entities import ApiKey, Customer
from backend.api.auth import _hash_api_key


DEFAULT_CUSTOMER_NAME = "default"
DEFAULT_PLAN = "free"


async def seed() -> None:
    config = load_config()
    raw_key = secrets.token_urlsafe(32)
    key_hash = _hash_api_key(raw_key, config.api_key_hash_salt)

    async with async_session_maker() as session:
        result = await session.execute(
            select(Customer).where(Customer.name == DEFAULT_CUSTOMER_NAME)
        )
        customer = result.scalar_one_or_none()
        if customer is None:
            customer = Customer(name=DEFAULT_CUSTOMER_NAME, plan=DEFAULT_PLAN)
            session.add(customer)
            await session.flush()

        api_key = ApiKey(
            key_hash=key_hash,
            customer_id=customer.id,
            status="active",
        )
        session.add(api_key)
        await session.commit()

    print("Seeded API key (store securely, shown once):")
    print(raw_key)


if __name__ == "__main__":
    asyncio.run(seed())
