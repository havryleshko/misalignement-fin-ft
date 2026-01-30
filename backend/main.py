import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import redis.asyncio as redis

from backend.api.auth import authenticate_api_key
from backend.api import routes

from backend.config import load_config
from backend.models.db import async_session_maker


def create_app() -> FastAPI:
    config = load_config()
    app = FastAPI()
    app.state.config = config
    app.state.redis = redis.from_url(config.redis_url, decode_responses=True)

    logger = logging.getLogger("misalignment")

    @app.middleware("http")
    async def auth_rate_limit(request: Request, call_next):
        async with async_session_maker() as session:
            try:
                api_key_context = await authenticate_api_key(request, session)
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                )

        rate_limit = api_key_context.rate_limit or config.rate_limit_rpm
        current_window = int(time.time() // 60)
        redis_key = f"rate:{api_key_context.id}:{current_window}"

        try:
            current = await app.state.redis.incr(redis_key)
            if current == 1:
                await app.state.redis.expire(redis_key, 60)
        except Exception:
            return JSONResponse(
                status_code=503,
                content={"detail": "Rate limit service unavailable"},
            )

        if current > rate_limit:
            retry_after = 60 - int(time.time() % 60)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        start_time = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return response

    app.include_router(routes.router)
    return app


app = create_app()
