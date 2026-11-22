"""FastAPI application factory: auth, CORS, error envelopes and lifespan."""

from __future__ import annotations

import asyncio
import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .config import ServerConfig
from .routers import datasets, inference, models, sessions, system
from .schemas import API_VERSION, ApiError, error_envelope
from .worker import ComputeWorker, ModelManager, SessionManager

MAX_BODY_BYTES = 32 * 1024 * 1024


@dataclass
class ServerContext:
    config: ServerConfig
    worker: ComputeWorker
    models: ModelManager
    sessions: SessionManager
    start_time: float


def _request_id() -> str:
    return "req_" + uuid.uuid4().hex[:8]


class AuthMiddleware:
    """Constant-time Bearer-token gate (health remains public)."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not self.token:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        if path == "/api/v1/health" or not path.startswith("/api/v1") or method == "OPTIONS":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        supplied = ""
        auth = headers.get(b"authorization", b"").decode("latin-1")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
        if supplied and hmac.compare_digest(supplied, self.token):
            await self.app(scope, receive, send)
            return
        body = json.dumps(
            error_envelope(
                "UNAUTHORIZED",
                "Valid Authorization: Bearer <token> required.",
                None,
                _request_id(),
            )
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class BodyLimitMiddleware:
    def __init__(self, app: Any, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            headers = {key.lower(): value for key, value in scope.get("headers", [])}
            raw_len = headers.get(b"content-length")
            if raw_len is not None:
                try:
                    length = int(raw_len)
                except ValueError:
                    length = 0
                if length > self.limit:
                    body = json.dumps(
                        error_envelope(
                            "RESOLUTION_LIMIT_EXCEEDED",
                            "Request payload exceeds 32 MB.",
                            {"max_bytes": self.limit},
                            _request_id(),
                        )
                    ).encode("utf-8")
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 400,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                    return
        await self.app(scope, receive, send)


def create_app(
    config: Optional[ServerConfig] = None,
    model_factory: Optional[Callable[[str, str], object]] = None,
) -> FastAPI:
    if config is None:
        config = ServerConfig()
    if model_factory is not None:
        config.model_factory = model_factory
    config.resolve()
    worker = ComputeWorker()
    models_manager = ModelManager(config)
    session_manager = SessionManager(config, worker, models_manager)
    ctx = ServerContext(
        config=config,
        worker=worker,
        models=models_manager,
        sessions=session_manager,
        start_time=time.monotonic(),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx.start_time = time.monotonic()
        session_manager.bind_loop(asyncio.get_running_loop())
        worker.start()
        try:
            yield
        finally:
            session_manager.shutdown()
            worker.stop()

    app = FastAPI(
        title="AlphaBuilder Compute Engine",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/openapi.json",
    )
    app.state.ctx = ctx

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        request_id = _request_id()
        return JSONResponse(exc.envelope(request_id), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            error_envelope(
                "INVALID_INPUT",
                "Request validation failed.",
                {"errors": jsonable_encoder(exc.errors())},
                _request_id(),
            ),
            status_code=400,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "NOT_FOUND" if exc.status_code == 404 else "INVALID_INPUT"
        return JSONResponse(
            error_envelope(code, str(exc.detail), None, _request_id()),
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            error_envelope(
                "INTERNAL_ERROR",
                "Unexpected server error.",
                {"type": type(exc).__name__},
                _request_id(),
            ),
            status_code=500,
        )

    app.add_middleware(AuthMiddleware, token=config.api_token)
    app.add_middleware(BodyLimitMiddleware, limit=MAX_BODY_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    prefix = "/api/v1"
    app.include_router(system.router, prefix=prefix)
    app.include_router(models.router, prefix=prefix)
    app.include_router(inference.router, prefix=prefix)
    app.include_router(sessions.router, prefix=prefix)
    app.include_router(datasets.router, prefix=prefix)
    return app
