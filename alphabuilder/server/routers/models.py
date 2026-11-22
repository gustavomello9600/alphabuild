"""Checkpoint discovery and model lifecycle endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from ..schemas import (
    ApiError,
    LoadedModelInfo,
    LoadModelRequest,
    ModelListResponse,
    UnloadModelResponse,
)

router = APIRouter(tags=["models"])


def _ensure_not_running(request: Request) -> None:
    if request.app.state.ctx.sessions.has_running():
        raise ApiError(409, "ENGINE_BUSY", "A live MCTS session is RUNNING; pause it first.")


@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    ctx = request.app.state.ctx
    models = await asyncio.wrap_future(ctx.worker.submit(ctx.models.list_models, priority=0))
    return ModelListResponse(checkpoints_dir=str(ctx.config.checkpoints_dir), models=models)


@router.post("/models/load", response_model=LoadedModelInfo)
async def load_model(request: Request, body: LoadModelRequest) -> LoadedModelInfo:
    ctx = request.app.state.ctx
    _ensure_not_running(request)
    info = await asyncio.wrap_future(
        ctx.worker.submit(
            ctx.models.load,
            body.model_id,
            body.device,
            body.openvino_hint,
            priority=0,
        )
    )
    return LoadedModelInfo(**info)


@router.get("/models/active")
async def active_model(request: Request) -> dict:
    info = request.app.state.ctx.models.info
    if not info:
        return {"loaded": False}
    return info


@router.post("/models/unload", response_model=UnloadModelResponse)
async def unload_model(request: Request) -> UnloadModelResponse:
    ctx = request.app.state.ctx
    _ensure_not_running(request)
    await asyncio.wrap_future(ctx.worker.submit(ctx.models.unload, priority=0))
    return UnloadModelResponse(loaded=False)
