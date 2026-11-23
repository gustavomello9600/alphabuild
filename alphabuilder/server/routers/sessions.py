"""Interactive MCTS session CRUD and the ABWIRE01 WebSocket bridge."""

from __future__ import annotations

import asyncio
import hmac
import json
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from ..schemas import ApiError, SessionCommandRequest, SessionCreateRequest, SessionCreateResponse

router = APIRouter(tags=["sessions"])


def _manager(websocket_or_request: Any):
    app = (
        websocket_or_request.app
        if isinstance(websocket_or_request, WebSocket)
        else websocket_or_request.app
    )
    return app.state.ctx


def _ws_token_ok(websocket: WebSocket, expected: str) -> bool:
    if not expected:
        return True
    header = websocket.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        supplied = header[7:].strip()
        if hmac.compare_digest(supplied, expected):
            return True
    query_token = websocket.query_params.get("token", "")
    return bool(query_token) and hmac.compare_digest(query_token, expected)


@router.post("/sessions", status_code=201, response_model=SessionCreateResponse)
async def create_session(request: Request, body: SessionCreateRequest) -> SessionCreateResponse:
    ctx = request.app.state.ctx
    if body.kind == "MCTS_LIVE" and not ctx.models.loaded:
        raise ApiError(
            409,
            "MODEL_NOT_LOADED",
            "Nenhum checkpoint carregado no servidor de inferência.",
            {"endpoint": "/api/v1/sessions"},
        )
    session = ctx.sessions.create(body)
    return SessionCreateResponse(
        session_id=session.session_id,
        game_id=session.game_id,
        status=session.status,
        fem_enabled=session.fem_enabled,
        ws_url=f"/api/v1/ws/sessions/{session.session_id}",
    )


@router.get("/sessions")
async def list_sessions(request: Request) -> dict:
    return {"sessions": request.app.state.ctx.sessions.list()}


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str) -> dict:
    session = request.app.state.ctx.sessions.get(session_id)
    payload = session.state_payload()
    payload.update(
        {
            "fem_enabled": session.fem_enabled,
            "end_reason": session.end_reason,
            "final_score": session.final_score,
            "final_compliance": session.final_compliance,
            "final_volume": session.final_volume,
            "frames_available": sorted(session.frames.keys()),
            "history": session.history,
        }
    )
    return payload


@router.post("/sessions/{session_id}/command")
async def session_command(request: Request, session_id: str, body: SessionCommandRequest) -> dict:
    session = request.app.state.ctx.sessions.get(session_id)
    if body.op == "pause":
        session.pause()
    elif body.op == "resume":
        session.resume()
    elif body.op == "step_once":
        session.step_once()
    elif body.op == "cancel":
        session.cancel()
    elif body.op == "edit_state":
        session.edit_state(
            body.add_voxels, body.remove_voxels, body.load_config, body.bc_type
        )
    elif body.op == "update_config":
        session.update_config(body.phase1_mcts, body.phase2_mcts, body.use_guided_value)
    return {"ok": True, "status": session.status}


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> dict:
    request.app.state.ctx.sessions.delete(session_id)
    return {"deleted": True}


@router.websocket("/ws/sessions/{session_id}")
async def session_ws(websocket: WebSocket, session_id: str) -> None:
    ctx = _manager(websocket)
    if not _ws_token_ok(websocket, ctx.config.api_token):
        await websocket.close(code=4401, reason="UNAUTHORIZED")
        return
    try:
        session = ctx.sessions.get(session_id)
    except ApiError:
        await websocket.close(code=4404, reason="NOT_FOUND")
        return

    await websocket.accept()
    queue = session.subscribe()
    try:
        await websocket.send_json(session.state_payload())
        if session.frames:
            latest_step = max(session.frames.keys())
            await websocket.send_bytes(session.frames[latest_step])
        if session.status in {"COMPLETED", "FAILED", "CANCELLED"}:
            await websocket.send_json(
                {
                    "type": "session_ended",
                    "status": session.status,
                    "reason": session.end_reason or session.status,
                    "final_score": session.final_score,
                    "final_compliance": session.final_compliance,
                    "final_volume": session.final_volume,
                    "total_steps": session.step,
                }
            )

        async def receive_loop() -> None:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await queue.put(("text", {"type": "error", "code": "INVALID_INPUT", "message": "invalid JSON"}))
                    continue
                msg_type = message.get("type")
                if msg_type == "ping":
                    await queue.put(("text", {"type": "pong", "ts": message.get("ts")}))
                    continue
                if msg_type != "command":
                    await queue.put(
                        ("text", {"type": "error", "code": "INVALID_INPUT", "message": "expected type=command"})
                    )
                    continue
                payload = dict(message)
                payload.pop("type", None)
                try:
                    command = SessionCommandRequest.model_validate(payload)
                    if command.op == "pause":
                        session.pause()
                    elif command.op == "resume":
                        session.resume()
                    elif command.op == "step_once":
                        session.step_once()
                    elif command.op == "cancel":
                        session.cancel()
                    elif command.op == "edit_state":
                        session.edit_state(
                            command.add_voxels,
                            command.remove_voxels,
                            command.load_config,
                            command.bc_type,
                        )
                    elif command.op == "update_config":
                        session.update_config(
                            command.phase1_mcts,
                            command.phase2_mcts,
                            command.use_guided_value,
                        )
                except ApiError as exc:
                    await queue.put(
                        ("text", {"type": "error", "code": exc.code, "message": exc.message})
                    )
                except Exception as exc:  # noqa: BLE001
                    await queue.put(
                        ("text", {"type": "error", "code": "INVALID_INPUT", "message": str(exc)})
                    )

        receive_task = asyncio.create_task(receive_loop())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "text":
                    await websocket.send_json(payload)
                else:
                    await websocket.send_bytes(payload)
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
    except WebSocketDisconnect:
        pass
    finally:
        session.unsubscribe(queue)
