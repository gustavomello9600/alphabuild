"""Inference (Raio-X), FEM evaluation and SIMP startup endpoints."""

from __future__ import annotations

import asyncio
import base64
import time
import zlib
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from .. import wire
from ..schemas import (
    ApiError,
    FemEvalRequest,
    InferRequest,
    LoadConfig,
    SimpStartRequest,
)
from ..worker import (
    generate_bc_masks,
    generate_forces,
    has_fenics,
    load_region_bounds,
    validate_resolution,
)

router = APIRouter(tags=["inference"])


def _ensure_not_running(request: Request) -> None:
    if request.app.state.ctx.sessions.has_running():
        raise ApiError(409, "ENGINE_BUSY", "A live MCTS session is RUNNING; pause it first.")


def _build_from_json(req: InferRequest, config: Any):
    dims = validate_resolution(req.resolution, config.max_voxels)
    total = dims[0] * dims[1] * dims[2]
    density = np.zeros(dims, dtype=np.float32)
    if req.density_f32_b64_zlib:
        try:
            raw = zlib.decompress(base64.b64decode(req.density_f32_b64_zlib, validate=True))
        except Exception as exc:  # noqa: BLE001
            raise ApiError(400, "INVALID_INPUT", f"invalid density_f32_b64_zlib: {exc}") from exc
        arr = np.frombuffer(raw, dtype="<f4")
        if arr.size != total:
            raise ApiError(
                400,
                "INVALID_INPUT",
                f"density has {arr.size} values, expected {total}",
            )
        density = arr.reshape(dims).astype(np.float32).copy()
    else:
        for idx in req.solid_voxels:
            if not 0 <= idx < total:
                raise ApiError(400, "INVALID_INPUT", f"solid voxel {idx} outside domain")
            density.reshape(-1)[idx] = 1.0

    if req.custom_supports:
        masks = [np.zeros(dims, dtype=np.float32) for _ in range(3)]
        for support in req.custom_supports:
            if not (0 <= support.x < dims[0] and 0 <= support.y < dims[1] and 0 <= support.z < dims[2]):
                raise ApiError(400, "INVALID_INPUT", "custom support outside domain")
            masks[0][support.x, support.y, support.z] = 1.0 if support.mask_x else 0.0
            masks[1][support.x, support.y, support.z] = 1.0 if support.mask_y else 0.0
            masks[2][support.x, support.y, support.z] = 1.0 if support.mask_z else 0.0
        bc_masks = tuple(masks)
    else:
        bc_masks = generate_bc_masks(dims, req.bc_type)

    if req.custom_forces:
        forces = np.zeros((3, *dims), dtype=np.float32)
        for force in req.custom_forces:
            if not (0 <= force.x < dims[0] and 0 <= force.y < dims[1] and 0 <= force.z < dims[2]):
                raise ApiError(400, "INVALID_INPUT", "custom force outside domain")
            forces[0, force.x, force.y, force.z] = force.fx
            forces[1, force.x, force.y, force.z] = force.fy
            forces[2, force.x, force.y, force.z] = force.fz
    else:
        forces = generate_forces(dims, req.load_config)
    return dims, density, bc_masks, forces


def _build_from_binary(raw: bytes, config: Any):
    try:
        packet = wire.decode_packet(raw)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, "INVALID_INPUT", f"invalid ABWIRE01 body: {exc}") from exc
    header = packet.header or {}
    if "resolution" not in header:
        raise ApiError(400, "INVALID_INPUT", "ABWIRE01 header must include resolution")
    req = InferRequest(
        resolution=header["resolution"],
        bc_type=header.get("bc_type", "FULL_CLAMP"),
        load_config=LoadConfig(**header["load_config"]),
        custom_supports=None,
        custom_forces=None,
        include_blueprint=bool(header.get("include_blueprint", True)),
        policy_mode=header.get("policy_mode", "masked_probs"),
        policy_temperature=float(header.get("policy_temperature", 1.0)),
        policy_top_k=int(header.get("policy_top_k", 512)),
        response_format="abwire01",
    )
    dims = validate_resolution(req.resolution, config.max_voxels)
    tensors = {tensor.tag: tensor for tensor in packet.tensors}
    if wire.TAG_DENSITY not in tensors:
        raise ApiError(400, "INVALID_INPUT", "ABWIRE01 body is missing DENSITY")
    try:
        density_flat = wire.tensor_to_dense(tensors[wire.TAG_DENSITY])
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, "INVALID_INPUT", f"invalid DENSITY tensor: {exc}") from exc
    if density_flat.size != dims[0] * dims[1] * dims[2]:
        raise ApiError(400, "INVALID_INPUT", "DENSITY tensor shape mismatch")
    density = density_flat.reshape(dims).astype(np.float32)

    if wire.TAG_BC_MASKS in tensors:
        bc = wire.tensor_to_dense(tensors[wire.TAG_BC_MASKS]).reshape(3, *dims)
        bc_masks = tuple(np.asarray(bc[i], dtype=np.float32) for i in range(3))
    else:
        bc_masks = generate_bc_masks(dims, req.bc_type)
    if wire.TAG_FORCES in tensors:
        fc = wire.tensor_to_dense(tensors[wire.TAG_FORCES]).reshape(3, *dims)
        forces = np.asarray(fc, dtype=np.float32)
    else:
        forces = generate_forces(dims, req.load_config)
    return req, dims, density, bc_masks, forces


def _top_actions(prob_add: np.ndarray, prob_remove: np.ndarray, top_k: int) -> list[dict]:
    actions: list[dict] = []
    for channel, probs in ((0, prob_add), (1, prob_remove)):
        flat = probs.reshape(-1)
        indices = np.flatnonzero(flat > 1e-5)
        if indices.size == 0:
            continue
        vals = flat[indices]
        if vals.size > top_k:
            order = np.argpartition(vals, -top_k)[-top_k:]
            indices = indices[order]
            vals = vals[order]
        for idx, prob in sorted(zip(indices.tolist(), vals.tolist()), key=lambda x: x[1], reverse=True):
            x, y, z = np.unravel_index(idx, probs.shape)
            actions.append(
                {
                    "channel": channel,
                    "x": int(x),
                    "y": int(y),
                    "z": int(z),
                    "prob": float(prob),
                }
            )
    actions.sort(key=lambda item: item["prob"], reverse=True)
    return actions[:top_k]


def _run_infer(
    models: Any,
    dims: tuple[int, int, int],
    density: np.ndarray,
    bc_masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    forces: np.ndarray,
    load_config: dict,
    include_blueprint: bool,
    top_k: int,
) -> tuple[dict, list[wire.WireTensor]]:
    from alphabuilder.src.logic.mcts.engine import build_state_tensor
    from alphabuilder.src.logic.mcts.legal_moves import get_legal_moves, masked_softmax
    from alphabuilder.src.logic.selfplay.reward import (
        analyze_structure_islands,
        calculate_connectivity_reward,
        calculate_island_penalty,
    )

    started = time.perf_counter()
    state = build_state_tensor(density, bc_masks, forces)
    value, p_add, p_rem, blueprint = models.predict(state, include_blueprint=include_blueprint)
    inference_ms = (time.perf_counter() - started) * 1000.0

    valid_add, valid_remove = get_legal_moves(
        density, 0.5, bc_masks=bc_masks, forces=forces
    )
    prob_add, prob_remove = masked_softmax(p_add, p_rem, valid_add, valid_remove, 1.0)
    analysis = analyze_structure_islands(density, load_config, bc_masks=bc_masks)
    total_voxels = int((density > 0.5).sum())
    island_penalty = calculate_island_penalty(
        analysis["n_islands"], analysis["loose_voxels"], total_voxels
    )
    _, connected_fraction = calculate_connectivity_reward(density, bc_masks, forces)
    volume_fraction = float(np.mean(density > 0.5))

    add_idx, add_val = wire.sparse_from_dense(prob_add, 1e-5)
    rem_idx, rem_val = wire.sparse_from_dense(prob_remove, 1e-5)
    tensors = [
        wire.coo1_f32(wire.TAG_POLICY_ADD, dims, add_idx, add_val),
        wire.coo1_f32(wire.TAG_POLICY_REMOVE, dims, rem_idx, rem_val),
    ]
    if include_blueprint and blueprint is not None:
        tensors.append(
            wire.dense_f16(
                wire.TAG_BLUEPRINT_PROBS,
                dims,
                np.asarray(blueprint, dtype=np.float32).reshape(-1),
            )
        )
    info = models.info or {}
    header = {
        "value": float(value),
        "inference_ms": round(inference_ms, 2),
        "device": info.get("device_resolved", "cpu"),
        "model_id": info.get("model_id"),
        "resolution": list(dims),
        "legal_add_count": int(valid_add.sum()),
        "legal_remove_count": int(valid_remove.sum()),
        "volume_fraction": volume_fraction,
        "island_analysis": {
            "n_islands": int(analysis["n_islands"]),
            "loose_voxels": int(analysis["loose_voxels"]),
            "is_connected": bool(analysis["is_connected"]),
            "connected_load_fraction": float(connected_fraction),
            "island_penalty": float(island_penalty),
        },
        "top_policy_actions": _top_actions(prob_add, prob_remove, top_k),
        "policy_add_count": len(add_idx),
        "policy_remove_count": len(rem_idx),
    }
    return header, tensors


@router.post("/infer")
async def infer(request: Request):
    ctx = request.app.state.ctx
    _ensure_not_running(request)
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type == "application/octet-stream":
        body = await request.body()
        if len(body) > ctx.config.max_request_bytes:
            raise ApiError(400, "RESOLUTION_LIMIT_EXCEEDED", "Request payload exceeds 32 MB")
        req, dims, density, bc_masks, forces = _build_from_binary(body, ctx.config)
        include_blueprint = req.include_blueprint
        top_k = req.policy_top_k
        response_format = "abwire01"
    else:
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise ApiError(400, "INVALID_INPUT", f"invalid JSON body: {exc}") from exc
        try:
            req = InferRequest.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise ApiError(400, "INVALID_INPUT", str(exc)) from exc
        dims, density, bc_masks, forces = _build_from_json(req, ctx.config)
        include_blueprint = req.include_blueprint
        top_k = req.policy_top_k
        response_format = req.response_format

    header, tensors = await asyncio.wrap_future(
        ctx.worker.submit(
            _run_infer,
            ctx.models,
            dims,
            density,
            bc_masks,
            forces,
            req.load_config.model_dump(),
            include_blueprint,
            top_k,
            priority=0,
        )
    )
    # Recompute island analysis against the actual load config in the worker result.
    if response_format == "json":
        return JSONResponse(header)
    packet = wire.encode_packet(wire.PAYLOAD_INFER_RESULT, header, tensors, compress=True, flags=0x01)
    return Response(content=packet, media_type="application/octet-stream")


def _run_fem(
    models: Any,
    req: FemEvalRequest,
    dims: tuple[int, int, int],
    density: np.ndarray,
    bc_masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    forces: np.ndarray,
) -> tuple[dict, list[wire.WireTensor]]:
    from alphabuilder.src.logic.selfplay.evaluator import evaluate_main_island
    from alphabuilder.src.logic.selfplay.reward import (
        analyze_structure_islands,
        calculate_compliance_score,
        calculate_connectivity_reward,
        calculate_displacement_penalty,
        calculate_island_penalty,
        calculate_reward,
        calculate_volume_bonus,
    )

    started = time.perf_counter()
    analysis = analyze_structure_islands(
        density, req.load_config.model_dump(), bc_masks=bc_masks
    )
    vol_frac = float(np.mean(density > 0.5))
    total_voxels = int((density > 0.5).sum())
    island_penalty = calculate_island_penalty(
        analysis["n_islands"], analysis["loose_voxels"], total_voxels
    )
    _, connected_fraction = calculate_connectivity_reward(density, bc_masks, forces)
    if not analysis["is_connected"]:
        fem_result = None
    else:
        fem_result = evaluate_main_island(
            density,
            req.load_config.model_dump(),
            analysis["main_island_mask"],
            E=req.E,
            nu=req.nu,
            force_magnitude=req.force_magnitude,
            bc_type=req.bc_type,
        )
    solve_ms = (time.perf_counter() - started) * 1000.0
    valid = bool(fem_result is not None and fem_result.valid)
    compliance = float(fem_result.compliance) if valid else None
    max_disp = float(fem_result.max_displacement) if valid else None
    displacement_map = fem_result.displacement_map if valid else None
    if valid:
        reward = calculate_reward(compliance or 0.0, vol_frac, True, max_disp)
        compliance_score = calculate_compliance_score(compliance or 0.0)
        volume_bonus = calculate_volume_bonus(vol_frac)
        disp_penalty = calculate_displacement_penalty(max_disp)
    else:
        reward = -1.0
        compliance_score = None
        volume_bonus = None
        disp_penalty = 0.0
    header = {
        "valid": valid,
        "error_msg": None if valid else (fem_result.error_msg if fem_result else "disconnected"),
        "solve_time_ms": round(solve_ms, 2),
        "compliance": compliance,
        "max_displacement": max_disp,
        "volume_fraction": vol_frac,
        "reward_v33": {
            "compliance_score": compliance_score,
            "volume_bonus": volume_bonus,
            "fem_score": (
                float(np.clip(compliance_score + volume_bonus, -1.0, 1.0))
                if valid
                else None
            ),
            "displacement_penalty": float(disp_penalty),
            "island_penalty": float(island_penalty),
            "connectivity_bonus": float(connected_fraction),
            "total_reward": float(np.clip(reward - island_penalty, -1.0, 1.0)),
            "is_collapsed": reward == -1.0,
        },
        "island_analysis": {
            "n_islands": int(analysis["n_islands"]),
            "loose_voxels": int(analysis["loose_voxels"]),
            "is_connected": bool(analysis["is_connected"]),
            "connected_load_fraction": float(connected_fraction),
            "island_penalty": float(island_penalty),
        },
        "resolution": list(dims),
    }
    tensors: list[wire.WireTensor] = []
    if displacement_map is not None:
        tensors.append(
            wire.dense_f16(
                wire.TAG_DISPLACEMENT_MAG,
                dims,
                np.asarray(displacement_map, dtype=np.float32).reshape(-1),
            )
        )
    return header, tensors


@router.post("/fem/evaluate")
async def fem_evaluate(request: Request):
    ctx = request.app.state.ctx
    if not has_fenics():
        raise ApiError(501, "CAPABILITY_UNAVAILABLE", "FEniCSx is not installed on this host.")
    _ensure_not_running(request)
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, "INVALID_INPUT", f"invalid JSON body: {exc}") from exc
    try:
        req = FemEvalRequest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, "INVALID_INPUT", str(exc)) from exc
    dims, density, bc_masks, forces = _build_from_json(req, ctx.config)
    header, tensors = await asyncio.wrap_future(
        ctx.worker.submit(
            _run_fem,
            ctx.models,
            req,
            dims,
            density,
            bc_masks,
            forces,
            priority=0,
        )
    )
    if req.response_format == "json":
        return JSONResponse(header)
    packet = wire.encode_packet(wire.PAYLOAD_FEM_RESULT, header, tensors, compress=True, flags=0x01)
    return Response(content=packet, media_type="application/octet-stream")


@router.post("/simp/start", status_code=201)
async def simp_start(request: Request, body: SimpStartRequest) -> dict:
    if not has_fenics():
        raise ApiError(
            501, "CAPABILITY_UNAVAILABLE", "SIMP requires FEniCSx/fenitop on this host."
        )
    _ensure_not_running(request)
    raise ApiError(
        501,
        "CAPABILITY_UNAVAILABLE",
        "SIMP streaming requires the fenitop runtime (implemented in the Studio compute worker).",
    )
