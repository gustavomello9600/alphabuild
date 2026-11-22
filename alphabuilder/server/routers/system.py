"""System, capabilities and harvest supervisor endpoints."""

from __future__ import annotations

import asyncio
import os
import platform
import socket
import time
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Request

from .. import __version__
from ..schemas import (
    API_VERSION,
    REWARD_SPEC,
    ApiError,
    CapabilitiesResponse,
    DeviceInfo,
    FeatureFlags,
    HealthResponse,
    LimitsInfo,
)
from ..worker import has_fenics, has_fenitop_simp, has_monai, has_openvino, has_torch

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    ctx = request.app.state.ctx
    return HealthResponse(
        uptime_sec=round(time.monotonic() - ctx.start_time, 1),
        auth_required=ctx.config.auth_required,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _cpu_name() -> str:
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _devices() -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    cuda_available = False
    cuda_name = None
    xpu_available = False
    xpu_name = None
    if has_torch():
        try:
            import torch  # type: ignore

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_name = torch.cuda.get_device_name(0)
            if hasattr(torch, "xpu"):
                xpu_available = bool(torch.xpu.is_available())
                if xpu_available:
                    xpu_name = getattr(torch.xpu, "get_device_name", lambda _i: "Intel XPU")(0)
        except Exception:
            pass
    npu_available = False
    npu_name = None
    if has_openvino():
        try:
            import openvino  # type: ignore

            core = openvino.Core()
            available = list(core.available_devices)
            npu = [d for d in available if d.startswith("NPU")]
            npu_available = bool(npu)
            if npu:
                npu_name = "Intel NPU (OpenVINO)"
        except Exception:
            npu_available = False
    devices.append(DeviceInfo(id="cuda", available=cuda_available, name=cuda_name))
    devices.append(DeviceInfo(id="xpu", available=xpu_available, name=xpu_name))
    devices.append(DeviceInfo(id="npu", available=npu_available, name=npu_name))
    devices.append(DeviceInfo(id="cpu", available=True, name=f"{_cpu_name()} ({os.cpu_count()} cores)"))
    return devices


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities(request: Request) -> CapabilitiesResponse:
    ctx = request.app.state.ctx
    features = FeatureFlags(
        has_torch=has_torch(),
        has_monai=has_monai(),
        has_openvino=has_openvino(),
        has_fenics=has_fenics(),
        has_fenitop_simp=has_fenitop_simp(),
        has_harvest_supervisor=bool(ctx.config.harvest_supervisor_url),
    )
    return CapabilitiesResponse(
        hostname=socket.gethostname(),
        os=platform.system().lower() or os.name,
        arch=platform.machine(),
        devices=_devices(),
        features=features,
        limits=LimitsInfo(
            max_voxels=ctx.config.max_voxels,
            max_concurrent_sessions=ctx.config.max_concurrent_sessions,
        ),
        reward_spec=dict(REWARD_SPEC),
        active_model=ctx.models.info,
    )


@router.get("/cluster/harvest")
async def cluster_harvest(request: Request) -> dict:
    ctx = request.app.state.ctx
    url = ctx.config.harvest_supervisor_url
    if not url:
        return {"configured": False, "reachable": False}
    target = url.rstrip("/") + "/api/v1/status"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(target)
            response.raise_for_status()
            payload = response.json()
        if isinstance(payload, dict):
            payload = {**payload, "supervisor_url": url, "reachable": True, "configured": True}
            return payload
        return {
            "configured": True,
            "reachable": True,
            "supervisor_url": url,
            "payload": payload,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "configured": True,
            "reachable": False,
            "supervisor_url": url,
            "error": str(exc),
        }
