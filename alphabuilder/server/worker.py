"""Single-owner compute worker, model manager and live session engine.

PyTorch, OpenVINO, FEniCSx and MCTS are never executed on the asyncio event
loop: all calls are dispatched to one dedicated `ComputeWorker` thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import importlib.util
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import wire
from .config import ServerConfig
from .schemas import (
    ApiError,
    LoadConfig,
    MctsSettings,
    SessionCreateRequest,
    SessionEnded,
)


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def has_torch() -> bool:
    return has_module("torch")


def has_monai() -> bool:
    return has_module("monai")


def has_openvino() -> bool:
    return has_module("openvino")


def has_fenics() -> bool:
    return has_module("dolfinx")


def has_fenitop_simp() -> bool:
    return has_module("alphabuilder.src.logic.fenitop.topopt")


def load_region_bounds(
    resolution: tuple[int, int, int], load_config: LoadConfig | dict
) -> tuple[int, int, int, int, int]:
    cfg = load_config.model_dump() if isinstance(load_config, LoadConfig) else dict(load_config)
    nx, ny, nz = resolution
    lx = min(max(0, int(cfg["x"])), nx - 1)
    ly = float(cfg["y"])
    lz_center = (float(cfg["z_start"]) + float(cfg["z_end"])) / 2.0
    y_min = max(0, int(ly - 1.0))
    y_max = min(ny, int(ly + 1.0) + 1)
    z_min = max(0, int(lz_center - 1.0))
    z_max = min(nz, int(lz_center + 1.0) + 1)
    return lx, y_min, y_max, z_min, z_max


def generate_bc_masks(
    resolution: tuple[int, int, int], bc_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact copy of the v3.3 harvest generator semantics."""
    nx, ny, nz = resolution
    mask_x = np.zeros((nx, ny, nz), dtype=np.float32)
    mask_y = np.zeros((nx, ny, nz), dtype=np.float32)
    mask_z = np.zeros((nx, ny, nz), dtype=np.float32)
    if bc_type == "RAIL_XY":
        mask_x[0, :, :] = 1.0
        mask_y[0, :, :] = 1.0
    else:
        mask_x[0, :, :] = 1.0
        mask_y[0, :, :] = 1.0
        mask_z[0, :, :] = 1.0
    return mask_x, mask_y, mask_z


def generate_forces(resolution: tuple[int, int, int], load_config: LoadConfig | dict) -> np.ndarray:
    nx, ny, nz = resolution
    forces = np.zeros((3, nx, ny, nz), dtype=np.float32)
    lx, y_min, y_max, z_min, z_max = load_region_bounds(resolution, load_config)
    forces[1, lx, y_min:y_max, z_min:z_max] = -1.0
    return forces


def validate_resolution(resolution: Any, max_voxels: int) -> tuple[int, int, int]:
    try:
        dims = tuple(int(v) for v in resolution)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "INVALID_INPUT", f"invalid resolution: {resolution}") from exc
    if len(dims) != 3 or any(v < 1 or v > 256 for v in dims):
        raise ApiError(
            400,
            "RESOLUTION_LIMIT_EXCEEDED",
            f"resolution must satisfy 1 <= D,H,W <= 256: {dims}",
            {"resolution": list(dims)},
        )
    if dims[0] * dims[1] * dims[2] > max_voxels:
        raise ApiError(
            400,
            "RESOLUTION_LIMIT_EXCEEDED",
            f"voxel count exceeds {max_voxels}",
            {"resolution": list(dims), "max_voxels": max_voxels},
        )
    return dims  # type: ignore[return-value]


def read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Best-effort light metadata read; never required for server startup."""
    if not has_torch():
        return {}
    try:
        import torch  # type: ignore

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            return {}
        meta = dict(checkpoint.get("config") or {})
        meta["epoch"] = checkpoint.get("epoch")
        metrics = checkpoint.get("metrics") or {}
        meta["val_loss"] = checkpoint.get("val_loss", metrics.get("loss"))
        return meta
    except Exception:
        return {}


class ModelManager:
    """Owns the active checkpoint; only touched inside the ComputeWorker."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._model: Any = None
        self._info: Optional[dict[str, Any]] = None
        self._lock = threading.Lock()

    @property
    def info(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._info) if self._info else None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def list_models(self) -> list[dict[str, Any]]:
        base = self.config.checkpoints_dir
        base.mkdir(parents=True, exist_ok=True)
        models: list[dict[str, Any]] = []
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".pt", ".pth"}:
                continue
            try:
                relative = path.relative_to(base).as_posix()
            except ValueError:
                continue
            stat = path.stat()
            metadata = read_checkpoint_metadata(path)
            preview = {
                key: metadata[key]
                for key in ("use_swin", "feature_size", "epoch", "val_loss")
                if key in metadata and metadata[key] is not None
            }
            models.append(
                {
                    "model_id": relative,
                    "name": path.name,
                    "size_mb": round(stat.st_size / (1024.0 * 1024.0), 2),
                    "modified_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)
                    ),
                    "config_preview": preview,
                }
            )
        return models

    def load(self, model_id: str, device: str, openvino_hint: str) -> dict[str, Any]:
        path = self.config.safe_path(self.config.checkpoints_dir, model_id)
        if not path.is_file():
            raise ApiError(404, "NOT_FOUND", f"Checkpoint not found: {model_id}")
        if path.suffix.lower() not in {".pt", ".pth"}:
            raise ApiError(400, "INVALID_INPUT", "model_id must reference a .pt/.pth file")
        metadata = read_checkpoint_metadata(path)
        use_swin = bool(metadata.get("use_swin", False))
        if use_swin and not has_monai():
            raise ApiError(
                501,
                "CAPABILITY_UNAVAILABLE",
                "Checkpoint requires MONAI Swin-UNETR, which is not installed.",
            )
        started = time.perf_counter()
        if self.config.model_factory is not None:
            model = self.config.model_factory(str(path), device)
        else:
            if not has_torch():
                raise ApiError(
                    501,
                    "CAPABILITY_UNAVAILABLE",
                    "PyTorch is not installed; cannot load a checkpoint.",
                )
            from alphabuilder.src.neural.inference import AlphaBuilderInference

            model = AlphaBuilderInference(str(path), device=device)
        load_ms = (time.perf_counter() - started) * 1000.0
        resolved = getattr(model, "device_str", None) or getattr(model, "device", device)
        resolved = str(resolved).replace("torch.device(", "").strip("')\"") or str(device)
        info = {
            "loaded": True,
            "model_id": model_id,
            "architecture": "swin" if use_swin else "simple",
            "feature_size": int(
                metadata.get("feature_size", getattr(model, "feature_size", 24))
            ),
            "in_channels": int(metadata.get("in_channels", getattr(model, "in_channels", 7))),
            "has_blueprint_head": bool(metadata.get("has_blueprint_head", True)),
            "device_resolved": resolved,
            "epoch": int(metadata.get("epoch", getattr(model, "epoch", -1)) or -1),
            "val_loss": float(metadata.get("val_loss", getattr(model, "val_loss", float("inf"))) or float("inf")),
            "load_time_ms": round(load_ms, 2),
        }
        with self._lock:
            self._model = model
            self._info = info
        return info

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._info = None
        gc.collect()
        if has_torch():
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            except Exception:
                pass

    def _require_model(self) -> Any:
        with self._lock:
            model = self._model
        if model is None:
            raise ApiError(
                409,
                "MODEL_NOT_LOADED",
                "Nenhum checkpoint carregado no servidor de inferência.",
            )
        return model

    def predict(
        self, state: np.ndarray, include_blueprint: bool = False
    ) -> tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        model = self._require_model()
        if include_blueprint and hasattr(model, "predict_with_blueprint"):
            value, p_add, p_rem, blueprint = model.predict_with_blueprint(state)
            return float(value), np.asarray(p_add), np.asarray(p_rem), np.asarray(blueprint)
        result = model.predict(state)
        value, p_add, p_rem = result[0], result[1], result[2]
        return float(value), np.asarray(p_add), np.asarray(p_rem), None

    def predict_batch(
        self, states: list[np.ndarray]
    ) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
        model = self._require_model()
        if hasattr(model, "predict_batch"):
            values, p_adds, p_rems = model.predict_batch(states)
            return (
                [float(v) for v in values],
                [np.asarray(v) for v in p_adds],
                [np.asarray(v) for v in p_rems],
            )
        values, p_adds, p_rems = [], [], []
        for state in states:
            value, p_add, p_rem, _ = self.predict(state, include_blueprint=False)
            values.append(value)
            p_adds.append(p_add)
            p_rems.append(p_rem)
        return values, p_adds, p_rems


class ComputeWorker:
    """One priority queue consumed by exactly one dedicated thread."""

    _STOP = object()

    def __init__(self) -> None:
        self._queue: "queue.PriorityQueue[Any]" = queue.PriorityQueue()
        self._thread = threading.Thread(target=self._run, name="ab-compute-worker", daemon=True)
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._started = False
        self._stopped = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._queue.put((1000, 0, None, self._STOP, (), {}))
        if self._started:
            self._thread.join(timeout=timeout)

    def submit(self, fn: Callable[..., Any], *args: Any, priority: int = 1, **kwargs: Any) -> concurrent.futures.Future:
        if self._stopped:
            raise ApiError(503, "ENGINE_BUSY", "Compute worker is shutting down")
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        self._queue.put((priority, seq, future, fn, args, kwargs))
        return future

    def _run(self) -> None:
        while True:
            _priority, _seq, future, fn, args, kwargs = self._queue.get()
            if fn is self._STOP:
                return
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - propagate to the awaiting caller
                if future is not None:
                    future.set_exception(exc)
            else:
                if future is not None:
                    future.set_result(result)


@dataclass
class _StepOutcome:
    density: np.ndarray
    phase: str
    step: int
    metadata: dict[str, Any]
    frame: bytes
    end_reason: Optional[str] = None
    fem_enabled: bool = False
    final_score: Optional[float] = None
    final_compliance: Optional[float] = None
    final_volume: Optional[float] = None


class LiveSession:
    """A stateful MCTS live session driven by a scheduler thread."""

    def __init__(self, request: SessionCreateRequest, config: ServerConfig, manager: "SessionManager") -> None:
        self.request = request
        self.config = config
        self.manager = manager
        self.session_id = "sess_" + uuid.uuid4().hex[:20]
        self.game_id = request.resume_game_id or str(uuid.uuid4())
        self.dims = validate_resolution(request.resolution, config.max_voxels)
        self.bc_type = request.bc_type
        self.load_config = request.load_config
        self.include_blueprint = request.include_blueprint
        self.fem_enabled = has_fenics()
        current = np.zeros(self.dims, dtype=np.float32)
        for idx in request.initial_solid_voxels:
            if not 0 <= idx < current.size:
                raise ApiError(400, "INVALID_INPUT", f"initial voxel {idx} outside domain")
            current.reshape(-1)[idx] = 1.0
        self.density = current
        self.bc_masks = generate_bc_masks(self.dims, self.bc_type)
        self.forces = generate_forces(self.dims, self.load_config)
        self.step = 0
        self.phase = "GROWTH"
        self.status = "PAUSED" if request.start_paused else "CREATED"
        self.end_reason: Optional[str] = None
        self.final_score: Optional[float] = None
        self.final_compliance: Optional[float] = None
        self.final_volume: Optional[float] = None
        self.frames: dict[int, bytes] = {}
        self.history: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._pause = threading.Event()
        if request.start_paused:
            self._pause.set()
        self._cancel = threading.Event()
        self._one_step = False
        self._mcts: Any = None
        # Incremented on external state edits so an in-flight step computed
        # from the previous density can be discarded instead of overwriting it.
        self._state_generation = 0
        self._runner = threading.Thread(
            target=self._run_loop, name=f"ab-session-{self.session_id[-6:]}", daemon=True
        )
        self._subscribers: set[asyncio.Queue] = set()

    # ---------------------------------------------------------------- events
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2048)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def _publish(self, item: Any) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        loop = self.manager.loop
        for q in subscribers:
            try:
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(q.put_nowait, item)
                else:
                    q.put_nowait(item)
            except (RuntimeError, asyncio.QueueFull):
                pass

    def publish_state(self) -> None:
        self._publish(("text", self.state_payload()))

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "type": "session_state",
                "session_id": self.session_id,
                "game_id": self.game_id,
                "status": self.status,
                "step": self.step,
                "phase": self.phase,
                "resolution": list(self.dims),
                "bc_type": self.bc_type,
                "load_config": self.load_config.model_dump(),
            }

    # ---------------------------------------------------------------- control
    def start(self) -> None:
        with self._lock:
            if self.status == "CREATED":
                self.status = "RUNNING"
        self._runner.start()
        self.publish_state()

    def pause(self) -> None:
        with self._lock:
            if self.status in {"COMPLETED", "FAILED", "CANCELLED"}:
                return
            self.status = "PAUSED"
            self._pause.set()
        self.publish_state()

    def resume(self) -> None:
        with self._lock:
            if self.status != "PAUSED":
                return
            self.status = "RUNNING"
            self._pause.clear()
        self.publish_state()

    def step_once(self) -> None:
        with self._lock:
            if self.status != "PAUSED":
                raise ApiError(400, "INVALID_INPUT", "step_once requires a PAUSED session")
            self._one_step = True
            self.status = "RUNNING"
            self._pause.clear()
        self.publish_state()

    def cancel(self) -> None:
        self._cancel.set()
        self._pause.clear()

    def edit_state(
        self,
        add_voxels: list[int],
        remove_voxels: list[int],
        load_config: Optional[LoadConfig],
        bc_type: Optional[str],
    ) -> None:
        with self._lock:
            if self.status != "PAUSED":
                raise ApiError(400, "INVALID_INPUT", "edit_state is only accepted while PAUSED")
            flat = self.density.reshape(-1)
            for idx in add_voxels:
                if not 0 <= idx < flat.size:
                    raise ApiError(400, "INVALID_INPUT", f"add voxel {idx} outside domain")
                flat[idx] = 1.0
            for idx in remove_voxels:
                if not 0 <= idx < flat.size:
                    raise ApiError(400, "INVALID_INPUT", f"remove voxel {idx} outside domain")
                flat[idx] = 0.0
            if load_config is not None:
                self.load_config = load_config
            if bc_type is not None:
                self.bc_type = bc_type
            self.bc_masks = generate_bc_masks(self.dims, self.bc_type)
            self.forces = generate_forces(self.dims, self.load_config)
            self._state_generation += 1
            self._mcts = None
        self.publish_state()

    def update_config(
        self,
        phase1: Optional[MctsSettings],
        phase2: Optional[MctsSettings],
        use_guided_value: Optional[bool],
    ) -> None:
        with self._lock:
            if phase1 is not None:
                self.request.phase1_mcts = phase1
            if phase2 is not None:
                self.request.phase2_mcts = phase2
            if use_guided_value is not None:
                if use_guided_value != self.request.episode_config.use_guided_value:
                    self.request.episode_config.use_guided_value = use_guided_value
                    if self._mcts is not None:
                        self._mcts.reset()
        self.publish_state()

    # ---------------------------------------------------------------- runner
    def _run_loop(self) -> None:
        if self.request.kind == "SIMP":
            self._run_simp_loop()
            return
        try:
            while not self._cancel.is_set():
                with self._lock:
                    if self.status in {"COMPLETED", "FAILED", "CANCELLED"}:
                        return
                    paused = self._pause.is_set()
                    one_step = self._one_step
                    self._one_step = False
                    phase = self.phase
                    step = self.step
                    density = self.density.copy()
                    bc_masks = tuple(m.copy() for m in self.bc_masks)
                    forces = self.forces.copy()
                    include_blueprint = self.include_blueprint
                    generation = self._state_generation
                if paused and not one_step:
                    self._pause.wait(0.05)
                    continue
                if step >= self.request.episode_config.max_steps:
                    self._finish("COMPLETED", "MAX_STEPS")
                    return
                outcome = self.manager.worker.submit(
                    self._compute_step,
                    density,
                    bc_masks,
                    forces,
                    phase,
                    step,
                    include_blueprint,
                    priority=1,
                ).result()
                with self._lock:
                    if generation != self._state_generation:
                        # The user edited the state while this step was running:
                        # discard the stale result and recompute from the edit.
                        continue
                    self.density = outcome.density
                    self.phase = outcome.phase
                    self.step = outcome.step
                    self.frames[outcome.metadata["step"]] = outcome.frame
                    self.history.append(outcome.metadata)
                    self.final_score = outcome.final_score
                    self.final_compliance = outcome.final_compliance
                    self.final_volume = outcome.final_volume
                    if one_step and outcome.end_reason is None:
                        self.status = "PAUSED"
                        self._pause.set()
                progress = {
                    "type": "step_progress",
                    "step": outcome.metadata["step"],
                    "phase": outcome.phase,
                    "sims_done": outcome.metadata.get("mcts_stats", {}).get(
                        "num_simulations", 0
                    )
                    if outcome.metadata.get("mcts_stats")
                    else 0,
                    "sims_total": outcome.metadata.get("mcts_stats", {}).get(
                        "num_simulations", 0
                    )
                    if outcome.metadata.get("mcts_stats")
                    else 0,
                    "cache_hits": outcome.metadata.get("mcts_stats", {}).get("cache_hits", 0)
                    if outcome.metadata.get("mcts_stats")
                    else 0,
                    "elapsed_ms": outcome.metadata.get("step_elapsed_ms", 0.0),
                }
                self._publish(("text", progress))
                self._publish(("binary", outcome.frame))
                self.publish_state()
                if outcome.end_reason is not None:
                    self._finish(
                        "COMPLETED" if outcome.end_reason != "COLLAPSE" else "COMPLETED",
                        outcome.end_reason,
                    )
                    return
            self._finish("CANCELLED", "CANCELLED")
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            self._finish("FAILED", f"{type(exc).__name__}: {exc}", error=str(exc))

    def _run_simp_loop(self) -> None:
        try:
            outcomes = self.manager.worker.submit(self._compute_simp, priority=1).result()
            if not outcomes:
                self._finish("FAILED", "SIMP_EMPTY_HISTORY")
                return
            for outcome in outcomes:
                if self._cancel.is_set():
                    self._finish("CANCELLED", "CANCELLED")
                    return
                while self._pause.is_set() and not self._cancel.is_set():
                    self._pause.wait(0.05)
                if self._cancel.is_set():
                    self._finish("CANCELLED", "CANCELLED")
                    return
                with self._lock:
                    self.density = outcome.density
                    self.phase = "REFINEMENT"
                    self.step = outcome.step
                    self.frames[outcome.step] = outcome.frame
                    self.history.append(outcome.metadata)
                    self.final_score = outcome.final_score
                    self.final_compliance = outcome.final_compliance
                    self.final_volume = outcome.final_volume
                self._publish(
                    (
                        "text",
                        {
                            "type": "step_progress",
                            "step": outcome.step,
                            "phase": "REFINEMENT",
                            "sims_done": outcome.step,
                            "sims_total": outcomes[-1].step,
                            "cache_hits": 0,
                            "elapsed_ms": outcome.metadata.get("step_elapsed_ms", 0.0),
                        },
                    )
                )
                self._publish(("binary", outcome.frame))
                self.publish_state()
            self._finish("COMPLETED", "CONVERGENCE")
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc()
            self._finish("FAILED", f"{type(exc).__name__}: {exc}", error=str(exc))

    def _compute_simp(self) -> list[_StepOutcome]:
        from alphabuilder.src.core.physics_model import PhysicalProperties
        from alphabuilder.src.logic.harvest.config import SIMPConfig
        from alphabuilder.src.logic.harvest.optimization import run_fenitop_optimization
        from alphabuilder.src.logic.mcts.legal_moves import get_legal_moves
        from alphabuilder.src.logic.selfplay.reward import (
            calculate_compliance_score,
            calculate_displacement_penalty,
            calculate_reward,
            calculate_volume_bonus,
        )

        settings = self.request.simp_config
        if settings is None:
            from .schemas import SimpSettings

            settings = SimpSettings()
        started = time.perf_counter()
        load_config = self.load_config.model_dump()
        load_config["bc_type"] = self.bc_type
        props = PhysicalProperties(E=100.0, nu=0.25)
        simp_config = SIMPConfig(
            vol_frac=settings.target_vol_frac,
            max_iter=settings.max_iter,
            r_min=settings.r_min,
            load_config=load_config,
        )
        initial = (self.density > 0.5).astype(np.float32)
        history = run_fenitop_optimization(
            self.dims,
            props,
            simp_config,
            initial_density=initial,
            strategy=settings.strategy,
            record_displacement_map=True,
        )
        outcomes: list[_StepOutcome] = []
        for entry in history:
            density = np.asarray(entry.get("density_map"), dtype=np.float32)
            if density.shape != self.dims:
                density = np.asarray(self.density, dtype=np.float32)
            volume = float(entry.get("vol_frac", float(np.mean(density > 0.5))))
            compliance = float(entry.get("compliance", 0.0))
            max_disp = float(entry.get("max_displacement", 0.0))
            fem_score = float(entry.get("fem_score", 0.0))
            displacement = entry.get("displacement_map")
            reward = calculate_reward(compliance, volume, True, max_disp)
            valid_add, valid_remove = get_legal_moves(
                density,
                0.5,
                bc_masks=self.bc_masks,
                forces=(self.forces[0], self.forces[1], self.forces[2]),
            )
            matrix: dict[str, Any] = {
                "step": int(entry.get("step", len(outcomes))),
                "phase": "REFINEMENT",
                "value": fem_score,
                "value_target": entry.get("final_fem_score"),
                "fem_reward": reward,
                "compliance_fem": compliance,
                "max_displacement": max_disp,
                "volume_fraction": volume,
                "is_connected": True,
                "n_islands": 1,
                "loose_voxels": 0,
                "connected_load_fraction": 1.0,
                "island_penalty": 0.0,
                "legal_add_count": int(valid_add.sum()),
                "legal_remove_count": int(valid_remove.sum()),
                "reward_components": {},
                "reward_v33": {
                    "compliance_score": float(calculate_compliance_score(compliance)),
                    "volume_bonus": float(calculate_volume_bonus(volume)),
                    "fem_score": fem_score,
                    "displacement_penalty": float(calculate_displacement_penalty(max_disp)),
                    "island_penalty": 0.0,
                    "connectivity_bonus": 0.0,
                    "pbrs_delta": None,
                    "total_reward": float(reward),
                    "is_collapsed": reward == -1.0,
                },
                "mcts_stats": None,
                "selected_actions": [],
                "step_elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "fem_enabled": True,
                "resolution": list(self.dims),
                "bc_type": self.bc_type,
                "load_config": self.load_config.model_dump(),
            }
            tensors: list[wire.WireTensor] = [
                wire.dense_f16(wire.TAG_DENSITY, self.dims, density.reshape(-1))
            ]
            if displacement is not None:
                disp = np.asarray(displacement, dtype=np.float32)
                if disp.shape == self.dims:
                    tensors.append(wire.dense_f16(wire.TAG_DISPLACEMENT_MAG, self.dims, disp.reshape(-1)))
            frame = wire.encode_packet(
                wire.PAYLOAD_STEP_FRAME, matrix, tensors, compress=True, flags=0x01
            )
            outcomes.append(
                _StepOutcome(
                    density=density,
                    phase="REFINEMENT",
                    step=int(matrix["step"]),
                    metadata=matrix,
                    frame=frame,
                    end_reason=None,
                    fem_enabled=True,
                    final_score=float(reward),
                    final_compliance=compliance,
                    final_volume=volume,
                )
            )
        return outcomes

    def _finish(self, status: str, reason: str, error: Optional[str] = None) -> None:
        with self._lock:
            if self.status in {"COMPLETED", "FAILED", "CANCELLED"}:
                return
            self.status = status
            self.end_reason = reason
            final_score = self.final_score
            final_compliance = self.final_compliance
            final_volume = self.final_volume
            total_steps = self.step
        event = {
            "type": "session_ended",
            "status": status,
            "reason": reason,
            "final_score": final_score,
            "final_compliance": final_compliance,
            "final_volume": final_volume,
            "total_steps": total_steps,
        }
        if error:
            event["error"] = error
        self._publish(("text", event))

    # ---------------------------------------------------------------- compute
    def _mcts_config(self, phase: str) -> Any:
        from alphabuilder.src.logic.mcts.engine import MCTSConfig

        settings = (
            self.request.phase1_mcts if phase == "GROWTH" else self.request.phase2_mcts
        )
        return MCTSConfig(
            num_simulations=max(1, settings.num_simulations),
            batch_size=max(1, settings.batch_size),
            action_batch_size=max(1, settings.action_batch_size),
            c_puct=settings.c_puct,
            top_k_expansion=settings.top_k_expansion,
            phase=phase,
            mixmax_beta=settings.mixmax_beta,
            min_volume_fraction=0.0 if phase == "GROWTH" else 0.01,
            max_volume_fraction=0.9,
        )

    def _compute_step(
        self,
        density: np.ndarray,
        bc_masks: tuple[np.ndarray, np.ndarray, np.ndarray],
        forces: np.ndarray,
        phase: str,
        step: int,
        include_blueprint: bool,
    ) -> _StepOutcome:
        from alphabuilder.src.logic.mcts.engine import (
            AlphaBuilderMCTS,
            build_state_tensor,
            select_pv_branches,
        )
        from alphabuilder.src.logic.mcts.legal_moves import get_legal_moves, masked_softmax
        from alphabuilder.src.logic.selfplay.reward import (
            analyze_structure_islands,
            calculate_compliance_score,
            calculate_connectivity_reward,
            calculate_displacement_penalty,
            calculate_island_penalty,
            calculate_pbrs_potential,
            calculate_reward,
            calculate_volume_bonus,
            get_guidance_alpha,
        )

        load_cfg_dict = {**self.load_config.model_dump(), "bc_type": self.bc_type}
        force_tuple = (forces[0], forces[1], forces[2])

        started = time.perf_counter()
        state_tensor = build_state_tensor(density, bc_masks, force_tuple)
        value, p_add, p_rem, blueprint = self.manager.models.predict(
            state_tensor, include_blueprint=include_blueprint
        )
        inference_ms = (time.perf_counter() - started) * 1000.0

        ep_cfg = self.request.episode_config
        alpha_guide = (
            get_guidance_alpha(
                generation=ep_cfg.generation,
                decay_generations=ep_cfg.guidance_decay_generations,
            )
            if ep_cfg.use_guided_value
            else 0.0
        )
        phi_before, _ = calculate_pbrs_potential(
            density, bc_masks, force_tuple, load_cfg_dict
        )

        def _shape_value(
            net_val: float,
            density_grid: np.ndarray,
            masks: tuple[np.ndarray, np.ndarray, np.ndarray],
            force_vecs: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> float:
            if phase == "REFINEMENT":
                isl = analyze_structure_islands(density_grid, load_cfg_dict, bc_masks=masks)
                if not isl["is_connected"]:
                    return -1.0
                total_vox = int((density_grid > 0.5).sum())
                isl_pen = calculate_island_penalty(
                    isl["n_islands"], isl["loose_voxels"], total_vox
                )
                return float(np.clip(float(net_val) - isl_pen, -1.0, 1.0))
            if alpha_guide > 0.0:
                phi_leaf, _ = calculate_pbrs_potential(
                    density_grid, masks, force_vecs, load_cfg_dict
                )
                return float(
                    np.clip(float(net_val) + alpha_guide * (phi_leaf - phi_before), -1.0, 1.0)
                )
            return float(np.clip(float(net_val), -1.0, 1.0))

        def predict_fn(arr: np.ndarray):
            nv, pa, pr, _ = self.manager.models.predict(arr, include_blueprint=False)
            masks = (arr[1], arr[2], arr[3])
            fvecs = (arr[4], arr[5], arr[6])
            return _shape_value(nv, arr[0], masks, fvecs), pa, pr

        def predict_batch_fn(states: list[np.ndarray]):
            nvs, pas, prs = self.manager.models.predict_batch(states)
            shaped = [
                _shape_value(nv, st[0], (st[1], st[2], st[3]), (st[4], st[5], st[6]))
                for nv, st in zip(nvs, states)
            ]
            return shaped, pas, prs

        if self._mcts is None:
            self._mcts = AlphaBuilderMCTS(
                predict_fn=predict_fn,
                num_simulations=self.request.phase1_mcts.num_simulations,
                batch_size=self.request.phase1_mcts.batch_size,
                action_batch_size=self.request.phase1_mcts.action_batch_size,
                c_puct=self.request.phase1_mcts.c_puct,
                top_k_expansion=self.request.phase1_mcts.top_k_expansion,
                predict_batch_fn=predict_batch_fn,
            )
        # Hold a local reference: an edit can reset self._mcts while this step
        # is still running, and the stale outcome is discarded by generation.
        mcts = self._mcts
        mcts.predict_fn = predict_fn
        mcts.predict_batch_fn = predict_batch_fn
        mcts.config = self._mcts_config(phase)
        result = mcts.search(density, bc_masks, force_tuple)

        valid_add, valid_remove = get_legal_moves(
            density, 0.5, bc_masks=bc_masks, forces=force_tuple
        )
        prob_add, prob_remove = masked_softmax(
            p_add,
            p_rem,
            valid_add,
            valid_remove,
            temperature=1.0,
        )

        pv_branches = select_pv_branches(
            result.root,
            max_actions=mcts.config.action_batch_size,
            max_depth=4,
        )
        if pv_branches:
            actions = [act for branch in pv_branches for act, _, _ in branch]
        elif result.actions:
            actions = list(result.actions)
        else:
            actions = []

        new_density = density.copy()
        for channel, x, y, z in actions:
            new_density[x, y, z] = 1.0 if channel == 0 else 0.0

        analysis = analyze_structure_islands(
            new_density, load_cfg_dict, bc_masks=bc_masks
        )
        vol_frac = float(np.mean(new_density > 0.5))
        total_voxels = int((new_density > 0.5).sum())
        island_penalty = calculate_island_penalty(
            analysis["n_islands"], analysis["loose_voxels"], total_voxels
        )
        conn_bonus, connected_load_fraction = calculate_connectivity_reward(
            new_density, bc_masks, force_tuple
        )
        connectivity_bonus = conn_bonus if phase == "GROWTH" else 0.0
        next_valid_add, next_valid_remove = get_legal_moves(
            new_density, 0.5, bc_masks=bc_masks, forces=force_tuple
        )
        legal_add_count = int(next_valid_add.sum())
        legal_remove_count = int(next_valid_remove.sum())
        phi_after, _ = calculate_pbrs_potential(
            new_density, bc_masks, force_tuple, load_cfg_dict
        )
        pbrs_delta = (
            float(alpha_guide * (phi_after - phi_before))
            if (phase == "GROWTH" and alpha_guide > 0.0)
            else None
        )

        new_phase = phase
        reset_tree = False
        if phase == "GROWTH":
            if analysis["is_connected"]:
                new_phase = "REFINEMENT"
                reset_tree = True
            elif step + 1 >= self.request.episode_config.max_phase1_steps:
                new_phase = "REFINEMENT"
                reset_tree = True

        compliance: Optional[float] = None
        max_disp: Optional[float] = None
        displacement_map: Optional[np.ndarray] = None
        fem_reward: Optional[float] = None
        collapsed = False
        fem_enabled = self.fem_enabled

        if new_phase == "REFINEMENT":
            if self.fem_enabled:
                if not analysis["is_connected"]:
                    fem_reward = -1.0
                    collapsed = True
                else:
                    from alphabuilder.src.logic.selfplay.evaluator import evaluate_main_island

                    fem_result = evaluate_main_island(
                        new_density,
                        self.load_config.model_dump(),
                        analysis["main_island_mask"],
                        bc_type=self.bc_type,
                    )
                    if fem_result.valid:
                        compliance = float(fem_result.compliance)
                        max_disp = float(fem_result.max_displacement)
                        displacement_map = fem_result.displacement_map
                        fem_reward = calculate_reward(
                            compliance=compliance,
                            vol_frac=vol_frac,
                            is_valid=True,
                            max_displacement=max_disp,
                        )
                        collapsed = fem_reward == -1.0
                    else:
                        fem_reward = -1.0
                        collapsed = True
            elif not self.request.allow_neural_only_phase2:
                raise ApiError(
                    501,
                    "CAPABILITY_UNAVAILABLE",
                    "Phase 2 requires FEniCSx and allow_neural_only_phase2 is false.",
                )

        if new_phase == "REFINEMENT":
            if fem_reward is not None:
                step_value = float(np.clip(0.5 * fem_reward + 0.5 * value - island_penalty, -1.0, 1.0))
            else:
                step_value = float(np.clip(value - island_penalty, -1.0, 1.0))
        else:
            step_value = float(np.clip(value, -1.0, 1.0))

        if fem_reward is not None:
            comp_value = compliance if compliance is not None else 0.0
            compliance_score = calculate_compliance_score(comp_value)
            volume_bonus = calculate_volume_bonus(vol_frac)
            reward_v33 = {
                "compliance_score": float(compliance_score),
                "volume_bonus": float(volume_bonus),
                "fem_score": float(np.clip(compliance_score + volume_bonus, -1.0, 1.0)),
                "displacement_penalty": float(calculate_displacement_penalty(max_disp)),
                "island_penalty": float(island_penalty),
                "connectivity_bonus": float(connectivity_bonus),
                "pbrs_delta": pbrs_delta,
                "total_reward": float(
                    np.clip(fem_reward - island_penalty, -1.0, 1.0)
                ),
                "is_collapsed": bool(collapsed),
            }
        else:
            reward_v33 = {
                "compliance_score": None,
                "volume_bonus": None,
                "fem_score": None,
                "displacement_penalty": 0.0,
                "island_penalty": float(island_penalty),
                "connectivity_bonus": float(connectivity_bonus),
                "pbrs_delta": pbrs_delta,
                "total_reward": float(step_value),
                "is_collapsed": False,
            }

        root = result.root
        stats = {
            "num_simulations": int(result.num_simulations),
            "nodes_expanded": 0,
            "max_depth": 0,
            "cache_hits": len(getattr(mcts, "eval_cache", {})),
            "top8_concentration": 0.0,
            "refutation": False,
        }
        if root is not None and hasattr(root, "tree_stats"):
            nodes, depth = root.tree_stats()
            stats["nodes_expanded"] = int(nodes)
            stats["max_depth"] = int(depth)
        visits = result.visit_distribution or {}
        total_visits = sum(visits.values())
        if total_visits:
            sorted_visits = sorted(visits.values(), reverse=True)
            stats["top8_concentration"] = float(sum(sorted_visits[:8]) / total_visits)
            best_visit_action = max(visits.items(), key=lambda item: item[1])[0]
            all_pol = np.concatenate([p_add.reshape(-1), p_rem.reshape(-1)])
            best_idx = int(np.argmax(all_pol))
            spatial = p_add.size
            if best_idx < spatial:
                best_policy = (0,) + np.unravel_index(best_idx, p_add.shape)
            else:
                best_policy = (1,) + np.unravel_index(best_idx - spatial, p_rem.shape)
            stats["refutation"] = bool(tuple(best_visit_action) != tuple(best_policy))

        selected = []
        if pv_branches:
            for branch_index, branch in enumerate(pv_branches):
                for action, visit_count, mean_value in branch:
                    channel, x, y, z = action
                    selected.append(
                        {
                            "channel": int(channel),
                            "x": int(x),
                            "y": int(y),
                            "z": int(z),
                            "visits": int(visit_count),
                            "q_value": float(mean_value),
                            "branch_index": branch_index,
                        }
                    )
        else:
            for branch_index, action in enumerate(actions):
                channel, x, y, z = action
                child = root.children.get(action) if root is not None else None
                selected.append(
                    {
                        "channel": int(channel),
                        "x": int(x),
                        "y": int(y),
                        "z": int(z),
                        "visits": int(child.visit_count) if child is not None else int(visits.get(action, 0)),
                        "q_value": float(child.mean_value) if child is not None else 0.0,
                        "branch_index": branch_index,
                    }
                )

        step_index = step + 1
        metadata = {
            "step": step_index,
            "phase": new_phase,
            "value": float(value),
            "value_target": float(fem_reward) if fem_reward is not None else None,
            "fem_reward": float(fem_reward) if fem_reward is not None else None,
            "compliance_fem": compliance,
            "max_displacement": max_disp,
            "volume_fraction": vol_frac,
            "is_connected": bool(analysis["is_connected"]),
            "n_islands": int(analysis["n_islands"]),
            "loose_voxels": int(analysis["loose_voxels"]),
            "connected_load_fraction": float(connected_load_fraction),
            "island_penalty": float(island_penalty),
            "reward_components": {
                "connectivity_bonus": float(connectivity_bonus),
                "island_penalty": float(island_penalty),
                "total": float(step_value),
            },
            "reward_v33": reward_v33,
            "mcts_stats": stats,
            "selected_actions": selected,
            "step_elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "inference_ms": round(inference_ms, 2),
            "legal_add_count": legal_add_count,
            "legal_remove_count": legal_remove_count,
            "fem_enabled": fem_enabled,
            "resolution": list(self.dims),
            "bc_type": self.bc_type,
            "load_config": self.load_config.model_dump(),
        }

        tensors: list[wire.WireTensor] = [
            wire.dense_u8(TAG_DENSITY_ALIAS, self.dims, (new_density > 0.5).astype(np.uint8).reshape(-1))
        ]
        add_idx, add_val = wire.sparse_from_dense(prob_add, 1e-5)
        rem_idx, rem_val = wire.sparse_from_dense(prob_remove, 1e-5)
        tensors.append(wire.coo1_f32(wire.TAG_POLICY_ADD, self.dims, add_idx, add_val))
        tensors.append(wire.coo1_f32(wire.TAG_POLICY_REMOVE, self.dims, rem_idx, rem_val))

        visit_add = np.zeros(self.dims, dtype=np.float32)
        visit_rem = np.zeros(self.dims, dtype=np.float32)
        q_add = np.zeros(self.dims, dtype=np.float32)
        q_rem = np.zeros(self.dims, dtype=np.float32)
        if root is not None:
            for action, count in root.collect_subtree_visits().items():
                channel, x, y, z = action
                if channel == 0:
                    visit_add[x, y, z] += float(count)
                else:
                    visit_rem[x, y, z] += float(count)
            for action, child in root.children.items():
                channel, x, y, z = action
                if channel == 0:
                    q_add[x, y, z] = float(child.mean_value)
                else:
                    q_rem[x, y, z] = float(child.mean_value)
        vi_a, vv_a = wire.sparse_from_dense(visit_add, 0.0)
        vi_r, vv_r = wire.sparse_from_dense(visit_rem, 0.0)
        tensors.append(wire.coo1_f32(wire.TAG_MCTS_VISIT_ADD, self.dims, vi_a, vv_a))
        tensors.append(wire.coo1_f32(wire.TAG_MCTS_VISIT_REMOVE, self.dims, vi_r, vv_r))
        qi_a = [idx for idx, val in zip(vi_a, vv_a) if val > 0]
        qv_a = [float(q_add.reshape(-1)[idx]) for idx in qi_a]
        qi_r = [idx for idx, val in zip(vi_r, vv_r) if val > 0]
        qv_r = [float(q_rem.reshape(-1)[idx]) for idx in qi_r]
        tensors.append(wire.coo1_f32(wire.TAG_MCTS_Q_ADD, self.dims, qi_a, qv_a))
        tensors.append(wire.coo1_f32(wire.TAG_MCTS_Q_REMOVE, self.dims, qi_r, qv_r))

        if include_blueprint and blueprint is not None:
            tensors.append(
                wire.dense_f16(
                    wire.TAG_BLUEPRINT_PROBS,
                    self.dims,
                    np.asarray(blueprint, dtype=np.float32).reshape(-1),
                )
            )
        if displacement_map is not None:
            tensors.append(
                wire.dense_f16(
                    wire.TAG_DISPLACEMENT_MAG,
                    self.dims,
                    np.asarray(displacement_map, dtype=np.float32).reshape(-1),
                )
            )

        frame = wire.encode_packet(
            wire.PAYLOAD_STEP_FRAME,
            metadata,
            tensors,
            compress=True,
            flags=0x01,
        )

        end_reason = None
        final_score = float(step_value)
        if new_phase == "REFINEMENT" and fem_reward is not None:
            final_score = float(np.clip(fem_reward - island_penalty, -1.0, 1.0))
        if collapsed:
            end_reason = "COLLAPSE"
        elif step_index >= self.request.episode_config.max_steps:
            end_reason = "MAX_STEPS"

        if reset_tree:
            mcts.reset()
        elif actions:
            if len(pv_branches) == 1:
                for act, _, _ in pv_branches[0]:
                    mcts.step(act)
                if phase == "GROWTH" and alpha_guide > 0.0 and abs(phi_after - phi_before) > 1e-9:
                    mcts.eval_cache = {}
            elif not pv_branches and len(actions) == 1:
                mcts.step(actions[0])
                if phase == "GROWTH" and alpha_guide > 0.0 and abs(phi_after - phi_before) > 1e-9:
                    mcts.eval_cache = {}
            else:
                mcts.reset()
        else:
            mcts.clear_tree()

        return _StepOutcome(
            density=new_density,
            phase=new_phase,
            step=step_index,
            metadata=metadata,
            frame=frame,
            end_reason=end_reason,
            fem_enabled=fem_enabled,
            final_score=final_score,
            final_compliance=compliance,
            final_volume=vol_frac,
        )


TAG_DENSITY_ALIAS = wire.TAG_DENSITY


class SessionManager:
    """Thread-safe registry of live sessions."""

    def __init__(self, config: ServerConfig, worker: ComputeWorker, models: ModelManager) -> None:
        self.config = config
        self.worker = worker
        self.models = models
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def has_running(self) -> bool:
        with self._lock:
            return any(s.status == "RUNNING" for s in self._sessions.values())

    def create(self, request: SessionCreateRequest) -> LiveSession:
        if request.kind == "MCTS_LIVE":
            if not has_fenics() and not request.allow_neural_only_phase2:
                raise ApiError(
                    501,
                    "CAPABILITY_UNAVAILABLE",
                    "Phase 2 requires FEniCSx (allow_neural_only_phase2=false).",
                )
        with self._lock:
            active = [
                s
                for s in self._sessions.values()
                if s.status not in {"COMPLETED", "FAILED", "CANCELLED"}
            ]
            if len(active) >= self.config.max_concurrent_sessions:
                raise ApiError(409, "ENGINE_BUSY", "Maximum concurrent sessions reached.")
        session = LiveSession(request, self.config, self)
        with self._lock:
            self._sessions[session.session_id] = session
        session.start()
        return session

    def get(self, session_id: str) -> LiveSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ApiError(404, "NOT_FOUND", f"Session not found: {session_id}")
        return session

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [s.state_payload() for s in sessions]

    def delete(self, session_id: str) -> None:
        session = self.get(session_id)
        session.cancel()
        time.sleep(0.01)
        with self._lock:
            self._sessions.pop(session_id, None)

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.cancel()
