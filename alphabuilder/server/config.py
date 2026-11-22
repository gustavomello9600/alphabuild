"""Server configuration and sandbox helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

MAX_VOXELS = 262_144
MAX_TENSOR_BYTES = 32 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@dataclass
class ServerConfig:
    """Runtime configuration for the compute engine."""

    host: str = "127.0.0.1"
    port: int = 8765
    checkpoints_dir: Path = Path("checkpoints")
    data_dir: Path = Path("data")
    api_token: str = ""
    harvest_supervisor_url: str = ""
    allow_insecure_remote: bool = False
    device: str = "auto"
    max_voxels: int = MAX_VOXELS
    max_concurrent_sessions: int = 1
    max_request_bytes: int = MAX_TENSOR_BYTES
    # Test/embedding hook: (checkpoint_path, device) -> model with predict methods.
    model_factory: Optional[Callable[[str, str], object]] = field(default=None, repr=False)

    def is_loopback(self) -> bool:
        return self.host.strip().lower() in LOOPBACK_HOSTS

    def validate(self) -> None:
        """Refuse an unauthenticated non-loopback bind unless explicitly allowed."""
        if not self.is_loopback() and not self.api_token and not self.allow_insecure_remote:
            raise RuntimeError(
                "Refusing to bind a non-loopback address without --api-token. "
                "Pass --allow-insecure-remote only for trusted isolated networks."
            )
        if "://" in self.harvest_supervisor_url and not self.harvest_supervisor_url.startswith(
            ("http://", "https://")
        ):
            raise RuntimeError("--harvest-supervisor-url must be http(s)")

    def resolve(self) -> "ServerConfig":
        self.checkpoints_dir = Path(self.checkpoints_dir).expanduser().resolve()
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.validate()
        return self

    @property
    def auth_required(self) -> bool:
        return bool(self.api_token)

    def safe_path(self, base: Path, relative: str) -> Path:
        """Resolve `relative` under `base`, rejecting traversal/symlink escapes."""
        from .schemas import ApiError

        normalized = str(relative).replace("\\", "/")
        parts = [part for part in normalized.split("/") if part not in {"", "."}]
        is_absolute = normalized.startswith("/") or (
            len(normalized) >= 2 and normalized[1] == ":"
        )
        if is_absolute or any(part == ".." for part in parts):
            raise ApiError(
                403,
                "PATH_OUTSIDE_SANDBOX",
                f"Path escapes the configured sandbox: {relative}",
                {"path": relative},
            )
        candidate = (base.joinpath(*parts)).resolve()
        base_resolved = base.resolve()
        if candidate != base_resolved and not candidate.is_relative_to(base_resolved):
            raise ApiError(
                403,
                "PATH_OUTSIDE_SANDBOX",
                f"Path escapes the configured sandbox: {relative}",
                {"path": relative},
            )
        return candidate
