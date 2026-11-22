"""CLI entry point: `python -m alphabuilder.server`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from . import __version__
from .app import create_app
from .config import ServerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alphabuilder.server",
        description="AlphaBuilder Compute & Inference Server v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--api-token", default="")
    parser.add_argument("--harvest-supervisor-url", default="")
    parser.add_argument("--allow-insecure-remote", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ServerConfig(
        host=args.host,
        port=args.port,
        checkpoints_dir=Path(args.checkpoints_dir),
        data_dir=Path(args.data_dir),
        api_token=args.api_token or "",
        harvest_supervisor_url=args.harvest_supervisor_url or "",
        allow_insecure_remote=bool(args.allow_insecure_remote),
        device=args.device,
    )
    try:
        config.resolve()
    except RuntimeError as exc:
        print(f"alphabuilder-engine: {exc}", file=sys.stderr)
        return 2
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
