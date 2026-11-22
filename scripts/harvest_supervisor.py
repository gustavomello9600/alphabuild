#!/usr/bin/env python3
"""
Pull-based Harvest Supervisor for AlphaBuilder v3.3.

Coordinates deterministic seed shards across heterogeneous workers over Tailscale/SSH
using SQLite WAL + BEGIN IMMEDIATE transactions and lease tokens.
Stdlib-only (no third-party packages required on Oracle supervisor host).
"""

import argparse
import json
import signal
import sqlite3
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_HOST = "100.104.121.28"
DEFAULT_PORT = 8420
DEFAULT_LEASE_TTL = 2400
MAX_ATTEMPTS = 5


def _now() -> float:
    return time.time()


class SupervisorState:
    """Thread-safe SQLite-backed coordinator for harvest shard leases."""

    def __init__(
        self,
        db_path: Path,
        expected_commit: str,
        active_wave: str,
        lease_ttl: int = DEFAULT_LEASE_TTL,
    ) -> None:
        self.db_path = db_path
        self.expected_commit = expected_commit
        self.active_wave = active_wave
        self.lease_ttl = int(lease_ttl)
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS shards (
                        shard_id TEXT PRIMARY KEY,
                        wave TEXT NOT NULL,
                        seed_start INTEGER NOT NULL,
                        seed_count INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        assigned_worker TEXT,
                        lease_token TEXT,
                        lease_expires_at REAL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        revision INTEGER NOT NULL DEFAULT 1,
                        manifest_json TEXT,
                        last_error TEXT,
                        updated_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_shards_wave_status
                        ON shards(wave, status, seed_start);

                    CREATE TABLE IF NOT EXISTS workers (
                        worker_id TEXT PRIMARY KEY,
                        arch TEXT,
                        dolfinx_version TEXT,
                        code_commit TEXT,
                        slots INTEGER,
                        cpu_busy REAL,
                        ram_free_mb REAL,
                        disk_free_gb REAL,
                        last_seen REAL NOT NULL
                    );
                    """
                )
            finally:
                conn.close()

    def seed_wave(
        self,
        wave: str,
        prefix: str,
        seed_start: int,
        num_episodes: int,
        shard_size: int = 10,
    ) -> int:
        """Idempotently populate shard rows for a harvest wave."""
        if num_episodes <= 0 or shard_size <= 0:
            return 0
        now = _now()
        rows = []
        idx = 0
        cur = seed_start
        end = seed_start + num_episodes
        while cur < end:
            count = min(shard_size, end - cur)
            shard_id = f"{prefix}{idx:04d}" if wave == "pilot" else f"{prefix}{idx:06d}"
            rows.append((shard_id, wave, cur, count, "PENDING", now))
            cur += count
            idx += 1

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                before = conn.total_changes
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO shards
                    (shard_id, wave, seed_start, seed_count, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                inserted = conn.total_changes - before
                conn.execute("COMMIT")
                return inserted
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def _upsert_worker(self, conn: sqlite3.Connection, payload: Dict[str, Any], now: float) -> None:
        worker_id = payload.get("worker_id")
        if not worker_id:
            return
        code_commit = payload.get("code_git_commit") or payload.get("code_commit")
        conn.execute(
            """
            INSERT INTO workers (
                worker_id, arch, dolfinx_version, code_commit, slots,
                cpu_busy, ram_free_mb, disk_free_gb, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                arch = COALESCE(excluded.arch, workers.arch),
                dolfinx_version = COALESCE(excluded.dolfinx_version, workers.dolfinx_version),
                code_commit = COALESCE(excluded.code_commit, workers.code_commit),
                slots = COALESCE(excluded.slots, workers.slots),
                cpu_busy = COALESCE(excluded.cpu_busy, workers.cpu_busy),
                ram_free_mb = COALESCE(excluded.ram_free_mb, workers.ram_free_mb),
                disk_free_gb = COALESCE(excluded.disk_free_gb, workers.disk_free_gb),
                last_seen = excluded.last_seen
            """,
            (
                str(worker_id),
                payload.get("arch"),
                payload.get("dolfinx_version"),
                code_commit,
                payload.get("slots"),
                payload.get("cpu_busy"),
                payload.get("ram_free_mb"),
                payload.get("disk_free_gb"),
                now,
            ),
        )

    def acquire_lease(self, payload: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]]]:
        worker_id = str(payload.get("worker_id") or "").strip()
        if not worker_id:
            return 400, {"error": "worker_id is required"}

        req_commit = str(payload.get("code_git_commit") or payload.get("code_commit") or "").strip()
        now = _now()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                self._upsert_worker(conn, payload, now)

                if self.expected_commit and req_commit != self.expected_commit:
                    conn.execute("COMMIT")
                    return 409, {
                        "error": "CODE_MISMATCH",
                        "expected_commit": self.expected_commit,
                        "received_commit": req_commit,
                    }

                if not self.active_wave or self.active_wave == "none":
                    conn.execute("COMMIT")
                    return 204, None

                # Reclaim expired leases on active wave
                conn.execute(
                    """
                    UPDATE shards
                    SET status = 'PENDING',
                        assigned_worker = NULL,
                        lease_token = NULL,
                        lease_expires_at = NULL,
                        last_error = 'lease_expired',
                        updated_at = ?
                    WHERE wave = ?
                      AND status IN ('LEASED', 'RUNNING')
                      AND lease_expires_at < ?
                      AND attempt_count < ?
                    """,
                    (now, self.active_wave, now, MAX_ATTEMPTS),
                )
                conn.execute(
                    """
                    UPDATE shards
                    SET status = 'FAILED',
                        assigned_worker = NULL,
                        lease_token = NULL,
                        lease_expires_at = NULL,
                        last_error = 'max_attempts_exceeded',
                        updated_at = ?
                    WHERE wave = ?
                      AND status IN ('LEASED', 'RUNNING')
                      AND lease_expires_at < ?
                      AND attempt_count >= ?
                    """,
                    (now, self.active_wave, now, MAX_ATTEMPTS),
                )

                row = conn.execute(
                    """
                    SELECT shard_id, wave, seed_start, seed_count, attempt_count, revision
                    FROM shards
                    WHERE wave = ? AND status = 'PENDING'
                    ORDER BY seed_start ASC
                    LIMIT 1
                    """,
                    (self.active_wave,),
                ).fetchone()

                if row is None:
                    conn.execute("COMMIT")
                    return 204, None

                lease_token = str(uuid.uuid4())
                expires_at = now + self.lease_ttl
                attempt_next = int(row["attempt_count"]) + 1

                conn.execute(
                    """
                    UPDATE shards
                    SET status = 'LEASED',
                        assigned_worker = ?,
                        lease_token = ?,
                        lease_expires_at = ?,
                        attempt_count = ?,
                        updated_at = ?
                    WHERE shard_id = ? AND status = 'PENDING'
                    """,
                    (worker_id, lease_token, expires_at, attempt_next, now, row["shard_id"]),
                )
                conn.execute("COMMIT")

                return 200, {
                    "shard_id": row["shard_id"],
                    "wave": row["wave"],
                    "seed_start": int(row["seed_start"]),
                    "seed_count": int(row["seed_count"]),
                    "lease_token": lease_token,
                    "lease_expires_at": expires_at,
                    "attempt_count": attempt_next,
                    "revision": int(row["revision"]),
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def heartbeat(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        now = _now()
        shard_id = payload.get("shard_id")
        lease_token = payload.get("lease_token")

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                self._upsert_worker(conn, payload, now)

                if not shard_id:
                    conn.execute("COMMIT")
                    return 200, {"action": "OK"}

                row = conn.execute(
                    "SELECT status, lease_token FROM shards WHERE shard_id = ?",
                    (shard_id,),
                ).fetchone()

                if row is None:
                    conn.execute("COMMIT")
                    return 200, {"action": "ABORT_LEASE_LOST", "reason": "unknown_shard"}

                if row["status"] not in ("LEASED", "RUNNING") or row["lease_token"] != lease_token:
                    conn.execute("COMMIT")
                    return 200, {
                        "action": "ABORT_LEASE_LOST",
                        "reason": f"status={row['status']}",
                    }

                new_expires = now + self.lease_ttl
                conn.execute(
                    """
                    UPDATE shards
                    SET status = 'RUNNING',
                        lease_expires_at = ?,
                        updated_at = ?
                    WHERE shard_id = ? AND lease_token = ?
                    """,
                    (new_expires, now, shard_id, lease_token),
                )
                conn.execute("COMMIT")
                return 200, {"action": "CONTINUE", "lease_expires_at": new_expires}
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def complete_shard(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        shard_id = str(payload.get("shard_id") or "").strip()
        lease_token = str(payload.get("lease_token") or "").strip()
        worker_id = str(payload.get("worker_id") or "").strip()
        manifest = payload.get("manifest")

        if not shard_id or not isinstance(manifest, dict):
            return 400, {"error": "shard_id and manifest dict are required"}

        manifest_str = json.dumps(manifest)
        now = _now()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT status, lease_token, seed_start, seed_count, manifest_json FROM shards WHERE shard_id = ?",
                    (shard_id,),
                ).fetchone()

                if row is None:
                    conn.execute("COMMIT")
                    return 404, {"error": f"Unknown shard {shard_id}"}

                if row["status"] == "VERIFIED":
                    existing = json.loads(row["manifest_json"]) if row["manifest_json"] else {}
                    if (
                        (lease_token and row["lease_token"] == lease_token)
                        or existing.get("db_sha256") == manifest.get("db_sha256")
                    ):
                        conn.execute("COMMIT")
                        return 200, {"status": "ACK_VERIFIED", "note": "already_verified"}
                    conn.execute("COMMIT")
                    return 409, {"error": "Shard already verified with different artifact"}

                valid_token = bool(lease_token and row["lease_token"] == lease_token)
                valid_unclaimed = (
                    row["status"] == "PENDING"
                    and manifest.get("shard_id") == shard_id
                    and int(manifest.get("seed_start", -1)) == int(row["seed_start"])
                    and not manifest.get("seeds_failed")
                )
                if not (valid_token or valid_unclaimed):
                    conn.execute("COMMIT")
                    return 409, {
                        "error": "LEASE_MISMATCH",
                        "current_status": row["status"],
                    }

                conn.execute(
                    """
                    UPDATE shards
                    SET status = 'VERIFIED',
                        assigned_worker = COALESCE(NULLIF(?, ''), assigned_worker),
                        lease_expires_at = NULL,
                        manifest_json = ?,
                        last_error = NULL,
                        updated_at = ?
                    WHERE shard_id = ?
                    """,
                    (worker_id, manifest_str, now, shard_id),
                )
                conn.execute("COMMIT")
                return 200, {"status": "ACK_VERIFIED", "shard_id": shard_id}
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def fail_shard(self, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        shard_id = str(payload.get("shard_id") or "").strip()
        lease_token = str(payload.get("lease_token") or "").strip()
        error_msg = str(payload.get("error") or "worker_reported_failure")[:1000]
        now = _now()

        if not shard_id:
            return 400, {"error": "shard_id is required"}

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT status, lease_token, attempt_count FROM shards WHERE shard_id = ?",
                    (shard_id,),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return 404, {"error": f"Unknown shard {shard_id}"}

                if row["status"] == "VERIFIED":
                    conn.execute("COMMIT")
                    return 200, {"status": "VERIFIED"}

                if row["lease_token"] != lease_token:
                    conn.execute("COMMIT")
                    return 409, {"error": "LEASE_MISMATCH"}

                new_status = "FAILED" if int(row["attempt_count"]) >= MAX_ATTEMPTS else "PENDING"
                conn.execute(
                    """
                    UPDATE shards
                    SET status = ?,
                        assigned_worker = NULL,
                        lease_token = NULL,
                        lease_expires_at = NULL,
                        last_error = ?,
                        updated_at = ?
                    WHERE shard_id = ?
                    """,
                    (new_status, error_msg, now, shard_id),
                )
                conn.execute("COMMIT")
                return 200, {"status": new_status, "shard_id": shard_id}
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def status_summary(self) -> Dict[str, Any]:
        now = _now()
        with self._lock:
            conn = self._connect()
            try:
                counts_rows = conn.execute(
                    "SELECT wave, status, COUNT(*) AS cnt FROM shards GROUP BY wave, status"
                ).fetchall()
                by_wave: Dict[str, Dict[str, int]] = {}
                for r in counts_rows:
                    by_wave.setdefault(r["wave"], {})[r["status"]] = int(r["cnt"])

                verified_rows = conn.execute(
                    "SELECT wave, manifest_json FROM shards WHERE status = 'VERIFIED'"
                ).fetchall()
                verified_episodes = 0
                verified_records = 0
                verified_duration = 0.0
                for r in verified_rows:
                    if r["manifest_json"]:
                        m = json.loads(r["manifest_json"])
                        verified_episodes += len(m.get("seeds_completed", []))
                        verified_records += int(m.get("record_count", 0))
                        verified_duration += float(m.get("duration_total_sec", 0.0))

                active_rows = conn.execute(
                    """
                    SELECT shard_id, wave, seed_start, seed_count, status,
                           assigned_worker, attempt_count, lease_expires_at
                    FROM shards
                    WHERE status IN ('LEASED', 'RUNNING')
                    ORDER BY seed_start ASC
                    """
                ).fetchall()
                active_leases = [
                    {
                        "shard_id": r["shard_id"],
                        "wave": r["wave"],
                        "seed_range": [int(r["seed_start"]), int(r["seed_start"]) + int(r["seed_count"]) - 1],
                        "status": r["status"],
                        "worker": r["assigned_worker"],
                        "attempt": int(r["attempt_count"]),
                        "ttl_remaining_sec": round(float(r["lease_expires_at"] or 0.0) - now, 1),
                    }
                    for r in active_rows
                ]

                worker_rows = conn.execute(
                    "SELECT * FROM workers ORDER BY last_seen DESC"
                ).fetchall()
                workers = [
                    {
                        "worker_id": r["worker_id"],
                        "arch": r["arch"],
                        "dolfinx_version": r["dolfinx_version"],
                        "code_commit": r["code_commit"],
                        "slots": r["slots"],
                        "cpu_busy": r["cpu_busy"],
                        "ram_free_mb": r["ram_free_mb"],
                        "disk_free_gb": r["disk_free_gb"],
                        "sec_since_seen": round(now - float(r["last_seen"]), 1),
                    }
                    for r in worker_rows
                ]

                eps_per_slot_hour = (
                    round(3600.0 * verified_episodes / verified_duration, 2)
                    if verified_duration > 0.0
                    else 0.0
                )

                return {
                    "ok": True,
                    "timestamp": now,
                    "expected_commit": self.expected_commit,
                    "active_wave": self.active_wave,
                    "shards_by_wave": by_wave,
                    "verified_episodes": verified_episodes,
                    "verified_records": verified_records,
                    "verified_slot_sec": round(verified_duration, 1),
                    "eps_per_slot_hour": eps_per_slot_hour,
                    "active_leases": active_leases,
                    "workers": workers,
                }
            finally:
                conn.close()


def make_handler(state: SupervisorState):
    class HarvestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, code: int, payload: Optional[Dict[str, Any]]) -> None:
            if code == 204 or payload is None:
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Optional[Dict[str, Any]]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
                return data if isinstance(data, dict) else {}
            except Exception:
                return None

        def log_message(self, format: str, *args: Any) -> None:
            sys.stdout.write(f"[supervisor] {self.address_string()} - {format % args}\n")
            sys.stdout.flush()

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/v1/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "expected_commit": state.expected_commit,
                        "active_wave": state.active_wave,
                        "lease_ttl_sec": state.lease_ttl,
                        "timestamp": _now(),
                    },
                )
            elif path == "/api/v1/status":
                self._send_json(200, state.status_summary())
            else:
                self._send_json(404, {"error": f"Not found: {path}"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            payload = self._read_json()
            if payload is None:
                self._send_json(400, {"error": "Invalid JSON body"})
                return

            if path == "/api/v1/lease":
                code, resp = state.acquire_lease(payload)
                self._send_json(code, resp)
            elif path == "/api/v1/heartbeat":
                code, resp = state.heartbeat(payload)
                self._send_json(code, resp)
            elif path == "/api/v1/complete":
                code, resp = state.complete_shard(payload)
                self._send_json(code, resp)
            elif path == "/api/v1/fail":
                code, resp = state.fail_shard(payload)
                self._send_json(code, resp)
            else:
                self._send_json(404, {"error": f"Not found: {path}"})

    return HarvestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaBuilder v3.3 Pull-Based Harvest Supervisor")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Bind IP (default Tailscale Oracle IP)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind TCP port (default 8420)")
    parser.add_argument("--db-path", type=str, default="supervisor_state.db", help="Supervisor SQLite DB path")
    parser.add_argument("--expected-commit", type=str, default="", help="Required worker git short commit hash")
    parser.add_argument(
        "--active-wave",
        type=str,
        choices=["pilot", "prod", "none"],
        default="pilot",
        help="Active wave served on POST /api/v1/lease",
    )
    parser.add_argument("--lease-ttl", type=int, default=DEFAULT_LEASE_TTL, help="Lease TTL seconds (default 2400)")
    parser.add_argument("--shard-size", type=int, default=10, help="Episodes per shard (default 10)")
    parser.add_argument("--init-pilot", action="store_true", help="Idempotently seed pilot wave (seeds 0..99)")
    parser.add_argument("--pilot-episodes", type=int, default=100, help="Number of pilot episodes (default 100)")
    parser.add_argument("--init-prod", action="store_true", help="Idempotently seed prod wave (seeds 100000..103999)")
    parser.add_argument("--prod-episodes", type=int, default=4000, help="Number of prod episodes (default 4000)")
    parser.add_argument("--prod-seed-start", type=int, default=100000, help="First prod seed (default 100000)")
    parser.add_argument("--init-only", action="store_true", help="Initialize DB/waves and exit without serving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host == "0.0.0.0":
        raise SystemExit("Refusing to bind supervisor to 0.0.0.0; bind to tailscale0 or 127.0.0.1.")

    state = SupervisorState(
        db_path=Path(args.db_path),
        expected_commit=args.expected_commit.strip(),
        active_wave=args.active_wave,
        lease_ttl=args.lease_ttl,
    )

    if args.init_pilot:
        added = state.seed_wave("pilot", "shard_pilot_", 0, args.pilot_episodes, args.shard_size)
        print(f"Seeded pilot wave: {added} new shard(s) added.", flush=True)

    if args.init_prod:
        added = state.seed_wave("prod", "shard_", args.prod_seed_start, args.prod_episodes, args.shard_size)
        print(f"Seeded prod wave: {added} new shard(s) added.", flush=True)

    if args.init_only:
        print(json.dumps(state.status_summary(), indent=2))
        return

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))

    def _shutdown(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(
        f"Supervisor listening on http://{args.host}:{args.port} | "
        f"wave={args.active_wave} | expected_commit={args.expected_commit or '<any>'}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
