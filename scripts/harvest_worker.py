#!/usr/bin/env python3
"""
Pull-based Harvest Worker Agent for AlphaBuilder v3.3.

Runs N independent single-rank worker slots (--executor native | docker), enforces
disk/RAM/pause guards, renews heartbeats, audits SQLite shards, verifies MD5 via rclone,
and reports completion to the Oracle Supervisor.
Stdlib-only (no third-party packages required on the host).
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_STOP_EVENT = threading.Event()


def _http_json(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 15.0,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if resp.status == 204 or not raw:
                return resp.status, None
            return resp.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {"error": str(e)}
        except Exception:
            body = {"error": raw.decode("utf-8", errors="replace")}
        return e.code, body


def _read_ram_free_mb() -> float:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemAvailable:"):
                kb = float(line.split()[1])
                return round(kb / 1024.0, 1)
    return 0.0


def _read_cpu_busy_pct() -> float:
    try:
        load1 = os.getloadavg()[0]
        cpus = os.cpu_count() or 1
        return round(min(100.0, (load1 / cpus) * 100.0), 1)
    except Exception:
        return 0.0


def _read_disk_free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return round(shutil.disk_usage(str(path)).free / (1024.0 ** 3), 2)


def _detect_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _probe_runtime_versions(args: argparse.Namespace) -> Tuple[str, str]:
    code = "import dolfinx, numpy; print(dolfinx.__version__, numpy.__version__)"
    try:
        if args.executor == "docker":
            cmd = ["docker", "run", "--rm", args.docker_image, "-c", code]
        else:
            cmd = [args.python_bin, "-c", code]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=60).strip()
        parts = out.splitlines()[-1].split()
        if len(parts) >= 2:
            return parts[0], parts[1]
    except Exception as e:
        print(f"[worker] WARNING: could not probe dolfinx/numpy version: {e}", flush=True)
    return "unknown", "unknown"


def _compute_file_hashes(path: Path) -> Tuple[int, str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            sha.update(chunk)
            md5.update(chunk)
    return size, sha.hexdigest(), md5.hexdigest()


def _finalize_sqlite_file(part_db: Path) -> None:
    conn = sqlite3.connect(str(part_db), timeout=60.0)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
        conn.commit()
    finally:
        conn.close()


def _collect_db_manifest(
    db_path: Path,
    shard: Dict[str, Any],
    code_commit: str,
    dolfinx_ver: str,
    numpy_ver: str,
    worker_id: str,
    duration_sec: float,
) -> Dict[str, Any]:
    size_bytes, sha256_hex, md5_hex = _compute_file_hashes(db_path)

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        ep_rows = conn.execute(
            """
            SELECT episode_id, seed, strategy, bc_type,
                   final_compliance, final_volume, final_max_displacement, final_is_connected
            FROM episodes
            ORDER BY seed ASC
            """
        ).fetchall()

        summaries = []
        completed_seeds: List[int] = []
        total_records = 0

        for r in ep_rows:
            seed = int(r["seed"])
            completed_seeds.append(seed)
            rec_stats = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN phase='GROWTH' THEN 1 ELSE 0 END) AS n_p1,
                    SUM(CASE WHEN phase='REFINEMENT' THEN 1 ELSE 0 END) AS n_p2,
                    MIN(fitness_score) AS f_min,
                    MAX(fitness_score) AS f_max
                FROM records
                WHERE episode_id = ?
                """,
                (r["episode_id"],),
            ).fetchone()
            n_p1 = int(rec_stats["n_p1"] or 0)
            n_p2 = int(rec_stats["n_p2"] or 0)
            total_records += n_p1 + n_p2
            summaries.append(
                {
                    "seed": seed,
                    "episode_id": r["episode_id"],
                    "strategy": r["strategy"],
                    "bc_type": r["bc_type"],
                    "final_compliance": r["final_compliance"],
                    "final_volume": r["final_volume"],
                    "final_max_displacement": r["final_max_displacement"],
                    "is_connected": int(r["final_is_connected"] or 0),
                    "n_phase1": n_p1,
                    "n_phase2": n_p2,
                    "fitness_min": rec_stats["f_min"],
                    "fitness_max": rec_stats["f_max"],
                }
            )
    finally:
        conn.close()

    seed_start = int(shard["seed_start"])
    seed_count = int(shard["seed_count"])
    expected_seeds = list(range(seed_start, seed_start + seed_count))
    completed_set = set(completed_seeds)
    failed_seeds = [s for s in expected_seeds if s not in completed_set]

    return {
        "schema_version": "v3.3",
        "shard_id": shard["shard_id"],
        "wave": shard["wave"],
        "revision": int(shard.get("revision", 1)),
        "code_git_commit": code_commit,
        "dolfinx_version": dolfinx_ver,
        "numpy_version": numpy_ver,
        "worker_id": worker_id,
        "worker_arch": platform.machine(),
        "resolution": [64, 32, 8],
        "seed_start": seed_start,
        "seed_count": seed_count,
        "seeds_completed": completed_seeds,
        "seeds_failed": failed_seeds,
        "record_count": total_records,
        "db_filename": db_path.name,
        "db_size_bytes": size_bytes,
        "db_sha256": sha256_hex,
        "db_md5": md5_hex,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_total_sec": round(duration_sec, 2),
        "episodes_summary": summaries,
    }


def _upload_and_verify_rclone(
    rclone_bin: str,
    remote_base: str,
    wave: str,
    db_path: Path,
    manifest_path: Path,
    expected_size: int,
    expected_md5: str,
) -> None:
    sub = "pilot" if wave == "pilot" else ("shards" if wave == "prod" else wave)
    base = remote_base.rstrip("/")
    sep = "" if base.endswith(":") else "/"
    remote_dir = f"{base}{sep}{sub}"
    remote_db = f"{remote_dir}/{db_path.name}"
    remote_manifest = f"{remote_dir}/{manifest_path.name}"

    def _lsjson_check() -> Optional[Dict[str, Any]]:
        res = subprocess.run(
            [rclone_bin, "lsjson", "--hash", remote_db],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if res.returncode in (3, 4):
            return None
        if res.returncode != 0:
            raise RuntimeError(f"rclone lsjson failed ({res.returncode}): {res.stderr.strip()}")
        out = res.stdout.strip()
        items = json.loads(out) if out else []
        return items[0] if items else None

    # Check if exact file already exists on remote before overwriting
    existing = _lsjson_check()
    if existing:
        rem_md5 = (existing.get("Hashes") or {}).get("md5", "").lower()
        rem_size = int(existing.get("Size", -1))
        if rem_size == expected_size and rem_md5 == expected_md5.lower():
            pass
        else:
            subprocess.run(
                [
                    rclone_bin,
                    "copyto",
                    str(db_path),
                    remote_db,
                    "--checksum",
                    "--retries",
                    "5",
                    "--low-level-retries",
                    "10",
                    "--timeout",
                    "120s",
                ],
                check=True,
                timeout=600,
            )
    else:
        subprocess.run(
            [
                rclone_bin,
                "copyto",
                str(db_path),
                remote_db,
                "--ignore-existing",
                "--checksum",
                "--retries",
                "5",
                "--low-level-retries",
                "10",
                "--timeout",
                "120s",
            ],
            check=True,
            timeout=600,
        )

    verified = _lsjson_check()
    if not verified:
        raise RuntimeError(f"rclone lsjson returned empty after upload of {remote_db}")
    rem_md5 = (verified.get("Hashes") or {}).get("md5", "").lower()
    rem_size = int(verified.get("Size", -1))
    if rem_size != expected_size or rem_md5 != expected_md5.lower():
        raise RuntimeError(
            f"Remote checksum/size mismatch for {remote_db}: "
            f"remote=({rem_size}, {rem_md5}) vs local=({expected_size}, {expected_md5})"
        )

    subprocess.run(
        [
            rclone_bin,
            "copyto",
            str(manifest_path),
            remote_manifest,
            "--checksum",
            "--retries",
            "5",
            "--timeout",
            "60s",
        ],
        check=True,
        timeout=180,
    )


class RetentionManager:
    def __init__(self, retain_count: int) -> None:
        self.retain_count = int(retain_count)
        self._lock = threading.Lock()
        self._recent: List[Path] = []

    def register_and_prune(self, db_path: Path) -> None:
        if self.retain_count < 0:
            return
        with self._lock:
            self._recent.append(db_path)
            while len(self._recent) > self.retain_count:
                old = self._recent.pop(0)
                if old.exists():
                    try:
                        old.unlink()
                    except OSError as e:
                        print(f"[worker] WARNING: failed to delete retained {old}: {e}", flush=True)


def run_slot(
    slot_idx: int,
    args: argparse.Namespace,
    code_commit: str,
    dolfinx_ver: str,
    numpy_ver: str,
    retention: RetentionManager,
    lease_lock: threading.Lock,
    completed_counter: Dict[str, Any],
) -> None:
    slot_tag = f"{args.worker_id}#slot{slot_idx}"
    work_dir = Path(args.work_dir).resolve()
    runs_dir = work_dir / "runs"
    done_dir = work_dir / "done"
    runs_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    pause_path = Path(args.pause_file).expanduser()

    while not _STOP_EVENT.is_set():
        with lease_lock:
            if args.max_shards is not None and completed_counter["claimed"] >= args.max_shards:
                return

        if pause_path.exists():
            print(f"[{slot_tag}] Pause file {pause_path} present; sleeping {args.poll_interval}s.", flush=True)
            _STOP_EVENT.wait(args.poll_interval)
            continue

        disk_free = _read_disk_free_gb(work_dir)
        ram_free = _read_ram_free_mb()
        if disk_free < args.min_disk_gb or (ram_free > 0 and ram_free < args.min_ram_mb):
            print(
                f"[{slot_tag}] Guard tripped (disk={disk_free:.2f}GB < {args.min_disk_gb}GB or "
                f"ram={ram_free:.0f}MB < {args.min_ram_mb}MB); waiting {args.poll_interval}s.",
                flush=True,
            )
            _STOP_EVENT.wait(args.poll_interval)
            continue

        telemetry = {
            "worker_id": args.worker_id,
            "arch": platform.machine(),
            "dolfinx_version": dolfinx_ver,
            "code_git_commit": code_commit,
            "slots": args.slots,
            "cpu_busy": _read_cpu_busy_pct(),
            "ram_free_mb": ram_free,
            "disk_free_gb": disk_free,
        }

        with lease_lock:
            if args.max_shards is not None and completed_counter["claimed"] >= args.max_shards:
                return
            try:
                code, lease = _http_json(
                    f"{args.supervisor_url.rstrip('/')}/api/v1/lease",
                    method="POST",
                    payload=telemetry,
                )
            except Exception as e:
                print(f"[{slot_tag}] Lease request error: {e}", flush=True)
                lease = None
                code = 0

            if code == 200 and lease:
                completed_counter["claimed"] += 1

        if code == 409:
            print(f"[{slot_tag}] Supervisor returned 409 CODE_MISMATCH: {lease}. Stopping.", flush=True)
            _STOP_EVENT.set()
            return

        if code == 204 or not lease:
            _STOP_EVENT.wait(args.poll_interval)
            continue

        shard_id = lease["shard_id"]
        wave = lease["wave"]
        seed_start = int(lease["seed_start"])
        seed_count = int(lease["seed_count"])
        lease_token = lease["lease_token"]
        attempt = int(lease.get("attempt_count", 1))
        print(
            f"[{slot_tag}] Leased {shard_id} (wave={wave}, seeds={seed_start}..{seed_start + seed_count - 1}, attempt={attempt})",
            flush=True,
        )

        shard_run_dir = runs_dir / shard_id
        shard_run_dir.mkdir(parents=True, exist_ok=True)
        part_db = shard_run_dir / f"{shard_id}.part.db"
        container_name = f"ab_{args.worker_id}_{shard_id}_a{attempt}_{slot_idx}".replace("/", "_")

        abort_flag = threading.Event()
        stop_hb = threading.Event()
        proc_holder: Dict[str, Optional[subprocess.Popen]] = {"proc": None}

        def _terminate_child() -> None:
            p = proc_holder.get("proc")
            if args.executor == "docker":
                subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=10)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass

        def _hb_loop() -> None:
            while not stop_hb.is_set() and not _STOP_EVENT.is_set():
                if stop_hb.wait(60.0):
                    break
                hb_payload = {
                    "worker_id": args.worker_id,
                    "shard_id": shard_id,
                    "lease_token": lease_token,
                    "cpu_busy": _read_cpu_busy_pct(),
                    "ram_free_mb": _read_ram_free_mb(),
                    "disk_free_gb": _read_disk_free_gb(work_dir),
                }
                try:
                    hb_code, hb_resp = _http_json(
                        f"{args.supervisor_url.rstrip('/')}/api/v1/heartbeat",
                        method="POST",
                        payload=hb_payload,
                    )
                    if hb_code == 200 and hb_resp and hb_resp.get("action") == "ABORT_LEASE_LOST":
                        print(f"[{slot_tag}] Lease lost for {shard_id}: {hb_resp}. Aborting subprocess.", flush=True)
                        abort_flag.set()
                        _terminate_child()
                        break
                except Exception as e:
                    print(f"[{slot_tag}] Heartbeat warning ({shard_id}): {e}", flush=True)

        hb_thread = threading.Thread(target=_hb_loop, daemon=True)
        hb_thread.start()

        t_start = time.time()
        try:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"

            if args.executor == "docker":
                uid_gid = f"{os.getuid()}:{os.getgid()}"
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    container_name,
                    "--user",
                    uid_gid,
                    "-e",
                    "HOME=/tmp",
                    "-e",
                    "XDG_CACHE_HOME=/tmp/.cache",
                    "--cpus",
                    str(args.docker_cpus),
                    "--memory",
                    str(args.docker_memory),
                    "-v",
                    f"{shard_run_dir}:/data",
                    args.docker_image,
                    "/workspace/alphabuild/run_data_harvest.py",
                    "--episodes",
                    str(seed_count),
                    "--seed-offset",
                    str(seed_start),
                    "--max-iter",
                    str(args.max_iter),
                    "--db-path",
                    f"/data/{part_db.name}",
                    "--shard-id",
                    shard_id,
                    "--worker-id",
                    args.worker_id,
                    "--code-version",
                    f"{code_commit}+v3.3",
                    "--resume-shard",
                ]
            else:
                repo_dir = Path(args.repo_dir).resolve()
                cmd = [
                    args.python_bin,
                    str(repo_dir / "run_data_harvest.py"),
                    "--episodes",
                    str(seed_count),
                    "--seed-offset",
                    str(seed_start),
                    "--max-iter",
                    str(args.max_iter),
                    "--db-path",
                    str(part_db),
                    "--shard-id",
                    shard_id,
                    "--worker-id",
                    args.worker_id,
                    "--code-version",
                    f"{code_commit}+v3.3",
                    "--resume-shard",
                ]

            proc_holder["proc"] = subprocess.Popen(cmd, env=env)
            while proc_holder["proc"].poll() is None:
                if _STOP_EVENT.is_set() or abort_flag.is_set():
                    _terminate_child()
                    break
                time.sleep(1.0)

            rc = proc_holder["proc"].wait()
            if abort_flag.is_set() or _STOP_EVENT.is_set():
                raise RuntimeError("Shard execution aborted due to lease loss or shutdown")
            if rc != 0:
                raise RuntimeError(f"run_data_harvest exited with code {rc}")
            if not part_db.exists():
                raise RuntimeError(f"Expected part DB missing: {part_db}")

            _finalize_sqlite_file(part_db)

            # Audit generated shard
            if args.executor == "docker":
                uid_gid = f"{os.getuid()}:{os.getgid()}"
                audit_cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--user",
                    uid_gid,
                    "-e",
                    "HOME=/tmp",
                    "-e",
                    "XDG_CACHE_HOME=/tmp/.cache",
                    "-v",
                    f"{shard_run_dir}:/data",
                    args.docker_image,
                    "/workspace/alphabuild/scripts/audit_harvest_db.py",
                    f"/data/{part_db.name}",
                    "--sample-steps",
                    "250",
                ]
            else:
                repo_dir = Path(args.repo_dir).resolve()
                audit_cmd = [
                    args.python_bin,
                    str(repo_dir / "scripts" / "audit_harvest_db.py"),
                    str(part_db),
                    "--sample-steps",
                    "250",
                ]
            audit_res = subprocess.run(audit_cmd, capture_output=True, text=True, timeout=180)
            if audit_res.returncode != 0:
                raise RuntimeError(f"audit_harvest_db failed ({audit_res.returncode}): {audit_res.stdout}\n{audit_res.stderr}")

            final_db = done_dir / f"{shard_id}.db"
            shutil.move(str(part_db), str(final_db))
            duration = time.time() - t_start

            manifest = _collect_db_manifest(
                final_db,
                lease,
                code_commit=code_commit,
                dolfinx_ver=dolfinx_ver,
                numpy_ver=numpy_ver,
                worker_id=args.worker_id,
                duration_sec=duration,
            )
            if manifest["seeds_failed"]:
                raise RuntimeError(f"Incomplete shard: missing seeds {manifest['seeds_failed']}")

            manifest_path = done_dir / f"{shard_id}.manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            if args.rclone_remote:
                _upload_and_verify_rclone(
                    rclone_bin=args.rclone_bin,
                    remote_base=args.rclone_remote,
                    wave=wave,
                    db_path=final_db,
                    manifest_path=manifest_path,
                    expected_size=manifest["db_size_bytes"],
                    expected_md5=manifest["db_md5"],
                )

            comp_code, comp_resp = _http_json(
                f"{args.supervisor_url.rstrip('/')}/api/v1/complete",
                method="POST",
                payload={
                    "worker_id": args.worker_id,
                    "shard_id": shard_id,
                    "lease_token": lease_token,
                    "manifest": manifest,
                },
            )
            if comp_code != 200 or not comp_resp or comp_resp.get("status") != "ACK_VERIFIED":
                raise RuntimeError(f"Supervisor rejected completion ({comp_code}): {comp_resp}")

            retention.register_and_prune(final_db)
            shutil.rmtree(shard_run_dir, ignore_errors=True)
            print(
                f"[{slot_tag}] VERIFIED {shard_id}: {len(manifest['seeds_completed'])} eps, "
                f"{manifest['record_count']} records, {manifest['db_size_bytes']/1e6:.2f} MB in {duration:.1f}s",
                flush=True,
            )

        except Exception as e:
            err_str = str(e)
            print(f"[{slot_tag}] FAILED {shard_id}: {err_str}", flush=True)
            if not abort_flag.is_set() and not _STOP_EVENT.is_set():
                try:
                    _http_json(
                        f"{args.supervisor_url.rstrip('/')}/api/v1/fail",
                        method="POST",
                        payload={
                            "worker_id": args.worker_id,
                            "shard_id": shard_id,
                            "lease_token": lease_token,
                            "error": err_str[:1000],
                        },
                    )
                except Exception:
                    pass
        finally:
            stop_hb.set()
            hb_thread.join(timeout=5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaBuilder v3.3 Pull-Based Harvest Worker")
    parser.add_argument("--supervisor-url", type=str, required=True)
    parser.add_argument("--worker-id", type=str, default=socket.gethostname())
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--executor", type=str, choices=["native", "docker"], default="native")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--repo-dir", type=str, default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--docker-image", type=str, default="alphabuilder-harvest:v3.3")
    parser.add_argument("--docker-cpus", type=str, default="1.0")
    parser.add_argument("--docker-memory", type=str, default="2g")
    parser.add_argument("--work-dir", type=str, default="./alphabuild_work")
    parser.add_argument("--rclone-bin", type=str, default="rclone")
    parser.add_argument("--rclone-remote", type=str, default="gdrive_harvest:")
    parser.add_argument("--min-disk-gb", type=float, default=15.0)
    parser.add_argument("--min-ram-mb", type=float, default=2200.0)
    parser.add_argument("--retain-local-shards", type=int, default=0)
    parser.add_argument("--expected-commit", type=str, default="")
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--pause-file", type=str, default="~/.alphabuild_pause")
    parser.add_argument("--poll-interval", type=int, default=20)
    parser.add_argument("--max-shards", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    code_commit = args.expected_commit.strip() or _detect_git_commit(repo_dir)
    if not code_commit:
        raise SystemExit("Could not detect git commit; pass --expected-commit explicitly.")

    if args.rclone_remote:
        rclone_resolved = shutil.which(args.rclone_bin) or (args.rclone_bin if Path(args.rclone_bin).exists() else None)
        if not rclone_resolved:
            raise SystemExit(f"rclone binary not found: {args.rclone_bin}")
        args.rclone_bin = rclone_resolved

    dolfinx_ver, numpy_ver = _probe_runtime_versions(args)
    retention = RetentionManager(args.retain_local_shards)
    lease_lock = threading.Lock()
    completed_counter: Dict[str, Any] = {"claimed": 0}

    def _on_signal(signum, frame):
        print(f"[worker] Received signal {signum}, stopping cleanly...", flush=True)
        _STOP_EVENT.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    print(
        f"[worker] Starting {args.worker_id} ({platform.machine()}) | executor={args.executor} | "
        f"slots={args.slots} | commit={code_commit} | dolfinx={dolfinx_ver} | supervisor={args.supervisor_url}",
        flush=True,
    )

    threads: List[threading.Thread] = []
    for idx in range(args.slots):
        t = threading.Thread(
            target=run_slot,
            args=(idx, args, code_commit, dolfinx_ver, numpy_ver, retention, lease_lock, completed_counter),
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[worker] {args.worker_id} finished.", flush=True)


if __name__ == "__main__":
    main()
