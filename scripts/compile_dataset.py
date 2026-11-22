#!/usr/bin/env python3
"""
Deterministic Dataset Compiler for AlphaBuilder v3.3 Sharded Harvest.

Performs 5-stage streaming compilation from WORM SQLite shards + manifests:
1. Optional incremental download via rclone + SHA-256 & PRAGMA integrity verification.
2. Deterministic deduplication across seeds/episode_ids with invalid-shard quarantine.
3. Physics & policy legality gate auditing + header checks (`\\x93NUMPY` / `COO1`).
4. High-speed bulk streaming merge into a slim Kaggle-ready SQLite database
   (omitting `displacement_blob` and optionally `rho_cont_blob` while preserving
   `v_constructed_blob` for the Blueprint Head and raw physics scalars).
5. Deterministic stratified 90/10 episode-level split (`split_v3_3.json`), Markdown
   compilation report, and `TopologyDatasetV31` smoke verification.
"""

import argparse
import hashlib
import json
import random
import shutil
import sqlite3
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.logic.storage import (  # noqa: E402
    initialize_database,
    deserialize_array,
    deserialize_sparse,
    sparse_decode,
)
from alphabuilder.src.logic.harvest.generators import get_load_region_bounds  # noqa: E402
from alphabuilder.src.logic.mcts.legal_moves import (  # noqa: E402
    get_legal_add_moves,
    get_legal_remove_moves,
    get_protected_regions,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _quantiles(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": round(float(np.min(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
    }


def _sync_rclone(rclone_bin: str, rclone_remote: str, input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    remote = rclone_remote.rstrip("/")
    print(f"[compile] Syncing shards from {remote} -> {input_dir}...", flush=True)
    subprocess.run(
        [
            rclone_bin,
            "copy",
            remote,
            str(input_dir),
            "--include",
            "*.manifest.json",
            "--transfers",
            "8",
            "--checkers",
            "8",
        ],
        check=True,
    )
    subprocess.run(
        [
            rclone_bin,
            "copy",
            remote,
            str(input_dir),
            "--include",
            "shard_*.db",
            "--ignore-existing",
            "--checksum",
            "--transfers",
            "6",
            "--checkers",
            "8",
        ],
        check=True,
    )


def _quarantine(paths: List[Path], quarantine_dir: Path, reason: str) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compile] QUARANTINE: {reason}", flush=True)
    for p in paths:
        if p.exists():
            dest = quarantine_dir / p.name
            if dest.exists():
                dest = quarantine_dir / f"{p.stem}.{int(time.time())}{p.suffix}"
            shutil.move(str(p), str(dest))


def _verify_and_discover_shards(
    input_dir: Path,
    quarantine_dir: Path,
    expected_commit: Optional[str],
) -> Tuple[List[Tuple[Dict[str, Any], Path, Path]], List[str]]:
    manifest_files = sorted(
        p
        for p in input_dir.rglob("*.manifest.json")
        if quarantine_dir not in p.parents and p.parent != quarantine_dir
    )
    valid_candidates: List[Tuple[Dict[str, Any], Path, Path]] = []
    quarantined_notes: List[str] = []

    for mpath in manifest_files:
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as e:
            note = f"{mpath.name}: invalid JSON ({e})"
            _quarantine([mpath], quarantine_dir, note)
            quarantined_notes.append(note)
            continue

        schema_ver = manifest.get("schema_version")
        if schema_ver != "v3.3":
            note = f"{mpath.name}: unexpected schema_version={schema_ver}"
            quarantined_notes.append(note)
            continue

        if expected_commit:
            got_commit = str(manifest.get("code_git_commit") or "")
            if got_commit != expected_commit and not got_commit.startswith(expected_commit):
                note = f"{mpath.name}: commit mismatch ({got_commit} != {expected_commit})"
                quarantined_notes.append(note)
                continue

        db_name = manifest.get("db_filename") or mpath.name.replace(".manifest.json", ".db")
        db_path = mpath.parent / db_name
        if not db_path.exists():
            note = f"{mpath.name}: missing db {db_name}"
            quarantined_notes.append(note)
            continue

        expected_size = int(manifest.get("db_size_bytes", -1))
        actual_size = db_path.stat().st_size
        if expected_size >= 0 and actual_size != expected_size:
            note = f"{db_name}: size mismatch ({actual_size} != {expected_size})"
            _quarantine([db_path, mpath], quarantine_dir, note)
            quarantined_notes.append(note)
            continue

        expected_sha = str(manifest.get("db_sha256") or "").lower()
        actual_sha = _sha256_file(db_path).lower()
        if expected_sha and actual_sha != expected_sha:
            note = f"{db_name}: sha256 mismatch ({actual_sha[:12]} != {expected_sha[:12]})"
            _quarantine([db_path, mpath], quarantine_dir, note)
            quarantined_notes.append(note)
            continue

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"integrity_check={integrity}")
            ep_cnt = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            if ep_cnt == 0:
                raise RuntimeError("0 episodes in shard")
            conn.close()
        except Exception as e:
            note = f"{db_name}: SQLite verification failed ({e})"
            _quarantine([db_path, mpath], quarantine_dir, note)
            quarantined_notes.append(note)
            continue

        valid_candidates.append((manifest, mpath, db_path))

    # Best revision wins deterministically per shard_id
    valid_candidates.sort(
        key=lambda item: (
            0 if not item[0].get("seeds_failed") else 1,
            -int(item[0].get("revision", 1)),
            str(item[0].get("created_at_utc") or ""),
            str(item[0].get("shard_id") or ""),
        )
    )
    return valid_candidates, quarantined_notes


def _audit_compiled_database(
    output_db: Path,
    sample_audit: int,
) -> Dict[str, Any]:
    conn = sqlite3.connect(f"file:{output_db}?mode=ro", uri=True)
    cur = conn.cursor()

    integrity = cur.execute("PRAGMA integrity_check").fetchone()[0]
    ep_rows = cur.execute(
        """
        SELECT episode_id, seed, strategy, bc_type, load_config, resolution,
               bc_masks_blob, forces_blob, v_constructed_blob,
               final_compliance, final_volume, final_max_displacement, final_is_connected,
               worker_id
        FROM episodes
        ORDER BY COALESCE(seed, 0), episode_id
        """
    ).fetchall()

    total_records = cur.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    phase_counts = dict(cur.execute("SELECT phase, COUNT(*) FROM records GROUP BY phase").fetchall())
    orphan_records = cur.execute(
        "SELECT COUNT(*) FROM records r LEFT JOIN episodes e ON r.episode_id = e.episode_id WHERE e.episode_id IS NULL"
    ).fetchone()[0]

    non_finite_records = cur.execute(
        """
        SELECT COUNT(*) FROM records
        WHERE fitness_score IS NULL
           OR fitness_score != fitness_score
           OR ABS(fitness_score) > 1e300
           OR (compliance IS NOT NULL AND (compliance != compliance OR ABS(compliance) > 1e300))
           OR (vol_frac IS NOT NULL AND (vol_frac != vol_frac OR ABS(vol_frac) > 1e300))
           OR (max_displacement IS NOT NULL AND (max_displacement != max_displacement OR ABS(max_displacement) > 1e300))
        """
    ).fetchone()[0]

    errors: List[str] = []
    if integrity != "ok":
        errors.append(f"integrity_check={integrity}")
    if orphan_records:
        errors.append(f"orphan_records={orphan_records}")
    if non_finite_records:
        errors.append(f"non_finite_records={non_finite_records}")

    episodes_meta: Dict[str, Dict[str, Any]] = {}
    bc_comp: Dict[str, List[float]] = {"FULL_CLAMP": [], "RAIL_XY": []}
    bc_disp: Dict[str, List[float]] = {"FULL_CLAMP": [], "RAIL_XY": []}
    strat_comp: Dict[str, List[float]] = {"BEZIER": [], "FULL_DOMAIN": []}
    worker_counts: Dict[str, int] = {}
    connected_eps = 0
    numpy_header_failures = 0

    for row in ep_rows:
        (
            ep_id,
            seed,
            strat,
            bc,
            load_j,
            res_j,
            bc_b,
            f_b,
            vc_b,
            fc,
            fv,
            fd,
            fconn,
            wid,
        ) = row
        res = tuple(json.loads(res_j))
        lcfg = json.loads(load_j)
        for blob in (bc_b, f_b, vc_b):
            if blob is not None and zlib.decompress(blob)[:6] != b"\x93NUMPY":
                numpy_header_failures += 1
        bc_masks = deserialize_array(bc_b).astype(np.float32)
        forces = deserialize_array(f_b).astype(np.float32)
        episodes_meta[ep_id] = {
            "seed": seed,
            "strategy": strat,
            "bc_type": bc,
            "res": res,
            "bc_masks": bc_masks,
            "forces": forces,
        }
        worker_counts[str(wid or "unknown")] = worker_counts.get(str(wid or "unknown"), 0) + 1
        if fconn:
            connected_eps += 1
        if bc in bc_comp and fc is not None:
            bc_comp[bc].append(float(fc))
        if bc in bc_disp and fd is not None:
            bc_disp[bc].append(float(fd))
        if strat in strat_comp and fc is not None:
            strat_comp[strat].append(float(fc))

        # Load-patch alignment check
        lx, y_min, y_max, z_min, z_max = get_load_region_bounds(res, lcfg)
        fy = forces[1]
        if int(np.sum(np.abs(forces) > 1e-6)) != int(np.sum(fy[lx, y_min:y_max, z_min:z_max] < -0.5)):
            errors.append(f"Episode {ep_id}: force tensor misaligned with load_config")

    # Sequence continuity check per episode in SQL
    bad_seq = cur.execute(
        """
        SELECT episode_id, COUNT(*), MIN(step), MAX(step), SUM(is_final_step)
        FROM records
        GROUP BY episode_id
        HAVING MIN(step) != 0 OR MAX(step) != COUNT(*) - 1 OR SUM(is_final_step) != 1
        """
    ).fetchall()
    if bad_seq:
        errors.append(f"{len(bad_seq)} episode(s) failed step continuity / is_final_step uniqueness")

    # Value & physics distributions across all records
    p1_vals = [float(r[0]) for r in cur.execute("SELECT fitness_score FROM records WHERE phase='GROWTH'").fetchall()]
    p2_rows_scalars = cur.execute(
        "SELECT fitness_score, compliance, vol_frac, max_displacement FROM records WHERE phase='REFINEMENT'"
    ).fetchall()
    p2_vals = [float(r[0]) for r in p2_rows_scalars]
    p2_logc = [float(np.log10(max(float(r[1]), 1e-9))) for r in p2_rows_scalars if r[1] is not None and r[1] > 0]
    p2_vols = [float(r[2]) for r in p2_rows_scalars if r[2] is not None]
    p2_disps = [float(r[3]) for r in p2_rows_scalars if r[3] is not None]

    # Sampled tensor header + legal policy checks
    sampled = cur.execute(
        """
        SELECT episode_id, phase, density_blob, policy_add_blob, policy_remove_blob, is_final_step
        FROM records
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (sample_audit,),
    ).fetchall()

    coo1_header_failures = 0
    illegal_add = 0
    illegal_rem = 0

    for ep_id, phase, dens_b, add_b, rem_b, is_final in sampled:
        if zlib.decompress(dens_b)[:6] != b"\x93NUMPY":
            numpy_header_failures += 1
        for sb in (add_b, rem_b):
            if sb is not None and zlib.decompress(sb)[:4] != b"COO1":
                coo1_header_failures += 1

        ep = episodes_meta[ep_id]
        res = ep["res"]
        bc_tuple = (ep["bc_masks"][0], ep["bc_masks"][1], ep["bc_masks"][2])
        fmag = np.linalg.norm(ep["forces"], axis=0)
        dens = deserialize_array(dens_b).astype(np.float32)

        ai, av = deserialize_sparse(add_b)
        ri, rv = deserialize_sparse(rem_b)
        pol_add = sparse_decode(ai, av, res)
        pol_rem = sparse_decode(ri, rv, res)

        valid_add = get_legal_add_moves(dens, threshold=0.5, include_supports=True, bc_masks=bc_tuple)
        prot = get_protected_regions(bc_tuple, fmag)
        valid_rem = get_legal_remove_moves(dens, threshold=0.5) * (1.0 - prot)

        if float(np.sum(pol_add * (1.0 - valid_add))) > 1e-5:
            illegal_add += 1
        if float(np.sum(pol_rem * (1.0 - valid_rem))) > 1e-5:
            illegal_rem += 1

    conn.close()

    if numpy_header_failures:
        errors.append(f"numpy_header_failures={numpy_header_failures}")
    if coo1_header_failures:
        errors.append(f"coo1_header_failures={coo1_header_failures}")
    if illegal_add:
        errors.append(f"illegal_add_samples={illegal_add}")
    if illegal_rem:
        errors.append(f"illegal_remove_samples={illegal_rem}")

    return {
        "sqlite_integrity": integrity,
        "episodes": len(ep_rows),
        "connected_episodes": connected_eps,
        "connectivity_pct": round(100.0 * connected_eps / max(1, len(ep_rows)), 2),
        "total_records": total_records,
        "phase_counts": phase_counts,
        "sampled_steps_audited": len(sampled),
        "workers": worker_counts,
        "bc_stats": {
            bc: {"compliance": _quantiles(bc_comp[bc]), "max_displacement": _quantiles(bc_disp[bc])}
            for bc in ("FULL_CLAMP", "RAIL_XY")
        },
        "strategy_stats": {s: {"compliance": _quantiles(strat_comp[s])} for s in ("BEZIER", "FULL_DOMAIN")},
        "value_stats": {"GROWTH": _quantiles(p1_vals), "REFINEMENT": _quantiles(p2_vals)},
        "phase2_physics_stats": {
            "log10_compliance": _quantiles(p2_logc),
            "vol_frac": _quantiles(p2_vols),
            "max_displacement": _quantiles(p2_disps),
        },
        "errors": errors,
    }


def _build_stratified_split(
    output_db: Path,
    val_fraction: float,
    seed: int,
) -> Dict[str, Any]:
    conn = sqlite3.connect(f"file:{output_db}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT episode_id, seed, strategy, bc_type
        FROM episodes
        ORDER BY COALESCE(seed, 0), episode_id
        """
    ).fetchall()
    conn.close()

    strata: Dict[str, List[str]] = {}
    for ep_id, _, strat, bc in rows:
        key = f"{strat}:{bc}"
        strata.setdefault(key, []).append(str(ep_id))

    rng = random.Random(seed)
    train_ids: List[str] = []
    val_ids: List[str] = []
    strata_summary: Dict[str, Dict[str, int]] = {}

    for key in sorted(strata.keys()):
        ep_list = list(strata[key])
        rng.shuffle(ep_list)
        if len(ep_list) >= 2:
            n_val = max(1, int(round(len(ep_list) * val_fraction)))
            n_val = min(n_val, len(ep_list) - 1)
        else:
            n_val = 0
        v_sub = ep_list[:n_val]
        t_sub = ep_list[n_val:]
        val_ids.extend(v_sub)
        train_ids.extend(t_sub)
        strata_summary[key] = {"total": len(ep_list), "train": len(t_sub), "val": len(v_sub)}

    if set(train_ids) & set(val_ids):
        raise RuntimeError("Data leakage detected between train and val episode IDs!")

    return {
        "schema_version": "v3.3",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "counts": {
            "total_episodes": len(rows),
            "train_episodes": len(train_ids),
            "val_episodes": len(val_ids),
        },
        "strata_breakdown": strata_summary,
        "train_episode_ids": sorted(train_ids),
        "val_episode_ids": sorted(val_ids),
    }


def _smoke_test_dataset(output_db: Path) -> Dict[str, Any]:
    try:
        from alphabuilder.src.neural.dataset import TopologyDatasetV31
    except Exception as e:
        return {"skipped": True, "reason": str(e)}

    t0 = time.perf_counter()
    ds = TopologyDatasetV31(output_db, augment=False, preload_to_ram=False)
    n = len(ds)
    if n == 0:
        raise RuntimeError("TopologyDatasetV31 loaded 0 samples!")

    sample_indices = sorted(set([0, n // 2, n - 1]))
    shapes = []
    for idx in sample_indices:
        item = ds[idx]
        st = item["state"]
        po = item["policy"]
        val = item["value"]
        if not np.isfinite(st.numpy()).all() or not np.isfinite(po.numpy()).all() or not np.isfinite(val.numpy()).all():
            raise RuntimeError(f"Non-finite tensor returned by TopologyDatasetV31 at idx={idx}")
        shapes.append((tuple(st.shape), tuple(po.shape)))
    dt = time.perf_counter() - t0
    return {
        "skipped": False,
        "dataset_len": n,
        "checked_indices": sample_indices,
        "sample_shapes": shapes,
        "elapsed_sec": round(dt, 3),
    }


def _write_markdown_report(
    report_out: Path,
    output_db: Path,
    shards_merged: int,
    duplicates_skipped: int,
    quarantined: List[str],
    audit: Dict[str, Any],
    split_info: Dict[str, Any],
    smoke: Dict[str, Any],
    elapsed_sec: float,
) -> None:
    db_mb = output_db.stat().st_size / 1e6
    lines = [
        "# Relatório de Compilação do Dataset AlphaBuilder v3.3",
        "",
        f"- **Banco compilado:** `{output_db.name}` (`{db_mb:.2f} MB`, `{db_mb / max(1, audit['episodes']):.3f} MB/episódio`)",
        f"- **Shards processadas:** `{shards_merged}` mescladas | `{duplicates_skipped}` episódios duplicados ignorados | `{len(quarantined)}` quarentenadas",
        f"- **Tempo de compilação:** `{elapsed_sec:.2f} s` | **Integridade SQLite:** `{audit['sqlite_integrity']}` | **Erros de auditoria:** `{len(audit['errors'])}`",
        "",
        "## Cobertura e Conectividade",
        f"- **Episódios:** `{audit['episodes']}` (`{audit['connected_episodes']}/{audit['episodes']} = {audit['connectivity_pct']}%` conectados)",
        f"- **Registros:** `{audit['total_records']}` (`GROWTH`: `{audit['phase_counts'].get('GROWTH', 0)}`, `REFINEMENT`: `{audit['phase_counts'].get('REFINEMENT', 0)}`)",
        f"- **Proveniência por worker:** `{json.dumps(audit['workers'])}`",
        "",
        "## Separação Física e Distribuições",
        f"- **`FULL_CLAMP` compliance:** `{json.dumps(audit['bc_stats']['FULL_CLAMP']['compliance'])}`",
        f"- **`RAIL_XY` compliance:** `{json.dumps(audit['bc_stats']['RAIL_XY']['compliance'])}`",
        f"- **Phase 2 `log10(C)`:** `{json.dumps(audit['phase2_physics_stats']['log10_compliance'])}`",
        f"- **Value `GROWTH`:** `{json.dumps(audit['value_stats']['GROWTH'])}`",
        f"- **Value `REFINEMENT`:** `{json.dumps(audit['value_stats']['REFINEMENT'])}`",
        "",
        "## Split Estratificado & Teste Smoke (`TopologyDatasetV31`)",
        f"- **Split (`{split_info['seed']}`, val={split_info['val_fraction']}):** `{json.dumps(split_info['counts'])}`",
        f"- **Estratos:** `{json.dumps(split_info['strata_breakdown'])}`",
        f"- **Smoke `TopologyDatasetV31`:** `{json.dumps(smoke)}`",
    ]
    if quarantined:
        lines.extend(["", "## Shards em Quarentena", *[f"- {q}" for q in quarantined]])
    if audit["errors"]:
        lines.extend(["", "## Erros Detectados", *[f"- {e}" for e in audit["errors"]]])

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile sharded AlphaBuilder v3.3 SQLite dataset for training")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory holding shard_*.db and .manifest.json")
    parser.add_argument("--rclone-remote", type=str, default="", help="Optional rclone remote path to sync before compiling")
    parser.add_argument("--rclone-bin", type=str, default="rclone")
    parser.add_argument("--quarantine-dir", type=str, default="")
    parser.add_argument("--output-db", type=str, required=True, help="Path for compiled Kaggle-ready SQLite database")
    parser.add_argument("--split-out", type=str, required=True, help="Path for stratified split_v3_3.json")
    parser.add_argument("--report-out", type=str, required=True, help="Path for Markdown compilation report")
    parser.add_argument("--keep-rho-cont", action="store_true", help="Keep rho_cont_blob in output DB (default: omitted)")
    parser.add_argument("--min-connected-only", action="store_true", help="Include only final_is_connected=1 episodes")
    parser.add_argument("--expected-commit", type=str, default="", help="Optional git commit filter for manifests")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-audit", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    quarantine_dir = Path(args.quarantine_dir).resolve() if args.quarantine_dir else (input_dir / "quarantine")
    output_db = Path(args.output_db).resolve()
    split_out = Path(args.split_out).resolve()
    report_out = Path(args.report_out).resolve()

    if args.rclone_remote:
        _sync_rclone(args.rclone_bin, args.rclone_remote, input_dir)

    candidates, quarantined = _verify_and_discover_shards(
        input_dir, quarantine_dir, expected_commit=(args.expected_commit.strip() or None)
    )
    if not candidates:
        raise SystemExit(f"No valid v3.3 shards found in {input_dir}")

    if output_db.exists():
        output_db.unlink()
    for suffix in ("-wal", "-shm", "-journal"):
        p_aux = Path(str(output_db) + suffix)
        if p_aux.exists():
            p_aux.unlink()

    initialize_database(output_db, wal_mode=False)

    dst = sqlite3.connect(output_db)
    dst.execute("PRAGMA synchronous = OFF")
    dst.execute("PRAGMA journal_mode = OFF")
    dst.execute("PRAGMA temp_store = MEMORY")
    dst.execute("PRAGMA cache_size = -262144")

    seen_episode_ids = set()
    seen_seeds = set()
    shards_merged = 0
    duplicates_skipped = 0
    record_batch: List[tuple] = []
    batch_size = 5000

    insert_ep_sql = """
        INSERT INTO episodes (
            episode_id, seed, shard_id, worker_id, code_version,
            bc_masks_blob, forces_blob, v_constructed_blob,
            load_config, bc_type, strategy, resolution,
            final_compliance, final_volume, final_max_displacement,
            final_is_connected, simp_iters, duration_sec, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    insert_rec_sql = """
        INSERT INTO records (
            episode_id, step, phase, density_blob, rho_cont_blob,
            policy_add_blob, policy_remove_blob, fitness_score,
            compliance, vol_frac, max_displacement, beta,
            is_final_step, is_connected, displacement_blob,
            reward_components_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    for manifest, _, shard_db in candidates:
        src = sqlite3.connect(f"file:{shard_db}?mode=ro", uri=True)
        ep_rows = src.execute(
            """
            SELECT episode_id, seed, shard_id, worker_id, code_version,
                   bc_masks_blob, forces_blob, v_constructed_blob,
                   load_config, bc_type, strategy, resolution,
                   final_compliance, final_volume, final_max_displacement,
                   final_is_connected, simp_iters, duration_sec, created_at
            FROM episodes
            ORDER BY COALESCE(seed, 0), episode_id
            """
        ).fetchall()

        accepted_ep_ids = set()
        for ep in ep_rows:
            ep_id = ep[0]
            seed = ep[1]
            fconn = ep[15]
            if args.min_connected_only and not fconn:
                continue
            if ep_id in seen_episode_ids or (seed is not None and seed in seen_seeds):
                duplicates_skipped += 1
                continue
            seen_episode_ids.add(ep_id)
            if seed is not None:
                seen_seeds.add(seed)
            accepted_ep_ids.add(ep_id)
            dst.execute(insert_ep_sql, ep)

        if accepted_ep_ids:
            shards_merged += 1
            placeholders = ",".join("?" for _ in accepted_ep_ids)
            rec_rows = src.execute(
                f"""
                SELECT episode_id, step, phase, density_blob, rho_cont_blob,
                       policy_add_blob, policy_remove_blob, fitness_score,
                       compliance, vol_frac, max_displacement, beta,
                       is_final_step, is_connected, displacement_blob,
                       reward_components_json, created_at
                FROM records
                WHERE episode_id IN ({placeholders})
                ORDER BY episode_id, step ASC
                """,
                tuple(accepted_ep_ids),
            ).fetchall()

            for r in rec_rows:
                row_list = list(r)
                if not args.keep_rho_cont:
                    row_list[4] = None  # Omit rho_cont_blob for slim Kaggle dataset
                row_list[14] = None     # Always omit heavy displacement_blob in compiled dataset
                record_batch.append(tuple(row_list))
                if len(record_batch) >= batch_size:
                    dst.executemany(insert_rec_sql, record_batch)
                    record_batch.clear()

        src.close()

    if record_batch:
        dst.executemany(insert_rec_sql, record_batch)
        record_batch.clear()

    dst.execute("CREATE INDEX IF NOT EXISTS idx_records_episode ON records(episode_id)")
    dst.execute("CREATE INDEX IF NOT EXISTS idx_records_phase ON records(phase)")
    dst.execute("CREATE INDEX IF NOT EXISTS idx_episodes_seed ON episodes(seed)")
    dst.commit()
    dst.execute("ANALYZE")
    dst.execute("VACUUM")
    dst.execute("PRAGMA journal_mode = DELETE")
    dst.close()

    audit = _audit_compiled_database(output_db, sample_audit=args.sample_audit)
    split_info = _build_stratified_split(output_db, val_fraction=args.val_fraction, seed=args.seed)
    split_out.parent.mkdir(parents=True, exist_ok=True)
    split_out.write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    smoke = _smoke_test_dataset(output_db)
    elapsed_sec = time.perf_counter() - t0
    _write_markdown_report(
        report_out=report_out,
        output_db=output_db,
        shards_merged=shards_merged,
        duplicates_skipped=duplicates_skipped,
        quarantined=quarantined,
        audit=audit,
        split_info=split_info,
        smoke=smoke,
        elapsed_sec=elapsed_sec,
    )

    summary = {
        "output_db": str(output_db),
        "size_mb": round(output_db.stat().st_size / 1e6, 2),
        "shards_merged": shards_merged,
        "duplicates_skipped": duplicates_skipped,
        "quarantined": len(quarantined),
        "audit": audit,
        "split_counts": split_info["counts"],
        "smoke": smoke,
        "elapsed_sec": round(elapsed_sec, 2),
    }
    print(json.dumps(summary, indent=2))
    if audit["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
