"""
PyTorch Datasets & Multi-Resolution Batching for AlphaBuilder v4.0 Stage 6 Self-Play.

Provides:
- `SelfPlayDataset`: Reads Self-Play `games`, `game_steps`, and `fem_probes` SQLite databases,
  extracting normalized MCTS visit distributions (with advantage gating), real FEM
  backpropagated value targets, `log10(Compliance)`, connectivity labels, optional `domain_mask`
  (keep-out regions), and optional `sed_target = log1p(SED_map)` fields.
- `MixedReplayDataset`: Combines `SelfPlayDataset` + `TopologyDatasetV31` (warmup anchor)
  with consistent batch keys and per-epoch synchronized index sampling.
- `BucketedResolutionBatchSampler`: Yields DDP-safe batches grouped by spatial resolution
  `(D, H, W)` so Tier-3 multi-resolution domains train together with 0% zero-padding.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from alphabuilder.src.logic.mcts.legal_moves import get_legal_moves
from alphabuilder.src.logic.storage import (
    deserialize_array,
    deserialize_sparse,
    sparse_decode,
)
from alphabuilder.src.neural.augmentation import flip_x, flip_y, flip_z, load_multiplier


_LOG10_9 = float(2.0 * math.log10(3.0))


def _flip_spatial_only(arr: np.ndarray, axis_4d: int) -> np.ndarray:
    ax = axis_4d if arr.ndim == 4 else (axis_4d - 1)
    return np.flip(arr, axis=ax).copy()


def collect_game_ids_from_dbs(db_paths: Sequence[Union[str, Path]]) -> List[str]:
    """Return sorted unique game IDs across one or more self-play SQLite databases."""
    seen: set = set()
    ordered: List[str] = []
    for dbp in db_paths:
        p = Path(dbp)
        if not p.exists():
            continue
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT game_id FROM games ORDER BY created_at ASC, game_id ASC").fetchall()
            for (gid,) in rows:
                s = str(gid)
                if s not in seen:
                    seen.add(s)
                    ordered.append(s)
        finally:
            conn.close()
    return ordered


class SelfPlayDataset(Dataset):
    """
    Memory-efficient compressed-blob dataset over Self-Play `game_steps` + `fem_probes`.
    Safe under multi-worker `DataLoader` and `DistributedSampler` / `BucketedResolutionBatchSampler`.
    """

    def __init__(
        self,
        db_paths: Sequence[Union[str, Path]],
        allowed_game_ids: Optional[Iterable[str]] = None,
        augment: bool = True,
        include_probes: bool = True,
        phase_filter: Optional[str] = None,
        policy_delta_tol: float = 0.005,
        compute_legal_mask: bool = True,
        load_mult_prob: float = 0.05,
    ):
        self.db_paths = [Path(p) for p in db_paths]
        self.allowed_game_ids = {str(g) for g in allowed_game_ids} if allowed_game_ids is not None else None
        self.augment = bool(augment)
        self.include_probes = bool(include_probes)
        self.phase_filter = phase_filter
        self.policy_delta_tol = float(policy_delta_tol)
        self.compute_legal_mask = bool(compute_legal_mask)
        self.load_mult_prob = float(load_mult_prob)

        self.games: Dict[str, dict] = {}
        self._game_idx_to_id: List[str] = []
        self._blob_buf: bytes = b""
        # Offsets columns: (d0, d1, a0, a1, r0, r1, s0, s1)
        self._offsets: np.ndarray = np.zeros((0, 8), dtype=np.int64)
        self._game_indices: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._phases: np.ndarray = np.zeros((0,), dtype=np.uint8)
        self._is_final: np.ndarray = np.zeros((0,), dtype=np.uint8)
        self._is_connected: np.ndarray = np.zeros((0,), dtype=np.uint8)
        self._values: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._compliances: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._policy_valid_flag: np.ndarray = np.zeros((0,), dtype=np.uint8)
        self._is_probe: np.ndarray = np.zeros((0,), dtype=np.uint8)

        self._load_all()

    def _load_all(self) -> None:
        buf = bytearray()
        offsets_list: List[Tuple[int, int, int, int, int, int, int, int]] = []
        game_indices_list: List[int] = []
        phases_list: List[int] = []
        final_list: List[int] = []
        conn_list: List[int] = []
        values_list: List[float] = []
        comp_list: List[float] = []
        pol_valid_list: List[int] = []
        probe_list: List[int] = []

        gid_to_idx: Dict[str, int] = {}

        for db_path in self.db_paths:
            if not db_path.exists():
                continue
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                cur = conn.cursor()
                g_cols = {r[1] for r in cur.execute("PRAGMA table_info(games)").fetchall()}
                dm_expr = "domain_mask_blob" if "domain_mask_blob" in g_cols else "NULL AS domain_mask_blob"
                cur.execute(
                    f"""
                    SELECT game_id, bc_masks_blob, forces_blob, resolution, final_score, {dm_expr}
                    FROM games
                    """
                )
                game_meta: Dict[str, Optional[float]] = {}
                for gid, bc_blob, f_blob, res_json, fin_score, dm_blob in cur.fetchall():
                    gkey = str(gid)
                    if self.allowed_game_ids is not None and gkey not in self.allowed_game_ids:
                        continue
                    game_meta[gkey] = float(fin_score) if fin_score is not None else None
                    if gkey not in gid_to_idx:
                        bc_raw = deserialize_array(bc_blob)
                        bc_stored = (
                            bc_raw.astype(np.uint8)
                            if np.all((bc_raw == 0) | (bc_raw == 1))
                            else bc_raw.astype(np.float32)
                        )
                        forces = deserialize_array(f_blob).astype(np.float32)
                        resolution = tuple(int(v) for v in json.loads(res_json))
                        dm_arr = (
                            (deserialize_array(dm_blob) > 0.5).astype(np.uint8)
                            if dm_blob
                            else None
                        )
                        gid_to_idx[gkey] = len(self._game_idx_to_id)
                        self._game_idx_to_id.append(gkey)
                        self.games[gkey] = {
                            "bc_masks": bc_stored,
                            "forces": forces,
                            "resolution": resolution,
                            "domain_mask": dm_arr,
                        }

                if not game_meta:
                    continue

                s_cols = {r[1] for r in cur.execute("PRAGMA table_info(game_steps)").fetchall()}
                s_sed_expr = "sed_map_blob" if "sed_map_blob" in s_cols else "NULL AS sed_map_blob"
                cur.execute(
                    f"""
                    SELECT game_id, step, phase, density_blob,
                           mcts_visit_add_blob, mcts_visit_remove_blob,
                           is_connected, compliance_fem, value_target, fem_reward, final_reward, {s_sed_expr}
                    FROM game_steps
                    ORDER BY game_id ASC, step ASC
                    """
                )
                steps_by_game: Dict[str, List[tuple]] = {}
                for row in cur:
                    gkey = str(row[0])
                    if gkey in game_meta:
                        steps_by_game.setdefault(gkey, []).append(row)

                for gkey, rows in steps_by_game.items():
                    g_idx = gid_to_idx[gkey]
                    fin_score = game_meta[gkey]
                    reached_phase2 = any(r[2] == "REFINEMENT" for r in rows)
                    n_rows = len(rows)

                    for i, row in enumerate(rows):
                        _, _step, phase_str, d_blob, va_blob, vr_blob, is_conn, comp_fem, val_tgt, fem_r, fin_r, sed_blob = row
                        if self.phase_filter and phase_str != self.phase_filter:
                            continue

                        has_visits = bool((va_blob and len(va_blob) > 0) or (vr_blob and len(vr_blob) > 0))
                        if val_tgt is not None and np.isfinite(val_tgt):
                            v_val = float(val_tgt)
                        elif fem_r is not None and np.isfinite(fem_r):
                            v_val = float(fem_r)
                        elif fin_r is not None and np.isfinite(fin_r):
                            v_val = float(fin_r)
                        elif fin_score is not None and np.isfinite(fin_score):
                            v_val = float(fin_score)
                        else:
                            v_val = -1.0
                        v_val = float(np.clip(v_val, -1.0, 1.0))

                        pol_ok = False
                        if has_visits:
                            if phase_str == "GROWTH":
                                pol_ok = bool(reached_phase2 and (fin_score is None or fin_score > -0.80))
                            else:
                                next_fem = None
                                if i + 1 < n_rows and rows[i + 1][2] == "REFINEMENT":
                                    nxt = rows[i + 1][9]
                                    if nxt is not None and np.isfinite(nxt):
                                        next_fem = float(nxt)
                                cur_ok = bool(is_conn and fem_r is not None and np.isfinite(fem_r) and float(fem_r) > -0.99)
                                delta_ok = (next_fem is None) or (fem_r is not None and next_fem >= float(fem_r) - self.policy_delta_tol)
                                pol_ok = cur_ok and delta_ok

                        d0 = len(buf)
                        buf.extend(d_blob)
                        d1 = len(buf)
                        a0 = len(buf)
                        if va_blob and pol_ok:
                            buf.extend(va_blob)
                        a1 = len(buf)
                        r0 = len(buf)
                        if vr_blob and pol_ok:
                            buf.extend(vr_blob)
                        r1 = len(buf)
                        s0 = len(buf)
                        if sed_blob and is_conn:
                            buf.extend(sed_blob)
                        s1 = len(buf)

                        offsets_list.append((d0, d1, a0, a1, r0, r1, s0, s1))
                        game_indices_list.append(g_idx)
                        phases_list.append(0 if phase_str == "GROWTH" else 1)
                        final_list.append(1 if i == n_rows - 1 else 0)
                        conn_list.append(1 if is_conn else 0)
                        values_list.append(v_val)
                        comp_val = float(comp_fem) if (comp_fem is not None and np.isfinite(comp_fem) and float(comp_fem) > 0.0) else float("nan")
                        comp_list.append(comp_val)
                        pol_valid_list.append(1 if pol_ok else 0)
                        probe_list.append(0)

                if self.include_probes and self.phase_filter in (None, "REFINEMENT"):
                    has_probes = cur.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='fem_probes'"
                    ).fetchone()[0]
                    if has_probes:
                        p_cols = {r[1] for r in cur.execute("PRAGMA table_info(fem_probes)").fetchall()}
                        p_sed_expr = "sed_map_blob" if "sed_map_blob" in p_cols else "NULL AS sed_map_blob"
                        cur.execute(
                            f"""
                            SELECT game_id, density_blob, is_connected, compliance_fem, fem_reward, {p_sed_expr}
                            FROM fem_probes
                            ORDER BY id ASC
                            """
                        )
                        for gkey_raw, d_blob, is_conn, comp_fem, fem_r, sed_blob in cur:
                            gkey = str(gkey_raw)
                            if gkey not in gid_to_idx:
                                continue
                            if fem_r is None or not np.isfinite(fem_r):
                                continue
                            g_idx = gid_to_idx[gkey]
                            d0 = len(buf)
                            buf.extend(d_blob)
                            d1 = len(buf)
                            s0 = len(buf)
                            if sed_blob and is_conn:
                                buf.extend(sed_blob)
                            s1 = len(buf)
                            offsets_list.append((d0, d1, d1, d1, d1, d1, s0, s1))
                            game_indices_list.append(g_idx)
                            phases_list.append(1)
                            final_list.append(0)
                            conn_list.append(1 if is_conn else 0)
                            values_list.append(float(np.clip(float(fem_r), -1.0, 1.0)))
                            comp_val = float(comp_fem) if (comp_fem is not None and np.isfinite(comp_fem) and float(comp_fem) > 0.0) else float("nan")
                            comp_list.append(comp_val)
                            pol_valid_list.append(0)
                            probe_list.append(1)
            finally:
                conn.close()

        self._blob_buf = bytes(buf)
        self._offsets = np.asarray(offsets_list, dtype=np.int64) if offsets_list else np.zeros((0, 8), dtype=np.int64)
        self._game_indices = np.asarray(game_indices_list, dtype=np.int32)
        self._phases = np.asarray(phases_list, dtype=np.uint8)
        self._is_final = np.asarray(final_list, dtype=np.uint8)
        self._is_connected = np.asarray(conn_list, dtype=np.uint8)
        self._values = np.asarray(values_list, dtype=np.float32)
        self._compliances = np.asarray(comp_list, dtype=np.float32)
        self._policy_valid_flag = np.asarray(pol_valid_list, dtype=np.uint8)
        self._is_probe = np.asarray(probe_list, dtype=np.uint8)

    def get_sample_resolution(self, idx: int) -> Tuple[int, int, int]:
        gid = self._game_idx_to_id[int(self._game_indices[idx])]
        return self.games[gid]["resolution"]

    def __len__(self) -> int:
        return int(self._values.shape[0])

    def get_memory_usage_mb(self) -> float:
        total = len(self._blob_buf)
        for g in self.games.values():
            total += g["bc_masks"].nbytes + g["forces"].nbytes
            if g["domain_mask"] is not None:
                total += g["domain_mask"].nbytes
        for arr in (self._offsets, self._game_indices, self._phases, self._values, self._compliances):
            total += arr.nbytes
        return total / (1024.0 * 1024.0)

    def __getitem__(self, idx: int) -> dict:
        gid = self._game_idx_to_id[int(self._game_indices[idx])]
        ginfo = self.games[gid]
        resolution = ginfo["resolution"]

        d0, d1, a0, a1, r0, r1, s0, s1 = [int(v) for v in self._offsets[idx]]
        density = deserialize_array(self._blob_buf[d0:d1]).astype(np.float32)
        if a1 > a0:
            add_idx, add_val = deserialize_sparse(self._blob_buf[a0:a1])
            policy_add = sparse_decode(add_idx, add_val, resolution)
        else:
            policy_add = np.zeros(resolution, dtype=np.float32)
        if r1 > r0:
            rem_idx, rem_val = deserialize_sparse(self._blob_buf[r0:r1])
            policy_remove = sparse_decode(rem_idx, rem_val, resolution)
        else:
            policy_remove = np.zeros(resolution, dtype=np.float32)

        if s1 > s0:
            sed_raw = deserialize_array(self._blob_buf[s0:s1]).astype(np.float32)
            sed_target = np.log1p(np.maximum(0.0, sed_raw))[None]
            sed_valid = True
        else:
            sed_target = np.zeros((1, *resolution), dtype=np.float32)
            sed_valid = False

        if ginfo["domain_mask"] is not None:
            domain_mask = ginfo["domain_mask"].astype(np.float32)[None]
        else:
            domain_mask = np.ones((1, *resolution), dtype=np.float32)

        bc_masks = ginfo["bc_masks"].astype(np.float32)
        forces = ginfo["forces"].astype(np.float32)
        state = np.concatenate([density[None], bc_masks, forces], axis=0).astype(np.float32)
        policy = np.stack([policy_add, policy_remove], axis=0).astype(np.float32)
        blueprint = np.zeros((1, *resolution), dtype=np.float32)

        value = float(self._values[idx])
        phase = "GROWTH" if int(self._phases[idx]) == 0 else "REFINEMENT"
        is_final = bool(self._is_final[idx])
        is_connected = bool(self._is_connected[idx])
        comp_raw = float(self._compliances[idx])
        comp_valid = bool(np.isfinite(comp_raw) and comp_raw > 0.0 and is_connected)
        log_compliance = float(math.log10(comp_raw)) if comp_valid else 0.0
        conn_target = 1.0 if is_connected else 0.0
        conn_valid = True
        pol_ok = bool(self._policy_valid_flag[idx])

        if self.augment:
            if (
                phase == "REFINEMENT"
                and is_connected
                and self.load_mult_prob > 0.0
                and np.random.random() < self.load_mult_prob
            ):
                new_state, new_pol, new_val = load_multiplier(state, policy, value, k=3.0)
                if new_val != value:
                    state, policy, value = new_state, new_pol, new_val
                    pol_ok = True
                    if comp_valid:
                        log_compliance += _LOG10_9
                    if sed_valid:
                        # Linear elasticity scaling under 3F: SED(3F) = 9 * SED(F)
                        sed_raw_scaled = np.expm1(sed_target) * 9.0
                        sed_target = np.log1p(np.maximum(0.0, sed_raw_scaled)).astype(np.float32)

            if np.random.random() < 0.5:
                state, policy, domain_mask = flip_x(state, policy, domain_mask)
                sed_target = _flip_spatial_only(sed_target, 1)
            if np.random.random() < 0.5:
                state, policy, domain_mask = flip_y(state, policy, domain_mask)
                sed_target = _flip_spatial_only(sed_target, 2)
            if np.random.random() < 0.5:
                state, policy, domain_mask = flip_z(state, policy, domain_mask)
                sed_target = _flip_spatial_only(sed_target, 3)

        if self.compute_legal_mask and state.shape[0] >= 7:
            valid_add, valid_remove = get_legal_moves(
                state[0],
                threshold=0.5,
                bc_masks=(state[1], state[2], state[3]),
                forces=(state[4], state[5], state[6]),
                domain_mask=domain_mask[0],
            )
            legal_mask = np.stack([valid_add, valid_remove], axis=0).astype(np.float32)
        else:
            legal_mask = np.ones_like(policy, dtype=np.float32)

        masked_pol = np.maximum(0.0, policy) * legal_mask
        add_mass = float(masked_pol[0].sum())
        rem_mass = float(masked_pol[1].sum())
        pol_sum = add_mass + rem_mass
        if pol_ok and pol_sum > 1e-6:
            policy = (masked_pol / pol_sum).astype(np.float32)
            policy_mask = 1.0
            op_target = np.array([add_mass / pol_sum, rem_mass / pol_sum, 0.0], dtype=np.float32)
            op_mask = 1.0
        elif is_final and is_connected and (not bool(self._is_probe[idx])) and phase == "REFINEMENT" and value > -0.8:
            policy = np.zeros_like(policy, dtype=np.float32)
            policy_mask = 0.0
            op_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            op_mask = 1.0
        else:
            policy = np.zeros_like(policy, dtype=np.float32)
            policy_mask = 0.0
            op_target = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            op_mask = 0.0

        return {
            "state": torch.from_numpy(np.ascontiguousarray(state)).float(),
            "domain_mask": torch.from_numpy(np.ascontiguousarray(domain_mask)).float(),
            "policy": torch.from_numpy(np.ascontiguousarray(policy)).float(),
            "legal_mask": torch.from_numpy(np.ascontiguousarray(legal_mask)).float(),
            "policy_mask": torch.tensor(policy_mask, dtype=torch.float32),
            "value": torch.tensor([value], dtype=torch.float32),
            "blueprint": torch.from_numpy(np.ascontiguousarray(blueprint)).float(),
            "blueprint_mask": torch.tensor(0.0, dtype=torch.float32),
            "log_compliance": torch.tensor([log_compliance], dtype=torch.float32),
            "compliance_mask": torch.tensor(1.0 if comp_valid else 0.0, dtype=torch.float32),
            "conn_target": torch.tensor([conn_target], dtype=torch.float32),
            "conn_mask": torch.tensor(1.0 if conn_valid else 0.0, dtype=torch.float32),
            "sed_target": torch.from_numpy(np.ascontiguousarray(sed_target)).float(),
            "sed_mask": torch.tensor(1.0 if (sed_valid and is_connected) else 0.0, dtype=torch.float32),
            "op_target": torch.from_numpy(op_target).float(),
            "op_mask": torch.tensor(op_mask, dtype=torch.float32),
            "phase": phase,
            "is_final": is_final,
            "is_connected": is_connected,
        }


class MixedReplayDataset(Dataset):
    """
    Epoch-resampled replay buffer mixing `SelfPlayDataset` and `TopologyDatasetV31` (warmup anchor).

    Call `set_epoch(epoch)` before iterating each epoch so all DDP ranks share an identical,
    deterministic index permutation table and consistent tensor keys across sources.
    """

    def __init__(
        self,
        selfplay_ds: Optional[SelfPlayDataset],
        warmup_ds: Optional[Dataset],
        selfplay_fraction: float = 0.60,
        min_epoch_samples: int = 12000,
        max_sp_repeat: int = 4,
        seed: int = 42,
    ):
        self.selfplay_ds = selfplay_ds
        self.warmup_ds = warmup_ds
        self.selfplay_fraction = float(min(max(selfplay_fraction, 0.0), 1.0))
        self.min_epoch_samples = int(max(1, min_epoch_samples))
        self.max_sp_repeat = int(max(1, max_sp_repeat))
        self.seed = int(seed)

        self._plan_is_sp: np.ndarray = np.zeros((0,), dtype=bool)
        self._plan_idx: np.ndarray = np.zeros((0,), dtype=np.int64)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + int(epoch) * 1009)
        n_sp_avail = len(self.selfplay_ds) if self.selfplay_ds is not None else 0
        n_wu_avail = len(self.warmup_ds) if self.warmup_ds is not None else 0

        if n_sp_avail == 0 and n_wu_avail == 0:
            self._plan_is_sp = np.zeros((0,), dtype=bool)
            self._plan_idx = np.zeros((0,), dtype=np.int64)
            return

        if n_sp_avail == 0:
            n_wu = min(n_wu_avail, self.min_epoch_samples)
            self._plan_is_sp = np.zeros((n_wu,), dtype=bool)
            self._plan_idx = rng.choice(n_wu_avail, size=n_wu, replace=False).astype(np.int64)
            return

        if n_wu_avail == 0 or self.selfplay_fraction >= 1.0:
            target_sp = min(n_sp_avail * self.max_sp_repeat, max(n_sp_avail, self.min_epoch_samples))
            reps = (target_sp + n_sp_avail - 1) // n_sp_avail
            sp_idx = np.tile(np.arange(n_sp_avail, dtype=np.int64), reps)[:target_sp]
            rng.shuffle(sp_idx)
            self._plan_is_sp = np.ones((len(sp_idx),), dtype=bool)
            self._plan_idx = sp_idx
            return

        target_sp_raw = int(round(self.selfplay_fraction * self.min_epoch_samples))
        n_sp = max(n_sp_avail, min(target_sp_raw, n_sp_avail * self.max_sp_repeat))
        n_wu_desired = int(round(n_sp * (1.0 - self.selfplay_fraction) / max(self.selfplay_fraction, 1e-6)))
        n_wu = max(1, min(n_wu_avail, n_wu_desired))

        reps = (n_sp + n_sp_avail - 1) // n_sp_avail
        sp_idx = np.tile(np.arange(n_sp_avail, dtype=np.int64), reps)[:n_sp]
        wu_idx = rng.choice(n_wu_avail, size=n_wu, replace=False).astype(np.int64)

        is_sp = np.concatenate([np.ones(n_sp, dtype=bool), np.zeros(n_wu, dtype=bool)])
        indices = np.concatenate([sp_idx, wu_idx])
        perm = rng.permutation(len(indices))
        self._plan_is_sp = is_sp[perm]
        self._plan_idx = indices[perm]

    def get_sample_resolution(self, idx: int) -> Tuple[int, int, int]:
        src_idx = int(self._plan_idx[idx])
        if bool(self._plan_is_sp[idx]):
            assert self.selfplay_ds is not None
            return self.selfplay_ds.get_sample_resolution(src_idx)
        assert self.warmup_ds is not None
        if hasattr(self.warmup_ds, "schema_version") and getattr(self.warmup_ds, "schema_version") == "v2":
            ep_id = self.warmup_ds._ep_idx_to_id[int(self.warmup_ds._ep_indices[src_idx])]  # type: ignore[attr-defined]
            return self.warmup_ds.episodes[ep_id]["resolution"]  # type: ignore[attr-defined]
        return (64, 32, 8)

    def __len__(self) -> int:
        return int(self._plan_idx.shape[0])

    def __getitem__(self, idx: int) -> dict:
        src_idx = int(self._plan_idx[idx])
        if bool(self._plan_is_sp[idx]):
            assert self.selfplay_ds is not None
            return self.selfplay_ds[src_idx]
        assert self.warmup_ds is not None
        sample = self.warmup_ds[src_idx]
        _, D, H, W = sample["state"].shape
        if "domain_mask" not in sample:
            sample["domain_mask"] = torch.ones((1, D, H, W), dtype=torch.float32)
        if "sed_target" not in sample:
            sample["sed_target"] = torch.zeros((1, D, H, W), dtype=torch.float32)
        if "sed_mask" not in sample:
            sample["sed_mask"] = torch.tensor(0.0, dtype=torch.float32)
        if "op_target" not in sample:
            sample["op_target"] = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32)
        if "op_mask" not in sample:
            sample["op_mask"] = torch.tensor(0.0, dtype=torch.float32)
        return sample


class BucketedResolutionBatchSampler(Sampler[List[int]]):
    """
    Groups dataset indices by spatial `(D, H, W)` resolution before yielding batches so
    multi-resolution Tier-3 domains train cleanly alongside `(64, 32, 8)` with 0% padding.
    Supports distributed DDP sharding (`rank`, `world_size`).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_batches(self) -> List[List[int]]:
        n = len(self.dataset)
        buckets: Dict[Tuple[int, int, int], List[int]] = {}
        get_res = getattr(self.dataset, "get_sample_resolution", None)
        for idx in range(n):
            res = get_res(idx) if callable(get_res) else (64, 32, 8)
            buckets.setdefault(tuple(int(v) for v in res), []).append(idx)  # type: ignore[arg-type]

        rng = np.random.default_rng(self.seed + self.epoch * 7919)
        batches: List[List[int]] = []
        for _res, idxs in sorted(buckets.items()):
            arr = np.asarray(idxs, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(arr)
            for start in range(0, len(arr), self.batch_size):
                chunk = arr[start : start + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk.tolist())

        if self.shuffle and len(batches) > 1:
            perm = rng.permutation(len(batches))
            batches = [batches[int(p)] for p in perm]

        if self.world_size > 1 and batches:
            # Make batch count divisible across DDP ranks cleanly to prevent NCCL barriers from hanging
            if len(batches) < self.world_size:
                reps = (self.world_size + len(batches) - 1) // len(batches)
                batches = (batches * reps)[: self.world_size]
            usable = (len(batches) // self.world_size) * self.world_size
            if usable > 0:
                batches = batches[:usable]
            batches = batches[self.rank :: self.world_size]

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._build_batches())

    def __len__(self) -> int:
        return len(self._build_batches())
