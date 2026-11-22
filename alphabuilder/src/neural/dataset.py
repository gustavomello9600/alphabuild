"""
PyTorch Dataset & Symmetry Collate for AlphaBuilder v3.3.

Supports compact v2/v3.3 SQLite schemas (with compressed blobs packed in RAM for
near-zero CoW overhead under multiprocessing DataLoader workers) as well as legacy v1.
Provides episode-level filtering (`split_v3_3.json`), on-the-fly legal action masking,
auxiliary Blueprint Head targets (`v_constructed_blob`), reformulated negative sampling,
and full D4 x Z2 = 16x 3D rigid symmetry augmentation (8x flips per-sample + 2x rot90 per-batch).
"""

import json
import sqlite3
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from alphabuilder.src.logic.mcts.legal_moves import get_legal_moves
from alphabuilder.src.logic.storage import (
    deserialize_array as _storage_deserialize_array,
    deserialize_sparse as _storage_deserialize_sparse,
)
from alphabuilder.src.neural.augmentation import (
    erosion_attack,
    flip_x,
    flip_y,
    flip_z,
    load_multiplier,
    sabotage,
    saboteur,
)


def sparse_decode(indices: np.ndarray, values: np.ndarray, shape: tuple) -> np.ndarray:
    """Reconstruct dense array from sparse COO format."""
    flat = np.zeros(int(np.prod(shape)), dtype=np.float32)
    if len(indices) > 0:
        flat[indices] = values
    return flat.reshape(shape)


def deserialize_sparse(blob: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Deserialize sparse arrays from compressed bytes (COO1 and legacy pickle)."""
    return _storage_deserialize_sparse(blob)


def deserialize_array(blob: bytes) -> np.ndarray:
    """Deserialize numpy array from compressed bytes (.npy and legacy pickle)."""
    return _storage_deserialize_array(blob)


def deserialize_state_legacy(blob: bytes) -> np.ndarray:
    """Legacy alias for deserializing dense state/policy blobs."""
    return deserialize_array(blob)


def detect_schema_version(db_path: Path) -> str:
    """Detect whether the SQLite database uses v2/v3.3 (`episodes`/`records`) or legacy v1 schema."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM episodes")
        count = cursor.fetchone()[0]
        return "v2" if count > 0 else "v1"
    except sqlite3.OperationalError:
        return "v1"
    finally:
        conn.close()


def make_symmetry_collate_fn(enable_rot90: bool = True) -> Callable:
    """
    Create a DataLoader collate_fn completing the D4 x Z2 = 16x symmetry group.

    Per-sample `__getitem__` applies independent flips along X, Y, Z (8x).
    This collate_fn applies a batch-wide 90-degree Z-axis rotation with 50% probability (2x),
    mapping (B, C, 64, 32, 8) <-> (B, C, 32, 64, 8) cleanly with zero wasted padding.
    """

    def collate_fn(batch_list: List[dict]) -> dict:
        batch = default_collate(batch_list)
        if enable_rot90 and torch.rand(1).item() < 0.5:
            state = batch["state"]
            if state.ndim == 5 and state.shape[1] >= 7:
                state_rot = torch.rot90(state, k=1, dims=(2, 3)).contiguous()
                old_fx = state_rot[:, 4].clone()
                old_fy = state_rot[:, 5].clone()
                state_rot[:, 4] = -old_fy
                state_rot[:, 5] = old_fx

                old_mx = state_rot[:, 1].clone()
                old_my = state_rot[:, 2].clone()
                state_rot[:, 1] = old_my
                state_rot[:, 2] = old_mx
                batch["state"] = state_rot

                for key in ("policy", "legal_mask", "blueprint"):
                    if key in batch and isinstance(batch[key], torch.Tensor) and batch[key].ndim == 5:
                        batch[key] = torch.rot90(batch[key], k=1, dims=(2, 3)).contiguous()
        return batch

    return collate_fn


class TopologyDatasetV31(Dataset):
    """
    Memory-efficient dataset for topology optimization training.

    In `v2`/`v3.3` mode with `preload_to_ram=True`, raw compressed blobs are packed into an
    immutable `bytes` buffer (`_blob_buf`) with int64 slice offsets and decoded on-the-fly
    inside `__getitem__`. This shrinks RAM footprint by ~10-15x compared to dense float32
    caching and enables multiprocessing `DataLoader(num_workers=4)` cleanly.
    """

    _PHASE_NAMES = ("GROWTH", "REFINEMENT")

    def __init__(
        self,
        db_path: Union[str, Path],
        augment: bool = True,
        phase_filter: Optional[str] = None,
        preload_to_ram: bool = True,
        episode_ids: Optional[Iterable[str]] = None,
        split_json: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        compute_legal_mask: bool = True,
    ):
        self.db_path = Path(db_path)
        self.augment = augment
        self.phase_filter = phase_filter
        self.preload_to_ram = preload_to_ram
        self.compute_legal_mask = compute_legal_mask

        self.allowed_episode_ids: Optional[set] = None
        if episode_ids is not None:
            self.allowed_episode_ids = {str(ep) for ep in episode_ids}
        elif split_json is not None:
            if split not in ("train", "val"):
                raise ValueError(f"split must be 'train' or 'val' when split_json is provided, got {split!r}")
            split_data = json.loads(Path(split_json).read_text(encoding="utf-8"))
            key = f"{split}_episode_ids"
            if key not in split_data:
                raise KeyError(f"Key {key!r} not found in {split_json}")
            self.allowed_episode_ids = {str(ep) for ep in split_data[key]}

        self.schema_version = detect_schema_version(self.db_path)

        # Storage fields populated by _load_v2 / _load_v1
        self.episodes: Dict[str, dict] = {}
        self._ep_idx_to_id: List[str] = []
        self._blob_buf: bytes = b""
        self._offsets: Optional[np.ndarray] = None
        self._ep_indices: Optional[np.ndarray] = None
        self._steps: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._phases: Optional[np.ndarray] = None
        self._is_final: Optional[np.ndarray] = None
        self._is_connected: Optional[np.ndarray] = None

        # Legacy v1 containers
        self._v1_records: List[dict] = []
        self._v1_preloaded: List[dict] = []

        if self.schema_version == "v2":
            self._load_v2()
        else:
            self._load_v1()

    def _load_v2(self) -> None:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(episodes)")
        ep_cols = {row[1] for row in cursor.fetchall()}
        has_vconst = "v_constructed_blob" in ep_cols
        has_final_conn = "final_is_connected" in ep_cols

        select_cols = ["episode_id", "bc_masks_blob", "forces_blob", "resolution"]
        if has_vconst:
            select_cols.append("v_constructed_blob")
        if has_final_conn:
            select_cols.append("final_is_connected")

        cursor.execute(f"SELECT {', '.join(select_cols)} FROM episodes")
        episode_rows = cursor.fetchall()

        self.episodes = {}
        self._ep_idx_to_id = []
        ep_id_to_idx: Dict[str, int] = {}

        for row in episode_rows:
            ep_id = str(row[0])
            if self.allowed_episode_ids is not None and ep_id not in self.allowed_episode_ids:
                continue

            bc_raw = deserialize_array(row[1])
            if np.all((bc_raw == 0) | (bc_raw == 1)):
                bc_stored = bc_raw.astype(np.uint8)
            else:
                bc_stored = bc_raw.astype(np.float32)

            forces = deserialize_array(row[2]).astype(np.float32)
            resolution = tuple(int(v) for v in json.loads(row[3]))

            col_idx = 4
            v_const = None
            if has_vconst:
                vb = row[col_idx]
                col_idx += 1
                if vb is not None:
                    v_arr = deserialize_array(vb)
                    v_const = (v_arr > 0.5).astype(np.uint8)

            final_is_conn = True
            if has_final_conn:
                fc = row[col_idx]
                final_is_conn = bool(fc) if fc is not None else False

            ep_id_to_idx[ep_id] = len(self._ep_idx_to_id)
            self._ep_idx_to_id.append(ep_id)
            self.episodes[ep_id] = {
                "bc_masks": bc_stored,
                "forces": forces,
                "resolution": resolution,
                "v_constructed": v_const,
                "final_is_connected": final_is_conn,
            }

        query = (
            "SELECT episode_id, step, phase, density_blob, policy_add_blob, "
            "policy_remove_blob, fitness_score, is_final_step, is_connected "
            "FROM records"
        )
        if self.phase_filter:
            cursor.execute(query + " WHERE phase = ? ORDER BY episode_id, step ASC", (self.phase_filter,))
        else:
            cursor.execute(query + " ORDER BY episode_id, step ASC")

        buf = bytearray()
        offsets_list: List[Tuple[int, int, int, int, int, int]] = []
        ep_indices_list: List[int] = []
        steps_list: List[int] = []
        values_list: List[float] = []
        phases_list: List[int] = []
        final_list: List[int] = []
        conn_list: List[int] = []

        for row in cursor:
            ep_id, step, phase, density_blob, add_blob, rem_blob, fitness, is_final, is_conn = row
            ep_key = str(ep_id)
            ep_idx = ep_id_to_idx.get(ep_key)
            if ep_idx is None:
                continue

            ep_indices_list.append(ep_idx)
            steps_list.append(int(step))
            values_list.append(float(fitness))
            phases_list.append(0 if phase == "GROWTH" else 1)
            final_list.append(1 if is_final else 0)
            conn_list.append(1 if is_conn else 0)

            if self.preload_to_ram:
                d0 = len(buf)
                if density_blob:
                    buf.extend(density_blob)
                d1 = len(buf)

                a0 = len(buf)
                if add_blob:
                    buf.extend(add_blob)
                a1 = len(buf)

                r0 = len(buf)
                if rem_blob:
                    buf.extend(rem_blob)
                r1 = len(buf)

                offsets_list.append((d0, d1, a0, a1, r0, r1))

        conn.close()

        self._blob_buf = bytes(buf) if self.preload_to_ram else b""
        self._offsets = (
            np.asarray(offsets_list, dtype=np.int64)
            if self.preload_to_ram
            else None
        )
        if self.preload_to_ram and len(offsets_list) == 0:
            self._offsets = np.zeros((0, 6), dtype=np.int64)

        self._ep_indices = np.asarray(ep_indices_list, dtype=np.int32)
        self._steps = np.asarray(steps_list, dtype=np.int32)
        self._values = np.asarray(values_list, dtype=np.float32)
        self._phases = np.asarray(phases_list, dtype=np.uint8)
        self._is_final = np.asarray(final_list, dtype=np.uint8)
        self._is_connected = np.asarray(conn_list, dtype=np.uint8)

    def _load_v1(self) -> None:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        query = "SELECT id, episode_id, phase, metadata, fitness_score, state_blob, policy_blob FROM training_data"
        try:
            if self.phase_filter:
                cursor.execute(query + " WHERE phase = ?", (self.phase_filter,))
            else:
                cursor.execute(query)
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()

        self.episodes = {}
        self._v1_records = []
        self._v1_preloaded = []

        for row in rows:
            rec_id, ep_id, phase, meta_json, fitness, state_blob, policy_blob = row
            ep_key = str(ep_id)
            if self.allowed_episode_ids is not None and ep_key not in self.allowed_episode_ids:
                continue

            meta = json.loads(meta_json) if meta_json else {}
            is_final = bool(meta.get("is_final_step", False))
            is_conn = bool(meta.get("is_connected", False))

            if self.preload_to_ram:
                state = deserialize_state_legacy(state_blob).astype(np.float32)
                policy = (
                    deserialize_state_legacy(policy_blob).astype(np.float32)
                    if policy_blob
                    else np.zeros((2,) + state.shape[1:], dtype=np.float32)
                )
                self._v1_preloaded.append(
                    {
                        "state": state,
                        "policy": policy,
                        "value": float(fitness),
                        "phase": str(phase),
                        "is_final": is_final,
                        "is_connected": is_conn,
                    }
                )

            self._v1_records.append(
                {
                    "record_id": rec_id,
                    "episode_id": ep_key,
                    "phase": str(phase),
                    "value": float(fitness),
                    "is_final": is_final,
                    "is_connected": is_conn,
                }
            )

    def get_episode_ids(self) -> List[str]:
        """Return unique episode IDs present in this dataset."""
        if self.schema_version == "v2":
            return list(self._ep_idx_to_id)
        seen = []
        seen_set = set()
        for r in self._v1_records:
            ep = r["episode_id"]
            if ep not in seen_set:
                seen_set.add(ep)
                seen.append(ep)
        return seen

    def __len__(self) -> int:
        if self.schema_version == "v2":
            return 0 if self._steps is None else int(self._steps.shape[0])
        return len(self._v1_records)

    def _decode_v2_record(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, float, str, bool, bool]:
        ep_id = self._ep_idx_to_id[int(self._ep_indices[idx])]  # type: ignore[index]
        ep_info = self.episodes[ep_id]
        resolution = ep_info["resolution"]

        if self.preload_to_ram and self._offsets is not None:
            d0, d1, a0, a1, r0, r1 = [int(v) for v in self._offsets[idx]]
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
        else:
            step = int(self._steps[idx])  # type: ignore[index]
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            row = conn.execute(
                """
                SELECT density_blob, policy_add_blob, policy_remove_blob
                FROM records WHERE episode_id = ? AND step = ?
                """,
                (ep_id, step),
            ).fetchone()
            conn.close()
            if row is None:
                raise KeyError(f"Record not found for episode_id={ep_id}, step={step}")

            density = deserialize_array(row[0]).astype(np.float32)
            if row[1]:
                add_idx, add_val = deserialize_sparse(row[1])
                policy_add = sparse_decode(add_idx, add_val, resolution)
            else:
                policy_add = np.zeros(resolution, dtype=np.float32)
            if row[2]:
                rem_idx, rem_val = deserialize_sparse(row[2])
                policy_remove = sparse_decode(rem_idx, rem_val, resolution)
            else:
                policy_remove = np.zeros(resolution, dtype=np.float32)

        bc_masks = ep_info["bc_masks"].astype(np.float32)
        forces = ep_info["forces"].astype(np.float32)
        state = np.concatenate([density[None], bc_masks, forces], axis=0).astype(np.float32)
        policy = np.stack([policy_add, policy_remove], axis=0).astype(np.float32)

        v_const = ep_info["v_constructed"]
        if v_const is not None:
            blueprint = v_const.astype(np.float32)[None]
            bp_valid = bool(ep_info["final_is_connected"] and blueprint.sum() > 0)
        else:
            blueprint = np.zeros((1, *resolution), dtype=np.float32)
            bp_valid = False

        value = float(self._values[idx])  # type: ignore[index]
        phase = self._PHASE_NAMES[int(self._phases[idx])]  # type: ignore[index]
        is_final = bool(self._is_final[idx])  # type: ignore[index]
        is_connected = bool(self._is_connected[idx])  # type: ignore[index]
        return state, policy, blueprint, bp_valid, value, phase, is_final, is_connected

    def _decode_v1_record(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, float, str, bool, bool]:
        if self.preload_to_ram:
            d = self._v1_preloaded[idx]
            state = d["state"].copy()
            policy = d["policy"].copy()
            value = float(d["value"])
            phase = d["phase"]
            is_final = bool(d["is_final"])
            is_connected = bool(d["is_connected"])
        else:
            info = self._v1_records[idx]
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            row = conn.execute(
                "SELECT state_blob, policy_blob FROM training_data WHERE id = ?",
                (info["record_id"],),
            ).fetchone()
            conn.close()
            state = deserialize_state_legacy(row[0]).astype(np.float32)
            policy = (
                deserialize_state_legacy(row[1]).astype(np.float32)
                if row[1]
                else np.zeros((2,) + state.shape[1:], dtype=np.float32)
            )
            value = float(info["value"])
            phase = info["phase"]
            is_final = bool(info["is_final"])
            is_connected = bool(info["is_connected"])

        blueprint = np.zeros((1, *state.shape[1:]), dtype=np.float32)
        return state, policy, blueprint, False, value, phase, is_final, is_connected

    def __getitem__(self, idx: int) -> dict:
        if self.schema_version == "v2":
            state, policy, blueprint, bp_valid, value, phase, is_final, is_connected = self._decode_v2_record(idx)
        else:
            state, policy, blueprint, bp_valid, value, phase, is_final, is_connected = self._decode_v1_record(idx)

        if self.augment:
            state, policy, blueprint, bp_valid, value = self._apply_augmentation(
                state, policy, blueprint, bp_valid, value, phase, is_final, is_connected
            )

        if self.compute_legal_mask and state.shape[0] >= 7:
            valid_add, valid_remove = get_legal_moves(
                state[0],
                threshold=0.5,
                bc_masks=(state[1], state[2], state[3]),
                forces=(state[4], state[5], state[6]),
            )
            legal_mask = np.stack([valid_add, valid_remove], axis=0).astype(np.float32)
        else:
            legal_mask = np.ones_like(policy, dtype=np.float32)

        masked_pol = np.maximum(0.0, policy) * legal_mask
        pol_sum = float(masked_pol.sum())
        if pol_sum > 1e-6:
            policy = (masked_pol / pol_sum).astype(np.float32)
            policy_mask = 1.0
        else:
            policy = np.zeros_like(policy, dtype=np.float32)
            policy_mask = 0.0

        return {
            "state": torch.from_numpy(np.ascontiguousarray(state)).float(),
            "policy": torch.from_numpy(np.ascontiguousarray(policy)).float(),
            "legal_mask": torch.from_numpy(np.ascontiguousarray(legal_mask)).float(),
            "policy_mask": torch.tensor(policy_mask, dtype=torch.float32),
            "value": torch.tensor([value], dtype=torch.float32),
            "blueprint": torch.from_numpy(np.ascontiguousarray(blueprint)).float(),
            "blueprint_mask": torch.tensor(1.0 if bp_valid else 0.0, dtype=torch.float32),
            "phase": phase,
            "is_final": is_final,
            "is_connected": is_connected,
        }

    def _apply_augmentation(
        self,
        state: np.ndarray,
        policy: np.ndarray,
        blueprint: np.ndarray,
        bp_valid: bool,
        value: float,
        phase: str,
        is_final: bool,
        is_connected: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, float]:
        """
        Apply reformulated negative sampling followed by 8x per-sample rigid flips.
        (The final 2x rot90 is applied at batch level by `make_symmetry_collate_fn` -> 16x.)
        """
        # STEP 1: Reformulated Negative Sampling
        if is_final:
            if np.random.random() < 0.25:
                state, policy, value = erosion_attack(state, policy, value)
        elif phase == "REFINEMENT" and is_connected:
            r = np.random.random()
            if r < 0.04:
                new_state, new_policy, new_val = load_multiplier(state, policy, value, k=3.0)
                if new_val != value:
                    state, policy, value = new_state, new_policy, new_val
                    # Original 1x blueprint no longer describes optimal 3x design
                    bp_valid = False
            elif r < 0.08:
                state, policy, value = sabotage(state, policy, value)
            elif r < 0.14:
                state, policy, value = saboteur(state, policy, value)

        # STEP 2: 8x Rigid Reflections (preserve shape without square padding)
        if np.random.random() < 0.5:
            state, policy, blueprint = flip_x(state, policy, blueprint)
        if np.random.random() < 0.5:
            state, policy, blueprint = flip_y(state, policy, blueprint)
        if np.random.random() < 0.5:
            state, policy, blueprint = flip_z(state, policy, blueprint)

        return state, policy, blueprint, bp_valid, value

    def get_memory_usage_mb(self) -> float:
        """Estimate RAM usage of preloaded buffers in megabytes."""
        if not self.preload_to_ram:
            return 0.0

        total_bytes = 0
        if self.schema_version == "v2":
            for ep_info in self.episodes.values():
                total_bytes += ep_info["bc_masks"].nbytes
                total_bytes += ep_info["forces"].nbytes
                if ep_info["v_constructed"] is not None:
                    total_bytes += ep_info["v_constructed"].nbytes
            total_bytes += len(self._blob_buf)
            for arr in (
                self._offsets,
                self._ep_indices,
                self._steps,
                self._values,
                self._phases,
                self._is_final,
                self._is_connected,
            ):
                if arr is not None:
                    total_bytes += arr.nbytes
        else:
            for d in self._v1_preloaded:
                total_bytes += d["state"].nbytes + d["policy"].nbytes

        return total_bytes / (1024 * 1024)
