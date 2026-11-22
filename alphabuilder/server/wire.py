"""Normative `ABWIRE01` wire codec (little-endian, 4-byte aligned)."""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

MAGIC = b"ABWIRE01"
HEADER_LEN = 32

PAYLOAD_INFER_RESULT = 1
PAYLOAD_FEM_RESULT = 2
PAYLOAD_STEP_FRAME = 3
PAYLOAD_STEP_BATCH = 4

FLAG_ZLIB = 0x01

TAG_DENSITY = 1
TAG_RHO_CONT = 2
TAG_BC_MASKS = 3
TAG_FORCES = 4
TAG_POLICY_ADD = 5
TAG_POLICY_REMOVE = 6
TAG_MCTS_VISIT_ADD = 7
TAG_MCTS_VISIT_REMOVE = 8
TAG_MCTS_Q_ADD = 9
TAG_MCTS_Q_REMOVE = 10
TAG_BLUEPRINT_PROBS = 11
TAG_DISPLACEMENT_MAG = 12
TAG_DISPLACEMENT_VEC = 13

ENCODING_DENSE_LE = 0
ENCODING_COO1_RAW = 1

DTYPE_U8 = 0
DTYPE_F16 = 1
DTYPE_F32 = 2
DTYPE_I32 = 3


def _pad4(n: int) -> int:
    return (n + 3) & ~3


def _shape4(shape: Sequence[int], ndim: int) -> list[int]:
    if not 1 <= ndim <= 4:
        raise ValueError(f"ndim must be 1..4, got {ndim}")
    if len(shape) != ndim:
        raise ValueError(f"shape {shape} does not match ndim {ndim}")
    out = [int(v) for v in shape]
    if any(v < 0 or v > 0xFFFF for v in out):
        raise ValueError(f"shape dimension out of u16 range: {out}")
    return out + [0] * (4 - ndim)


@dataclass
class WireTensor:
    tag: int
    encoding: int
    dtype: int
    ndim: int
    shape: list[int]
    data: bytes


def dense_f32(tag: int, shape: Sequence[int], values: Iterable[float]) -> WireTensor:
    arr = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    ndim = len(shape)
    expected = int(np.prod(shape)) if shape else 0
    if arr.size != expected:
        raise ValueError(f"shape {tuple(shape)} expects {expected} values, got {arr.size}")
    return WireTensor(tag, ENCODING_DENSE_LE, DTYPE_F32, ndim, _shape4(shape, ndim), arr.tobytes())


def dense_f16(tag: int, shape: Sequence[int], values: Iterable[float]) -> WireTensor:
    arr = np.ascontiguousarray(np.asarray(values, dtype="<f2"))
    ndim = len(shape)
    expected = int(np.prod(shape)) if shape else 0
    if arr.size != expected:
        raise ValueError(f"shape {tuple(shape)} expects {expected} values, got {arr.size}")
    return WireTensor(tag, ENCODING_DENSE_LE, DTYPE_F16, ndim, _shape4(shape, ndim), arr.tobytes())


def dense_u8(tag: int, shape: Sequence[int], values: Iterable[int]) -> WireTensor:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.uint8))
    ndim = len(shape)
    expected = int(np.prod(shape)) if shape else 0
    if arr.size != expected:
        raise ValueError(f"shape {tuple(shape)} expects {expected} values, got {arr.size}")
    return WireTensor(tag, ENCODING_DENSE_LE, DTYPE_U8, ndim, _shape4(shape, ndim), arr.tobytes())


def coo1_bytes(indices: Sequence[int], values: Sequence[float]) -> bytes:
    if len(indices) != len(values):
        raise ValueError("indices/values length mismatch")
    n = len(indices)
    idx = np.ascontiguousarray(np.asarray(indices, dtype="<i4")).tobytes()
    val = np.ascontiguousarray(np.asarray(values, dtype="<f4")).tobytes()
    return b"COO1" + struct.pack("<I", n) + idx + val


def coo1_f32(tag: int, shape: Sequence[int], indices: Sequence[int], values: Sequence[float]) -> WireTensor:
    ndim = len(shape)
    return WireTensor(
        tag,
        ENCODING_COO1_RAW,
        DTYPE_F32,
        ndim,
        _shape4(shape, ndim),
        coo1_bytes(indices, values),
    )


def sparse_from_dense(dense: np.ndarray, threshold: float = 1e-7) -> tuple[list[int], list[float]]:
    flat = np.asarray(dense, dtype=np.float32).reshape(-1)
    mask = np.abs(flat) > threshold
    return np.flatnonzero(mask).astype(np.int32).tolist(), flat[mask].tolist()


def encode_packet(
    payload_type: int,
    header: dict,
    tensors: Sequence[WireTensor],
    compress: bool = True,
    flags: int = 0,
) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header_padded = _pad4(len(header_bytes))

    block = bytearray()
    for tensor in tensors:
        data = bytes(tensor.data)
        padded = _pad4(len(data))
        block += struct.pack("<HBB", tensor.tag, tensor.encoding, tensor.dtype)
        block += bytes([tensor.ndim, 0, 0, 0])
        block += struct.pack("<4H", *tensor.shape)
        block += struct.pack("<I", len(data))
        block += data
        block += b"\x00" * (padded - len(data))

    wire_flags = flags
    wire_block = bytes(block)
    if compress:
        wire_flags |= FLAG_ZLIB
        compressed = zlib.compress(wire_block, level=6)
        wire_block = compressed + b"\x00" * (_pad4(len(compressed)) - len(compressed))

    return (
        MAGIC
        + struct.pack(
            "<6I",
            payload_type,
            wire_flags,
            len(header_bytes),
            header_padded,
            len(tensors),
            len(wire_block),
        )
        + header_bytes
        + b"\x00" * (header_padded - len(header_bytes))
        + wire_block
    )


def encode_raw_packet(payload_type: int, header: dict, raw_block: bytes, flags: int = 0) -> bytes:
    """Encode a packet whose tensor block is an already-built byte sequence (STEP_BATCH)."""
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header_padded = _pad4(len(header_bytes))
    return (
        MAGIC
        + struct.pack(
            "<6I",
            payload_type,
            flags,
            len(header_bytes),
            header_padded,
            0,
            len(raw_block),
        )
        + header_bytes
        + b"\x00" * (header_padded - len(header_bytes))
        + raw_block
    )


def encode_step_batch(frames: Sequence[bytes]) -> bytes:
    header = {
        "count": len(frames),
        "frame_byte_lengths": [len(frame) for frame in frames],
    }
    return encode_raw_packet(PAYLOAD_STEP_BATCH, header, b"".join(frames))


@dataclass
class DecodedPacket:
    payload_type: int
    flags: int
    header: dict
    tensors: list[WireTensor]


def decode_packet(raw: bytes) -> DecodedPacket:
    if len(raw) < HEADER_LEN:
        raise ValueError("ABWIRE01 buffer smaller than header")
    if raw[:8] != MAGIC:
        raise ValueError("invalid ABWIRE01 magic")
    payload_type, flags, header_len, header_padded, num_tensors, block_len = struct.unpack_from(
        "<6I", raw, 8
    )
    header_end = HEADER_LEN + header_padded
    block_end = header_end + block_len
    if header_padded < header_len or header_padded % 4 or block_end > len(raw):
        raise ValueError("invalid ABWIRE01 lengths")
    header = json.loads(raw[HEADER_LEN : HEADER_LEN + header_len].decode("utf-8"))
    block = raw[header_end:block_end]
    if flags & FLAG_ZLIB:
        block = zlib.decompress(block)
    tensors: list[WireTensor] = []
    cursor = 0
    for _ in range(num_tensors):
        if cursor + 20 > len(block):
            raise ValueError("truncated TensorEntry")
        tag, encoding, dtype, ndim = struct.unpack_from("<HBBB", block, cursor)
        shape = list(struct.unpack_from("<4H", block, cursor + 8))
        data_len = struct.unpack_from("<I", block, cursor + 16)[0]
        cursor += 20
        data = block[cursor : cursor + data_len]
        cursor += _pad4(data_len)
        tensors.append(WireTensor(tag, encoding, dtype, ndim, shape, data))
    return DecodedPacket(payload_type, flags, header, tensors)


def tensor_to_dense(tensor: WireTensor) -> np.ndarray:
    shape = tensor.shape[: tensor.ndim]
    total = int(np.prod(shape)) if shape else 0
    if tensor.encoding == ENCODING_COO1_RAW:
        if tensor.data[:4] != b"COO1":
            raise ValueError("missing COO1 magic")
        n = struct.unpack_from("<I", tensor.data, 4)[0]
        idx = np.frombuffer(tensor.data, dtype="<i4", count=n, offset=8)
        vals = np.frombuffer(tensor.data, dtype="<f4", count=n, offset=8 + n * 4)
        out = np.zeros(total, dtype=np.float32)
        out[idx] = vals
        return out.reshape(shape)
    if tensor.encoding != ENCODING_DENSE_LE:
        raise ValueError(f"unsupported encoding {tensor.encoding}")
    if tensor.dtype == DTYPE_U8:
        arr = np.frombuffer(tensor.data, dtype=np.uint8)
    elif tensor.dtype == DTYPE_F16:
        arr = np.frombuffer(tensor.data, dtype="<f2").astype(np.float32)
    elif tensor.dtype == DTYPE_F32:
        arr = np.frombuffer(tensor.data, dtype="<f4")
    elif tensor.dtype == DTYPE_I32:
        arr = np.frombuffer(tensor.data, dtype="<i4").astype(np.float32)
    else:
        raise ValueError(f"unsupported dtype {tensor.dtype}")
    return arr.reshape(shape)
