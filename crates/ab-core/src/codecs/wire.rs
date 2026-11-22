//! Normative `ABWIRE01` (network/WS) and `ABMESH01` (Tauri IPC) protocols.
//!
//! All integers/floats are little-endian and every section is 4-byte aligned.

use serde::{Deserialize, Serialize};

use crate::codecs::coo1::Coo1;
use crate::coords::Dims;
use crate::error::{CoreError, CoreResult};

pub const ABWIRE01_MAGIC: &[u8; 8] = b"ABWIRE01";
pub const ABMESH01_MAGIC: &[u8; 8] = b"ABMESH01";
pub const ABWIRE01_HEADER_LEN: usize = 32;
pub const ABMESH01_HEADER_LEN: usize = 96;

pub mod abmesh_flags {
    pub const DISPLACEMENT: u32 = 0x01;
    pub const BLUEPRINT: u32 = 0x02;
    pub const POLICY: u32 = 0x04;
    pub const MCTS: u32 = 0x08;
    pub const DIFF: u32 = 0x10;
    pub const RHO_CONT: u32 = 0x20;
}

pub mod payload_type {
    pub const INFER_RESULT: u32 = 1;
    pub const FEM_RESULT: u32 = 2;
    pub const STEP_FRAME: u32 = 3;
    pub const STEP_BATCH: u32 = 4;
}

pub mod tensor_tag {
    pub const DENSITY: u16 = 1;
    pub const RHO_CONT: u16 = 2;
    pub const BC_MASKS: u16 = 3;
    pub const FORCES: u16 = 4;
    pub const POLICY_ADD: u16 = 5;
    pub const POLICY_REMOVE: u16 = 6;
    pub const MCTS_VISIT_ADD: u16 = 7;
    pub const MCTS_VISIT_REMOVE: u16 = 8;
    pub const MCTS_Q_ADD: u16 = 9;
    pub const MCTS_Q_REMOVE: u16 = 10;
    pub const BLUEPRINT_PROBS: u16 = 11;
    pub const DISPLACEMENT_MAG: u16 = 12;
    pub const DISPLACEMENT_VEC: u16 = 13;
}

pub mod encoding {
    pub const DENSE_LE: u8 = 0;
    pub const COO1_RAW: u8 = 1;
}

pub mod dtype {
    pub const U8: u8 = 0;
    pub const F16: u8 = 1;
    pub const F32: u8 = 2;
    pub const I32: u8 = 3;
}

pub mod wire_flags {
    pub const ZLIB_TENSOR_BLOCK: u32 = 0x01;
}

fn checked_shape_product(shape: [u16; 4], ndim: u8) -> CoreResult<usize> {
    if !matches!(ndim, 1..=4) {
        return Err(CoreError::Wire {
            message: format!("ndim must be 1..4, got {ndim}"),
        });
    }
    let mut total = 1_usize;
    for &dim in shape.iter().take(ndim as usize) {
        total = total
            .checked_mul(dim as usize)
            .ok_or_else(|| CoreError::Wire {
                message: "shape product overflow".to_string(),
            })?;
    }
    Ok(total)
}

/// One `TensorEntry` of an `ABWIRE01` packet.
#[derive(Debug, Clone, PartialEq)]
pub struct WireTensor {
    pub tag: u16,
    pub encoding: u8,
    pub dtype: u8,
    pub ndim: u8,
    pub shape: [u16; 4],
    /// Raw little-endian payload (unpadded).
    pub data: Vec<u8>,
}

impl WireTensor {
    pub fn dense_f32(tag: u16, shape: [u16; 4], fdims: usize, values: &[f32]) -> CoreResult<Self> {
        if !matches!(fdims, 1..=4) {
            return Err(CoreError::Wire {
                message: format!("ndim must be 1..4, got {fdims}"),
            });
        }
        let expected = shape.iter().take(fdims).product::<u16>() as usize;
        if expected != values.len() {
            return Err(CoreError::Wire {
                message: format!(
                    "shape {:?} expects {} values, got {}",
                    &shape[..fdims],
                    expected,
                    values.len()
                ),
            });
        }
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Ok(Self {
            tag,
            encoding: encoding::DENSE_LE,
            dtype: dtype::F32,
            ndim: fdims as u8,
            shape,
            data,
        })
    }

    pub fn dense_f16(tag: u16, shape: [u16; 4], fdims: usize, values: &[f32]) -> CoreResult<Self> {
        let expected = shape.iter().take(fdims).product::<u16>() as usize;
        if expected != values.len() {
            return Err(CoreError::Wire {
                message: format!(
                    "shape {:?} expects {} values, got {}",
                    &shape[..fdims],
                    expected,
                    values.len()
                ),
            });
        }
        let mut data = Vec::with_capacity(values.len() * 2);
        for v in values {
            data.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
        }
        Ok(Self {
            tag,
            encoding: encoding::DENSE_LE,
            dtype: dtype::F16,
            ndim: fdims as u8,
            shape,
            data,
        })
    }

    pub fn dense_u8(tag: u16, shape: [u16; 4], fdims: usize, values: &[u8]) -> CoreResult<Self> {
        let expected = shape.iter().take(fdims).product::<u16>() as usize;
        if expected != values.len() {
            return Err(CoreError::Wire {
                message: format!(
                    "shape {:?} expects {} values, got {}",
                    &shape[..fdims],
                    expected,
                    values.len()
                ),
            });
        }
        Ok(Self {
            tag,
            encoding: encoding::DENSE_LE,
            dtype: dtype::U8,
            ndim: fdims as u8,
            shape,
            data: values.to_vec(),
        })
    }

    pub fn coo1_f32(tag: u16, shape: [u16; 4], fdims: usize, coo: &Coo1) -> CoreResult<Self> {
        let total = checked_shape_product(shape, fdims as u8)?;
        Ok(Self {
            tag,
            encoding: encoding::COO1_RAW,
            dtype: dtype::F32,
            ndim: fdims as u8,
            shape,
            data: coo.encode(total)?,
        })
    }

    /// Decode any supported entry into a dense `f32` vector.
    pub fn to_dense_f32(&self) -> CoreResult<Vec<f32>> {
        if self.encoding == encoding::COO1_RAW {
            let total = checked_shape_product(self.shape, self.ndim)?;
            let coo = Coo1::parse(&self.data, total)?;
            return coo.to_dense(total);
        }
        if self.encoding != encoding::DENSE_LE {
            return Err(CoreError::Wire {
                message: format!("unsupported encoding {}", self.encoding),
            });
        }
        let total = checked_shape_product(self.shape, self.ndim)?;
        let expected = match self.dtype {
            dtype::U8 => total,
            dtype::F16 => total * 2,
            dtype::F32 | dtype::I32 => total * 4,
            other => {
                return Err(CoreError::Wire {
                    message: format!("unsupported dtype {other}"),
                })
            }
        };
        if self.data.len() != expected {
            return Err(CoreError::Wire {
                message: format!(
                    "tensor tag {} data_len {} != expected {}",
                    self.tag,
                    self.data.len(),
                    expected
                ),
            });
        }
        let mut out = Vec::with_capacity(total);
        match self.dtype {
            dtype::U8 => out.extend(self.data.iter().map(|&b| b as f32)),
            dtype::F16 => {
                for chunk in self.data.as_chunks::<2>().0 {
                    out.push(half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
                }
            }
            dtype::F32 => {
                for chunk in self.data.as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            dtype::I32 => {
                for chunk in self.data.as_chunks::<4>().0 {
                    out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32);
                }
            }
            _ => unreachable!(),
        }
        Ok(out)
    }
}

/// `ABWIRE01` packet (header + metadata JSON + tensor block).
#[derive(Debug, Clone, PartialEq)]
pub struct WirePacket {
    pub payload_type: u32,
    pub flags: u32,
    pub header_json: String,
    pub tensors: Vec<WireTensor>,
}

impl WirePacket {
    pub fn new(payload_type: u32, header_json: String, tensors: Vec<WireTensor>) -> Self {
        Self {
            payload_type,
            flags: 0,
            header_json,
            tensors,
        }
    }

    pub fn encode(&self) -> CoreResult<Vec<u8>> {
        let header_bytes = self.header_json.as_bytes();
        let header_len = header_bytes.len();
        let header_padded = (header_len + 3) & !3;
        if header_len > u32::MAX as usize {
            return Err(CoreError::Wire {
                message: "header JSON too large".to_string(),
            });
        }

        let mut block = Vec::new();
        for tensor in &self.tensors {
            let padded_len = (tensor.data.len() + 3) & !3;
            if tensor.data.len() > u32::MAX as usize {
                return Err(CoreError::Wire {
                    message: "tensor payload too large".to_string(),
                });
            }
            block.extend_from_slice(&tensor.tag.to_le_bytes());
            block.push(tensor.encoding);
            block.push(tensor.dtype);
            block.push(tensor.ndim);
            block.extend_from_slice(&[0, 0, 0]);
            for dim in tensor.shape {
                block.extend_from_slice(&dim.to_le_bytes());
            }
            block.extend_from_slice(&(tensor.data.len() as u32).to_le_bytes());
            block.extend_from_slice(&tensor.data);
            block.resize(block.len() + (padded_len - tensor.data.len()), 0);
        }

        let on_wire_block = if self.flags & wire_flags::ZLIB_TENSOR_BLOCK != 0 {
            let mut compressed = crate::codecs::zlib_compress(&block)?;
            let padded = (compressed.len() + 3) & !3;
            compressed.resize(padded, 0);
            compressed
        } else {
            block
        };

        let total = ABWIRE01_HEADER_LEN + header_padded + on_wire_block.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(ABWIRE01_MAGIC);
        out.extend_from_slice(&self.payload_type.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&(header_len as u32).to_le_bytes());
        out.extend_from_slice(&(header_padded as u32).to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
        out.extend_from_slice(&(on_wire_block.len() as u32).to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.resize(out.len() + (header_padded - header_len), 0);
        out.extend_from_slice(&on_wire_block);
        debug_assert_eq!(out.len(), total);
        Ok(out)
    }

    /// Decode a standalone packet; trailing bytes are rejected.
    pub fn decode(raw: &[u8]) -> CoreResult<Self> {
        let (packet, consumed) = Self::decode_prefix(raw)?;
        if consumed != raw.len() {
            return Err(CoreError::Wire {
                message: format!(
                    "ABWIRE01 trailing bytes: consumed {consumed} of {}",
                    raw.len()
                ),
            });
        }
        Ok(packet)
    }

    /// Decode one packet from a possibly larger buffer (used by STEP_BATCH).
    pub fn decode_prefix(raw: &[u8]) -> CoreResult<(Self, usize)> {
        if raw.len() < ABWIRE01_HEADER_LEN {
            return Err(CoreError::Wire {
                message: "buffer smaller than 32-byte header".to_string(),
            });
        }
        if &raw[0..8] != ABWIRE01_MAGIC {
            return Err(CoreError::Wire {
                message: "invalid ABWIRE01 magic".to_string(),
            });
        }
        let payload_type = read_u32(raw, 8)?;
        let flags = read_u32(raw, 12)?;
        let header_len = read_u32(raw, 16)? as usize;
        let header_padded = read_u32(raw, 20)? as usize;
        let num_tensors = read_u32(raw, 24)? as usize;
        let tensor_block_len = read_u32(raw, 28)? as usize;
        if header_padded < header_len || !header_padded.is_multiple_of(4) {
            return Err(CoreError::Wire {
                message: format!("invalid header lengths: {header_len} / {header_padded}"),
            });
        }
        let header_end = ABWIRE01_HEADER_LEN + header_padded;
        let block_end =
            header_end
                .checked_add(tensor_block_len)
                .ok_or_else(|| CoreError::Wire {
                    message: "tensor block length overflow".to_string(),
                })?;
        if block_end > raw.len() {
            return Err(CoreError::Wire {
                message: format!("truncated packet: need {block_end}, have {}", raw.len()),
            });
        }
        let header_json =
            std::str::from_utf8(&raw[ABWIRE01_HEADER_LEN..ABWIRE01_HEADER_LEN + header_len])
                .map_err(|e| CoreError::Wire {
                    message: format!("header JSON is not UTF-8: {e}"),
                })?
                .to_string();
        serde_json::from_str::<serde_json::Value>(&header_json).map_err(|e| CoreError::Wire {
            message: format!("invalid header JSON: {e}"),
        })?;

        let block_raw = &raw[header_end..block_end];
        let block = if flags & wire_flags::ZLIB_TENSOR_BLOCK != 0 {
            crate::codecs::zlib_decompress(block_raw, 128 * 1024 * 1024)?
        } else {
            block_raw.to_vec()
        };

        let mut tensors = Vec::with_capacity(num_tensors.min(64));
        let mut cursor = 0_usize;
        for _ in 0..num_tensors {
            if cursor + 20 > block.len() {
                return Err(CoreError::Wire {
                    message: "truncated TensorEntry header".to_string(),
                });
            }
            let tag = u16::from_le_bytes([block[cursor], block[cursor + 1]]);
            let enc = block[cursor + 2];
            let dt = block[cursor + 3];
            let ndim = block[cursor + 4];
            if !matches!(ndim, 1..=4) {
                return Err(CoreError::Wire {
                    message: format!("invalid ndim {ndim}"),
                });
            }
            let mut shape = [0_u16; 4];
            for (i, dim) in shape.iter_mut().enumerate() {
                let off = cursor + 8 + i * 2;
                *dim = u16::from_le_bytes([block[off], block[off + 1]]);
            }
            let data_len = u32::from_le_bytes([
                block[cursor + 16],
                block[cursor + 17],
                block[cursor + 18],
                block[cursor + 19],
            ]) as usize;
            cursor += 20;
            let padded = (data_len + 3) & !3;
            if cursor + padded > block.len() {
                return Err(CoreError::Wire {
                    message: "truncated TensorEntry payload".to_string(),
                });
            }
            let data = block[cursor..cursor + data_len].to_vec();
            cursor += padded;
            tensors.push(WireTensor {
                tag,
                encoding: enc,
                dtype: dt,
                ndim,
                shape,
                data,
            });
        }
        if cursor != block.len() {
            return Err(CoreError::Wire {
                message: format!(
                    "tensor block cursor {} != decompressed length {}",
                    cursor,
                    block.len()
                ),
            });
        }
        Ok((
            Self {
                payload_type,
                flags,
                header_json,
                tensors,
            },
            block_end,
        ))
    }

    pub fn tensor(&self, tag: u16) -> Option<&WireTensor> {
        self.tensors.iter().find(|t| t.tag == tag)
    }
}

/// Concatenate complete `STEP_FRAME` packets into a `STEP_BATCH` packet.
pub fn encode_step_batch(frames: &[Vec<u8>]) -> CoreResult<Vec<u8>> {
    let lengths: Vec<usize> = frames.iter().map(|frame| frame.len()).collect();
    let header = serde_json::json!({
        "count": frames.len(),
        "frame_byte_lengths": lengths,
    });
    let header_bytes = serde_json::to_vec(&header)?;
    let header_padded = (header_bytes.len() + 3) & !3;
    let mut block = Vec::new();
    for frame in frames {
        if !frame.len().is_multiple_of(4) {
            return Err(CoreError::Wire {
                message: "STEP_FRAME packets must be 4-byte aligned".to_string(),
            });
        }
        block.extend_from_slice(frame);
    }
    let mut out = Vec::with_capacity(ABWIRE01_HEADER_LEN + header_padded + block.len());
    out.extend_from_slice(ABWIRE01_MAGIC);
    out.extend_from_slice(&payload_type::STEP_BATCH.to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes());
    out.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&(header_padded as u32).to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes());
    out.extend_from_slice(&(block.len() as u32).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.resize(out.len() + (header_padded - header_bytes.len()), 0);
    out.extend_from_slice(&block);
    Ok(out)
}

/// Decode a `STEP_BATCH` packet into its constituent `STEP_FRAME` packets.
pub fn decode_step_batch(raw: &[u8]) -> CoreResult<Vec<WirePacket>> {
    if raw.len() < ABWIRE01_HEADER_LEN || &raw[0..8] != ABWIRE01_MAGIC {
        return Err(CoreError::Wire {
            message: "invalid STEP_BATCH magic".to_string(),
        });
    }
    let kind = read_u32(raw, 8)?;
    if kind != payload_type::STEP_BATCH {
        return Err(CoreError::Wire {
            message: format!("expected STEP_BATCH (4), got {kind}"),
        });
    }
    let flags = read_u32(raw, 12)?;
    let header_len = read_u32(raw, 16)? as usize;
    let header_padded = read_u32(raw, 20)? as usize;
    let block_len = read_u32(raw, 28)? as usize;
    if header_padded < header_len || !header_padded.is_multiple_of(4) {
        return Err(CoreError::Wire {
            message: "invalid STEP_BATCH header lengths".to_string(),
        });
    }
    let header_end = ABWIRE01_HEADER_LEN + header_padded;
    let block_end = header_end + block_len;
    if block_end > raw.len() {
        return Err(CoreError::Wire {
            message: "truncated STEP_BATCH".to_string(),
        });
    }
    let header_text = std::str::from_utf8(
        &raw[ABWIRE01_HEADER_LEN..ABWIRE01_HEADER_LEN + header_len],
    )
    .map_err(|e| CoreError::Wire {
        message: format!("STEP_BATCH header is not UTF-8: {e}"),
    })?;
    let header: serde_json::Value = serde_json::from_str(header_text)?;
    let count = header.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let lengths: Vec<usize> = header
        .get("frame_byte_lengths")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect()
        })
        .unwrap_or_default();
    if lengths.len() != count {
        return Err(CoreError::Wire {
            message: "STEP_BATCH frame_byte_lengths/count mismatch".to_string(),
        });
    }
    let block_raw = &raw[header_end..block_end];
    let block = if flags & wire_flags::ZLIB_TENSOR_BLOCK != 0 {
        crate::codecs::zlib_decompress(block_raw, 128 * 1024 * 1024)?
    } else {
        block_raw.to_vec()
    };
    if lengths.iter().sum::<usize>() != block.len() {
        return Err(CoreError::Wire {
            message: "STEP_BATCH lengths do not cover the tensor block".to_string(),
        });
    }
    let mut frames = Vec::with_capacity(count);
    let mut cursor = 0_usize;
    for length in lengths {
        let end = cursor + length;
        let slice = &block[cursor..end];
        let (packet, consumed) = WirePacket::decode_prefix(slice)?;
        if consumed != length {
            return Err(CoreError::Wire {
                message: format!("STEP_FRAME length mismatch: {consumed} != {length}"),
            });
        }
        frames.push(packet);
        cursor = end;
    }
    Ok(frames)
}

/// `ABMESH01` packet: header + surface-culled mesh + overlays + metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct AbMeshPacket {
    pub flags: u32,
    pub step_index: u32,
    pub dims: Dims,
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    /// `[ao, density, disp_norm, strain_norm]` per vertex.
    pub ao_and_scalars: Vec<f32>,
    pub voxel_ids: Vec<u32>,
    pub indices: Vec<u32>,
    pub blueprint_positions: Vec<f32>,
    pub blueprint_normals: Vec<f32>,
    pub blueprint_indices: Vec<u32>,
    pub policy_add: Vec<(u32, f32)>,
    pub policy_rem: Vec<(u32, f32)>,
    pub mcts_add: Vec<(u32, f32, f32)>,
    pub mcts_rem: Vec<(u32, f32, f32)>,
    pub diff_added: Vec<u32>,
    pub diff_removed: Vec<u32>,
    pub disp_p99: f32,
    pub strain_p99: f32,
    pub meta_json: String,
}

impl AbMeshPacket {
    pub fn empty(dims: Dims, step_index: u32) -> Self {
        Self {
            flags: 0,
            step_index,
            dims,
            positions: Vec::new(),
            normals: Vec::new(),
            ao_and_scalars: Vec::new(),
            voxel_ids: Vec::new(),
            indices: Vec::new(),
            blueprint_positions: Vec::new(),
            blueprint_normals: Vec::new(),
            blueprint_indices: Vec::new(),
            policy_add: Vec::new(),
            policy_rem: Vec::new(),
            mcts_add: Vec::new(),
            mcts_rem: Vec::new(),
            diff_added: Vec::new(),
            diff_removed: Vec::new(),
            disp_p99: 0.0,
            strain_p99: 0.0,
            meta_json: "{}".to_string(),
        }
    }

    fn validate(&self) -> CoreResult<()> {
        self.dims.validate()?;
        if !self.positions.len().is_multiple_of(3) {
            return Err(CoreError::AbMesh {
                message: "positions length is not a multiple of 3".to_string(),
            });
        }
        let vertex_count = self.positions.len() / 3;
        if self.normals.len() != self.positions.len() {
            return Err(CoreError::AbMesh {
                message: "normals length differs from positions".to_string(),
            });
        }
        if self.ao_and_scalars.len() != vertex_count * 4 {
            return Err(CoreError::AbMesh {
                message: "ao_and_scalars must hold 4 floats per vertex".to_string(),
            });
        }
        if self.voxel_ids.len() != vertex_count {
            return Err(CoreError::AbMesh {
                message: "voxel_ids must hold 1 entry per vertex".to_string(),
            });
        }
        if !self.indices.len().is_multiple_of(3) {
            return Err(CoreError::AbMesh {
                message: "indices length is not a multiple of 3".to_string(),
            });
        }
        if !self.blueprint_positions.len().is_multiple_of(3)
            || self.blueprint_normals.len() != self.blueprint_positions.len()
            || !self.blueprint_indices.len().is_multiple_of(3)
        {
            return Err(CoreError::AbMesh {
                message: "invalid blueprint mesh section".to_string(),
            });
        }
        if self.meta_json.is_empty() {
            return Err(CoreError::AbMesh {
                message: "meta_json must contain at least '{}'".to_string(),
            });
        }
        serde_json::from_str::<serde_json::Value>(&self.meta_json).map_err(|e| {
            CoreError::AbMesh {
                message: format!("meta_json is not valid JSON: {e}"),
            }
        })?;
        let fits = |n: usize| n <= u32::MAX as usize;
        if !fits(vertex_count)
            || !fits(self.indices.len())
            || !fits(self.blueprint_positions.len() / 3)
            || !fits(self.blueprint_indices.len())
        {
            return Err(CoreError::AbMesh {
                message: "section count exceeds u32".to_string(),
            });
        }
        Ok(())
    }

    pub fn encode(&self) -> CoreResult<Vec<u8>> {
        self.validate()?;
        let v = self.positions.len() / 3;
        let i = self.indices.len();
        let vb = self.blueprint_positions.len() / 3;
        let ib = self.blueprint_indices.len();
        let meta_len = self.meta_json.len();
        let meta_padded = (meta_len + 3) & !3;

        let total = ABMESH01_HEADER_LEN
            + v * 12
            + v * 12
            + v * 16
            + v * 4
            + i * 4
            + vb * 12
            + vb * 12
            + ib * 4
            + self.policy_add.len() * 8
            + self.policy_rem.len() * 8
            + self.mcts_add.len() * 12
            + self.mcts_rem.len() * 12
            + self.diff_added.len() * 4
            + self.diff_removed.len() * 4
            + meta_padded;
        let total_u32 = u32::try_from(total).map_err(|_| CoreError::AbMesh {
            message: "ABMESH01 packet exceeds 4 GiB".to_string(),
        })?;

        let mut out = vec![0_u8; ABMESH01_HEADER_LEN];
        out[0..8].copy_from_slice(ABMESH01_MAGIC);
        write_u32(&mut out, 8, self.flags);
        write_u32(&mut out, 12, self.step_index);
        write_u32(&mut out, 16, self.dims.d);
        write_u32(&mut out, 20, self.dims.h);
        write_u32(&mut out, 24, self.dims.w);
        write_u32(&mut out, 28, v as u32);
        write_u32(&mut out, 32, i as u32);
        write_u32(&mut out, 36, self.policy_add.len() as u32);
        write_u32(&mut out, 40, self.policy_rem.len() as u32);
        write_u32(&mut out, 44, self.mcts_add.len() as u32);
        write_u32(&mut out, 48, self.mcts_rem.len() as u32);
        write_u32(&mut out, 52, vb as u32);
        write_u32(&mut out, 56, ib as u32);
        write_u32(&mut out, 60, self.diff_added.len() as u32);
        write_u32(&mut out, 64, self.diff_removed.len() as u32);
        write_f32(&mut out, 68, self.disp_p99);
        write_f32(&mut out, 72, self.strain_p99);
        write_u32(&mut out, 76, meta_len as u32);
        write_u32(&mut out, 80, meta_padded as u32);
        write_u32(&mut out, 84, 0);
        write_u32(&mut out, 88, 0);
        write_u32(&mut out, 92, total_u32);

        let mut body = Vec::with_capacity(total - ABMESH01_HEADER_LEN);
        push_f32s(&mut body, &self.positions);
        push_f32s(&mut body, &self.normals);
        push_f32s(&mut body, &self.ao_and_scalars);
        push_u32s(&mut body, &self.voxel_ids);
        push_u32s(&mut body, &self.indices);
        push_f32s(&mut body, &self.blueprint_positions);
        push_f32s(&mut body, &self.blueprint_normals);
        push_u32s(&mut body, &self.blueprint_indices);
        for &(idx, prob) in &self.policy_add {
            body.extend_from_slice(&idx.to_le_bytes());
            body.extend_from_slice(&prob.to_le_bytes());
        }
        for &(idx, prob) in &self.policy_rem {
            body.extend_from_slice(&idx.to_le_bytes());
            body.extend_from_slice(&prob.to_le_bytes());
        }
        for &(idx, visits, q) in &self.mcts_add {
            body.extend_from_slice(&idx.to_le_bytes());
            body.extend_from_slice(&visits.to_le_bytes());
            body.extend_from_slice(&q.to_le_bytes());
        }
        for &(idx, visits, q) in &self.mcts_rem {
            body.extend_from_slice(&idx.to_le_bytes());
            body.extend_from_slice(&visits.to_le_bytes());
            body.extend_from_slice(&q.to_le_bytes());
        }
        push_u32s(&mut body, &self.diff_added);
        push_u32s(&mut body, &self.diff_removed);
        body.extend_from_slice(self.meta_json.as_bytes());
        body.resize(body.len() + (meta_padded - meta_len), 0);
        debug_assert_eq!(body.len(), total - ABMESH01_HEADER_LEN);
        out.extend_from_slice(&body);
        debug_assert_eq!(out.len(), total);
        Ok(out)
    }

    pub fn decode(raw: &[u8]) -> CoreResult<Self> {
        if raw.len() < ABMESH01_HEADER_LEN {
            return Err(CoreError::AbMesh {
                message: "buffer smaller than 96-byte header".to_string(),
            });
        }
        if &raw[0..8] != ABMESH01_MAGIC {
            return Err(CoreError::AbMesh {
                message: "invalid ABMESH01 magic".to_string(),
            });
        }
        let total_byte_len = read_u32(raw, 92)? as usize;
        if total_byte_len != raw.len() {
            return Err(CoreError::AbMesh {
                message: format!(
                    "total_byte_len {total_byte_len} != buffer length {}",
                    raw.len()
                ),
            });
        }
        let counts = AbMeshHeader {
            flags: read_u32(raw, 8)?,
            step_index: read_u32(raw, 12)?,
            dims: Dims::new(read_u32(raw, 16)?, read_u32(raw, 20)?, read_u32(raw, 24)?),
            vertex_count: read_u32(raw, 28)? as usize,
            index_count: read_u32(raw, 32)? as usize,
            policy_add_count: read_u32(raw, 36)? as usize,
            policy_rem_count: read_u32(raw, 40)? as usize,
            mcts_add_count: read_u32(raw, 44)? as usize,
            mcts_rem_count: read_u32(raw, 48)? as usize,
            blueprint_vertex_count: read_u32(raw, 52)? as usize,
            blueprint_index_count: read_u32(raw, 56)? as usize,
            diff_added_count: read_u32(raw, 60)? as usize,
            diff_removed_count: read_u32(raw, 64)? as usize,
            disp_p99: f32::from_bits(read_u32(raw, 68)?),
            strain_p99: f32::from_bits(read_u32(raw, 72)?),
            meta_json_len: read_u32(raw, 76)? as usize,
            meta_json_padded_len: read_u32(raw, 80)? as usize,
        };
        if counts.meta_json_padded_len < counts.meta_json_len
            || !counts.meta_json_padded_len.is_multiple_of(4)
        {
            return Err(CoreError::AbMesh {
                message: "invalid meta_json lengths".to_string(),
            });
        }
        let required = ABMESH01_HEADER_LEN
            + counts.vertex_count * 12
            + counts.vertex_count * 12
            + counts.vertex_count * 16
            + counts.vertex_count * 4
            + counts.index_count * 4
            + counts.blueprint_vertex_count * 12
            + counts.blueprint_vertex_count * 12
            + counts.blueprint_index_count * 4
            + counts.policy_add_count * 8
            + counts.policy_rem_count * 8
            + counts.mcts_add_count * 12
            + counts.mcts_rem_count * 12
            + counts.diff_added_count * 4
            + counts.diff_removed_count * 4
            + counts.meta_json_padded_len;
        if required != total_byte_len {
            return Err(CoreError::AbMesh {
                message: format!(
                    "sections require {required} bytes, header declares {total_byte_len}"
                ),
            });
        }

        let mut cursor = ABMESH01_HEADER_LEN;
        let positions = take_f32s(raw, &mut cursor, counts.vertex_count * 3)?;
        let normals = take_f32s(raw, &mut cursor, counts.vertex_count * 3)?;
        let ao_and_scalars = take_f32s(raw, &mut cursor, counts.vertex_count * 4)?;
        let voxel_ids = take_u32s(raw, &mut cursor, counts.vertex_count)?;
        let indices = take_u32s(raw, &mut cursor, counts.index_count)?;
        let blueprint_positions = take_f32s(raw, &mut cursor, counts.blueprint_vertex_count * 3)?;
        let blueprint_normals = take_f32s(raw, &mut cursor, counts.blueprint_vertex_count * 3)?;
        let blueprint_indices = take_u32s(raw, &mut cursor, counts.blueprint_index_count)?;

        let mut policy_add = Vec::with_capacity(counts.policy_add_count);
        for _ in 0..counts.policy_add_count {
            let idx = read_u32(raw, cursor)?;
            cursor += 4;
            let prob = f32::from_bits(read_u32(raw, cursor)?);
            cursor += 4;
            policy_add.push((idx, prob));
        }
        let mut policy_rem = Vec::with_capacity(counts.policy_rem_count);
        for _ in 0..counts.policy_rem_count {
            let idx = read_u32(raw, cursor)?;
            cursor += 4;
            let prob = f32::from_bits(read_u32(raw, cursor)?);
            cursor += 4;
            policy_rem.push((idx, prob));
        }
        let mut mcts_add = Vec::with_capacity(counts.mcts_add_count);
        for _ in 0..counts.mcts_add_count {
            let idx = read_u32(raw, cursor)?;
            cursor += 4;
            let visits = f32::from_bits(read_u32(raw, cursor)?);
            cursor += 4;
            let q = f32::from_bits(read_u32(raw, cursor)?);
            cursor += 4;
            mcts_add.push((idx, visits, q));
        }
        let mut mcts_rem = Vec::with_capacity(counts.mcts_rem_count);
        for _ in 0..counts.mcts_rem_count {
            let idx = read_u32(raw, cursor)?;
            cursor += 4;
            let visits = f32::from_bits(read_u32(raw, cursor)?);
            cursor += 4;
            let q = f32::from_bits(read_u32(raw, cursor)?);
            cursor += 4;
            mcts_rem.push((idx, visits, q));
        }
        let diff_added = take_u32s(raw, &mut cursor, counts.diff_added_count)?;
        let diff_removed = take_u32s(raw, &mut cursor, counts.diff_removed_count)?;
        if cursor + counts.meta_json_padded_len != total_byte_len {
            return Err(CoreError::AbMesh {
                message: "metadata cursor mismatch".to_string(),
            });
        }
        let meta_bytes = &raw[cursor..cursor + counts.meta_json_len];
        let meta_json = std::str::from_utf8(meta_bytes)
            .map_err(|e| CoreError::AbMesh {
                message: format!("meta_json is not UTF-8: {e}"),
            })?
            .to_string();
        serde_json::from_str::<serde_json::Value>(&meta_json).map_err(|e| CoreError::AbMesh {
            message: format!("meta_json is not valid JSON: {e}"),
        })?;
        cursor += counts.meta_json_padded_len;
        if cursor != total_byte_len {
            return Err(CoreError::AbMesh {
                message: format!("cursor {cursor} != total_byte_len {total_byte_len}"),
            });
        }

        let packet = Self {
            flags: counts.flags,
            step_index: counts.step_index,
            dims: counts.dims,
            positions,
            normals,
            ao_and_scalars,
            voxel_ids,
            indices,
            blueprint_positions,
            blueprint_normals,
            blueprint_indices,
            policy_add,
            policy_rem,
            mcts_add,
            mcts_rem,
            diff_added,
            diff_removed,
            disp_p99: counts.disp_p99,
            strain_p99: counts.strain_p99,
            meta_json,
        };
        packet.validate()?;
        Ok(packet)
    }

    /// Parse the fixed 96-byte header without materializing the body.
    pub fn header(raw: &[u8]) -> CoreResult<AbMeshHeader> {
        if raw.len() < ABMESH01_HEADER_LEN {
            return Err(CoreError::AbMesh {
                message: "buffer smaller than 96-byte header".to_string(),
            });
        }
        if &raw[0..8] != ABMESH01_MAGIC {
            return Err(CoreError::AbMesh {
                message: "invalid ABMESH01 magic".to_string(),
            });
        }
        Ok(AbMeshHeader {
            flags: read_u32(raw, 8)?,
            step_index: read_u32(raw, 12)?,
            dims: Dims::new(read_u32(raw, 16)?, read_u32(raw, 20)?, read_u32(raw, 24)?),
            vertex_count: read_u32(raw, 28)? as usize,
            index_count: read_u32(raw, 32)? as usize,
            policy_add_count: read_u32(raw, 36)? as usize,
            policy_rem_count: read_u32(raw, 40)? as usize,
            mcts_add_count: read_u32(raw, 44)? as usize,
            mcts_rem_count: read_u32(raw, 48)? as usize,
            blueprint_vertex_count: read_u32(raw, 52)? as usize,
            blueprint_index_count: read_u32(raw, 56)? as usize,
            diff_added_count: read_u32(raw, 60)? as usize,
            diff_removed_count: read_u32(raw, 64)? as usize,
            disp_p99: f32::from_bits(read_u32(raw, 68)?),
            strain_p99: f32::from_bits(read_u32(raw, 72)?),
            meta_json_len: read_u32(raw, 76)? as usize,
            meta_json_padded_len: read_u32(raw, 80)? as usize,
        })
    }
}

/// Decoded `ABMESH01` fixed header.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AbMeshHeader {
    pub flags: u32,
    pub step_index: u32,
    pub dims: Dims,
    pub vertex_count: usize,
    pub index_count: usize,
    pub policy_add_count: usize,
    pub policy_rem_count: usize,
    pub mcts_add_count: usize,
    pub mcts_rem_count: usize,
    pub blueprint_vertex_count: usize,
    pub blueprint_index_count: usize,
    pub diff_added_count: usize,
    pub diff_removed_count: usize,
    pub disp_p99: f32,
    pub strain_p99: f32,
    pub meta_json_len: usize,
    pub meta_json_padded_len: usize,
}

fn read_u32(raw: &[u8], offset: usize) -> CoreResult<u32> {
    let bytes = raw.get(offset..offset + 4).ok_or_else(|| CoreError::Wire {
        message: format!("read_u32 out of bounds at {offset}"),
    })?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn write_u32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_f32(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn push_u32s(out: &mut Vec<u8>, values: &[u32]) {
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn take_f32s(raw: &[u8], cursor: &mut usize, count: usize) -> CoreResult<Vec<f32>> {
    let bytes = count.checked_mul(4).ok_or_else(|| CoreError::AbMesh {
        message: "float section length overflow".to_string(),
    })?;
    let end = cursor.checked_add(bytes).ok_or_else(|| CoreError::AbMesh {
        message: "float section length overflow".to_string(),
    })?;
    if end > raw.len() {
        return Err(CoreError::AbMesh {
            message: "truncated float section".to_string(),
        });
    }
    let mut out = Vec::with_capacity(count);
    for chunk in raw[*cursor..end].as_chunks::<4>().0 {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    *cursor = end;
    Ok(out)
}

fn take_u32s(raw: &[u8], cursor: &mut usize, count: usize) -> CoreResult<Vec<u32>> {
    let bytes = count.checked_mul(4).ok_or_else(|| CoreError::AbMesh {
        message: "u32 section length overflow".to_string(),
    })?;
    let end = cursor.checked_add(bytes).ok_or_else(|| CoreError::AbMesh {
        message: "u32 section length overflow".to_string(),
    })?;
    if end > raw.len() {
        return Err(CoreError::AbMesh {
            message: "truncated u32 section".to_string(),
        });
    }
    let mut out = Vec::with_capacity(count);
    for chunk in raw[*cursor..end].as_chunks::<4>().0 {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    *cursor = end;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_wire() -> WirePacket {
        let density = vec![0_u8, 1, 0, 1, 1, 1, 0, 0];
        let coo = Coo1 {
            indices: vec![0, 3, 7],
            values: vec![0.25, 0.5, 1.0],
        };
        let policy = WireTensor::dense_f16(
            tensor_tag::BLUEPRINT_PROBS,
            [2, 2, 2, 0],
            3,
            &[0.0, 0.5, 1.0, 0.25, 0.75, 0.125, 0.875, 0.0],
        )
        .unwrap();
        WirePacket {
            payload_type: payload_type::INFER_RESULT,
            flags: wire_flags::ZLIB_TENSOR_BLOCK,
            header_json: r#"{"value":0.5,"device":"cpu"}"#.to_string(),
            tensors: vec![
                WireTensor::dense_u8(tensor_tag::DENSITY, [2, 2, 2, 0], 3, &density).unwrap(),
                WireTensor::coo1_f32(tensor_tag::POLICY_ADD, [2, 2, 2, 0], 3, &coo).unwrap(),
                policy,
            ],
        }
    }

    #[test]
    fn abwire_round_trip_compressed_and_plain() {
        for compressed in [false, true] {
            let mut packet = sample_wire();
            packet.flags = if compressed {
                wire_flags::ZLIB_TENSOR_BLOCK
            } else {
                0
            };
            let raw = packet.encode().unwrap();
            assert_eq!(&raw[0..8], ABWIRE01_MAGIC);
            assert!(raw.len().is_multiple_of(4));
            let header_json_len = u32::from_le_bytes([raw[16], raw[17], raw[18], raw[19]]) as usize;
            let header_json_padded =
                u32::from_le_bytes([raw[20], raw[21], raw[22], raw[23]]) as usize;
            assert!(header_json_padded.is_multiple_of(4));
            assert!(header_json_padded >= header_json_len);
            let decoded = WirePacket::decode(&raw).unwrap();
            assert_eq!(decoded, packet);
            assert_eq!(
                decoded
                    .tensor(tensor_tag::DENSITY)
                    .unwrap()
                    .to_dense_f32()
                    .unwrap(),
                vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
            );
            let policy = decoded
                .tensor(tensor_tag::POLICY_ADD)
                .unwrap()
                .to_dense_f32()
                .unwrap();
            assert_eq!(policy.len(), 8);
            assert_eq!(policy[7], 1.0);
            let blueprint = decoded
                .tensor(tensor_tag::BLUEPRINT_PROBS)
                .unwrap()
                .to_dense_f32()
                .unwrap();
            assert!((blueprint[1] - 0.5).abs() < 1e-3);
        }
    }

    #[test]
    fn abwire_rejects_bad_magic_and_trailing() {
        let raw = sample_wire().encode().unwrap();
        assert!(WirePacket::decode(&raw[..31]).is_err());
        let mut bad = raw.clone();
        bad[0] = b'X';
        assert!(WirePacket::decode(&bad).is_err());
        let mut trailing = raw;
        trailing.push(0xAA);
        assert!(WirePacket::decode(&trailing).is_err());
    }

    fn sample_abmesh() -> AbMeshPacket {
        let mut packet = AbMeshPacket::empty(Dims::new(2, 2, 2), 7);
        packet.flags = abmesh_flags::POLICY
            | abmesh_flags::BLUEPRINT
            | abmesh_flags::MCTS
            | abmesh_flags::DIFF;
        packet.positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        packet.normals = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        packet.ao_and_scalars = vec![
            1.0, 1.0, 0.0, 0.0, 0.78, 1.0, 0.5, 0.25, 0.56, 1.0, 0.25, 0.5, 0.35, 1.0, 0.0, 1.0,
        ];
        packet.voxel_ids = vec![0, 0, 1, 1];
        packet.indices = vec![0, 1, 2, 0, 2, 3];
        packet.blueprint_positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        packet.blueprint_normals = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        packet.blueprint_indices = vec![0, 1, 2];
        packet.policy_add = vec![(3, 0.5), (5, 0.25)];
        packet.policy_rem = vec![(0, 0.125)];
        packet.mcts_add = vec![(3, 48.0, 0.61), (5, 12.0, -0.2)];
        packet.mcts_rem = vec![(0, 3.0, 0.1)];
        packet.diff_added = vec![3, 5];
        packet.diff_removed = vec![0];
        packet.disp_p99 = 41.2;
        packet.strain_p99 = 0.75;
        packet.meta_json =
            r#"{"step":7,"phase":"GROWTH","value":0.5,"selected_actions":[]}"#.to_string();
        packet
    }

    #[test]
    fn abmesh_round_trip_and_layout() {
        let packet = sample_abmesh();
        let raw = packet.encode().unwrap();
        assert!(raw.len().is_multiple_of(4));
        let header = AbMeshPacket::header(&raw).unwrap();
        assert_eq!(header.dims, Dims::new(2, 2, 2));
        assert_eq!(header.vertex_count, 4);
        assert_eq!(header.index_count, 6);
        assert_eq!(header.policy_add_count, 2);
        assert!(header.meta_json_padded_len.is_multiple_of(4));
        assert_eq!(header.step_index, 7);
        assert_eq!(
            u32::from_le_bytes([raw[92], raw[93], raw[94], raw[95]]) as usize,
            raw.len()
        );
        let decoded = AbMeshPacket::decode(&raw).unwrap();
        assert_eq!(decoded, packet);
    }

    #[test]
    fn abmesh_rejects_corruption() {
        let raw = sample_abmesh().encode().unwrap();
        assert!(AbMeshPacket::decode(&raw[..40]).is_err());
        let mut bad = raw.clone();
        bad[92] = bad[92].wrapping_add(1);
        assert!(AbMeshPacket::decode(&bad).is_err());
        let mut bad_meta = raw;
        let header = AbMeshPacket::header(&bad_meta).unwrap();
        let meta_start = bad_meta.len() - header.meta_json_padded_len;
        bad_meta[meta_start + header.meta_json_len - 1] = 0;
        assert!(AbMeshPacket::decode(&bad_meta).is_err());
    }
}
