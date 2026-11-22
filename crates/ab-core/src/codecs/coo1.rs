//! `COO1` sparse tensor codec (decompressed payload only).

use crate::error::{CoreError, CoreResult};

pub const COO1_MAGIC: &[u8; 4] = b"COO1";

/// Sparse coordinate list with `f32` values.
#[derive(Debug, Clone, PartialEq)]
pub struct Coo1 {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl Coo1 {
    /// Parse a raw `COO1` payload. `total_voxels` bounds every index.
    pub fn parse(raw: &[u8], total_voxels: usize) -> CoreResult<Self> {
        if raw.len() < 8 || &raw[0..4] != COO1_MAGIC {
            return Err(CoreError::Coo1 {
                message: "missing COO1 magic".to_string(),
            });
        }
        let n = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
        let expected = 8_usize
            .checked_add(n.checked_mul(8).ok_or_else(|| CoreError::Coo1 {
                message: "COO1 length overflow".to_string(),
            })?)
            .ok_or_else(|| CoreError::Coo1 {
                message: "COO1 length overflow".to_string(),
            })?;
        if raw.len() < expected {
            return Err(CoreError::Coo1 {
                message: format!(
                    "COO1 declares {n} entries but payload has {} bytes",
                    raw.len()
                ),
            });
        }
        let idx_end = 8 + n * 4;
        let mut indices = Vec::with_capacity(n);
        for k in 0..n {
            let off = 8 + k * 4;
            let idx = i32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            if idx < 0 || idx as usize >= total_voxels {
                return Err(CoreError::Coo1 {
                    message: format!("index {idx} outside [0, {total_voxels})"),
                });
            }
            indices.push(idx as u32);
        }
        let mut values = Vec::with_capacity(n);
        for k in 0..n {
            let off = idx_end + k * 4;
            values.push(f32::from_le_bytes([
                raw[off],
                raw[off + 1],
                raw[off + 2],
                raw[off + 3],
            ]));
        }
        Ok(Self { indices, values })
    }

    /// Encode a sparse payload (`u32(n) + i32[n] + f32[n]`), without zlib.
    pub fn encode(&self, total_voxels: usize) -> CoreResult<Vec<u8>> {
        if self.indices.len() != self.values.len() {
            return Err(CoreError::Coo1 {
                message: format!(
                    "indices ({}) and values ({}) length mismatch",
                    self.indices.len(),
                    self.values.len()
                ),
            });
        }
        let mut out = Vec::with_capacity(8 + self.indices.len() * 8);
        out.extend_from_slice(COO1_MAGIC);
        out.extend_from_slice(&(self.indices.len() as u32).to_le_bytes());
        for &idx in &self.indices {
            if idx as usize >= total_voxels {
                return Err(CoreError::Coo1 {
                    message: format!("index {idx} outside [0, {total_voxels})"),
                });
            }
            out.extend_from_slice(&(idx as i32).to_le_bytes());
        }
        for &value in &self.values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(out)
    }

    /// Expand into a dense `f32` array of length `total_voxels`.
    pub fn to_dense(&self, total_voxels: usize) -> CoreResult<Vec<f32>> {
        let mut dense = vec![0.0_f32; total_voxels];
        for (&idx, &value) in self.indices.iter().zip(self.values.iter()) {
            let i = idx as usize;
            if i >= total_voxels {
                return Err(CoreError::Coo1 {
                    message: format!("index {idx} outside [0, {total_voxels})"),
                });
            }
            dense[i] = value;
        }
        Ok(dense)
    }

    /// Compress a dense grid into `COO1` (|value| > 1e-7, mirroring v3.3).
    pub fn from_dense(dense: &[f32]) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in dense.iter().enumerate() {
            if v.abs() > 1e-7 {
                indices.push(i as u32);
                values.push(v);
            }
        }
        Self { indices, values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coo1_round_trip() {
        let coo = Coo1 {
            indices: vec![0, 5, 42, 16_383],
            values: vec![1.0, -0.5, 0.25, 0.125],
        };
        let raw = coo.encode(64 * 32 * 8).unwrap();
        assert_eq!(&raw[0..4], COO1_MAGIC);
        assert_eq!(raw.len(), 8 + 4 * 8);
        let parsed = Coo1::parse(&raw, 64 * 32 * 8).unwrap();
        assert_eq!(parsed, coo);

        let dense = parsed.to_dense(64 * 32 * 8).unwrap();
        assert_eq!(dense[42], 0.25);
        assert_eq!(dense[1], 0.0);
        let again = Coo1::from_dense(&dense);
        assert_eq!(again.indices, coo.indices);
    }

    #[test]
    fn coo1_rejects_invalid() {
        assert!(Coo1::parse(b"XXXX", 10).is_err());
        let mut raw = Coo1 {
            indices: vec![1],
            values: vec![1.0],
        }
        .encode(10)
        .unwrap();
        raw[4] = 9;
        raw[5] = 0;
        raw[6] = 0;
        raw[7] = 0;
        assert!(Coo1::parse(&raw, 10).is_err());

        let out_of_range = Coo1 {
            indices: vec![10],
            values: vec![1.0],
        }
        .encode(10);
        assert!(out_of_range.is_err());

        let negative = [
            b'C', b'O', b'O', b'1', 1, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0,
        ];
        assert!(Coo1::parse(&negative, 10).is_err());
    }
}
