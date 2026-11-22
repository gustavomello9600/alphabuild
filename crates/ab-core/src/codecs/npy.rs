//! Safe parser/encoder for NumPy `.npy` v1/v2/v3 files (no pickle).

use half::f16;

use crate::error::{CoreError, CoreResult};

const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Supported dtype set (converted to `f32` when decoding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDtype {
    U8,
    F16,
    F32,
    F64,
    I32,
}

impl NpyDtype {
    pub fn descr(&self) -> &'static str {
        match self {
            NpyDtype::U8 => "|u1",
            NpyDtype::F16 => "<f2",
            NpyDtype::F32 => "<f4",
            NpyDtype::F64 => "<f8",
            NpyDtype::I32 => "<i4",
        }
    }

    pub fn itemsize(&self) -> usize {
        match self {
            NpyDtype::U8 => 1,
            NpyDtype::F16 => 2,
            NpyDtype::F32 | NpyDtype::I32 => 4,
            NpyDtype::F64 => 8,
        }
    }
}

/// Decoded array: C-order `f32` data plus its logical shape.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl DecodedArray {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn validate_len(&self, expected: usize) -> CoreResult<()> {
        if self.data.len() != expected {
            return Err(CoreError::Npy {
                message: format!(
                    "decoded {} values but expected {} for declared shape",
                    self.data.len(),
                    expected
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum HeaderValue {
    Str(String),
    Bool(bool),
    Shape(Vec<usize>),
}

/// Parse an `.npy` payload into C-order `f32` values.
pub fn parse_npy(raw: &[u8]) -> CoreResult<DecodedArray> {
    if raw.len() < 10 || &raw[0..6] != MAGIC {
        return Err(CoreError::Npy {
            message: "missing \\x93NUMPY magic".to_string(),
        });
    }
    let major = raw[6];
    let _minor = raw[7];
    let (header_len, header_start) = match major {
        1 => {
            if raw.len() < 10 {
                return Err(CoreError::Npy {
                    message: "truncated v1 header".to_string(),
                });
            }
            (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10_usize)
        }
        2 | 3 => {
            if raw.len() < 12 {
                return Err(CoreError::Npy {
                    message: "truncated v2/v3 header".to_string(),
                });
            }
            (
                u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize,
                12_usize,
            )
        }
        other => {
            return Err(CoreError::Npy {
                message: format!("unsupported .npy version {other}.{_minor}"),
            })
        }
    };
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| CoreError::Npy {
            message: "header length overflow".to_string(),
        })?;
    if header_end > raw.len() {
        return Err(CoreError::Npy {
            message: "header length exceeds payload".to_string(),
        });
    }
    let header_text =
        std::str::from_utf8(&raw[header_start..header_end]).map_err(|e| CoreError::Npy {
            message: format!("header is not UTF-8 ASCII: {e}"),
        })?;

    let entries = parse_header_dict(header_text)?;
    let mut descr: Option<String> = None;
    let mut fortran: Option<bool> = None;
    let mut shape: Option<Vec<usize>> = None;
    for (key, value) in entries {
        match (key.as_str(), value) {
            ("descr", HeaderValue::Str(s)) => descr = Some(s),
            ("fortran_order", HeaderValue::Bool(b)) => fortran = Some(b),
            ("shape", HeaderValue::Shape(s)) => shape = Some(s),
            _ => {}
        }
    }

    if fortran == Some(true) {
        return Err(CoreError::FortranOrder);
    }
    let descr = descr.ok_or_else(|| CoreError::Npy {
        message: "missing 'descr' in header".to_string(),
    })?;
    let shape = shape.ok_or_else(|| CoreError::Npy {
        message: "missing 'shape' in header".to_string(),
    })?;

    let dtype = parse_dtype(&descr)?;
    let expected = shape.iter().try_fold(1_usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| CoreError::Npy {
            message: "shape product overflow".to_string(),
        })
    })?;
    let byte_len = expected
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| CoreError::Npy {
            message: "byte length overflow".to_string(),
        })?;
    if raw.len() < header_end + byte_len {
        return Err(CoreError::Npy {
            message: format!(
                "payload needs {} bytes at offset {}, buffer has {}",
                byte_len,
                header_end,
                raw.len()
            ),
        });
    }
    let payload = &raw[header_end..header_end + byte_len];

    let mut data = Vec::with_capacity(expected);
    match dtype {
        NpyDtype::U8 => data.extend(payload.iter().map(|&v| v as f32)),
        NpyDtype::F16 => {
            for chunk in payload.as_chunks::<2>().0 {
                data.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
        }
        NpyDtype::F32 => {
            for chunk in payload.as_chunks::<4>().0 {
                data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        NpyDtype::F64 => {
            for chunk in payload.as_chunks::<8>().0 {
                data.push(f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32);
            }
        }
        NpyDtype::I32 => {
            for chunk in payload.as_chunks::<4>().0 {
                data.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32);
            }
        }
    }
    Ok(DecodedArray { shape, data })
}

fn parse_dtype(descr: &str) -> CoreResult<NpyDtype> {
    let trimmed = descr.trim();
    if trimmed.starts_with('>') {
        return Err(CoreError::UnsupportedDtype {
            dtype: descr.to_string(),
        });
    }
    let normalized = trimmed.trim_start_matches(['<', '|', '=']);
    match normalized {
        "u1" | "b1" => Ok(NpyDtype::U8),
        "f2" => Ok(NpyDtype::F16),
        "f4" => Ok(NpyDtype::F32),
        "f8" => Ok(NpyDtype::F64),
        "i4" => Ok(NpyDtype::I32),
        other => Err(CoreError::UnsupportedDtype {
            dtype: other.to_string(),
        }),
    }
}

/// Encode `f32` values into a v1 `.npy` buffer with 64-byte data alignment.
pub fn encode_npy(data: &[f32], shape: &[usize], dtype: NpyDtype) -> CoreResult<Vec<u8>> {
    let expected = shape.iter().try_fold(1_usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| CoreError::Npy {
            message: "shape product overflow".to_string(),
        })
    })?;
    if expected != data.len() {
        return Err(CoreError::Npy {
            message: format!(
                "shape {:?} requires {} values, got {}",
                shape,
                expected,
                data.len()
            ),
        });
    }

    let shape_repr = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", dims.join(", "))
    };
    let base = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype.descr(),
        shape_repr
    );

    // v1: magic(6) + version(2) + u16 header_len(2) + header.
    const PREFIX: usize = 10;
    let mut header_len = base.len() + 1; // trailing newline
    while !(PREFIX + header_len).is_multiple_of(64) {
        header_len += 1;
    }
    if header_len > u16::MAX as usize {
        return Err(CoreError::Npy {
            message: "header too large for v1".to_string(),
        });
    }
    let mut out = Vec::with_capacity(PREFIX + header_len + expected * dtype.itemsize());
    out.extend_from_slice(MAGIC);
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(base.as_bytes());
    out.resize(PREFIX + header_len - 1, b' ');
    out.push(b'\n');

    match dtype {
        NpyDtype::U8 => out.extend(data.iter().map(|v| v.round().clamp(0.0, 255.0) as u8)),
        NpyDtype::F16 => {
            for v in data {
                out.extend_from_slice(&f16::from_f32(*v).to_le_bytes());
            }
        }
        NpyDtype::F32 => {
            for v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        NpyDtype::F64 => {
            for v in data {
                out.extend_from_slice(&(*v as f64).to_le_bytes());
            }
        }
        NpyDtype::I32 => {
            for v in data {
                out.extend_from_slice(&(v.round() as i32).to_le_bytes());
            }
        }
    }
    debug_assert_eq!(out.len(), PREFIX + header_len + expected * dtype.itemsize());
    Ok(out)
}

fn parse_header_dict(text: &str) -> CoreResult<Vec<(String, HeaderValue)>> {
    let mut parser = LiteralParser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    parser.skip_ws();
    parser.expect(b'{')?;
    let mut entries = Vec::new();
    loop {
        parser.skip_ws();
        if parser.peek() == Some(b'}') {
            break;
        }
        let key = parser.parse_string()?;
        parser.skip_ws();
        parser.expect(b':')?;
        parser.skip_ws();
        let value = if parser.peek() == Some(b'\'') || parser.peek() == Some(b'"') {
            HeaderValue::Str(parser.parse_string()?)
        } else if parser.peek() == Some(b'(') {
            HeaderValue::Shape(parser.parse_shape()?)
        } else {
            let ident = parser.parse_ident()?;
            match ident.as_str() {
                "True" => HeaderValue::Bool(true),
                "False" => HeaderValue::Bool(false),
                other => {
                    return Err(CoreError::Npy {
                        message: format!("unsupported header literal '{other}'"),
                    })
                }
            }
        };
        entries.push((key, value));
        parser.skip_ws();
        match parser.peek() {
            Some(b',') => {
                parser.pos += 1;
            }
            Some(b'}') => {
                break;
            }
            other => {
                return Err(CoreError::Npy {
                    message: format!("expected ',' or '}}' in header, got {other:?}"),
                })
            }
        }
    }
    Ok(entries)
}

struct LiteralParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> LiteralParser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(
            self.peek(),
            Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n')
        ) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> CoreResult<()> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(CoreError::Npy {
                message: format!("expected '{}' at header byte {}", byte as char, self.pos),
            })
        }
    }

    fn parse_string(&mut self) -> CoreResult<String> {
        let quote = self.peek().ok_or_else(|| CoreError::Npy {
            message: "unexpected end of header".to_string(),
        })?;
        if quote != b'\'' && quote != b'"' {
            return Err(CoreError::Npy {
                message: format!("expected string quote at header byte {}", self.pos),
            });
        }
        self.pos += 1;
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b == quote {
                let s = std::str::from_utf8(&self.bytes[start..self.pos])
                    .map_err(|e| CoreError::Npy {
                        message: format!("invalid UTF-8 in header string: {e}"),
                    })?
                    .to_string();
                self.pos += 1;
                return Ok(s);
            }
            if b == b'\\' {
                return Err(CoreError::Npy {
                    message: "escape sequences are not supported in .npy headers".to_string(),
                });
            }
            self.pos += 1;
        }
        Err(CoreError::Npy {
            message: "unterminated header string".to_string(),
        })
    }

    fn parse_shape(&mut self) -> CoreResult<Vec<usize>> {
        self.expect(b'(')?;
        let mut dims = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(b')') {
                self.pos += 1;
                break;
            }
            let start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if start == self.pos {
                return Err(CoreError::Npy {
                    message: format!("expected integer in shape at header byte {}", start),
                });
            }
            let value: usize = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|e| CoreError::Npy {
                    message: format!("invalid shape: {e}"),
                })?
                .parse()
                .map_err(|e| CoreError::Npy {
                    message: format!("invalid shape integer: {e}"),
                })?;
            dims.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b')') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    return Err(CoreError::Npy {
                        message: "expected ',' or ')' in shape".to_string(),
                    })
                }
            }
        }
        Ok(dims)
    }

    fn parse_ident(&mut self) -> CoreResult<String> {
        let start = self.pos;
        while matches!(self.peek(), Some(b) if b.is_ascii_alphanumeric() || b == b'_') {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(CoreError::Npy {
                message: format!("expected identifier at header byte {}", start),
            });
        }
        Ok(std::str::from_utf8(&self.bytes[start..self.pos])
            .unwrap_or_default()
            .to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(dtype: NpyDtype, values: &[f32], tolerance: f32) {
        let shape = [2, 2, 2];
        let raw = encode_npy(values, &shape, dtype).unwrap();
        assert_eq!(&raw[0..6], MAGIC);
        let header_len = u16::from_le_bytes([raw[8], raw[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0, "data must be 64-byte aligned");
        let decoded = parse_npy(&raw).unwrap();
        assert_eq!(decoded.shape, shape.to_vec());
        assert_eq!(decoded.len(), values.len());
        for (a, b) in decoded.data.iter().zip(values.iter()) {
            assert!(
                (a - b).abs() <= tolerance,
                "dtype {dtype:?}: {a} != {b} (tol {tolerance})"
            );
        }
        decoded.validate_len(8).unwrap();
    }

    #[test]
    fn npy_round_trip_all_supported_dtypes() {
        let values = [0.0, 1.0, 0.5, 0.25, 3.0, 42.0, 7.0, 255.0];
        round_trip(NpyDtype::F32, &values, 0.0);
        let u8_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 255.0];
        round_trip(NpyDtype::U8, &u8_values, 0.0);
        round_trip(NpyDtype::F64, &values, 0.0);
        let f16_values = [0.0, 1.0, 0.5, 0.25, 3.0, 2.0, 7.0, 8.0];
        round_trip(NpyDtype::F16, &f16_values, 1e-3);
        let i32_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        round_trip(NpyDtype::I32, &i32_values, 0.0);
    }

    #[test]
    fn npy_rejects_fortran_order() {
        let base = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 2), }";
        let mut header_len = base.len() + 1;
        while !(10 + header_len).is_multiple_of(64) {
            header_len += 1;
        }
        let mut raw = Vec::new();
        raw.extend_from_slice(MAGIC);
        raw.extend_from_slice(&[1, 0]);
        raw.extend_from_slice(&(header_len as u16).to_le_bytes());
        raw.extend_from_slice(base.as_bytes());
        raw.resize(10 + header_len - 1, b' ');
        raw.push(b'\n');
        raw.extend_from_slice(&[0_u8; 16]);
        assert!(matches!(parse_npy(&raw), Err(CoreError::FortranOrder)));
    }

    #[test]
    fn npy_rejects_corrupt_and_big_endian() {
        let bad = b"not a npy file".to_vec();
        assert!(parse_npy(&bad).is_err());
        let values = [1.0_f32; 4];
        let raw = encode_npy(&values, &[2, 2], NpyDtype::F32).unwrap();
        let mut corrupt = raw.clone();
        let n = corrupt.len();
        corrupt.truncate(n - 3);
        assert!(parse_npy(&corrupt).is_err());

        let mut bad_endian = raw;
        let text = String::from_utf8_lossy(&bad_endian[10..]).to_string();
        if let Some(pos) = text.find("<f4") {
            bad_endian[10 + pos] = b'>';
            assert!(matches!(
                parse_npy(&bad_endian),
                Err(CoreError::UnsupportedDtype { .. })
            ));
        }
    }

    #[test]
    fn npy_v2_header_supported() {
        let values = [1.5_f32; 4];
        let v1 = encode_npy(&values, &[2, 2], NpyDtype::F32).unwrap();
        let header_len_v1 = u16::from_le_bytes([v1[8], v1[9]]) as usize;
        let header = &v1[10..10 + header_len_v1];
        let mut v2 = Vec::new();
        v2.extend_from_slice(MAGIC);
        v2.extend_from_slice(&[2, 0]);
        v2.extend_from_slice(&(header_len_v1 as u32).to_le_bytes());
        v2.extend_from_slice(header);
        v2.extend_from_slice(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let decoded = parse_npy(&v2).unwrap();
        assert_eq!(decoded.data, values.to_vec());
    }
}
