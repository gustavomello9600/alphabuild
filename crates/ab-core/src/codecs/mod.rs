//! Binary codecs: NumPy `.npy`, sparse `COO1`, and the two wire protocols.

pub mod coo1;
pub mod npy;
pub mod wire;

use std::io::{Read, Write};

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;

use crate::error::{CoreError, CoreResult};

/// zlib level used by the scientific core for SQLite blobs and ABWIRE01.
pub const ZLIB_LEVEL: u32 = 6;

/// Compress bytes with zlib (level 6).
pub fn zlib_compress(bytes: &[u8]) -> CoreResult<Vec<u8>> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(ZLIB_LEVEL));
    encoder.write_all(bytes)?;
    Ok(encoder.finish()?)
}

/// Decompress a zlib stream with a hard output limit.
pub fn zlib_decompress(bytes: &[u8], max_output: usize) -> CoreResult<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(bytes);
    let mut out = Vec::new();
    let mut buf = [0_u8; 16 * 1024];
    loop {
        let n = decoder.read(&mut buf)?;
        if n == 0 {
            break;
        }
        if out.len() + n > max_output {
            return Err(CoreError::InvalidInput {
                message: format!("decompressed payload exceeds {max_output} bytes"),
            });
        }
        out.extend_from_slice(&buf[..n]);
    }
    Ok(out)
}
