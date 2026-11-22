#![forbid(unsafe_code)]
//! AlphaBuilder Studio v4.0 Rust core.
//!
//! Pure, dependency-light implementations of the v3.3 mathematics, binary
//! codecs and read-only SQLite access used by the desktop application.
//! This crate must stay free of Tauri (and of `unsafe`).

pub mod codecs;
pub mod coords;
pub mod error;
pub mod exporter;
pub mod mesher;
pub mod reward;
pub mod sqlite_reader;
pub mod topology;

pub use error::{CoreError, CoreResult};

/// Maximum number of voxels accepted by the v4 protocols.
pub const MAX_VOXELS: usize = 262_144;
/// Maximum value of any single domain dimension.
pub const MAX_DIMENSION: u32 = 256;
