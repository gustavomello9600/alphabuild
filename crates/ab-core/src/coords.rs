//! Canonical `(D, H, W)` indexing and Three.js world-space conventions (v3.3).

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreResult};
use crate::{MAX_DIMENSION, MAX_VOXELS};

/// Domain resolution `(D, H, W)` = `(nx, ny, nz)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dims {
    pub d: u32,
    pub h: u32,
    pub w: u32,
}

impl Dims {
    pub const fn new(d: u32, h: u32, w: u32) -> Self {
        Self { d, h, w }
    }

    /// Validate `1 <= D,H,W <= 256` and `D*H*W <= 262_144`.
    pub fn validate(&self) -> CoreResult<()> {
        if self.d == 0 || self.h == 0 || self.w == 0 {
            return Err(CoreError::InvalidInput {
                message: format!("invalid resolution ({}, {}, {})", self.d, self.h, self.w),
            });
        }
        if self.d > MAX_DIMENSION || self.h > MAX_DIMENSION || self.w > MAX_DIMENSION {
            return Err(CoreError::ResolutionLimitExceeded {
                message: format!(
                    "dimension exceeds {}: ({}, {}, {})",
                    MAX_DIMENSION, self.d, self.h, self.w
                ),
            });
        }
        if self.len() > MAX_VOXELS {
            return Err(CoreError::ResolutionLimitExceeded {
                message: format!("voxel count {} exceeds {}", self.len(), MAX_VOXELS),
            });
        }
        Ok(())
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.d as usize * self.h as usize * self.w as usize
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub const fn plane_hw(&self) -> usize {
        self.h as usize * self.w as usize
    }

    #[inline]
    pub const fn contains(&self, x: i64, y: i64, z: i64) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as u32) < self.d
            && (y as u32) < self.h
            && (z as u32) < self.w
    }

    /// C-order flat index, or `None` when outside the domain.
    #[inline]
    pub fn flat_idx(&self, x: u32, y: u32, z: u32) -> Option<usize> {
        if x >= self.d || y >= self.h || z >= self.w {
            return None;
        }
        Some(x as usize * self.plane_hw() + y as usize * self.w as usize + z as usize)
    }

    #[inline]
    pub fn flat_idx_checked(&self, x: u32, y: u32, z: u32) -> CoreResult<usize> {
        self.flat_idx(x, y, z)
            .ok_or_else(|| CoreError::InvalidInput {
                message: format!(
                    "voxel ({}, {}, {}) outside domain ({}, {}, {})",
                    x, y, z, self.d, self.h, self.w
                ),
            })
    }

    /// Inverse C-order mapping. `idx` must be `< self.len()`.
    #[inline]
    pub fn inverse(&self, idx: usize) -> Option<(u32, u32, u32)> {
        if idx >= self.len() {
            return None;
        }
        let plane = self.plane_hw();
        let w = self.w as usize;
        let x = idx / plane;
        let rem = idx % plane;
        let y = rem / w;
        let z = rem % w;
        Some((x as u32, y as u32, z as u32))
    }

    /// Center of voxel `(x,y,z)` in right-handed world space (+Y up).
    #[inline]
    pub fn world_center(&self, x: u32, y: u32, z: u32) -> [f32; 3] {
        [
            x as f32 - self.d as f32 / 2.0 + 0.5,
            y as f32 + 0.5,
            z as f32 - self.w as f32 / 2.0 + 0.5,
        ]
    }

    /// Axis-aligned bounding box `[min, max]` of a voxel.
    #[inline]
    pub fn voxel_aabb(&self, x: u32, y: u32, z: u32) -> ([f32; 3], [f32; 3]) {
        let x0 = x as f32 - self.d as f32 / 2.0;
        let z0 = z as f32 - self.w as f32 / 2.0;
        ([x0, y as f32, z0], [x0 + 1.0, y as f32 + 1.0, z0 + 1.0])
    }
}

/// Boundary condition families accepted by the v3.3 engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BcType {
    #[default]
    FullClamp,
    RailXy,
    RollerY,
    RollerZ,
    Custom,
}

impl BcType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BcType::FullClamp => "FULL_CLAMP",
            BcType::RailXy => "RAIL_XY",
            BcType::RollerY => "ROLLER_Y",
            BcType::RollerZ => "ROLLER_Z",
            BcType::Custom => "CUSTOM",
        }
    }
}

/// Canonical load patch descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadConfig {
    pub x: f64,
    pub y: f64,
    pub z_start: f64,
    pub z_end: f64,
    #[serde(default)]
    pub bc_type: Option<BcType>,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z_start: 0.0,
            z_end: 0.0,
            bc_type: None,
        }
    }
}

/// Canonical load region bounds `(lx, y_min, y_max, z_min, z_max)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadRegionBounds {
    pub lx: u32,
    pub y_min: u32,
    pub y_max: u32,
    pub z_min: u32,
    pub z_max: u32,
}

/// Exact port of `alphabuilder/src/logic/harvest/generators.py::get_load_region_bounds`.
pub fn get_load_region_bounds(dims: Dims, load_config: &LoadConfig) -> LoadRegionBounds {
    let nx = dims.d;
    let ny = dims.h;
    let nz = dims.w;
    let lx = (load_config.x as i64).clamp(0, nx as i64 - 1) as u32;
    let ly = load_config.y;
    let lz_center = (load_config.z_start + load_config.z_end) / 2.0;
    let half = 1.0_f64;

    let y_min = ((ly - half).floor() as i64).max(0) as u32;
    let y_max = ((ly + half).floor() as i64 + 1).min(ny as i64) as u32;
    let z_min = ((lz_center - half).floor() as i64).max(0) as u32;
    let z_max = ((lz_center + half).floor() as i64 + 1).min(nz as i64) as u32;
    LoadRegionBounds {
        lx,
        y_min,
        y_max,
        z_min,
        z_max,
    }
}

impl LoadRegionBounds {
    pub fn voxel_count(&self) -> usize {
        ((self.y_max - self.y_min) * (self.z_max - self.z_min)) as usize
    }

    /// Iterate over the flat indices of the load patch.
    pub fn for_each_flat_idx<F: FnMut(usize)>(&self, dims: Dims, mut f: F) {
        let plane = dims.plane_hw();
        for y in self.y_min..self.y_max {
            for z in self.z_min..self.z_max {
                f(self.lx as usize * plane + y as usize * dims.w as usize + z as usize);
            }
        }
    }
}

/// Generate the three BC masks exactly as the v3.3 harvest generator does.
///
/// Only `RAIL_XY` relaxes `mask_z`; every other type defaults to a full clamp,
/// matching `generate_bc_masks` in the scientific core.
pub fn generate_bc_masks(dims: Dims, bc_type: BcType) -> [Vec<f32>; 3] {
    let mut mask_x = vec![0.0_f32; dims.len()];
    let mut mask_y = vec![0.0_f32; dims.len()];
    let mut mask_z = vec![0.0_f32; dims.len()];
    let plane = dims.plane_hw();
    match bc_type {
        BcType::RailXy => {
            for y in 0..dims.h {
                for z in 0..dims.w {
                    let idx = y as usize * dims.w as usize + z as usize;
                    mask_x[idx] = 1.0;
                    mask_y[idx] = 1.0;
                }
            }
        }
        BcType::Custom => {}
        _ => {
            for y in 0..dims.h {
                for z in 0..dims.w {
                    let idx = y as usize * dims.w as usize + z as usize;
                    mask_x[idx] = 1.0;
                    mask_y[idx] = 1.0;
                    mask_z[idx] = 1.0;
                }
            }
        }
    }
    debug_assert_eq!(mask_x.len(), dims.len());
    debug_assert!(plane > 0);
    [mask_x, mask_y, mask_z]
}

/// Build the canonical force tensor `(3, D, H, W)` with `fy = -1` on the load patch.
pub fn generate_forces(dims: Dims, load_config: &LoadConfig) -> [Vec<f32>; 3] {
    let mut fx = vec![0.0_f32; dims.len()];
    let mut fy = vec![0.0_f32; dims.len()];
    let fz = vec![0.0_f32; dims.len()];
    let bounds = get_load_region_bounds(dims, load_config);
    bounds.for_each_flat_idx(dims, |idx| fy[idx] = -1.0);
    debug_assert_eq!(fx.len(), dims.len());
    fx[..0].fill(0.0);
    [fx, fy, fz]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_vectors_from_spec() {
        let dims = Dims::new(64, 32, 8);
        assert_eq!(dims.flat_idx(0, 0, 0), Some(0));
        let [x, y, z] = dims.world_center(0, 0, 0);
        assert_eq!([x, y, z], [-31.5, 0.5, -3.5]);

        let idx = dims.flat_idx(63, 16, 4).unwrap();
        assert_eq!(idx, 63 * 256 + 16 * 8 + 4);
        assert_eq!(idx, 16260);
        assert_eq!(dims.inverse(idx), Some((63, 16, 4)));
        assert_eq!(dims.world_center(63, 16, 4), [31.5, 16.5, 0.5]);
    }

    #[test]
    fn load_region_matches_v33_python() {
        let dims = Dims::new(64, 32, 8);
        let cfg = LoadConfig {
            x: 63.0,
            y: 16.0,
            z_start: 3.0,
            z_end: 5.0,
            bc_type: None,
        };
        let b = get_load_region_bounds(dims, &cfg);
        assert_eq!(
            b,
            LoadRegionBounds {
                lx: 63,
                y_min: 15,
                y_max: 18,
                z_min: 3,
                z_max: 6
            }
        );
        assert_eq!(b.voxel_count(), 9);
    }

    #[test]
    fn bc_and_forces() {
        let dims = Dims::new(8, 4, 4);
        let masks = generate_bc_masks(dims, BcType::RailXy);
        assert_eq!(masks[0][0], 1.0);
        assert_eq!(masks[1][0], 1.0);
        assert_eq!(masks[2][0], 0.0);
        let full = generate_bc_masks(dims, BcType::FullClamp);
        assert_eq!(full[2][0], 1.0);

        let cfg = LoadConfig {
            x: 7.0,
            y: 2.0,
            z_start: 1.0,
            z_end: 3.0,
            bc_type: None,
        };
        let forces = generate_forces(dims, &cfg);
        let b = get_load_region_bounds(dims, &cfg);
        let mut count = 0;
        b.for_each_flat_idx(dims, |idx| {
            assert_eq!(forces[1][idx], -1.0);
            count += 1;
        });
        assert_eq!(count, b.voxel_count());
    }

    #[test]
    fn resolution_bounds_are_enforced() {
        assert!(Dims::new(0, 1, 1).validate().is_err());
        assert!(Dims::new(257, 1, 1).validate().is_err());
        assert!(Dims::new(256, 256, 256).validate().is_err());
        assert!(Dims::new(64, 32, 8).validate().is_ok());
    }
}
