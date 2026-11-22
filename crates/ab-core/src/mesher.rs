//! Surface-culled voxel mesher with 4-corner ambient occlusion.
//!
//! The six face tables, AO curve and anti-anisotropy triangulation are
//! normative (spec Section 6.3).

use crate::coords::Dims;
use crate::error::{CoreError, CoreResult};

pub const DEFAULT_THRESHOLD: f32 = 0.5;
pub const AO_LUT: [f32; 4] = [1.00, 0.78, 0.56, 0.35];

#[derive(Debug, Clone, Copy)]
struct Face {
    normal: [i32; 3],
    corners: [[i32; 3]; 4],
    u: [i32; 3],
    v: [i32; 3],
}

/// Exact face table from the specification.
const FACES: [Face; 6] = [
    Face {
        normal: [1, 0, 0],
        corners: [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        u: [0, 0, -1],
        v: [0, 1, 0],
    },
    Face {
        normal: [-1, 0, 0],
        corners: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        u: [0, 0, 1],
        v: [0, 1, 0],
    },
    Face {
        normal: [0, 1, 0],
        corners: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],
        u: [1, 0, 0],
        v: [0, 0, -1],
    },
    Face {
        normal: [0, -1, 0],
        corners: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        u: [1, 0, 0],
        v: [0, 0, 1],
    },
    Face {
        normal: [0, 0, 1],
        corners: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        u: [1, 0, 0],
        v: [0, 1, 0],
    },
    Face {
        normal: [0, 0, -1],
        corners: [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        u: [-1, 0, 0],
        v: [0, 1, 0],
    },
];

/// Signs of the four AO corners in the tangent plane, matching vertex order.
const AO_SIGNS: [(i32, i32); 4] = [(-1, -1), (1, -1), (1, 1), (-1, 1)];

/// Surface-culled mesh, ready for `ABMESH01` and the exporters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    /// `[ao, density, disp_norm, strain_norm]` per vertex.
    pub ao_and_scalars: Vec<f32>,
    pub voxel_ids: Vec<u32>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn face_count(&self) -> usize {
        self.vertex_count() / 4
    }
}

#[inline]
fn is_solid(density: &[f32], dims: Dims, threshold: f32, x: i64, y: i64, z: i64) -> bool {
    if x < 0 || y < 0 || z < 0 {
        return false;
    }
    let (x, y, z) = (x as u32, y as u32, z as u32);
    match dims.flat_idx(x, y, z) {
        Some(idx) => density[idx] > threshold,
        None => false,
    }
}

/// Build the mesh with optional displacement/strain normalizations for the shaders.
pub fn build_mesh(
    density: &[f32],
    dims: Dims,
    threshold: f32,
    disp_norm: Option<&[f32]>,
    strain_norm: Option<&[f32]>,
) -> CoreResult<Mesh> {
    if density.len() != dims.len() {
        return Err(CoreError::InvalidInput {
            message: format!(
                "density has {} values, expected {} for {:?}",
                density.len(),
                dims.len(),
                dims
            ),
        });
    }
    if let Some(disp) = disp_norm {
        if disp.len() != dims.len() {
            return Err(CoreError::InvalidInput {
                message: "disp_norm length mismatch".to_string(),
            });
        }
    }
    if let Some(strain) = strain_norm {
        if strain.len() != dims.len() {
            return Err(CoreError::InvalidInput {
                message: "strain_norm length mismatch".to_string(),
            });
        }
    }

    let mut mesh = Mesh::default();
    let half_d = dims.d as f32 / 2.0;
    let half_w = dims.w as f32 / 2.0;

    for x in 0..dims.d {
        for y in 0..dims.h {
            for z in 0..dims.w {
                let flat = dims.flat_idx(x, y, z).expect("in-bounds loop");
                let density_val = density[flat];
                if density_val <= threshold {
                    continue;
                }
                let x0 = x as f32 - half_d;
                let y0 = y as f32;
                let z0 = z as f32 - half_w;
                let disp_val = disp_norm.map(|v| v[flat]).unwrap_or(0.0);
                let strain_val = strain_norm.map(|v| v[flat]).unwrap_or(0.0);

                for face in FACES.iter() {
                    if is_solid(
                        density,
                        dims,
                        threshold,
                        x as i64 + face.normal[0] as i64,
                        y as i64 + face.normal[1] as i64,
                        z as i64 + face.normal[2] as i64,
                    ) {
                        continue;
                    }

                    let ax = x as i64 + face.normal[0] as i64;
                    let ay = y as i64 + face.normal[1] as i64;
                    let az = z as i64 + face.normal[2] as i64;
                    let mut ao = [1.0_f32; 4];
                    for (c, (su, sv)) in AO_SIGNS.iter().enumerate() {
                        let side1 = is_solid(
                            density,
                            dims,
                            threshold,
                            ax + (*su * face.u[0]) as i64,
                            ay + (*su * face.u[1]) as i64,
                            az + (*su * face.u[2]) as i64,
                        ) as u8;
                        let side2 = is_solid(
                            density,
                            dims,
                            threshold,
                            ax + (*sv * face.v[0]) as i64,
                            ay + (*sv * face.v[1]) as i64,
                            az + (*sv * face.v[2]) as i64,
                        ) as u8;
                        let corner = is_solid(
                            density,
                            dims,
                            threshold,
                            ax + (*su * face.u[0] + *sv * face.v[0]) as i64,
                            ay + (*su * face.u[1] + *sv * face.v[1]) as i64,
                            az + (*su * face.u[2] + *sv * face.v[2]) as i64,
                        ) as u8;
                        let occ = if side1 == 1 && side2 == 1 {
                            3
                        } else {
                            side1 + side2 + corner
                        };
                        ao[c] = AO_LUT[occ as usize];
                    }

                    let base = (mesh.positions.len() / 3) as u32;
                    for (c, corner) in face.corners.iter().enumerate() {
                        mesh.positions.push(x0 + corner[0] as f32);
                        mesh.positions.push(y0 + corner[1] as f32);
                        mesh.positions.push(z0 + corner[2] as f32);
                        mesh.normals.push(face.normal[0] as f32);
                        mesh.normals.push(face.normal[1] as f32);
                        mesh.normals.push(face.normal[2] as f32);
                        mesh.ao_and_scalars.push(ao[c]);
                        mesh.ao_and_scalars.push(density_val);
                        mesh.ao_and_scalars.push(disp_val);
                        mesh.ao_and_scalars.push(strain_val);
                        mesh.voxel_ids.push(flat as u32);
                    }

                    // Anti-anisotropy diagonal choice.
                    if ao[0] + ao[2] > ao[1] + ao[3] {
                        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
                        mesh.indices.extend_from_slice(&[base, base + 2, base + 3]);
                    } else {
                        mesh.indices
                            .extend_from_slice(&[base + 1, base + 2, base + 3]);
                        mesh.indices.extend_from_slice(&[base + 1, base + 3, base]);
                    }
                }
            }
        }
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_dims(dims: Dims) -> Vec<f32> {
        vec![1.0_f32; dims.len()]
    }

    #[test]
    fn unit_cube_emits_24_vertices_and_36_indices() {
        let dims = Dims::new(1, 1, 1);
        let mesh = build_mesh(&solid_dims(dims), dims, DEFAULT_THRESHOLD, None, None).unwrap();
        assert_eq!(mesh.vertex_count(), 24);
        assert_eq!(mesh.indices.len(), 36);
        assert_eq!(mesh.face_count(), 6);
        assert!(mesh
            .ao_and_scalars
            .as_chunks::<4>()
            .0
            .iter()
            .all(|a| a[0] == 1.0));
        assert!(mesh.voxel_ids.iter().all(|&id| id == 0));
    }

    #[test]
    fn solid_2x2x2_culls_internal_faces() {
        let dims = Dims::new(2, 2, 2);
        let mesh = build_mesh(&solid_dims(dims), dims, DEFAULT_THRESHOLD, None, None).unwrap();
        assert_eq!(mesh.face_count(), 24);
        assert_eq!(mesh.vertex_count(), 96);
        assert_eq!(mesh.indices.len(), 144);
        assert_eq!(mesh.triangle_count(), 48);
        assert!(
            mesh.ao_and_scalars
                .as_chunks::<4>()
                .0
                .iter()
                .all(|a| a[0] == 1.0),
            "external corners of a 2x2x2 block must have AO=1"
        );

        // Every emitted triangle must be CCW with respect to its face normal.
        for tri in mesh.indices.as_chunks::<3>().0 {
            let p = |i: u32| {
                let o = i as usize * 3;
                [
                    mesh.positions[o],
                    mesh.positions[o + 1],
                    mesh.positions[o + 2],
                ]
            };
            let a = p(tri[0]);
            let b = p(tri[1]);
            let c = p(tri[2]);
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let o = tri[0] as usize * 3;
            let n = [mesh.normals[o], mesh.normals[o + 1], mesh.normals[o + 2]];
            let dot = cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2];
            assert!(dot > 0.0, "triangle winding is inverted (dot={dot})");
        }
    }

    #[test]
    fn concave_l_corner_has_occlusion() {
        let dims = Dims::new(2, 2, 2);
        let mut density = solid_dims(dims);
        density[dims.flat_idx(0, 0, 0).unwrap()] = 0.0;
        let mesh = build_mesh(&density, dims, DEFAULT_THRESHOLD, None, None).unwrap();
        let min_ao = mesh
            .ao_and_scalars
            .as_chunks::<4>()
            .0
            .iter()
            .map(|a| a[0])
            .fold(1.0_f32, f32::min);
        assert!(
            min_ao < 1.0,
            "concave corner must be occluded, min AO={min_ao}"
        );
        assert!(min_ao >= 0.35);
    }

    #[test]
    fn attributes_propagate_to_every_vertex() {
        let dims = Dims::new(1, 1, 1);
        let density = solid_dims(dims);
        let disp = vec![0.5_f32];
        let strain = vec![0.25_f32];
        let mesh = build_mesh(
            &density,
            dims,
            DEFAULT_THRESHOLD,
            Some(&disp),
            Some(&strain),
        )
        .unwrap();
        for a in mesh.ao_and_scalars.as_chunks::<4>().0 {
            assert_eq!(a[1], 1.0);
            assert_eq!(a[2], 0.5);
            assert_eq!(a[3], 0.25);
        }
    }

    #[test]
    fn empty_domain_is_empty_mesh() {
        let dims = Dims::new(4, 4, 4);
        let mesh = build_mesh(&vec![0.0; dims.len()], dims, 0.5, None, None).unwrap();
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.indices.len(), 0);
    }

    #[test]
    fn nonstandard_resolution_is_supported() {
        let dims = Dims::new(3, 5, 7);
        let mesh = build_mesh(&solid_dims(dims), dims, 0.5, None, None).unwrap();
        let expected_faces = 2 * (3 * 5 + 3 * 7 + 5 * 7);
        assert_eq!(mesh.face_count(), expected_faces);
    }
}
