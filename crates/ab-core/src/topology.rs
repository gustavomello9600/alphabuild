//! Voxel topology, legal actions and lightweight strain analysis.
//!
//! Mirrors `alphabuilder/src/logic/mcts/legal_moves.py` and
//! `alphabuilder/src/logic/selfplay/reward.py` (v3.3).

use crate::coords::{get_load_region_bounds, Dims, LoadConfig};
use crate::error::{CoreError, CoreResult};
use crate::reward::calculate_island_penalty;

/// Three co-registered `(D,H,W)` volumes (BC masks or force components).
pub type Tensor3<'a> = [&'a [f32]; 3];

const NEIGHBORS_6: [(i64, i64, i64); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

fn check_len(name: &str, data: &[f32], dims: Dims) -> CoreResult<()> {
    if data.len() != dims.len() {
        return Err(CoreError::InvalidInput {
            message: format!(
                "{name} has {} values, expected {} for {:?}",
                data.len(),
                dims.len(),
                dims
            ),
        });
    }
    Ok(())
}

/// Result of the exact v3.3 island analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct IslandAnalysis {
    pub n_islands: usize,
    pub main_island_label: u32,
    pub main_island_mask: Vec<bool>,
    pub main_island_voxels: usize,
    pub loose_voxels: usize,
    pub is_connected: bool,
    pub total_voxels: usize,
    pub island_penalty: f64,
}

/// 6-connected component labeling (C-order), labels `1..=n`.
pub fn label_islands(density: &[f32], dims: Dims, threshold: f32) -> (Vec<u32>, usize) {
    let mut labels = vec![0_u32; density.len()];
    let mut queue: Vec<usize> = Vec::new();
    let mut n = 0_u32;
    let plane = dims.plane_hw();
    let stride_y = dims.w as usize;
    let d = dims.d as i64;
    let h = dims.h as i64;
    let w = dims.w as i64;

    for seed in 0..density.len() {
        if density[seed] <= threshold || labels[seed] != 0 {
            continue;
        }
        n += 1;
        labels[seed] = n;
        queue.clear();
        queue.push(seed);
        let mut head = 0;
        while head < queue.len() {
            let idx = queue[head];
            head += 1;
            let x = (idx / plane) as i64;
            let rem = idx % plane;
            let y = (rem / stride_y) as i64;
            let z = (rem % stride_y) as i64;
            for (dx, dy, dz) in NEIGHBORS_6 {
                let nx = x + dx;
                let ny = y + dy;
                let nz = z + dz;
                if nx < 0 || ny < 0 || nz < 0 || nx >= d || ny >= h || nz >= w {
                    continue;
                }
                let n_idx = nx as usize * plane + ny as usize * stride_y + nz as usize;
                if density[n_idx] > threshold && labels[n_idx] == 0 {
                    labels[n_idx] = n;
                    queue.push(n_idx);
                }
            }
        }
    }
    (labels, n as usize)
}

fn support_mask(bc_masks: Option<Tensor3<'_>>, dims: Dims) -> Vec<bool> {
    let mut support = vec![false; dims.len()];
    if let Some(masks) = bc_masks {
        let mut any = false;
        for idx in 0..dims.len() {
            let active = masks[0][idx] > 0.5 || masks[1][idx] > 0.5 || masks[2][idx] > 0.5;
            support[idx] = active;
            any |= active;
        }
        if any {
            return support;
        }
    }
    // Fallback: the x == 0 wall.
    let plane = dims.plane_hw();
    for value in support.iter_mut().take(plane) {
        *value = true;
    }
    support
}

/// Exact port of `analyze_structure_islands` from the v3.3 reward module.
///
/// `forces` is accepted for API parity with the normative signature; the
/// Python analysis uses masks only.
pub fn analyze_islands(
    density: &[f32],
    dims: Dims,
    bc_masks: Option<Tensor3<'_>>,
    _forces: Option<Tensor3<'_>>,
    load_config: &LoadConfig,
    threshold: f32,
) -> CoreResult<IslandAnalysis> {
    check_len("density", density, dims)?;
    if let Some(masks) = bc_masks {
        for m in masks {
            check_len("bc_mask", m, dims)?;
        }
    }
    let (labels, n_islands) = label_islands(density, dims, threshold);
    if n_islands == 0 {
        return Ok(IslandAnalysis {
            n_islands: 0,
            main_island_label: 0,
            main_island_mask: vec![false; dims.len()],
            main_island_voxels: 0,
            loose_voxels: 0,
            is_connected: false,
            total_voxels: 0,
            island_penalty: 0.0,
        });
    }

    let total_voxels = density.iter().filter(|&&v| v > threshold).count();
    let support = support_mask(bc_masks, dims);
    let mut support_set = vec![false; n_islands + 1];
    for (idx, &active) in support.iter().enumerate() {
        if active && labels[idx] > 0 {
            support_set[labels[idx] as usize] = true;
        }
    }
    let has_support = support_set.iter().skip(1).any(|&v| v);
    if !has_support {
        return Ok(IslandAnalysis {
            n_islands,
            main_island_label: 0,
            main_island_mask: vec![false; dims.len()],
            main_island_voxels: 0,
            loose_voxels: total_voxels,
            is_connected: false,
            total_voxels,
            island_penalty: calculate_island_penalty(n_islands, total_voxels, total_voxels),
        });
    }

    let anchored_mask: Vec<bool> = labels
        .iter()
        .map(|&l| l > 0 && support_set[l as usize])
        .collect();
    let anchored_voxels = anchored_mask.iter().filter(|&&v| v).count();
    let unanchored_voxels = total_voxels - anchored_voxels;

    let bounds = get_load_region_bounds(dims, load_config);
    let stride_w = dims.w as usize;
    let mut labels_at_load = vec![false; n_islands + 1];
    let mut any_at_load = false;
    let mut filled = 0_usize;
    let mut expected = 0_usize;
    for y in bounds.y_min..bounds.y_max {
        for z in bounds.z_min..bounds.z_max {
            expected += 1;
            let idx = bounds.lx as usize * dims.plane_hw() + y as usize * stride_w + z as usize;
            if density[idx] > threshold {
                filled += 1;
            }
            let label = labels[idx];
            if label > 0 {
                labels_at_load[label as usize] = true;
                any_at_load = true;
            }
        }
    }

    if !any_at_load {
        return Ok(IslandAnalysis {
            n_islands,
            main_island_label: 0,
            main_island_mask: anchored_mask,
            main_island_voxels: anchored_voxels,
            loose_voxels: unanchored_voxels,
            is_connected: false,
            total_voxels,
            island_penalty: calculate_island_penalty(n_islands, unanchored_voxels, total_voxels),
        });
    }

    let mut load_label_count = 0_usize;
    let mut main_label = 0_u32;
    for (label, &present) in labels_at_load.iter().enumerate().skip(1) {
        if present {
            load_label_count += 1;
            if support_set[label] {
                let size = labels.iter().filter(|&&l| l as usize == label).count();
                let best_size = if main_label == 0 {
                    0
                } else {
                    labels.iter().filter(|&&l| l == main_label).count()
                };
                if size > best_size {
                    main_label = label as u32;
                }
            }
        }
    }

    if main_label == 0 {
        return Ok(IslandAnalysis {
            n_islands,
            main_island_label: 0,
            main_island_mask: anchored_mask,
            main_island_voxels: anchored_voxels,
            loose_voxels: unanchored_voxels,
            is_connected: false,
            total_voxels,
            island_penalty: calculate_island_penalty(n_islands, unanchored_voxels, total_voxels),
        });
    }

    let main_mask: Vec<bool> = labels.iter().map(|&l| l == main_label).collect();
    let main_voxels = main_mask.iter().filter(|&&v| v).count();
    let loose_voxels = total_voxels - main_voxels;
    let connected = load_label_count == 1 && filled == expected;
    Ok(IslandAnalysis {
        n_islands,
        main_island_label: main_label,
        main_island_mask: main_mask,
        main_island_voxels: main_voxels,
        loose_voxels,
        is_connected: connected,
        total_voxels,
        island_penalty: calculate_island_penalty(n_islands, loose_voxels, total_voxels),
    })
}

/// Connectivity reward `(bonus, connected_load_fraction)` from v3.3 reward.py.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConnectivityReport {
    pub bonus: f64,
    pub connected_load_fraction: f64,
}

pub fn calculate_connectivity_reward(
    density: &[f32],
    dims: Dims,
    bc_masks: Tensor3<'_>,
    forces: Tensor3<'_>,
    threshold: f32,
) -> CoreResult<ConnectivityReport> {
    check_len("density", density, dims)?;
    for m in bc_masks {
        check_len("bc_mask", m, dims)?;
    }
    for f in forces {
        check_len("force", f, dims)?;
    }
    let (labels, n_islands) = label_islands(density, dims, threshold);
    if n_islands == 0 {
        return Ok(ConnectivityReport {
            bonus: 0.0,
            connected_load_fraction: 0.0,
        });
    }
    let support = support_mask(Some(bc_masks), dims);
    if !support.iter().any(|&v| v) {
        return Ok(ConnectivityReport {
            bonus: 0.0,
            connected_load_fraction: 0.0,
        });
    }
    let mut support_set = vec![false; n_islands + 1];
    for (idx, &active) in support.iter().enumerate() {
        if active && labels[idx] > 0 {
            support_set[labels[idx] as usize] = true;
        }
    }
    if !support_set.iter().skip(1).any(|&v| v) {
        return Ok(ConnectivityReport {
            bonus: 0.0,
            connected_load_fraction: 0.0,
        });
    }
    let mut total_load = 0_usize;
    let mut connected_load = 0_usize;
    for idx in 0..dims.len() {
        let force_mag = forces[0][idx].abs() + forces[1][idx].abs() + forces[2][idx].abs();
        if force_mag > 1e-6 {
            total_load += 1;
            let label = labels[idx];
            if density[idx] > threshold && label > 0 && support_set[label as usize] {
                connected_load += 1;
            }
        }
    }
    let fraction = if total_load > 0 {
        connected_load as f64 / total_load as f64
    } else {
        0.0
    };
    let mut bonus = 0.5 * fraction;
    if fraction >= 0.999 {
        bonus += 0.5;
    }
    Ok(ConnectivityReport {
        bonus,
        connected_load_fraction: fraction,
    })
}

/// Legal add/remove action masks, exactly mirroring `legal_moves.py`.
#[derive(Debug, Clone, PartialEq)]
pub struct LegalMoves {
    pub valid_add: Vec<bool>,
    pub valid_remove: Vec<bool>,
}

impl LegalMoves {
    pub fn add_count(&self) -> usize {
        self.valid_add.iter().filter(|&&v| v).count()
    }

    pub fn remove_count(&self) -> usize {
        self.valid_remove.iter().filter(|&&v| v).count()
    }
}

pub fn compute_legal_moves(
    density: &[f32],
    dims: Dims,
    bc_masks: Option<Tensor3<'_>>,
    forces: Option<Tensor3<'_>>,
    threshold: f32,
) -> CoreResult<LegalMoves> {
    check_len("density", density, dims)?;
    if let Some(masks) = bc_masks {
        for m in masks {
            check_len("bc_mask", m, dims)?;
        }
    }
    if let Some(force_components) = forces {
        for f in force_components {
            check_len("force", f, dims)?;
        }
    }

    let solid: Vec<bool> = density.iter().map(|&v| v > threshold).collect();
    let mut valid_add = vec![false; dims.len()];
    let plane = dims.plane_hw();
    let stride_y = dims.w as usize;
    let d = dims.d as i64;
    let h = dims.h as i64;
    let w = dims.w as i64;

    let mark_neighborhood = |idx: usize, valid_add: &mut Vec<bool>| {
        let x = (idx / plane) as i64;
        let rem = idx % plane;
        let y = (rem / stride_y) as i64;
        let z = (rem % stride_y) as i64;
        for (dx, dy, dz) in NEIGHBORS_6 {
            let nx = x + dx;
            let ny = y + dy;
            let nz = z + dz;
            if nx < 0 || ny < 0 || nz < 0 || nx >= d || ny >= h || nz >= w {
                continue;
            }
            let n_idx = nx as usize * plane + ny as usize * stride_y + nz as usize;
            if !solid[n_idx] {
                valid_add[n_idx] = true;
            }
        }
    };

    // Boundary of existing material + support initiation region.
    for (idx, &is_solid) in solid.iter().enumerate() {
        if is_solid {
            mark_neighborhood(idx, &mut valid_add);
        }
    }

    let mut support = vec![false; dims.len()];
    match bc_masks {
        Some(masks) => {
            for idx in 0..dims.len() {
                support[idx] = masks[0][idx] > 0.5 || masks[1][idx] > 0.5 || masks[2][idx] > 0.5;
            }
        }
        None => {
            let plane_len = dims.plane_hw();
            for y in 0..dims.h {
                for z in 0..dims.w {
                    let idx = y as usize * dims.w as usize + z as usize;
                    support[idx] = true;
                    if dims.d > 1 {
                        support[plane_len + idx] = true;
                    }
                }
            }
        }
    }
    for (idx, &active) in support.iter().enumerate() {
        if active && !solid[idx] {
            valid_add[idx] = true;
        }
        if active {
            mark_neighborhood(idx, &mut valid_add);
        }
    }
    for (idx, &is_solid) in solid.iter().enumerate() {
        if is_solid {
            valid_add[idx] = false;
        }
    }

    let mut valid_remove = solid.clone();
    if bc_masks.is_some() || forces.is_some() {
        for idx in 0..dims.len() {
            let bc_sum = bc_masks
                .map(|m| m[0][idx] + m[1][idx] + m[2][idx])
                .unwrap_or(0.0);
            let force_mag = forces
                .map(|f| f[0][idx].abs() + f[1][idx].abs() + f[2][idx].abs())
                .unwrap_or(0.0);
            if bc_sum > 0.0 || force_mag > 1e-6 {
                valid_remove[idx] = false;
            }
        }
    }

    Ok(LegalMoves {
        valid_add,
        valid_remove,
    })
}

/// Joint masked softmax over legal add/remove actions (v3.3 policy contract).
pub fn masked_softmax_policy(
    policy_add: &[f32],
    policy_remove: &[f32],
    valid_add: &[bool],
    valid_remove: &[bool],
    temperature: f32,
) -> CoreResult<(Vec<f32>, Vec<f32>)> {
    if policy_add.len() != policy_remove.len()
        || policy_add.len() != valid_add.len()
        || policy_remove.len() != valid_remove.len()
    {
        return Err(CoreError::InvalidInput {
            message: "masked_softmax tensor length mismatch".to_string(),
        });
    }
    let temp = temperature.max(1e-6);
    let mut combined = Vec::with_capacity(policy_add.len() + policy_remove.len());
    for (idx, &value) in policy_add.iter().enumerate() {
        combined.push(if valid_add[idx] {
            value / temp
        } else {
            f32::NEG_INFINITY
        });
    }
    for (idx, &value) in policy_remove.iter().enumerate() {
        combined.push(if valid_remove[idx] {
            value / temp
        } else {
            f32::NEG_INFINITY
        });
    }
    let max = combined.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Ok((vec![0.0; policy_add.len()], vec![0.0; policy_remove.len()]));
    }
    let mut total = 0.0_f64;
    let mut exps = vec![0.0_f32; combined.len()];
    for (idx, &value) in combined.iter().enumerate() {
        if value.is_finite() {
            let e = (value - max).exp();
            exps[idx] = e;
            total += e as f64;
        }
    }
    let spatial = policy_add.len();
    let (add_probs, rem_probs) = if total < 1e-12 {
        (vec![0.0; spatial], vec![0.0; policy_remove.len()])
    } else {
        let add: Vec<f32> = exps[..spatial]
            .iter()
            .map(|&v| (v as f64 / total) as f32)
            .collect();
        let rem: Vec<f32> = exps[spatial..]
            .iter()
            .map(|&v| (v as f64 / total) as f32)
            .collect();
        (add, rem)
    };
    Ok((add_probs, rem_probs))
}

/// Policy detection rule: negative logits or max > 1.001 mean raw logits.
pub fn policy_is_logits(policy_add: &[f32], policy_remove: &[f32]) -> bool {
    let mut max_value = f32::NEG_INFINITY;
    for &value in policy_add.iter().chain(policy_remove.iter()) {
        if value.abs() > 1e-7 {
            if value < -1e-4 {
                return true;
            }
            if value > max_value {
                max_value = value;
            }
        }
    }
    max_value > 1.001
}

/// Strain / displacement maps and normalized inputs for the GPU shader.
#[derive(Debug, Clone, PartialEq)]
pub struct StrainResult {
    pub strain: Vec<f32>,
    pub disp_norm: Vec<f32>,
    pub strain_norm: Vec<f32>,
    pub disp_p99: f32,
    pub strain_p99: f32,
}

fn percentile99(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 1e-6;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let idx = (0.99_f64 * (values.len() as f64 - 1.0)).floor() as usize;
    values[idx.min(values.len() - 1)].max(1e-6)
}

/// Central-difference strain `||grad |u|||` with robust p99 normalization.
pub fn compute_strain_and_p99(
    density: &[f32],
    disp_map: &[f32],
    dims: Dims,
) -> CoreResult<StrainResult> {
    check_len("density", density, dims)?;
    check_len("disp_map", disp_map, dims)?;
    let plane = dims.plane_hw();
    let stride_y = dims.w as usize;
    let d = dims.d as i64;
    let h = dims.h as i64;
    let w = dims.w as i64;
    let at = |x: i64, y: i64, z: i64| -> f32 {
        if x < 0 || y < 0 || z < 0 || x >= d || y >= h || z >= w {
            return 0.0;
        }
        disp_map[x as usize * plane + y as usize * stride_y + z as usize]
    };

    let mut strain = vec![0.0_f32; dims.len()];
    let mut disp_values = Vec::new();
    let mut strain_values = Vec::new();
    for (idx, &rho) in density.iter().enumerate() {
        if rho <= 0.5 {
            continue;
        }
        let x = (idx / plane) as i64;
        let rem = idx % plane;
        let y = (rem / stride_y) as i64;
        let z = (rem % stride_y) as i64;
        let gx = (at(x + 1, y, z) - at(x - 1, y, z)) * 0.5;
        let gy = (at(x, y + 1, z) - at(x, y - 1, z)) * 0.5;
        let gz = (at(x, y, z + 1) - at(x, y, z - 1)) * 0.5;
        let value = (gx * gx + gy * gy + gz * gz).sqrt();
        strain[idx] = value;
        let disp = disp_map[idx].abs();
        if disp.is_finite() {
            disp_values.push(disp);
        }
        if value.is_finite() {
            strain_values.push(value);
        }
    }

    let mut disp_sorted = disp_values.clone();
    let mut strain_sorted = strain_values.clone();
    let disp_p99 = percentile99(&mut disp_sorted);
    let strain_p99 = percentile99(&mut strain_sorted);
    let disp_norm = disp_map
        .iter()
        .enumerate()
        .map(|(idx, &v)| {
            if density[idx] > 0.5 {
                (v.abs() / disp_p99).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    let strain_norm = strain
        .iter()
        .map(|&v| (v / strain_p99).clamp(0.0, 1.0))
        .collect();
    Ok(StrainResult {
        strain,
        disp_norm,
        strain_norm,
        disp_p99,
        strain_p99,
    })
}

/// Volumetric IoU used by the A/B comparator (`intersection / max(1, union)`).
pub fn compute_iou_3d(density_a: &[f32], density_b: &[f32], threshold: f32) -> f32 {
    let mut intersection = 0_usize;
    let mut union = 0_usize;
    for (&a, &b) in density_a.iter().zip(density_b.iter()) {
        let sa = a > threshold;
        let sb = b > threshold;
        if sa && sb {
            intersection += 1;
        }
        if sa || sb {
            union += 1;
        }
    }
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union.max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{generate_bc_masks, generate_forces, BcType};

    fn fill_range(density: &mut [f32], dims: Dims, x0: u32, x1: u32) {
        for x in x0..x1 {
            for y in 0..dims.h {
                for z in 0..dims.w {
                    density[dims.flat_idx(x, y, z).unwrap()] = 1.0;
                }
            }
        }
    }

    fn test_config() -> LoadConfig {
        LoadConfig {
            x: 7.0,
            y: 1.0,
            z_start: 1.0,
            z_end: 2.0,
            bc_type: Some(BcType::FullClamp),
        }
    }

    #[test]
    fn disconnected_beam_has_two_islands() {
        let dims = Dims::new(8, 3, 3);
        let mut density = vec![0.0_f32; dims.len()];
        fill_range(&mut density, dims, 0, 2);
        fill_range(&mut density, dims, 4, 8);
        let masks = generate_bc_masks(dims, BcType::FullClamp);
        let bc = [
            masks[0].as_slice(),
            masks[1].as_slice(),
            masks[2].as_slice(),
        ];
        let forces = generate_forces(dims, &test_config());
        let f = [
            forces[0].as_slice(),
            forces[1].as_slice(),
            forces[2].as_slice(),
        ];
        let analysis =
            analyze_islands(&density, dims, Some(bc), Some(f), &test_config(), 0.5).unwrap();
        assert_eq!(analysis.n_islands, 2);
        assert!(!analysis.is_connected);
        // The load island is not anchored, so the Python analysis reports main_label = 0.
        assert_eq!(analysis.main_island_label, 0);
        assert_eq!(analysis.main_island_voxels, 18);
        assert_eq!(analysis.loose_voxels, 36);
        assert!(analysis.island_penalty > 0.0);
    }

    #[test]
    fn connected_beam_is_connected() {
        let dims = Dims::new(8, 3, 3);
        let mut density = vec![0.0_f32; dims.len()];
        fill_range(&mut density, dims, 0, 8);
        let masks = generate_bc_masks(dims, BcType::FullClamp);
        let bc = [
            masks[0].as_slice(),
            masks[1].as_slice(),
            masks[2].as_slice(),
        ];
        let forces = generate_forces(dims, &test_config());
        let f = [
            forces[0].as_slice(),
            forces[1].as_slice(),
            forces[2].as_slice(),
        ];
        let analysis =
            analyze_islands(&density, dims, Some(bc), Some(f), &test_config(), 0.5).unwrap();
        assert_eq!(analysis.n_islands, 1);
        assert!(analysis.is_connected);
        assert_eq!(analysis.loose_voxels, 0);

        let connectivity = calculate_connectivity_reward(&density, dims, bc, f, 0.5).unwrap();
        assert_eq!(connectivity.connected_load_fraction, 1.0);
        assert_eq!(connectivity.bonus, 1.0);
    }

    #[test]
    fn support_and_load_are_protected_from_removal() {
        let dims = Dims::new(8, 3, 3);
        let density = vec![1.0_f32; dims.len()];
        let masks = generate_bc_masks(dims, BcType::FullClamp);
        let bc = [
            masks[0].as_slice(),
            masks[1].as_slice(),
            masks[2].as_slice(),
        ];
        let forces = generate_forces(dims, &test_config());
        let f = [
            forces[0].as_slice(),
            forces[1].as_slice(),
            forces[2].as_slice(),
        ];
        let legal = compute_legal_moves(&density, dims, Some(bc), Some(f), 0.5).unwrap();
        let support_idx = dims.flat_idx(0, 1, 1).unwrap();
        let load_idx = dims.flat_idx(7, 1, 1).unwrap();
        let free_idx = dims.flat_idx(4, 1, 1).unwrap();
        assert!(!legal.valid_remove[support_idx]);
        assert!(!legal.valid_remove[load_idx]);
        assert!(legal.valid_remove[free_idx]);
        assert_eq!(legal.add_count(), 0);
    }

    #[test]
    fn support_neighborhood_is_addable() {
        let dims = Dims::new(8, 3, 3);
        let density = vec![0.0_f32; dims.len()];
        let masks = generate_bc_masks(dims, BcType::FullClamp);
        let bc = [
            masks[0].as_slice(),
            masks[1].as_slice(),
            masks[2].as_slice(),
        ];
        let legal = compute_legal_moves(&density, dims, Some(bc), None, 0.5).unwrap();
        assert!(legal.valid_add[dims.flat_idx(0, 1, 1).unwrap()]);
        assert!(legal.valid_add[dims.flat_idx(1, 1, 1).unwrap()]);
        assert!(!legal.valid_add[dims.flat_idx(2, 1, 1).unwrap()]);
        assert!(!legal.valid_add[dims.flat_idx(4, 1, 1).unwrap()]);
        assert_eq!(legal.remove_count(), 0);

        // Without masks, v3.3 seeds planes x=0 and x=1, so dilation reaches x=2.
        let default_legal = compute_legal_moves(&density, dims, None, None, 0.5).unwrap();
        assert!(default_legal.valid_add[dims.flat_idx(2, 1, 1).unwrap()]);
    }

    #[test]
    fn strain_uses_central_differences() {
        let dims = Dims::new(4, 1, 1);
        let density = vec![1.0_f32; dims.len()];
        let disp = vec![0.0_f32, 1.0, 2.0, 3.0];
        let result = compute_strain_and_p99(&density, &disp, dims).unwrap();
        // Interior voxels: (d(x+1)-d(x-1))/2 = 1. Boundary: 0.5.
        assert!((result.strain[1] - 1.0).abs() < 1e-6);
        assert!((result.strain[2] - 1.0).abs() < 1e-6);
        // p99 index = floor(0.99 * (N-1)) = 2 over the sorted [0,1,2,3] values.
        assert_eq!(result.disp_p99, 2.0);
        assert_eq!(result.disp_norm[3], 1.0);
        assert!(result.disp_norm[0] == 0.0);
    }

    #[test]
    fn iou_identity_and_disjoint() {
        let a = vec![1.0_f32, 0.0, 1.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0, 1.0];
        assert_eq!(compute_iou_3d(&a, &a, 0.5), 1.0);
        assert_eq!(compute_iou_3d(&a, &b, 0.5), 1.0 / 3.0);
        let empty = vec![0.0_f32; 4];
        assert_eq!(compute_iou_3d(&empty, &empty, 0.5), 0.0);
    }
}
