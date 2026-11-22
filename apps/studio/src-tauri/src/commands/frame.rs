//! ABMESH01 construction from local rows, studio state and ABWIRE01 frames.

use ab_core::codecs::wire::{abmesh_flags, tensor_tag, AbMeshPacket, WirePacket};
use ab_core::coords::{
    generate_bc_masks, generate_forces, BcType, Dims, LoadConfig as CoreLoadConfig,
};
use ab_core::error::{CoreError, CoreResult};
use ab_core::mesher::{build_mesh, DEFAULT_THRESHOLD};
use ab_core::topology::{
    compute_legal_moves, compute_strain_and_p99, masked_softmax_policy, policy_is_logits,
};
use serde_json::{json, Value};

use crate::contracts::{
    CachedInferOverlay, ForceVector, LoadConfig, MeshOptions, StudioVoxelState, SupportVoxel,
};

#[derive(Debug, Clone)]
pub struct FrameData {
    pub dims: Dims,
    pub density: Vec<f32>,
    pub rho_cont: Option<Vec<f32>>,
    pub displacement: Option<Vec<f32>>,
    pub blueprint: Option<Vec<f32>>,
    pub policy_add: Vec<f32>,
    pub policy_remove: Vec<f32>,
    pub mcts_visit_add: Vec<f32>,
    pub mcts_visit_remove: Vec<f32>,
    pub mcts_q_add: Vec<f32>,
    pub mcts_q_remove: Vec<f32>,
    pub diff_added: Vec<u32>,
    pub diff_removed: Vec<u32>,
    pub bc_masks: [Vec<f32>; 3],
    pub forces: [Vec<f32>; 3],
    pub metadata: Value,
}

impl FrameData {
    pub fn new(dims: Dims) -> Self {
        let total = dims.len();
        Self {
            dims,
            density: vec![0.0; total],
            rho_cont: None,
            displacement: None,
            blueprint: None,
            policy_add: vec![0.0; total],
            policy_remove: vec![0.0; total],
            mcts_visit_add: vec![0.0; total],
            mcts_visit_remove: vec![0.0; total],
            mcts_q_add: vec![0.0; total],
            mcts_q_remove: vec![0.0; total],
            diff_added: Vec::new(),
            diff_removed: Vec::new(),
            bc_masks: generate_bc_masks(dims, BcType::FullClamp),
            forces: [vec![0.0; total], vec![0.0; total], vec![0.0; total]],
            metadata: json!({}),
        }
    }

    pub fn empty_studio(state: &StudioVoxelState) -> CoreResult<Self> {
        let dims = Dims::new(
            state.resolution[0],
            state.resolution[1],
            state.resolution[2],
        );
        dims.validate()?;
        let mut data = Self::new(dims);
        for &idx in &state.solid_voxels {
            let idx = idx as usize;
            if idx >= dims.len() {
                return Err(CoreError::InvalidInput {
                    message: format!("solid voxel {idx} outside domain"),
                });
            }
            data.density[idx] = 1.0;
        }
        let bc = parse_bc_type(&state.bc_type);
        data.bc_masks = masks_for_mesh(state, dims, bc);
        data.forces = forces_for_mesh(state, dims);
        data.metadata = studio_metadata(state, &data);
        Ok(data)
    }

    pub fn apply_overlay(&mut self, overlay: &CachedInferOverlay) {
        if let Some(probs) = &overlay.blueprint_probs {
            if probs.len() == self.dims.len() {
                self.blueprint = Some(probs.clone());
            }
        }
        for entry in &overlay.policy_add {
            if let Some(slot) = self.policy_add.get_mut(entry.flat_idx as usize) {
                *slot = entry.prob;
            }
        }
        for entry in &overlay.policy_remove {
            if let Some(slot) = self.policy_remove.get_mut(entry.flat_idx as usize) {
                *slot = entry.prob;
            }
        }
        if let Some(value) = overlay.value {
            self.metadata["value"] = json!(value);
        }
    }
}

pub fn parse_bc_type(value: &str) -> BcType {
    match value {
        "RAIL_XY" => BcType::RailXy,
        "ROLLER_Y" => BcType::RollerY,
        "ROLLER_Z" => BcType::RollerZ,
        "CUSTOM" => BcType::Custom,
        _ => BcType::FullClamp,
    }
}

pub fn core_load_config(config: &LoadConfig) -> CoreLoadConfig {
    CoreLoadConfig {
        x: config.x,
        y: config.y,
        z_start: config.z_start,
        z_end: config.z_end,
        bc_type: Some(parse_bc_type(
            config.bc_type.as_deref().unwrap_or("FULL_CLAMP"),
        )),
    }
}

pub fn masks_for_mesh(state: &StudioVoxelState, dims: Dims, bc: BcType) -> [Vec<f32>; 3] {
    if let Some(supports) = &state.custom_supports {
        let mut masks = [
            vec![0.0; dims.len()],
            vec![0.0; dims.len()],
            vec![0.0; dims.len()],
        ];
        for support in supports {
            if let Some(idx) = dims.flat_idx(
                support.x.max(0) as u32,
                support.y.max(0) as u32,
                support.z.max(0) as u32,
            ) {
                masks[0][idx] = if support.mask_x { 1.0 } else { 0.0 };
                masks[1][idx] = if support.mask_y { 1.0 } else { 0.0 };
                masks[2][idx] = if support.mask_z { 1.0 } else { 0.0 };
            }
        }
        masks
    } else {
        generate_bc_masks(dims, bc)
    }
}

pub fn forces_for_mesh(state: &StudioVoxelState, dims: Dims) -> [Vec<f32>; 3] {
    if let Some(forces) = &state.custom_forces {
        if !forces.is_empty() {
            let mut out = [
                vec![0.0; dims.len()],
                vec![0.0; dims.len()],
                vec![0.0; dims.len()],
            ];
            for force in forces {
                if let Some(idx) = dims.flat_idx(
                    force.x.max(0) as u32,
                    force.y.max(0) as u32,
                    force.z.max(0) as u32,
                ) {
                    out[0][idx] = force.fx as f32;
                    out[1][idx] = force.fy as f32;
                    out[2][idx] = force.fz as f32;
                }
            }
            return out;
        }
    }
    generate_forces(dims, &core_load_config(&state.load_config))
}

pub fn supports_from_masks(dims: Dims, masks: &[Vec<f32>; 3]) -> Vec<SupportVoxel> {
    let mut supports = Vec::new();
    for (idx, &mask_x) in masks[0].iter().enumerate() {
        let mask_y = masks[1][idx];
        let mask_z = masks[2][idx];
        if mask_x > 0.5 || mask_y > 0.5 || mask_z > 0.5 {
            if let Some((x, y, z)) = dims.inverse(idx) {
                supports.push(SupportVoxel {
                    x: x as i32,
                    y: y as i32,
                    z: z as i32,
                    mask_x: mask_x > 0.5,
                    mask_y: mask_y > 0.5,
                    mask_z: mask_z > 0.5,
                });
            }
        }
    }
    supports
}

pub fn forces_from_arrays(dims: Dims, forces: &[Vec<f32>; 3]) -> Vec<ForceVector> {
    let mut out = Vec::new();
    for (idx, &fx) in forces[0].iter().enumerate() {
        let fy = forces[1][idx];
        let fz = forces[2][idx];
        if fx.abs() + fy.abs() + fz.abs() > 1e-6 {
            if let Some((x, y, z)) = dims.inverse(idx) {
                out.push(ForceVector {
                    x: x as i32,
                    y: y as i32,
                    z: z as i32,
                    fx: fx as f64,
                    fy: fy as f64,
                    fz: fz as f64,
                });
            }
        }
    }
    out
}

fn overlay_pairs(probs: &[f32]) -> Vec<(u32, f32)> {
    let max = probs.iter().copied().fold(0.0_f32, f32::max);
    let floor = (0.01 * max).max(1e-4);
    probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p >= floor && p > 0.0)
        .map(|(idx, &p)| (idx as u32, p))
        .collect()
}

fn mcts_triples(visits: &[f32], q_values: &[f32]) -> Vec<(u32, f32, f32)> {
    visits
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(idx, &v)| (idx as u32, v, q_values.get(idx).copied().unwrap_or(0.0)))
        .collect()
}

pub fn build_abmesh(data: &FrameData, options: &MeshOptions) -> CoreResult<Vec<u8>> {
    data.dims.validate()?;
    let total = data.dims.len();
    if data.density.len() != total {
        return Err(CoreError::InvalidInput {
            message: "density length mismatch".to_string(),
        });
    }
    let threshold = options.density_threshold.clamp(0.0, 1.0);
    let bc_refs = [
        data.bc_masks[0].as_slice(),
        data.bc_masks[1].as_slice(),
        data.bc_masks[2].as_slice(),
    ];
    let force_refs = [
        data.forces[0].as_slice(),
        data.forces[1].as_slice(),
        data.forces[2].as_slice(),
    ];

    let (policy_add, policy_remove) = if policy_is_logits(&data.policy_add, &data.policy_remove) {
        let legal = compute_legal_moves(
            &data.density,
            data.dims,
            Some(bc_refs),
            Some(force_refs),
            threshold,
        )?;
        masked_softmax_policy(
            &data.policy_add,
            &data.policy_remove,
            &legal.valid_add,
            &legal.valid_remove,
            1.0,
        )?
    } else {
        (data.policy_add.clone(), data.policy_remove.clone())
    };

    let (disp_norm, strain_norm, disp_p99, strain_p99) = match &data.displacement {
        Some(disp) if disp.len() == total => {
            let strain = compute_strain_and_p99(&data.density, disp, data.dims)?;
            (
                strain.disp_norm,
                strain.strain_norm,
                strain.disp_p99,
                strain.strain_p99,
            )
        }
        _ => (vec![0.0; total], vec![0.0; total], 0.0, 0.0),
    };

    let geometry = if options.use_rho_cont {
        data.rho_cont.as_deref().unwrap_or(&data.density)
    } else {
        &data.density
    };
    let mesh = build_mesh(
        geometry,
        data.dims,
        threshold,
        Some(&disp_norm),
        Some(&strain_norm),
    )?;

    let blueprint_mesh = if options.include_blueprint {
        match &data.blueprint {
            Some(blueprint) if blueprint.len() == total => Some(build_mesh(
                blueprint,
                data.dims,
                options.blueprint_threshold.clamp(0.0, 1.0),
                None,
                None,
            )?),
            _ => None,
        }
    } else {
        None
    };

    let mut flags = 0_u32;
    if data.displacement.is_some() {
        flags |= abmesh_flags::DISPLACEMENT;
    }
    if blueprint_mesh.is_some() {
        flags |= abmesh_flags::BLUEPRINT;
    }
    if policy_add.iter().any(|&p| p > 0.0) || policy_remove.iter().any(|&p| p > 0.0) {
        flags |= abmesh_flags::POLICY;
    }
    if data.mcts_visit_add.iter().any(|&v| v > 0.0)
        || data.mcts_visit_remove.iter().any(|&v| v > 0.0)
    {
        flags |= abmesh_flags::MCTS;
    }
    if !data.diff_added.is_empty() || !data.diff_removed.is_empty() {
        flags |= abmesh_flags::DIFF;
    }
    if options.use_rho_cont && data.rho_cont.is_some() {
        flags |= abmesh_flags::RHO_CONT;
    }

    let blueprint = blueprint_mesh.unwrap_or_default();
    let meta_json = serde_json::to_string(&data.metadata)?;
    let packet = AbMeshPacket {
        flags,
        step_index: data
            .metadata
            .get("step")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        dims: data.dims,
        positions: mesh.positions,
        normals: mesh.normals,
        ao_and_scalars: mesh.ao_and_scalars,
        voxel_ids: mesh.voxel_ids,
        indices: mesh.indices,
        blueprint_positions: blueprint.positions,
        blueprint_normals: blueprint.normals,
        blueprint_indices: blueprint.indices,
        policy_add: overlay_pairs(&policy_add),
        policy_rem: overlay_pairs(&policy_remove),
        mcts_add: mcts_triples(&data.mcts_visit_add, &data.mcts_q_add),
        mcts_rem: mcts_triples(&data.mcts_visit_remove, &data.mcts_q_remove),
        diff_added: data.diff_added.clone(),
        diff_removed: data.diff_removed.clone(),
        disp_p99,
        strain_p99,
        meta_json,
    };
    packet.encode()
}

fn tensor_dense(packet: &WirePacket, tag: u16, total: usize) -> CoreResult<Option<Vec<f32>>> {
    let Some(tensor) = packet.tensor(tag) else {
        return Ok(None);
    };
    let values = tensor.to_dense_f32()?;
    if values.len() != total {
        return Err(CoreError::InvalidInput {
            message: format!(
                "tensor {} has {} values, expected {}",
                tag,
                values.len(),
                total
            ),
        });
    }
    Ok(Some(values))
}

pub fn frame_from_wire(packet: &WirePacket, options: &MeshOptions) -> CoreResult<Vec<u8>> {
    let metadata = serde_json::from_str::<Value>(&packet.header_json)?;
    let density_tensor =
        packet
            .tensor(tensor_tag::DENSITY)
            .ok_or_else(|| CoreError::InvalidInput {
                message: "STEP_FRAME is missing DENSITY".to_string(),
            })?;
    let density_values = density_tensor.to_dense_f32()?;
    let dims = if let Some(res) = metadata.get("resolution").and_then(Value::as_array) {
        if res.len() >= 3 {
            Dims::new(
                res[0].as_u64().unwrap_or(64) as u32,
                res[1].as_u64().unwrap_or(32) as u32,
                res[2].as_u64().unwrap_or(8) as u32,
            )
        } else {
            let shape = density_tensor.shape;
            Dims::new(shape[0] as u32, shape[1] as u32, shape[2] as u32)
        }
    } else {
        let shape = density_tensor.shape;
        Dims::new(shape[0] as u32, shape[1] as u32, shape[2] as u32)
    };
    dims.validate()?;
    let total = dims.len();
    if density_values.len() != total {
        return Err(CoreError::InvalidInput {
            message: "density tensor shape mismatch".to_string(),
        });
    }
    let mut data = FrameData::new(dims);
    data.density = density_values;
    data.metadata = metadata.clone();

    if let Some(bc_type) = metadata.get("bc_type").and_then(Value::as_str) {
        data.bc_masks = generate_bc_masks(dims, parse_bc_type(bc_type));
    }
    if let Some(load) = metadata.get("load_config") {
        if let Ok(config) = serde_json::from_value::<LoadConfig>(load.clone()) {
            data.forces = generate_forces(dims, &core_load_config(&config));
        }
    }
    if let Some(t) = packet.tensor(tensor_tag::BC_MASKS) {
        let arr = t.to_dense_f32()?;
        if arr.len() == total * 3 {
            for c in 0..3 {
                data.bc_masks[c] = arr[c * total..(c + 1) * total].to_vec();
            }
        }
    }
    if let Some(t) = packet.tensor(tensor_tag::FORCES) {
        let arr = t.to_dense_f32()?;
        if arr.len() == total * 3 {
            for c in 0..3 {
                data.forces[c] = arr[c * total..(c + 1) * total].to_vec();
            }
        }
    }
    data.rho_cont = tensor_dense(packet, tensor_tag::RHO_CONT, total)?;
    data.blueprint = tensor_dense(packet, tensor_tag::BLUEPRINT_PROBS, total)?;
    data.displacement = tensor_dense(packet, tensor_tag::DISPLACEMENT_MAG, total)?;
    data.policy_add =
        tensor_dense(packet, tensor_tag::POLICY_ADD, total)?.unwrap_or_else(|| vec![0.0; total]);
    data.policy_remove =
        tensor_dense(packet, tensor_tag::POLICY_REMOVE, total)?.unwrap_or_else(|| vec![0.0; total]);
    data.mcts_visit_add = tensor_dense(packet, tensor_tag::MCTS_VISIT_ADD, total)?
        .unwrap_or_else(|| vec![0.0; total]);
    data.mcts_visit_remove = tensor_dense(packet, tensor_tag::MCTS_VISIT_REMOVE, total)?
        .unwrap_or_else(|| vec![0.0; total]);
    data.mcts_q_add =
        tensor_dense(packet, tensor_tag::MCTS_Q_ADD, total)?.unwrap_or_else(|| vec![0.0; total]);
    data.mcts_q_remove =
        tensor_dense(packet, tensor_tag::MCTS_Q_REMOVE, total)?.unwrap_or_else(|| vec![0.0; total]);
    build_abmesh(&data, options)
}

pub fn studio_metadata(state: &StudioVoxelState, data: &FrameData) -> Value {
    let solid = data.density.iter().filter(|&&v| v > 0.5).count();
    json!({
        "step": 0,
        "phase": "GROWTH",
        "value": 0.0,
        "value_target": null,
        "fem_reward": null,
        "compliance_fem": null,
        "max_displacement": null,
        "volume_fraction": solid as f64 / data.dims.len() as f64,
        "is_connected": false,
        "n_islands": 0,
        "loose_voxels": 0,
        "connected_load_fraction": 0.0,
        "island_penalty": 0.0,
        "reward_components": {},
        "reward_v33": {
            "compliance_score": null,
            "volume_bonus": null,
            "fem_score": null,
            "displacement_penalty": 0.0,
            "island_penalty": 0.0,
            "connectivity_bonus": 0.0,
            "pbrs_delta": null,
            "total_reward": 0.0,
            "is_collapsed": false
        },
        "mcts_stats": null,
        "selected_actions": [],
        "resolution": [state.resolution[0], state.resolution[1], state.resolution[2]],
        "bc_type": state.bc_type,
        "load_config": state.load_config,
        "default_threshold": DEFAULT_THRESHOLD
    })
}
