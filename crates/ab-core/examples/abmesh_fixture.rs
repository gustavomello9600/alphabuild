//! Generates the committed ABMESH01 fixture consumed by the Vitest suite.

use std::process::ExitCode;

use ab_core::codecs::wire::{abmesh_flags, AbMeshPacket};
use ab_core::coords::Dims;
use ab_core::mesher::build_mesh;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: abmesh_fixture <output.bin>");
        return ExitCode::from(2);
    };
    let dims = Dims::new(2, 2, 2);
    let density = vec![1.0_f32; dims.len()];
    let mesh = match build_mesh(&density, dims, 0.5, None, None) {
        Ok(mesh) => mesh,
        Err(error) => {
            eprintln!("mesh error: {error}");
            return ExitCode::from(3);
        }
    };
    let metadata = r#"{"step":3,"phase":"REFINEMENT","value":0.42,"value_target":0.35,"fem_reward":0.35,"compliance_fem":312.4,"max_displacement":41.2,"volume_fraction":0.112,"is_connected":true,"n_islands":1,"loose_voxels":0,"connected_load_fraction":1.0,"island_penalty":0.0,"reward_v33":{"compliance_score":0.58,"volume_bonus":-0.03,"fem_score":0.55,"displacement_penalty":0.0,"island_penalty":0.0,"connectivity_bonus":0.0,"pbrs_delta":null,"total_reward":0.55,"is_collapsed":false},"mcts_stats":{"num_simulations":160,"nodes_expanded":42,"max_depth":4,"cache_hits":12,"top8_concentration":0.64,"refutation":true},"selected_actions":[{"channel":0,"x":1,"y":1,"z":1,"visits":48,"q_value":0.61,"branch_index":0}],"resolution":[2,2,2],"bc_type":"FULL_CLAMP","load_config":{"x":1,"y":1,"z_start":0,"z_end":2}}"#;
    let packet = AbMeshPacket {
        flags: abmesh_flags::POLICY
            | abmesh_flags::BLUEPRINT
            | abmesh_flags::MCTS
            | abmesh_flags::DIFF
            | abmesh_flags::DISPLACEMENT,
        step_index: 3,
        dims,
        positions: mesh.positions.clone(),
        normals: mesh.normals.clone(),
        ao_and_scalars: mesh.ao_and_scalars.clone(),
        voxel_ids: mesh.voxel_ids.clone(),
        indices: mesh.indices.clone(),
        blueprint_positions: mesh.positions,
        blueprint_normals: mesh.normals,
        blueprint_indices: mesh.indices,
        policy_add: vec![(0, 0.5), (3, 0.25)],
        policy_rem: vec![(6, 0.125)],
        mcts_add: vec![(0, 48.0, 0.61), (3, 12.0, -0.2)],
        mcts_rem: vec![(6, 3.0, 0.1)],
        diff_added: vec![1, 3],
        diff_removed: vec![6],
        disp_p99: 41.2,
        strain_p99: 0.75,
        meta_json: metadata.to_string(),
    };
    match packet.encode() {
        Ok(bytes) => match std::fs::write(&path, bytes) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("write error: {error}");
                ExitCode::from(4)
            }
        },
        Err(error) => {
            eprintln!("encode error: {error}");
            ExitCode::from(5)
        }
    }
}
