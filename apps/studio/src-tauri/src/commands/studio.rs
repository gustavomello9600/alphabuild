//! Studio editing commands: instant Rust topology validation and ABMESH rebuilds.

use ab_core::coords::Dims;
use ab_core::topology::{analyze_islands, calculate_connectivity_reward, compute_legal_moves};
use tauri::ipc::Response;
use tauri::State;

use crate::commands::frame::{build_abmesh, core_load_config, FrameData};
use crate::contracts::{CachedInferOverlay, LocalTopologyReport, MeshOptions, StudioVoxelState};
use crate::state::AppState;

fn build_data(state: &StudioVoxelState) -> Result<FrameData, String> {
    let mut data = FrameData::empty_studio(state).map_err(|e| e.to_string())?;
    let dims = Dims::new(
        state.resolution[0],
        state.resolution[1],
        state.resolution[2],
    );
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
    let load = core_load_config(&state.load_config);
    let analysis = analyze_islands(
        &data.density,
        dims,
        Some(bc_refs),
        Some(force_refs),
        &load,
        0.5,
    )
    .map_err(|e| e.to_string())?;
    let legal = compute_legal_moves(&data.density, dims, Some(bc_refs), Some(force_refs), 0.5)
        .map_err(|e| e.to_string())?;
    let connectivity = calculate_connectivity_reward(&data.density, dims, bc_refs, force_refs, 0.5)
        .map_err(|e| e.to_string())?;
    data.metadata["is_connected"] = serde_json::json!(analysis.is_connected);
    data.metadata["n_islands"] = serde_json::json!(analysis.n_islands);
    data.metadata["loose_voxels"] = serde_json::json!(analysis.loose_voxels);
    data.metadata["island_penalty"] = serde_json::json!(analysis.island_penalty);
    data.metadata["connected_load_fraction"] =
        serde_json::json!(connectivity.connected_load_fraction);
    data.metadata["volume_fraction"] = serde_json::json!(
        data.density.iter().filter(|&&v| v > 0.5).count() as f64 / dims.len() as f64
    );
    data.metadata["legal_add_count"] = serde_json::json!(legal.add_count());
    data.metadata["legal_remove_count"] = serde_json::json!(legal.remove_count());
    Ok(data)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn studio_analyze_local(
    _state: State<'_, AppState>,
    state: StudioVoxelState,
) -> Result<LocalTopologyReport, String> {
    let data = build_data(&state)?;
    let dims = data.dims;
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
    let load = core_load_config(&state.load_config);
    let analysis = analyze_islands(
        &data.density,
        dims,
        Some(bc_refs),
        Some(force_refs),
        &load,
        0.5,
    )
    .map_err(|e| e.to_string())?;
    let legal = compute_legal_moves(&data.density, dims, Some(bc_refs), Some(force_refs), 0.5)
        .map_err(|e| e.to_string())?;
    let connectivity = calculate_connectivity_reward(&data.density, dims, bc_refs, force_refs, 0.5)
        .map_err(|e| e.to_string())?;
    Ok(LocalTopologyReport {
        is_connected: analysis.is_connected,
        n_islands: analysis.n_islands,
        loose_voxels: analysis.loose_voxels,
        volume_fraction: data.density.iter().filter(|&&v| v > 0.5).count() as f64
            / dims.len() as f64,
        connected_load_fraction: connectivity.connected_load_fraction,
        island_penalty: analysis.island_penalty,
        legal_add_count: legal.add_count(),
        legal_remove_count: legal.remove_count(),
    })
}

#[tauri::command(rename_all = "snake_case")]
pub async fn studio_build_mesh(
    _state: State<'_, AppState>,
    state: StudioVoxelState,
    overlay: Option<CachedInferOverlay>,
) -> Result<Response, String> {
    let mut data = build_data(&state)?;
    if let Some(overlay) = overlay {
        data.apply_overlay(&overlay);
    }
    let options = MeshOptions::default();
    let bytes = build_abmesh(&data, &options).map_err(|e| e.to_string())?;
    Ok(Response::new(bytes))
}
