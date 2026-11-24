//! Local/remote dataset commands and ABMESH frame construction.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use ab_core::codecs::coo1::Coo1;
use ab_core::codecs::wire::decode_step_batch;
use ab_core::coords::Dims;
use ab_core::reward::calculate_reward_breakdown;
use ab_core::sqlite_reader::{DbKind as CoreDbKind, DecodedBlob, SqlValue, SqliteReader};
use serde_json::{json, Value};
use tauri::ipc::Response;
use tauri::State;

use crate::commands::frame::{
    build_abmesh, forces_from_arrays, frame_from_wire, supports_from_masks, FrameData,
};
use crate::contracts::{
    ComparisonMetrics, DataSourceRef, DbKind, EntryListPage, EntryListQuery, EntryMetadata,
    EntrySummary, FrameRef, LoadConfig, MeshOptions,
};
use crate::state::{default_cache_key, AppState};

pub fn contract_kind(kind: CoreDbKind) -> DbKind {
    match kind {
        CoreDbKind::HarvestV33 => DbKind::HarvestV33,
        CoreDbKind::HarvestV2 => DbKind::HarvestV2,
        CoreDbKind::SelfPlay => DbKind::Selfplay,
        CoreDbKind::LegacyV1 => DbKind::LegacyV1,
    }
}

fn open_reader(source: &DataSourceRef) -> Result<SqliteReader, String> {
    SqliteReader::open(&source.db_id).map_err(|e| e.to_string())
}

fn value_text(value: Option<&SqlValue>) -> Option<String> {
    value.and_then(SqlValue::as_str).map(|s| s.to_string())
}

fn value_f64(value: Option<&SqlValue>) -> Option<f64> {
    value.and_then(SqlValue::as_f64)
}

fn value_i64(value: Option<&SqlValue>) -> Option<i64> {
    value.and_then(SqlValue::as_i64)
}

fn value_bool(value: Option<&SqlValue>) -> Option<bool> {
    value_i64(value).map(|v| v != 0)
}

fn value_blob<'a>(row: &'a HashMap<String, SqlValue>, key: &str) -> Option<&'a [u8]> {
    row.get(key).and_then(SqlValue::as_blob)
}

fn parse_resolution(value: Option<&SqlValue>) -> [u32; 3] {
    let Some(text) = value.and_then(SqlValue::as_str) else {
        return [64, 32, 8];
    };
    let parsed: Value = serde_json::from_str(text).unwrap_or(Value::Null);
    let arr = parsed.as_array();
    let get = |idx: usize, default: u32| {
        arr.and_then(|a| a.get(idx))
            .and_then(Value::as_u64)
            .unwrap_or(default as u64) as u32
    };
    [get(0, 64), get(1, 32), get(2, 8)]
}

fn parse_load_config(row: &HashMap<String, SqlValue>) -> LoadConfig {
    let text = value_text(row.get("load_config")).unwrap_or_else(|| "{}".to_string());
    let value: Value = serde_json::from_str(&text).unwrap_or_else(|_| json!({}));
    LoadConfig {
        x: value.get("x").and_then(Value::as_f64).unwrap_or(63.0),
        y: value.get("y").and_then(Value::as_f64).unwrap_or(16.0),
        z_start: value.get("z_start").and_then(Value::as_f64).unwrap_or(3.0),
        z_end: value.get("z_end").and_then(Value::as_f64).unwrap_or(5.0),
        bc_type: value
            .get("bc_type")
            .and_then(Value::as_str)
            .map(|s| s.to_string()),
    }
}

fn parse_bc_type_text(row: &HashMap<String, SqlValue>) -> String {
    value_text(row.get("bc_type")).unwrap_or_else(|| "FULL_CLAMP".to_string())
}

fn decode_dense(reader: &SqliteReader, blob: &[u8], total: usize) -> Result<Vec<f32>, String> {
    match reader.decode_blob(blob, total).map_err(|e| e.to_string())? {
        DecodedBlob::Dense(array) => {
            if array.data.len() != total {
                return Err(format!(
                    "dense blob has {} values, expected {}",
                    array.data.len(),
                    total
                ));
            }
            Ok(array.data)
        }
        DecodedBlob::Sparse(coo) => coo.to_dense(total).map_err(|e| e.to_string()),
    }
}

fn decode_coo(reader: &SqliteReader, blob: &[u8], total: usize) -> Result<Coo1, String> {
    match reader.decode_blob(blob, total).map_err(|e| e.to_string())? {
        DecodedBlob::Sparse(coo) => Ok(coo),
        DecodedBlob::Dense(array) => {
            if array.data.len() != total {
                return Err("dense policy length mismatch".to_string());
            }
            Ok(Coo1::from_dense(&array.data))
        }
    }
}

fn masks_and_forces(
    reader: &SqliteReader,
    row: &HashMap<String, SqlValue>,
    dims: Dims,
) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let total = dims.len();
    let mut masks = [vec![0.0; total], vec![0.0; total], vec![0.0; total]];
    let mut forces = [vec![0.0; total], vec![0.0; total], vec![0.0; total]];
    if let Some(blob) = value_blob(row, "bc_masks_blob") {
        if let Ok(values) = decode_dense(reader, blob, total * 3) {
            for c in 0..3 {
                masks[c].copy_from_slice(&values[c * total..(c + 1) * total]);
            }
        }
    }
    if let Some(blob) = value_blob(row, "forces_blob") {
        if let Ok(values) = decode_dense(reader, blob, total * 3) {
            for c in 0..3 {
                forces[c].copy_from_slice(&values[c * total..(c + 1) * total]);
            }
        }
    }
    (masks, forces)
}

fn volume_fraction(density: &[f32]) -> f64 {
    if density.is_empty() {
        return 0.0;
    }
    density.iter().filter(|&&v| v > 0.5).count() as f64 / density.len() as f64
}

fn reward_json(
    compliance: Option<f64>,
    volume: Option<f64>,
    max_disp: Option<f64>,
    connected: bool,
    island_penalty: f64,
) -> Value {
    match compliance {
        Some(c) => {
            let breakdown = calculate_reward_breakdown(
                c,
                volume.unwrap_or(0.0),
                connected,
                max_disp,
                island_penalty,
            );
            json!({
                "compliance_score": breakdown.compliance_score,
                "volume_bonus": breakdown.volume_bonus,
                "fem_score": breakdown.fem_score,
                "displacement_penalty": breakdown.displacement_penalty,
                "island_penalty": breakdown.island_penalty,
                "connectivity_bonus": 0.0,
                "pbrs_delta": null,
                "total_reward": breakdown.total_reward,
                "is_collapsed": breakdown.is_collapsed
            })
        }
        None => json!({
            "compliance_score": null,
            "volume_bonus": null,
            "fem_score": null,
            "displacement_penalty": 0.0,
            "island_penalty": island_penalty,
            "connectivity_bonus": 0.0,
            "pbrs_delta": null,
            "total_reward": if connected { 1.0 } else { -1.0 },
            "is_collapsed": !connected
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn base_metadata(
    dims: Dims,
    step: i64,
    phase: &str,
    value: f64,
    volume: f64,
    connected: bool,
    n_islands: i64,
    loose: i64,
    connected_fraction: f64,
    island_penalty: f64,
    compliance: Option<f64>,
    max_disp: Option<f64>,
    fem_reward: Option<f64>,
    value_target: Option<f64>,
    reward_components: Value,
    mcts_stats: Option<Value>,
    selected_actions: Value,
    bc_type: &str,
    load_config: &LoadConfig,
) -> Value {
    let reward = reward_json(
        compliance,
        Some(volume),
        max_disp,
        connected,
        island_penalty,
    );
    json!({
        "step": step,
        "phase": phase,
        "value": value,
        "value_target": value_target,
        "fem_reward": fem_reward,
        "compliance_fem": compliance,
        "max_displacement": max_disp,
        "volume_fraction": volume,
        "is_connected": connected,
        "n_islands": n_islands,
        "loose_voxels": loose,
        "connected_load_fraction": connected_fraction,
        "island_penalty": island_penalty,
        "reward_components": reward_components,
        "reward_v33": reward,
        "mcts_stats": mcts_stats,
        "selected_actions": selected_actions,
        "resolution": [dims.d, dims.h, dims.w],
        "bc_type": bc_type,
        "load_config": load_config
    })
}

#[allow(clippy::too_many_arguments)]
fn diff_next(
    reader: &SqliteReader,
    table: &str,
    id_column: &str,
    entry_id: &str,
    step: i64,
    dims: Dims,
    threshold: f32,
    current: &[f32],
) -> (Vec<u32>, Vec<u32>) {
    let sql =
        format!("SELECT density_blob FROM {table} WHERE {id_column} = ?1 AND step = ?2 LIMIT 1");
    let row = match reader.query_row(
        &sql,
        &[
            SqlValue::Text(entry_id.to_string()),
            SqlValue::Integer(step + 1),
        ],
    ) {
        Ok(Some(row)) => row,
        _ => return (Vec::new(), Vec::new()),
    };
    let Some(blob) = value_blob(&row, "density_blob") else {
        return (Vec::new(), Vec::new());
    };
    let Ok(next) = decode_dense(reader, blob, dims.len()) else {
        return (Vec::new(), Vec::new());
    };
    let mut added = Vec::new();
    let mut removed = Vec::new();
    for idx in 0..dims.len() {
        let was = current.get(idx).copied().unwrap_or(0.0) > threshold;
        let now = next.get(idx).copied().unwrap_or(0.0) > threshold;
        if !was && now {
            added.push(idx as u32);
        } else if was && !now {
            removed.push(idx as u32);
        }
    }
    (added, removed)
}

fn harvest_frame_data(
    reader: &SqliteReader,
    entry_id: &str,
    step: u32,
    options: &MeshOptions,
) -> Result<FrameData, String> {
    let episode = reader
        .query_row(
            "SELECT * FROM episodes WHERE episode_id = ?1 LIMIT 1",
            &[SqlValue::Text(entry_id.to_string())],
        )
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Episode not found: {entry_id}"))?;
    let dims_arr = parse_resolution(episode.get("resolution"));
    let dims = Dims::new(dims_arr[0], dims_arr[1], dims_arr[2]);
    let total = dims.len();
    let record = reader
        .query_row(
            "SELECT * FROM records WHERE episode_id = ?1 AND step = ?2 LIMIT 1",
            &[
                SqlValue::Text(entry_id.to_string()),
                SqlValue::Integer(step as i64),
            ],
        )
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Step {step} not found for {entry_id}"))?;
    let bc_type = parse_bc_type_text(&episode);
    let load_config = parse_load_config(&episode);
    let (masks, forces) = masks_and_forces(reader, &episode, dims);

    let density = match value_blob(&record, "density_blob") {
        Some(blob) => decode_dense(reader, blob, total)?,
        None => return Err("record has no density_blob".to_string()),
    };
    let rho_cont = match value_blob(&record, "rho_cont_blob") {
        Some(blob) => Some(decode_dense(reader, blob, total)?),
        None => None,
    };
    let displacement = match value_blob(&record, "displacement_blob") {
        Some(blob) => Some(decode_dense(reader, blob, total)?),
        None => None,
    };
    let blueprint = if options.include_blueprint {
        match value_blob(&episode, "v_constructed_blob") {
            Some(blob) => Some(decode_dense(reader, blob, total)?),
            None => None,
        }
    } else {
        None
    };
    let policy_add = match value_blob(&record, "policy_add_blob") {
        Some(blob) => decode_coo(reader, blob, total)?
            .to_dense(total)
            .map_err(|e| e.to_string())?,
        None => vec![0.0; total],
    };
    let policy_remove = match value_blob(&record, "policy_remove_blob") {
        Some(blob) => decode_coo(reader, blob, total)?
            .to_dense(total)
            .map_err(|e| e.to_string())?,
        None => vec![0.0; total],
    };
    let (diff_added, diff_removed) = if options.include_diff_next {
        diff_next(
            reader,
            "records",
            "episode_id",
            entry_id,
            step as i64,
            dims,
            options.density_threshold,
            &density,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let components = record
        .get("reward_components_json")
        .and_then(SqlValue::as_str)
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .unwrap_or_else(|| json!({}));
    let n_islands = components
        .get("n_islands")
        .and_then(Value::as_i64)
        .unwrap_or(1);
    let loose = components
        .get("loose_voxels")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let island_penalty = components
        .get("island_penalty")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let connected_fraction = components
        .get("connected_load_fraction")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let compliance = value_f64(record.get("compliance"));
    let max_disp = value_f64(record.get("max_displacement"));
    let volume = value_f64(record.get("vol_frac")).unwrap_or_else(|| volume_fraction(&density));
    let connected = value_bool(record.get("is_connected")).unwrap_or(false);
    let phase = value_text(record.get("phase")).unwrap_or_else(|| "GROWTH".to_string());
    let value = value_f64(record.get("fitness_score")).unwrap_or(0.0);
    let metadata = base_metadata(
        dims,
        step as i64,
        &phase,
        value,
        volume,
        connected,
        n_islands,
        loose,
        connected_fraction,
        island_penalty,
        compliance,
        max_disp,
        None,
        None,
        components,
        None,
        json!([]),
        &bc_type,
        &load_config,
    );
    let mut data = FrameData::new(dims);
    data.density = density;
    data.rho_cont = rho_cont;
    data.displacement = displacement;
    data.blueprint = blueprint;
    data.policy_add = policy_add;
    data.policy_remove = policy_remove;
    data.diff_added = diff_added;
    data.diff_removed = diff_removed;
    data.bc_masks = masks;
    data.forces = forces;
    data.metadata = metadata;
    Ok(data)
}

fn selfplay_frame_data(
    reader: &SqliteReader,
    entry_id: &str,
    step: u32,
    options: &MeshOptions,
) -> Result<FrameData, String> {
    let game = reader
        .query_row(
            "SELECT * FROM games WHERE game_id = ?1 LIMIT 1",
            &[SqlValue::Text(entry_id.to_string())],
        )
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Game not found: {entry_id}"))?;
    let dims_arr = parse_resolution(game.get("resolution"));
    let dims = Dims::new(dims_arr[0], dims_arr[1], dims_arr[2]);
    let total = dims.len();
    let record = reader
        .query_row(
            "SELECT * FROM game_steps WHERE game_id = ?1 AND step = ?2 LIMIT 1",
            &[
                SqlValue::Text(entry_id.to_string()),
                SqlValue::Integer(step as i64),
            ],
        )
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("Step {step} not found for {entry_id}"))?;
    let bc_type = parse_bc_type_text(&game);
    let load_config = parse_load_config(&game);
    let (masks, forces) = masks_and_forces(reader, &game, dims);
    let density = match value_blob(&record, "density_blob") {
        Some(blob) => decode_dense(reader, blob, total)?,
        None => return Err("game step has no density_blob".to_string()),
    };
    let policy_add = match value_blob(&record, "policy_add_blob") {
        Some(blob) => decode_coo(reader, blob, total)?
            .to_dense(total)
            .map_err(|e| e.to_string())?,
        None => vec![0.0; total],
    };
    let policy_remove = match value_blob(&record, "policy_remove_blob") {
        Some(blob) => decode_coo(reader, blob, total)?
            .to_dense(total)
            .map_err(|e| e.to_string())?,
        None => vec![0.0; total],
    };
    let visits_add = match value_blob(&record, "mcts_visit_add_blob") {
        Some(blob) => decode_coo(reader, blob, total)?
            .to_dense(total)
            .map_err(|e| e.to_string())?,
        None => vec![0.0; total],
    };
    let visits_rem = match value_blob(&record, "mcts_visit_remove_blob") {
        Some(blob) => decode_coo(reader, blob, total)?
            .to_dense(total)
            .map_err(|e| e.to_string())?,
        None => vec![0.0; total],
    };
    let q_add = match value_blob(&record, "mcts_q_add_blob") {
        Some(blob) => decode_coo(reader, blob, total)?
            .to_dense(total)
            .map_err(|e| e.to_string())?,
        None => vec![0.0; total],
    };
    let q_rem = match value_blob(&record, "mcts_q_remove_blob") {
        Some(blob) => decode_coo(reader, blob, total)?
            .to_dense(total)
            .map_err(|e| e.to_string())?,
        None => vec![0.0; total],
    };
    let displacement = match value_blob(&record, "displacement_map_blob") {
        Some(blob) => Some(decode_dense(reader, blob, total)?),
        None => None,
    };
    let (diff_added, diff_removed) = if options.include_diff_next {
        diff_next(
            reader,
            "game_steps",
            "game_id",
            entry_id,
            step as i64,
            dims,
            options.density_threshold,
            &density,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let stats = record
        .get("mcts_stats_json")
        .and_then(SqlValue::as_str)
        .and_then(|text| serde_json::from_str::<Value>(text).ok());
    let actions = record
        .get("selected_actions_json")
        .and_then(SqlValue::as_str)
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .unwrap_or_else(|| json!([]));
    let components = record
        .get("reward_components_json")
        .and_then(SqlValue::as_str)
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .unwrap_or_else(|| json!({}));
    let n_islands = value_i64(record.get("n_islands")).unwrap_or(1);
    let loose = value_i64(record.get("loose_voxels")).unwrap_or(0);
    let island_penalty = value_f64(record.get("island_penalty")).unwrap_or(0.0);
    let connected_fraction = value_f64(record.get("connected_load_fraction")).unwrap_or(0.0);
    let compliance = value_f64(record.get("compliance_fem"));
    let max_disp = value_f64(record.get("max_displacement"));
    let fem_reward = value_f64(record.get("fem_reward"));
    let value_target = value_f64(record.get("value_target"));
    let volume =
        value_f64(record.get("volume_fraction")).unwrap_or_else(|| volume_fraction(&density));
    let connected = value_bool(record.get("is_connected")).unwrap_or(false);
    let phase = value_text(record.get("phase")).unwrap_or_else(|| "GROWTH".to_string());
    let value = value_f64(record.get("value")).unwrap_or(0.0);
    let metadata = base_metadata(
        dims,
        step as i64,
        &phase,
        value,
        volume,
        connected,
        n_islands,
        loose,
        connected_fraction,
        island_penalty,
        compliance,
        max_disp,
        fem_reward,
        value_target,
        components,
        stats,
        actions,
        &bc_type,
        &load_config,
    );
    let mut data = FrameData::new(dims);
    data.density = density;
    data.displacement = displacement;
    data.policy_add = policy_add;
    data.policy_remove = policy_remove;
    data.mcts_visit_add = visits_add;
    data.mcts_visit_remove = visits_rem;
    data.mcts_q_add = q_add;
    data.mcts_q_remove = q_rem;
    data.diff_added = diff_added;
    data.diff_removed = diff_removed;
    data.bc_masks = masks;
    data.forces = forces;
    data.metadata = metadata;
    Ok(data)
}

pub fn frame_data_for_ref(
    source: &DataSourceRef,
    entry_id: &str,
    step: u32,
    options: &MeshOptions,
) -> Result<FrameData, String> {
    let reader = open_reader(source)?;
    let kind = reader.detect_kind().map_err(|e| e.to_string())?;
    match kind {
        CoreDbKind::SelfPlay => selfplay_frame_data(&reader, entry_id, step, options),
        CoreDbKind::HarvestV33 | CoreDbKind::HarvestV2 => {
            harvest_frame_data(&reader, entry_id, step, options)
        }
        CoreDbKind::LegacyV1 => {
            Err("Legacy v1 databases must be migrated to v3.3 before 3D inspection.".to_string())
        }
    }
}

pub fn build_local_frame(
    source: &DataSourceRef,
    entry_id: &str,
    step: u32,
    options: &MeshOptions,
) -> Result<Vec<u8>, String> {
    let data = frame_data_for_ref(source, entry_id, step, options)?;
    build_abmesh(&data, options).map_err(|e| e.to_string())
}

pub fn cache_frame(state: &AppState, key: &str, bytes: Vec<u8>) -> Arc<Vec<u8>> {
    let mut cache = state.cache.lock().unwrap_or_else(|e| e.into_inner());
    cache.insert(key.to_string(), bytes);
    cache.get(key).unwrap_or_else(|| Arc::new(Vec::new()))
}

fn pct(value: &str) -> String {
    let mut out = String::new();
    for byte in value.as_bytes() {
        let ch = *byte as char;
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '~') {
            out.push(ch);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

pub async fn fetch_remote_frame(
    state: &AppState,
    source: &DataSourceRef,
    entry_id: &str,
    step: u32,
    options: &MeshOptions,
) -> Result<Vec<u8>, String> {
    let connection = state.active_connection()?;
    let url = if source.kind == DbKind::Selfplay {
        format!(
            "/api/v1/selfplay/games/{}/steps?db_id={}",
            pct(entry_id),
            pct(&source.db_id)
        )
    } else {
        format!(
            "/api/v1/datasets/{}/episodes/{}/frames?start={}&end={}&include_rho_cont={}",
            pct(&source.db_id),
            pct(entry_id),
            step,
            step + 1,
            options.use_rho_cont
        )
    };
    let response = connection
        .request(reqwest::Method::GET, &url)
        .send()
        .await
        .map_err(|e| format!("Remote frame request failed: {e}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Remote frame request failed: HTTP {}",
            response.status()
        ));
    }
    let bytes = response.bytes().await.map_err(|e| e.to_string())?;
    let frames = decode_step_batch(&bytes).map_err(|e| e.to_string())?;
    let first = frames
        .into_iter()
        .next()
        .ok_or_else(|| "Remote STEP_BATCH is empty".to_string())?;
    frame_from_wire(&first, options).map_err(|e| e.to_string())
}

async fn build_frame(
    state: &AppState,
    source: &DataSourceRef,
    entry_id: &str,
    step: u32,
    options: &MeshOptions,
) -> Result<Vec<u8>, String> {
    let key = default_cache_key(&source.db_id, entry_id, step, options);
    {
        let mut cache = state.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(hit) = cache.get(&key) {
            return Ok((*hit).clone());
        }
    }
    let bytes = if source.origin == "remote" {
        fetch_remote_frame(state, source, entry_id, step, options).await?
    } else {
        let source = source.clone();
        let entry = entry_id.to_string();
        let options = options.clone();
        tauri::async_runtime::spawn_blocking(move || {
            build_local_frame(&source, &entry, step, &options)
        })
        .await
        .map_err(|e| e.to_string())??
    };
    cache_frame(state, &key, bytes.clone());
    Ok(bytes)
}

fn summarize_harvest_light(row: &HashMap<String, SqlValue>, kind: DbKind) -> Option<EntrySummary> {
    let entry_id = value_text(row.get("episode_id"))?;
    let fc = value_f64(row.get("final_compliance"));
    let fv = value_f64(row.get("final_volume"));
    let fdisp = value_f64(row.get("final_max_displacement"));
    let is_conn = value_bool(row.get("final_is_connected"));
    let est_reward = match (fc, fv) {
        (Some(c), Some(v)) => {
            Some(calculate_reward_breakdown(c, v, is_conn.unwrap_or(true), fdisp, 0.0).total_reward)
        }
        _ => None,
    };
    Some(EntrySummary {
        entry_id,
        kind,
        total_steps: 0,
        steps_phase1: 0,
        steps_phase2: 0,
        final_reward: est_reward,
        final_compliance: fc,
        final_volume: fv,
        bc_type: value_text(row.get("bc_type")),
        strategy: value_text(row.get("strategy")),
        neural_engine: None,
        checkpoint_version: value_text(row.get("code_version")),
        seed: value_i64(row.get("seed")),
        is_connected: is_conn,
        created_at: value_text(row.get("created_at")),
    })
}

fn hydrate_harvest_summary(reader: &SqliteReader, item: &mut EntrySummary) {
    let entry_id = item.entry_id.clone();
    let total = reader
        .query_row(
            "SELECT COUNT(*) AS c FROM records WHERE episode_id = ?1",
            &[SqlValue::Text(entry_id.clone())],
        )
        .ok()
        .flatten()
        .and_then(|r| value_i64(r.get("c")))
        .unwrap_or(0);
    let p1 = reader
        .query_row(
            "SELECT COUNT(*) AS c FROM records WHERE episode_id = ?1 AND phase = 'GROWTH'",
            &[SqlValue::Text(entry_id.clone())],
        )
        .ok()
        .flatten()
        .and_then(|r| value_i64(r.get("c")))
        .unwrap_or(0);
    let last = reader
        .query_row(
            "SELECT fitness_score, reward_components_json FROM records WHERE episode_id = ?1 ORDER BY step DESC LIMIT 1",
            &[SqlValue::Text(entry_id)],
        )
        .ok()
        .flatten();
    if let Some(r) = last.as_ref() {
        if let Some(rew) = r
            .get("reward_components_json")
            .and_then(SqlValue::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .and_then(|v| v.get("total").and_then(Value::as_f64))
            .or_else(|| value_f64(r.get("fitness_score")))
        {
            item.final_reward = Some(rew);
        }
    }
    item.total_steps = total;
    item.steps_phase1 = p1;
    item.steps_phase2 = total - p1;
}

fn summarize_game(reader: &SqliteReader, row: &HashMap<String, SqlValue>) -> Option<EntrySummary> {
    let entry_id = value_text(row.get("game_id"))?;
    let total = reader
        .query_row(
            "SELECT COUNT(*) AS c FROM game_steps WHERE game_id = ?1",
            &[SqlValue::Text(entry_id.clone())],
        )
        .ok()
        .flatten()
        .and_then(|r| value_i64(r.get("c")))
        .unwrap_or(0);
    let p1 = reader
        .query_row(
            "SELECT COUNT(*) AS c FROM game_steps WHERE game_id = ?1 AND phase = 'GROWTH'",
            &[SqlValue::Text(entry_id.clone())],
        )
        .ok()
        .flatten()
        .and_then(|r| value_i64(r.get("c")))
        .unwrap_or(0);
    Some(EntrySummary {
        entry_id,
        kind: DbKind::Selfplay,
        total_steps: total,
        steps_phase1: p1,
        steps_phase2: total - p1,
        final_reward: value_f64(row.get("final_score")),
        final_compliance: value_f64(row.get("final_compliance")),
        final_volume: value_f64(row.get("final_volume")),
        bc_type: value_text(row.get("bc_type")),
        strategy: None,
        neural_engine: value_text(row.get("neural_engine")),
        checkpoint_version: value_text(row.get("checkpoint_version")),
        seed: None,
        is_connected: None,
        created_at: value_text(row.get("created_at")),
    })
}

fn summarize_legacy(
    reader: &SqliteReader,
    row: &HashMap<String, SqlValue>,
) -> Option<EntrySummary> {
    let entry_id = value_text(row.get("episode_id"))?;
    let total = reader
        .query_row(
            "SELECT COUNT(*) AS c FROM training_data WHERE episode_id = ?1",
            &[SqlValue::Text(entry_id.clone())],
        )
        .ok()
        .flatten()
        .and_then(|r| value_i64(r.get("c")))
        .unwrap_or(0);
    Some(EntrySummary {
        entry_id,
        kind: DbKind::LegacyV1,
        total_steps: total,
        steps_phase1: 0,
        steps_phase2: 0,
        final_reward: None,
        final_compliance: None,
        final_volume: None,
        bc_type: None,
        strategy: None,
        neural_engine: None,
        checkpoint_version: None,
        seed: None,
        is_connected: None,
        created_at: None,
    })
}

fn list_local_entries(
    source: &DataSourceRef,
    query: &EntryListQuery,
) -> Result<EntryListPage, String> {
    let reader = open_reader(source)?;
    let kind = reader.detect_kind().map_err(|e| e.to_string())?;
    let rows = match kind {
        CoreDbKind::SelfPlay => {
            let select = reader
                .select_with_nulls(
                    "games",
                    &[
                        "game_id",
                        "final_score",
                        "final_compliance",
                        "final_volume",
                        "bc_type",
                        "neural_engine",
                        "checkpoint_version",
                        "created_at",
                    ],
                )
                .map_err(|e| e.to_string())?;
            reader
                .query_rows(&format!("SELECT {select} FROM games"), &[])
                .map_err(|e| e.to_string())?
        }
        CoreDbKind::HarvestV33 | CoreDbKind::HarvestV2 => {
            let select = reader
                .select_with_nulls(
                    "episodes",
                    &[
                        "episode_id",
                        "final_compliance",
                        "final_volume",
                        "final_max_displacement",
                        "final_is_connected",
                        "bc_type",
                        "strategy",
                        "code_version",
                        "seed",
                        "created_at",
                    ],
                )
                .map_err(|e| e.to_string())?;
            reader
                .query_rows(&format!("SELECT {select} FROM episodes"), &[])
                .map_err(|e| e.to_string())?
        }
        CoreDbKind::LegacyV1 => reader
            .query_rows("SELECT DISTINCT episode_id FROM training_data", &[])
            .map_err(|e| e.to_string())?,
    };
    let contract_db_kind = contract_kind(kind);
    let mut items: Vec<EntrySummary> = rows
        .iter()
        .filter_map(|row| match kind {
            CoreDbKind::SelfPlay => summarize_game(&reader, row),
            CoreDbKind::LegacyV1 => summarize_legacy(&reader, row),
            _ => summarize_harvest_light(row, contract_db_kind.clone()),
        })
        .collect();
    if let Some(search) = &query.search {
        let needle = search.to_lowercase();
        items.retain(|item| item.entry_id.to_lowercase().contains(&needle));
    }
    if let Some(strategy) = &query.strategy {
        items.retain(|item| item.strategy.as_deref() == Some(strategy.as_str()));
    }
    if let Some(bc_type) = &query.bc_type {
        items.retain(|item| item.bc_type.as_deref() == Some(bc_type.as_str()));
    }
    if let Some(engine) = &query.engine {
        items.retain(|item| item.neural_engine.as_deref() == Some(engine.as_str()));
    }
    if let Some(version) = &query.version {
        items.retain(|item| item.checkpoint_version.as_deref() == Some(version.as_str()));
    }
    if query.connected_only {
        items.retain(|item| item.is_connected == Some(true));
    }
    match query.sort.as_deref() {
        Some("seed_asc") => items.sort_by_key(|item| item.seed.unwrap_or(0)),
        Some("reward_desc") => items.sort_by(|a, b| {
            b.final_reward
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&a.final_reward.unwrap_or(f64::NEG_INFINITY))
        }),
        Some("compliance_asc") => items.sort_by(|a, b| {
            a.final_compliance
                .unwrap_or(f64::INFINITY)
                .total_cmp(&b.final_compliance.unwrap_or(f64::INFINITY))
        }),
        _ => items.sort_by(|a, b| b.created_at.cmp(&a.created_at)),
    }
    let total_count = items.len();
    let limit = if query.limit == 0 {
        50
    } else {
        query.limit as usize
    };
    let offset = query.offset as usize;
    let mut page: Vec<EntrySummary> = items.into_iter().skip(offset).take(limit).collect();
    if matches!(kind, CoreDbKind::HarvestV33 | CoreDbKind::HarvestV2) {
        for item in &mut page {
            hydrate_harvest_summary(&reader, item);
        }
    }
    Ok(EntryListPage {
        total_count,
        items: page,
    })
}

async fn list_remote_entries(
    state: &AppState,
    source: &DataSourceRef,
    query: &EntryListQuery,
) -> Result<EntryListPage, String> {
    let connection = state.active_connection()?;
    let (path, is_game) = if source.kind == DbKind::Selfplay {
        (
            format!("/api/v1/selfplay/games?db_id={}", pct(&source.db_id)),
            true,
        )
    } else {
        (
            format!("/api/v1/datasets/{}/episodes", pct(&source.db_id)),
            false,
        )
    };
    let mut request = connection.request(reqwest::Method::GET, &path);
    let limit = if query.limit == 0 { 50 } else { query.limit };
    request = request.query(&[("limit", limit), ("offset", query.offset)]);
    if let Some(search) = &query.search {
        request = request.query(&[("search", search)]);
    }
    if let Some(strategy) = &query.strategy {
        request = request.query(&[("strategy", strategy)]);
    }
    if let Some(bc_type) = &query.bc_type {
        request = request.query(&[("bc_type", bc_type)]);
    }
    if let Some(engine) = &query.engine {
        request = request.query(&[("engine", engine)]);
    }
    if let Some(version) = &query.version {
        request = request.query(&[("version", version)]);
    }
    if let Some(sort) = &query.sort {
        request = request.query(&[("sort", sort)]);
    }
    if query.connected_only {
        request = request.query(&[("connected_only", true)]);
    }
    let response = request.send().await.map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("Remote list failed: HTTP {}", response.status()));
    }
    let payload: Value = response.json().await.map_err(|e| e.to_string())?;
    let total_count = payload
        .get("total_count")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let mut items = Vec::new();
    if let Some(list) = payload.get("items").and_then(Value::as_array) {
        for item in list {
            let entry_id = if is_game {
                item.get("game_id").and_then(Value::as_str)
            } else {
                item.get("entry_id").and_then(Value::as_str)
            }
            .unwrap_or_default()
            .to_string();
            items.push(EntrySummary {
                entry_id,
                kind: source.kind.clone(),
                total_steps: item.get("total_steps").and_then(Value::as_i64).unwrap_or(0),
                steps_phase1: item
                    .get("steps_phase1")
                    .and_then(Value::as_i64)
                    .unwrap_or(0),
                steps_phase2: item
                    .get("steps_phase2")
                    .and_then(Value::as_i64)
                    .unwrap_or(0),
                final_reward: item.get("final_score").and_then(Value::as_f64),
                final_compliance: item.get("final_compliance").and_then(Value::as_f64),
                final_volume: item.get("final_volume").and_then(Value::as_f64),
                bc_type: item
                    .get("bc_type")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
                strategy: item
                    .get("strategy")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
                neural_engine: item
                    .get("neural_engine")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
                checkpoint_version: item
                    .get("checkpoint_version")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
                seed: item.get("seed").and_then(Value::as_i64),
                is_connected: item.get("is_connected").and_then(Value::as_bool),
                created_at: item
                    .get("created_at")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
            });
        }
    }
    Ok(EntryListPage { total_count, items })
}

#[allow(clippy::too_many_arguments)]
fn metadata_from_row(
    reader: &SqliteReader,
    row: &HashMap<String, SqlValue>,
    kind: DbKind,
    entry_id: &str,
) -> Result<EntryMetadata, String> {
    let resolution = parse_resolution(row.get("resolution"));
    let dims = Dims::new(resolution[0], resolution[1], resolution[2]);
    let load_config = parse_load_config(row);
    let bc_type = parse_bc_type_text(row);
    let (masks, forces) = masks_and_forces(reader, row, dims);
    let (phase_history, value_history, volume_history, compliance_history, refutation_steps) =
        if kind == DbKind::Selfplay {
            let rows = reader
                .query_rows(
                    "SELECT step, phase, value, volume_fraction, compliance_fem, mcts_stats_json FROM game_steps WHERE game_id = ?1 ORDER BY step",
                    &[SqlValue::Text(entry_id.to_string())],
                )
                .map_err(|e| e.to_string())?;
            let mut phases = Vec::new();
            let mut values = Vec::new();
            let mut volumes = Vec::new();
            let mut compliances = Vec::new();
            let mut refutations = Vec::new();
            for item in &rows {
                phases.push(value_text(item.get("phase")).unwrap_or_else(|| "GROWTH".to_string()));
                values.push(value_f64(item.get("value")).unwrap_or(0.0));
                volumes.push(value_f64(item.get("volume_fraction")));
                compliances.push(value_f64(item.get("compliance_fem")));
                if let Some(stats) = item
                    .get("mcts_stats_json")
                    .and_then(SqlValue::as_str)
                    .and_then(|text| serde_json::from_str::<Value>(text).ok())
                {
                    if stats
                        .get("refutation")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                    {
                        refutations.push(value_i64(item.get("step")).unwrap_or(0));
                    }
                }
            }
            (phases, values, volumes, compliances, refutations)
        } else {
            let rows = reader
                .query_rows(
                    "SELECT step, phase, fitness_score, vol_frac, compliance FROM records WHERE episode_id = ?1 ORDER BY step",
                    &[SqlValue::Text(entry_id.to_string())],
                )
                .map_err(|e| e.to_string())?;
            let mut phases = Vec::new();
            let mut values = Vec::new();
            let mut volumes = Vec::new();
            let mut compliances = Vec::new();
            for item in &rows {
                phases.push(value_text(item.get("phase")).unwrap_or_else(|| "GROWTH".to_string()));
                values.push(value_f64(item.get("fitness_score")).unwrap_or(0.0));
                volumes.push(value_f64(item.get("vol_frac")));
                compliances.push(value_f64(item.get("compliance")));
            }
            (phases, values, volumes, compliances, Vec::new())
        };
    let steps_phase1 = phase_history
        .iter()
        .filter(|p| p.as_str() == "GROWTH")
        .count() as i64;
    let total_steps = phase_history.len() as i64;
    let (final_score, final_compliance, final_volume, final_max_disp, real_scale, stiffness) =
        if kind == DbKind::Selfplay {
            (
                value_f64(row.get("final_score")),
                value_f64(row.get("final_compliance")),
                value_f64(row.get("final_volume")),
                None,
                value_f64(row.get("initial_cantilever_problem_real_scale_factor"))
                    .unwrap_or(500000.0),
                value_f64(row.get("stiffness_E")).unwrap_or(200e9),
            )
        } else {
            (
                None,
                value_f64(row.get("final_compliance")),
                value_f64(row.get("final_volume")),
                value_f64(row.get("final_max_displacement")),
                500000.0,
                200e9,
            )
        };
    let has_blueprint = value_blob(row, "v_constructed_blob").is_some();
    let has_rho_cont = if kind == DbKind::Selfplay {
        false
    } else {
        reader
            .query_row(
                "SELECT COUNT(*) AS c FROM records WHERE episode_id = ?1 AND rho_cont_blob IS NOT NULL",
                &[SqlValue::Text(entry_id.to_string())],
            )
            .ok()
            .flatten()
            .and_then(|r| value_i64(r.get("c")))
            .unwrap_or(0)
            > 0
    };
    let has_displacement = reader
        .query_row(
            if kind == DbKind::Selfplay {
                "SELECT COUNT(*) AS c FROM game_steps WHERE game_id = ?1 AND displacement_map_blob IS NOT NULL"
            } else {
                "SELECT COUNT(*) AS c FROM records WHERE episode_id = ?1 AND displacement_blob IS NOT NULL"
            },
            &[SqlValue::Text(entry_id.to_string())],
        )
        .ok()
        .flatten()
        .and_then(|r| value_i64(r.get("c")))
        .unwrap_or(0)
        > 0;
    Ok(EntryMetadata {
        entry_id: entry_id.to_string(),
        kind,
        resolution,
        bc_type,
        strategy: value_text(row.get("strategy")),
        neural_engine: value_text(row.get("neural_engine")),
        checkpoint_version: value_text(row.get("checkpoint_version"))
            .or_else(|| value_text(row.get("code_version"))),
        seed: value_i64(row.get("seed")),
        total_steps,
        steps_phase1,
        steps_phase2: total_steps - steps_phase1,
        final_score,
        final_compliance,
        final_volume,
        final_max_displacement: final_max_disp,
        real_scale_factor: real_scale,
        stiffness_e: stiffness,
        load_config,
        has_blueprint,
        has_rho_cont,
        has_displacement,
        supports: supports_from_masks(dims, &masks),
        forces: forces_from_arrays(dims, &forces),
        phase_history,
        value_history,
        volume_history,
        compliance_history,
        refutation_steps,
    })
}

fn get_local_metadata(source: &DataSourceRef, entry_id: &str) -> Result<EntryMetadata, String> {
    let reader = open_reader(source)?;
    let kind = reader.detect_kind().map_err(|e| e.to_string())?;
    let (row, contract) = match kind {
        CoreDbKind::SelfPlay => (
            reader
                .query_row(
                    "SELECT * FROM games WHERE game_id = ?1 LIMIT 1",
                    &[SqlValue::Text(entry_id.to_string())],
                )
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("Game not found: {entry_id}"))?,
            DbKind::Selfplay,
        ),
        CoreDbKind::HarvestV33 | CoreDbKind::HarvestV2 => (
            reader
                .query_row(
                    "SELECT * FROM episodes WHERE episode_id = ?1 LIMIT 1",
                    &[SqlValue::Text(entry_id.to_string())],
                )
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("Episode not found: {entry_id}"))?,
            contract_kind(kind),
        ),
        CoreDbKind::LegacyV1 => {
            return Err("Legacy v1 database: migrate to v3.3 for metadata".to_string())
        }
    };
    metadata_from_row(&reader, &row, contract, entry_id)
}

fn scan_databases(root: &Path, depth: u32, found: &mut HashSet<String>) {
    if depth > 3 || !root.is_dir() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if matches!(
                name,
                "target" | "node_modules" | ".git" | ".venv" | "__pycache__"
            ) {
                continue;
            }
            scan_databases(&path, depth + 1, found);
        } else if path.extension().and_then(|s| s.to_str()) == Some("db") {
            if let Ok(reader) = SqliteReader::open(&path) {
                if reader.detect_kind().is_ok() {
                    if let Ok(info) = reader.describe() {
                        found.insert(serde_json::to_string(&info).unwrap_or_default());
                    }
                }
            }
        }
    }
}

#[tauri::command(rename_all = "snake_case")]
pub async fn list_local_databases(
    state: State<'_, AppState>,
    extra_dirs: Vec<String>,
) -> Result<Vec<ab_core::sqlite_reader::DatabaseInfo>, String> {
    let mut found = HashSet::new();
    if let Ok(current) = std::env::current_dir() {
        scan_databases(&current, 0, &mut found);
    }
    for dir in extra_dirs {
        scan_databases(Path::new(&dir), 0, &mut found);
    }
    if let Ok(databases) = state.databases.lock() {
        for info in databases.values() {
            found.insert(serde_json::to_string(info).unwrap_or_default());
        }
    }
    let mut items: Vec<_> = found
        .into_iter()
        .filter_map(|text| serde_json::from_str(&text).ok())
        .collect();
    items.sort_by(|a: &ab_core::sqlite_reader::DatabaseInfo, b| a.name.cmp(&b.name));
    Ok(items)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn open_database_file(
    state: State<'_, AppState>,
    path: String,
) -> Result<ab_core::sqlite_reader::DatabaseInfo, String> {
    let reader = SqliteReader::open(&path).map_err(|e| e.to_string())?;
    let info = reader.describe().map_err(|e| e.to_string())?;
    let mut databases = state.databases.lock().map_err(|e| e.to_string())?;
    databases.insert(info.db_id.clone(), info.clone());
    Ok(info)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn list_entries(
    state: State<'_, AppState>,
    source: DataSourceRef,
    query: EntryListQuery,
) -> Result<EntryListPage, String> {
    if source.origin == "remote" {
        list_remote_entries(&state, &source, &query).await
    } else {
        list_local_entries(&source, &query)
    }
}

#[tauri::command(rename_all = "snake_case")]
pub async fn get_entry_metadata(
    state: State<'_, AppState>,
    source: DataSourceRef,
    entry_id: String,
) -> Result<EntryMetadata, String> {
    if source.origin == "remote" {
        let connection = state.active_connection()?;
        let path = if source.kind == DbKind::Selfplay {
            format!(
                "/api/v1/selfplay/games/{}/metadata?db_id={}",
                pct(&entry_id),
                pct(&source.db_id)
            )
        } else {
            format!(
                "/api/v1/datasets/{}/episodes/{}/metadata",
                pct(&source.db_id),
                pct(&entry_id)
            )
        };
        let response = connection
            .request(reqwest::Method::GET, &path)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!(
                "Remote metadata failed: HTTP {}",
                response.status()
            ));
        }
        return response.json().await.map_err(|e| e.to_string());
    }
    get_local_metadata(&source, &entry_id)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn get_frame_mesh(
    state: State<'_, AppState>,
    source: DataSourceRef,
    entry_id: String,
    step: u32,
    options: MeshOptions,
) -> Result<Response, String> {
    let bytes = build_frame(&state, &source, &entry_id, step, &options).await?;
    Ok(Response::new(bytes))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn prefetch_frames(
    state: State<'_, AppState>,
    source: DataSourceRef,
    entry_id: String,
    start: u32,
    count: u32,
    options: MeshOptions,
) -> Result<(), String> {
    for offset in 0..count {
        let step = start + offset;
        let key = default_cache_key(&source.db_id, &entry_id, step, &options);
        let cached = {
            let mut cache = state.cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.get(&key).is_some()
        };
        if cached {
            continue;
        }
        let bytes = build_frame(&state, &source, &entry_id, step, &options).await?;
        cache_frame(&state, &key, bytes);
    }
    Ok(())
}

async fn load_comparison_frame(
    state: &AppState,
    frame: &FrameRef,
    options: &MeshOptions,
) -> Result<(Vec<f32>, Value), String> {
    if frame.source.origin == "remote" {
        let connection = state.active_connection()?;
        let path = if frame.source.kind == DbKind::Selfplay {
            format!(
                "/api/v1/selfplay/games/{}/steps?db_id={}&start={}&end={}",
                pct(&frame.entry_id),
                pct(&frame.source.db_id),
                frame.step,
                frame.step + 1
            )
        } else {
            format!(
                "/api/v1/datasets/{}/episodes/{}/frames?start={}&end={}",
                pct(&frame.source.db_id),
                pct(&frame.entry_id),
                frame.step,
                frame.step + 1
            )
        };
        let response = connection
            .request(reqwest::Method::GET, &path)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!("Remote frame failed: HTTP {}", response.status()));
        }
        let bytes = response.bytes().await.map_err(|e| e.to_string())?;
        let frames = decode_step_batch(&bytes).map_err(|e| e.to_string())?;
        let packet = frames
            .into_iter()
            .next()
            .ok_or_else(|| "empty frame batch".to_string())?;
        let metadata =
            serde_json::from_str::<Value>(&packet.header_json).map_err(|e| e.to_string())?;
        let density = packet
            .tensor(ab_core::codecs::wire::tensor_tag::DENSITY)
            .ok_or_else(|| "missing density".to_string())?
            .to_dense_f32()
            .map_err(|e| e.to_string())?;
        Ok((density, metadata))
    } else {
        let source = frame.source.clone();
        let entry = frame.entry_id.clone();
        let step = frame.step;
        let options = options.clone();
        tauri::async_runtime::spawn_blocking(move || {
            let data = frame_data_for_ref(&source, &entry, step, &options)?;
            Ok::<(Vec<f32>, Value), String>((data.density, data.metadata))
        })
        .await
        .map_err(|e| e.to_string())?
    }
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_ab_comparison(
    state: State<'_, AppState>,
    a: FrameRef,
    b: FrameRef,
) -> Result<ComparisonMetrics, String> {
    let options = MeshOptions::default();
    let app_state = state.inner();
    let (density_a, meta_a) = load_comparison_frame(app_state, &a, &options).await?;
    let (density_b, meta_b) = load_comparison_frame(app_state, &b, &options).await?;
    let iou = ab_core::topology::compute_iou_3d(&density_a, &density_b, 0.5);
    let compliance_a = meta_a.get("compliance_fem").and_then(Value::as_f64);
    let compliance_b = meta_b.get("compliance_fem").and_then(Value::as_f64);
    let volume_a = meta_a.get("volume_fraction").and_then(Value::as_f64);
    let volume_b = meta_b.get("volume_fraction").and_then(Value::as_f64);
    let reward_a = meta_a
        .get("reward_v33")
        .and_then(|v| v.get("total_reward"))
        .and_then(Value::as_f64);
    let reward_b = meta_b
        .get("reward_v33")
        .and_then(|v| v.get("total_reward"))
        .and_then(Value::as_f64);
    let disp_a = meta_a.get("max_displacement").and_then(Value::as_f64);
    let disp_b = meta_b.get("max_displacement").and_then(Value::as_f64);
    let delta_compliance = match (compliance_a, compliance_b) {
        (Some(a), Some(b)) => Some(a - b),
        _ => None,
    };
    let delta_compliance_pct = match (compliance_a, delta_compliance) {
        (Some(a), Some(delta)) if a.abs() > 1e-12 => Some(delta / a * 100.0),
        _ => None,
    };
    Ok(ComparisonMetrics {
        iou_3d: iou,
        delta_compliance,
        delta_compliance_pct,
        delta_volume: match (volume_a, volume_b) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        },
        delta_reward: match (reward_a, reward_b) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        },
        delta_max_displacement: match (disp_a, disp_b) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        },
        delta_steps: meta_a.get("step").and_then(Value::as_i64).unwrap_or(0)
            - meta_b.get("step").and_then(Value::as_i64).unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ab_core::codecs::coo1::Coo1;
    use ab_core::codecs::npy::{encode_npy, NpyDtype};
    use ab_core::codecs::wire::AbMeshPacket;
    use ab_core::codecs::zlib_compress;
    use rusqlite::Connection;

    fn make_v33_db(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let total = 16_usize;
        let conn = Connection::open(path)?;
        conn.execute_batch(
            r#"
            CREATE TABLE episodes (
                episode_id TEXT PRIMARY KEY, seed INTEGER, shard_id TEXT, worker_id TEXT,
                code_version TEXT, bc_masks_blob BLOB NOT NULL, forces_blob BLOB NOT NULL,
                v_constructed_blob BLOB, load_config TEXT NOT NULL, bc_type TEXT NOT NULL,
                strategy TEXT NOT NULL, resolution TEXT NOT NULL, final_compliance REAL,
                final_volume REAL, final_max_displacement REAL, final_is_connected INTEGER,
                simp_iters INTEGER, duration_sec REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE records (
                id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL, step INTEGER NOT NULL,
                phase TEXT NOT NULL, density_blob BLOB NOT NULL, rho_cont_blob BLOB,
                policy_add_blob BLOB, policy_remove_blob BLOB, fitness_score REAL NOT NULL,
                compliance REAL, vol_frac REAL, max_displacement REAL, beta INTEGER,
                is_final_step INTEGER DEFAULT 0, is_connected INTEGER DEFAULT 0,
                displacement_blob BLOB, reward_components_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(episode_id, step)
            );
            "#,
        )?;
        let masks = vec![0.0_f32; total * 3];
        let forces = vec![0.0_f32; total * 3];
        let blueprint = vec![1.0_f32; total];
        let density = vec![1.0_f32; total];
        let displacement = vec![2.0_f32; total];
        let policy_add = Coo1 {
            indices: vec![0, 3],
            values: vec![0.5, 0.25],
        };
        let policy_rem = Coo1 {
            indices: vec![7],
            values: vec![0.125],
        };
        let npy = |values: &[f32], shape: &[usize], total: usize| {
            assert_eq!(values.len(), total);
            zlib_compress(&encode_npy(values, shape, NpyDtype::F32).unwrap()).unwrap()
        };
        conn.execute(
            "INSERT INTO episodes (episode_id, seed, code_version, bc_masks_blob, forces_blob, v_constructed_blob, load_config, bc_type, strategy, resolution, final_compliance, final_volume, final_max_displacement, final_is_connected)
             VALUES ('ep-1', 42, 'v3.3', ?1, ?2, ?3, ?4, 'FULL_CLAMP', 'FULL_DOMAIN', '[4, 2, 2]', 312.4, 0.5, 41.2, 1)",
            rusqlite::params![
                npy(&masks, &[3, 4, 2, 2], total * 3),
                npy(&forces, &[3, 4, 2, 2], total * 3),
                npy(&blueprint, &[4, 2, 2], total),
                "{\"x\":3,\"y\":1,\"z_start\":0,\"z_end\":2}"
            ],
        )?;
        for step in 0..2 {
            let mut step_density = density.clone();
            if step == 1 {
                step_density[3] = 0.0;
            }
            conn.execute(
                "INSERT INTO records (episode_id, step, phase, density_blob, rho_cont_blob, policy_add_blob, policy_remove_blob, fitness_score, compliance, vol_frac, max_displacement, is_connected, displacement_blob, reward_components_json)
                 VALUES ('ep-1', ?1, ?2, ?3, ?4, ?5, ?6, ?7, 312.4, 0.5, 41.2, 1, ?8, ?9)",
                rusqlite::params![
                    step,
                    if step == 0 { "GROWTH" } else { "REFINEMENT" },
                    npy(&step_density, &[4, 2, 2], total),
                    npy(&step_density, &[4, 2, 2], total),
                    zlib_compress(&policy_add.encode(total)?).unwrap(),
                    zlib_compress(&policy_rem.encode(total)?).unwrap(),
                    0.42,
                    npy(&displacement, &[4, 2, 2], total),
                    "{\"n_islands\":1,\"loose_voxels\":0,\"island_penalty\":0.0,\"connected_load_fraction\":1.0,\"total\":0.55}"
                ],
            )?;
        }
        Ok(())
    }

    #[test]
    fn builds_abmesh_from_synthetic_v33_harvest() {
        let path = std::env::temp_dir().join(format!(
            "ab-studio-frame-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        make_v33_db(&path).expect("fixture db");
        let source = DataSourceRef {
            origin: "local".to_string(),
            db_id: path.display().to_string(),
            kind: DbKind::HarvestV33,
        };
        let options = MeshOptions::default();
        let bytes = build_local_frame(&source, "ep-1", 0, &options).expect("frame");
        let packet = AbMeshPacket::decode(&bytes).expect("ABMESH01 decode");
        assert_eq!(packet.dims, Dims::new(4, 2, 2));
        assert_eq!(packet.step_index, 0);
        assert!(packet.flags & ab_core::codecs::wire::abmesh_flags::POLICY != 0);
        assert!(packet.flags & ab_core::codecs::wire::abmesh_flags::DIFF != 0);
        assert!(packet.flags & ab_core::codecs::wire::abmesh_flags::BLUEPRINT != 0);
        assert!(packet.flags & ab_core::codecs::wire::abmesh_flags::DISPLACEMENT != 0);
        assert!(!packet.indices.is_empty());
        assert_eq!(packet.policy_add.len(), 2);
        assert_eq!(packet.policy_rem.len(), 1);
        assert!(!packet.diff_removed.is_empty());
        let meta: Value = serde_json::from_str(&packet.meta_json).unwrap();
        assert_eq!(meta["step"], 0);
        assert_eq!(meta["phase"], "GROWTH");

        let data = frame_data_for_ref(&source, "ep-1", 1, &options).expect("frame data");
        assert_eq!(data.metadata["phase"], "REFINEMENT");
        assert!(data.blueprint.is_some());
        assert!(data.displacement.is_some());
        let _ = std::fs::remove_file(path);
    }
}
