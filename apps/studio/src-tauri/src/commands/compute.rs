//! Compute profiles, sidecar lifecycle, model proxies and live sessions.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Instant;

use ab_core::codecs::wire::WirePacket;
use serde_json::{json, Value};
use tauri::ipc::Response;
use tauri::{AppHandle, Emitter, State};

use crate::commands::frame::{build_abmesh, FrameData};
use crate::contracts::MeshOptions;
use crate::contracts::{
    ComputeConnectionState, ComputeProfile, FemEvalRequest, HarvestClusterStatus, InferRequest,
    LoadModelRequest, LoadedModelInfo, ModelSummary, SessionCommandPayload, SessionHandle,
    SidecarStatus, StartSessionRequest,
};
use crate::state::{ActiveConnection, AppState, SidecarProcess};
use crate::ws_bridge;

fn find_profile(state: &AppState, profile_id: &str) -> Result<ComputeProfile, String> {
    state
        .profiles
        .lock()
        .map_err(|e| e.to_string())?
        .iter()
        .find(|profile| profile.id == profile_id)
        .cloned()
        .ok_or_else(|| format!("Perfil não encontrado: {profile_id}"))
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

fn build_request(
    client: &reqwest::Client,
    profile: &ComputeProfile,
    method: reqwest::Method,
    path: &str,
) -> reqwest::RequestBuilder {
    let url = format!("{}{}", profile.base_url.trim_end_matches('/'), path);
    let builder = client.request(method, url);
    if profile.api_token.is_empty() {
        builder
    } else {
        builder.bearer_auth(&profile.api_token)
    }
}

async fn ensure_success(response: reqwest::Response) -> Result<reqwest::Response, String> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    let message = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(|s| s.to_string())
        })
        .unwrap_or(text);
    Err(format!("HTTP {status}: {message}"))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_list_profiles(
    state: State<'_, AppState>,
) -> Result<Vec<ComputeProfile>, String> {
    Ok(state.profiles.lock().map_err(|e| e.to_string())?.clone())
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_save_profile(
    state: State<'_, AppState>,
    profile: ComputeProfile,
) -> Result<ComputeProfile, String> {
    {
        let mut profiles = state.profiles.lock().map_err(|e| e.to_string())?;
        if let Some(existing) = profiles.iter_mut().find(|item| item.id == profile.id) {
            *existing = profile.clone();
        } else {
            profiles.push(profile.clone());
        }
    }
    state.save_profiles()?;
    Ok(profile)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_delete_profile(
    state: State<'_, AppState>,
    profile_id: String,
) -> Result<(), String> {
    {
        let mut profiles = state.profiles.lock().map_err(|e| e.to_string())?;
        profiles.retain(|profile| profile.id != profile_id);
    }
    state.save_profiles()
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_activate_profile(
    app: AppHandle,
    state: State<'_, AppState>,
    profile_id: String,
) -> Result<ComputeConnectionState, String> {
    let profile = find_profile(&state, &profile_id)?;
    let client = client();
    let started = Instant::now();
    let health = build_request(&client, &profile, reqwest::Method::GET, "/api/v1/health")
        .send()
        .await;
    let health = match health {
        Ok(response) => ensure_success(response).await,
        Err(err) => Err(format!("Sem conexão com {}: {err}", profile.base_url)),
    };
    match health {
        Ok(_) => {}
        Err(error) => {
            let state_payload = ComputeConnectionState {
                profile_id,
                connected: false,
                rtt_ms: 0.0,
                capabilities: None,
                active_model: None,
                error: Some(error),
            };
            let _ = app.emit("ab://compute-status", &state_payload);
            return Ok(state_payload);
        }
    }
    let capabilities = match build_request(
        &client,
        &profile,
        reqwest::Method::GET,
        "/api/v1/capabilities",
    )
    .send()
    .await
    {
        Ok(response) => match ensure_success(response).await {
            Ok(response) => response.json::<Value>().await.map_err(|e| e.to_string()),
            Err(error) => Err(error),
        },
        Err(error) => Err(error.to_string()),
    };
    let capabilities = match capabilities {
        Ok(value) => value,
        Err(error) => {
            let state_payload = ComputeConnectionState {
                profile_id,
                connected: false,
                rtt_ms: 0.0,
                capabilities: None,
                active_model: None,
                error: Some(error),
            };
            let _ = app.emit("ab://compute-status", &state_payload);
            return Ok(state_payload);
        }
    };
    let rtt_ms = started.elapsed().as_secs_f64() * 1000.0;
    let active_model = capabilities.get("active_model").cloned();
    {
        let mut active = state.active.lock().map_err(|e| e.to_string())?;
        *active = Some(ActiveConnection {
            profile: profile.clone(),
            client: client.clone(),
            capabilities: capabilities.clone(),
            rtt_ms,
        });
    }
    let result = ComputeConnectionState {
        profile_id,
        connected: true,
        rtt_ms,
        capabilities: Some(capabilities),
        active_model,
        error: None,
    };
    let _ = app.emit("ab://compute-status", &result);
    Ok(result)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_sidecar_control(
    app: AppHandle,
    state: State<'_, AppState>,
    action: String,
    python_bin: Option<String>,
    port: u16,
) -> Result<SidecarStatus, String> {
    match action.as_str() {
        "start" => {
            {
                let mut slot = state.sidecar.lock().map_err(|e| e.to_string())?;
                if let Some(process) = slot.as_mut() {
                    if process
                        .child
                        .try_wait()
                        .map_err(|e| e.to_string())?
                        .is_none()
                    {
                        return Ok(SidecarStatus {
                            running: true,
                            pid: Some(process.child.id()),
                            port: process.port,
                            message: "Sidecar já em execução.".to_string(),
                        });
                    }
                }
            }
            let python = python_bin.unwrap_or_else(|| "python3".to_string());
            let mut child = Command::new(&python)
                .args([
                    "-m",
                    "alphabuilder.server",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    &port.to_string(),
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("Falha ao iniciar {python}: {e}"))?;
            if let Some(stdout) = child.stdout.take() {
                let handle = app.clone();
                std::thread::spawn(move || {
                    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                        let _ =
                            handle.emit("ab://sidecar-log", json!({"level": "info", "line": line}));
                    }
                });
            }
            if let Some(stderr) = child.stderr.take() {
                let handle = app.clone();
                std::thread::spawn(move || {
                    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                        let _ = handle
                            .emit("ab://sidecar-log", json!({"level": "error", "line": line}));
                    }
                });
            }
            let pid = child.id();
            let mut slot = state.sidecar.lock().map_err(|e| e.to_string())?;
            *slot = Some(SidecarProcess { child, port });
            Ok(SidecarStatus {
                running: true,
                pid: Some(pid),
                port,
                message: "Sidecar iniciado.".to_string(),
            })
        }
        "stop" => {
            let mut slot = state.sidecar.lock().map_err(|e| e.to_string())?;
            if let Some(mut process) = slot.take() {
                let _ = process.child.kill();
                let _ = process.child.wait();
                Ok(SidecarStatus {
                    running: false,
                    pid: None,
                    port: process.port,
                    message: "Sidecar encerrado.".to_string(),
                })
            } else {
                Ok(SidecarStatus {
                    running: false,
                    pid: None,
                    port,
                    message: "Nenhum sidecar em execução.".to_string(),
                })
            }
        }
        _ => {
            let mut slot = state.sidecar.lock().map_err(|e| e.to_string())?;
            let status = match slot.as_mut() {
                Some(process) => match process.child.try_wait().map_err(|e| e.to_string())? {
                    None => SidecarStatus {
                        running: true,
                        pid: Some(process.child.id()),
                        port: process.port,
                        message: "Sidecar em execução.".to_string(),
                    },
                    Some(status) => {
                        let code = status.code().unwrap_or(-1);
                        *slot = None;
                        SidecarStatus {
                            running: false,
                            pid: None,
                            port,
                            message: format!("Sidecar terminou com código {code}."),
                        }
                    }
                },
                None => SidecarStatus {
                    running: false,
                    pid: None,
                    port,
                    message: "Nenhum sidecar em execução.".to_string(),
                },
            };
            Ok(status)
        }
    }
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_list_models(state: State<'_, AppState>) -> Result<Vec<ModelSummary>, String> {
    let connection = state.active_connection()?;
    let response = ensure_success(
        connection
            .request(reqwest::Method::GET, "/api/v1/models")
            .send()
            .await
            .map_err(|e| e.to_string())?,
    )
    .await?;
    let payload: Value = response.json().await.map_err(|e| e.to_string())?;
    let models = payload
        .get("models")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    models
        .into_iter()
        .map(|item| {
            Ok(ModelSummary {
                model_id: item
                    .get("model_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                size_mb: item.get("size_mb").and_then(Value::as_f64).unwrap_or(0.0),
                modified_at: item
                    .get("modified_at")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
                config_preview: item
                    .get("config_preview")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
            })
        })
        .collect()
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_load_model(
    app: AppHandle,
    state: State<'_, AppState>,
    req: LoadModelRequest,
) -> Result<LoadedModelInfo, String> {
    let connection = state.active_connection()?;
    let body = json!({
        "model_id": req.model_id,
        "device": req.device,
        "openvino_hint": req.openvino_hint
    });
    let response = ensure_success(
        connection
            .request(reqwest::Method::POST, "/api/v1/models/load")
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?,
    )
    .await?;
    let info: LoadedModelInfo = response.json().await.map_err(|e| e.to_string())?;
    let active_value = serde_json::to_value(&info).map_err(|e| e.to_string())?;
    let payload = {
        let mut active = state.active.lock().map_err(|e| e.to_string())?;
        if let Some(conn) = active.as_mut() {
            if let Some(map) = conn.capabilities.as_object_mut() {
                map.insert("active_model".to_string(), active_value.clone());
            }
            Some(ComputeConnectionState {
                profile_id: conn.profile.id.clone(),
                connected: true,
                rtt_ms: conn.rtt_ms,
                capabilities: Some(conn.capabilities.clone()),
                active_model: Some(active_value),
                error: None,
            })
        } else {
            None
        }
    };
    if let Some(payload) = payload {
        let _ = app.emit("ab://compute-status", &payload);
    }
    Ok(info)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_unload_model(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let connection = state.active_connection()?;
    ensure_success(
        connection
            .request(reqwest::Method::POST, "/api/v1/models/unload")
            .send()
            .await
            .map_err(|e| e.to_string())?,
    )
    .await?;
    let payload = {
        let mut active = state.active.lock().map_err(|e| e.to_string())?;
        if let Some(conn) = active.as_mut() {
            if let Some(map) = conn.capabilities.as_object_mut() {
                map.insert("active_model".to_string(), Value::Null);
            }
            Some(ComputeConnectionState {
                profile_id: conn.profile.id.clone(),
                connected: true,
                rtt_ms: conn.rtt_ms,
                capabilities: Some(conn.capabilities.clone()),
                active_model: None,
                error: None,
            })
        } else {
            None
        }
    };
    if let Some(payload) = payload {
        let _ = app.emit("ab://compute-status", &payload);
    }
    Ok(())
}

fn studio_state_from_infer(req: &InferRequest) -> crate::contracts::StudioVoxelState {
    crate::contracts::StudioVoxelState {
        resolution: req.resolution,
        bc_type: req.bc_type.clone(),
        load_config: req.load_config.clone(),
        custom_supports: req.custom_supports.clone(),
        custom_forces: req.custom_forces.clone(),
        solid_voxels: req.solid_voxels.clone(),
    }
}

fn frame_from_state(req: &InferRequest, metadata: Value) -> Result<FrameData, String> {
    let studio = studio_state_from_infer(req);
    let mut data = FrameData::empty_studio(&studio).map_err(|e| e.to_string())?;
    data.metadata = metadata;
    Ok(data)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_infer_xray(
    state: State<'_, AppState>,
    req: InferRequest,
) -> Result<Response, String> {
    let connection = state.active_connection()?;
    let body = json!({
        "resolution": req.resolution,
        "bc_type": req.bc_type,
        "load_config": req.load_config,
        "custom_supports": req.custom_supports,
        "custom_forces": req.custom_forces,
        "solid_voxels": req.solid_voxels,
        "density_f32_b64_zlib": null,
        "include_blueprint": req.include_blueprint,
        "policy_mode": req.policy_mode,
        "policy_temperature": req.policy_temperature,
        "policy_top_k": req.policy_top_k,
        "response_format": "abwire01"
    });
    let response = ensure_success(
        connection
            .request(reqwest::Method::POST, "/api/v1/infer")
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?,
    )
    .await?;
    let bytes = response.bytes().await.map_err(|e| e.to_string())?;
    let packet = WirePacket::decode(&bytes).map_err(|e| e.to_string())?;
    let metadata: Value = serde_json::from_str(&packet.header_json).map_err(|e| e.to_string())?;
    let mut data = frame_from_state(&req, metadata)?;
    let total = data.dims.len();
    if let Some(tensor) = packet.tensor(ab_core::codecs::wire::tensor_tag::POLICY_ADD) {
        data.policy_add = tensor.to_dense_f32().map_err(|e| e.to_string())?;
    }
    if let Some(tensor) = packet.tensor(ab_core::codecs::wire::tensor_tag::POLICY_REMOVE) {
        data.policy_remove = tensor.to_dense_f32().map_err(|e| e.to_string())?;
    }
    if let Some(tensor) = packet.tensor(ab_core::codecs::wire::tensor_tag::BLUEPRINT_PROBS) {
        let blueprint = tensor.to_dense_f32().map_err(|e| e.to_string())?;
        if blueprint.len() == total {
            data.blueprint = Some(blueprint);
        }
    }
    let bytes = build_abmesh(&data, &MeshOptions::default()).map_err(|e| e.to_string())?;
    Ok(Response::new(bytes))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_evaluate_fem(
    state: State<'_, AppState>,
    req: FemEvalRequest,
) -> Result<Response, String> {
    let connection = state.active_connection()?;
    let body = json!({
        "resolution": req.resolution,
        "bc_type": req.bc_type,
        "load_config": req.load_config,
        "custom_supports": req.custom_supports,
        "custom_forces": req.custom_forces,
        "solid_voxels": req.solid_voxels,
        "E": req.e,
        "nu": req.nu,
        "force_magnitude": req.force_magnitude,
        "include_blueprint": false,
        "include_displacement_map": req.include_displacement_map,
        "include_displacement_vector": req.include_displacement_vector,
        "policy_mode": "masked_probs",
        "policy_temperature": 1.0,
        "policy_top_k": 1,
        "response_format": "abwire01"
    });
    let response = ensure_success(
        connection
            .request(reqwest::Method::POST, "/api/v1/fem/evaluate")
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?,
    )
    .await?;
    let bytes = response.bytes().await.map_err(|e| e.to_string())?;
    let packet = WirePacket::decode(&bytes).map_err(|e| e.to_string())?;
    let metadata: Value = serde_json::from_str(&packet.header_json).map_err(|e| e.to_string())?;
    let infer = InferRequest {
        resolution: req.resolution,
        bc_type: req.bc_type,
        load_config: req.load_config,
        custom_supports: req.custom_supports,
        custom_forces: req.custom_forces,
        solid_voxels: req.solid_voxels,
        include_blueprint: false,
        policy_mode: "masked_probs".to_string(),
        policy_temperature: 1.0,
        policy_top_k: 1,
    };
    let mut data = frame_from_state(&infer, metadata)?;
    let total = data.dims.len();
    if let Some(tensor) = packet.tensor(ab_core::codecs::wire::tensor_tag::DISPLACEMENT_MAG) {
        let disp = tensor.to_dense_f32().map_err(|e| e.to_string())?;
        if disp.len() == total {
            data.displacement = Some(disp);
        }
    }
    let bytes = build_abmesh(&data, &MeshOptions::default()).map_err(|e| e.to_string())?;
    Ok(Response::new(bytes))
}

#[tauri::command(rename_all = "snake_case")]
pub async fn compute_harvest_status(
    supervisor_url: Option<String>,
) -> Result<HarvestClusterStatus, String> {
    let Some(url) = supervisor_url.filter(|value| !value.trim().is_empty()) else {
        return Ok(HarvestClusterStatus {
            configured: false,
            reachable: false,
            supervisor_url: None,
            payload: None,
            error: None,
        });
    };
    let target = format!("{}/api/v1/status", url.trim_end_matches('/'));
    let response = client().get(&target).send().await;
    match response {
        Ok(response) if response.status().is_success() => {
            let payload = response.json::<Value>().await.ok();
            Ok(HarvestClusterStatus {
                configured: true,
                reachable: true,
                supervisor_url: Some(url),
                payload,
                error: None,
            })
        }
        Ok(response) => Ok(HarvestClusterStatus {
            configured: true,
            reachable: false,
            supervisor_url: Some(url),
            payload: None,
            error: Some(format!("HTTP {}", response.status())),
        }),
        Err(error) => Ok(HarvestClusterStatus {
            configured: true,
            reachable: false,
            supervisor_url: Some(url),
            payload: None,
            error: Some(error.to_string()),
        }),
    }
}

fn websocket_url(ws_url: &str, base_url: &str) -> String {
    if ws_url.starts_with("ws://") || ws_url.starts_with("wss://") {
        ws_url.to_string()
    } else if ws_url.starts_with('/') {
        let base = base_url
            .trim_end_matches('/')
            .replace("https://", "wss://")
            .replace("http://", "ws://");
        format!("{base}{ws_url}")
    } else {
        ws_url.to_string()
    }
}

#[tauri::command(rename_all = "snake_case")]
pub async fn session_start(
    app: AppHandle,
    state: State<'_, AppState>,
    req: StartSessionRequest,
) -> Result<SessionHandle, String> {
    let connection = state.active_connection()?;
    let is_simp = req.kind == "SIMP";
    let path = if is_simp {
        "/api/v1/simp/start"
    } else {
        "/api/v1/sessions"
    };
    let body = if is_simp {
        json!({
            "resolution": req.resolution,
            "bc_type": req.bc_type,
            "load_config": req.load_config,
            "strategy": req.strategy.unwrap_or_else(|| "FULL_DOMAIN".to_string()),
            "target_vol_frac": req.target_vol_frac.unwrap_or(0.15),
            "max_iter": req.max_iter.unwrap_or(80),
            "r_min": req.r_min.unwrap_or(1.5),
            "seed": req.seed.unwrap_or(42)
        })
    } else {
        json!({
            "kind": if req.kind.is_empty() { "MCTS_LIVE".to_string() } else { req.kind.clone() },
            "resolution": req.resolution,
            "bc_type": req.bc_type,
            "load_config": req.load_config,
            "initial_solid_voxels": req.initial_solid_voxels,
            "resume_game_id": req.resume_game_id,
            "persist_to_db": req.persist_to_db,
            "db_path_id": req.db_path_id,
            "start_paused": req.start_paused,
            "include_blueprint": req.include_blueprint,
            "allow_neural_only_phase2": req.allow_neural_only_phase2,
            "episode_config": req.episode_config.unwrap_or_else(|| json!({})),
            "phase1_mcts": req.phase1_mcts.unwrap_or_else(|| json!({})),
            "phase2_mcts": req.phase2_mcts.unwrap_or_else(|| json!({}))
        })
    };
    let response = ensure_success(
        connection
            .request(reqwest::Method::POST, path)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?,
    )
    .await?;
    let payload: Value = response.json().await.map_err(|e| e.to_string())?;
    let handle = SessionHandle {
        session_id: payload
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        game_id: payload
            .get("game_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        status: payload
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("RUNNING")
            .to_string(),
        fem_enabled: payload
            .get("fem_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        ws_url: websocket_url(
            payload.get("ws_url").and_then(Value::as_str).unwrap_or(""),
            &connection.base_url(),
        ),
    };
    {
        let mut sessions = state.sessions.lock().map_err(|e| e.to_string())?;
        sessions.insert(handle.session_id.clone(), handle.clone());
    }
    let frames = Arc::clone(&state.session_frames);
    ws_bridge::spawn_ws_bridge(
        app,
        handle.session_id.clone(),
        handle.ws_url.clone(),
        connection.profile.api_token.clone(),
        frames,
    );
    Ok(handle)
}

#[tauri::command(rename_all = "snake_case")]
pub async fn session_send_command(
    state: State<'_, AppState>,
    session_id: String,
    cmd: SessionCommandPayload,
) -> Result<(), String> {
    let connection = state.active_connection()?;
    let body = json!({
        "op": cmd.op,
        "add_voxels": cmd.add_voxels,
        "remove_voxels": cmd.remove_voxels,
        "load_config": cmd.load_config,
        "bc_type": cmd.bc_type,
        "phase1_mcts": cmd.phase1_mcts,
        "phase2_mcts": cmd.phase2_mcts,
        "use_guided_value": cmd.use_guided_value
    });
    ensure_success(
        connection
            .request(
                reqwest::Method::POST,
                &format!("/api/v1/sessions/{session_id}/command"),
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?,
    )
    .await?;
    Ok(())
}

#[tauri::command(rename_all = "snake_case")]
pub async fn session_get_step_mesh(
    state: State<'_, AppState>,
    session_id: String,
    step: u32,
) -> Result<Response, String> {
    let frames = state.session_frames.lock().map_err(|e| e.to_string())?;
    let bytes = frames
        .get(&session_id, step)
        .ok_or_else(|| format!("Passo {step} ainda não está no cache da sessão."))?;
    Ok(Response::new((*bytes).clone()))
}
