//! tokio-tungstenite bridge: server WS -> Tauri events + ABMESH cache.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ab_core::codecs::wire::{payload_type, WirePacket};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tauri::{AppHandle, Emitter};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

use crate::commands::frame::frame_from_wire;
use crate::contracts::MeshOptions;
use crate::state::SessionFrames;

pub fn spawn_ws_bridge(
    app: AppHandle,
    session_id: String,
    url: String,
    token: String,
    frames: Arc<Mutex<SessionFrames>>,
) {
    tauri::async_runtime::spawn(async move {
        let mut attempt = 0_u32;
        let mut ended = false;
        while !ended && attempt < 5 {
            let result = run_once(&app, &session_id, &url, &token, &frames, &mut ended).await;
            if ended {
                break;
            }
            if let Err(error) = result {
                eprintln!("ws_bridge {session_id}: {error}");
            }
            attempt += 1;
            let backoff = Duration::from_millis(500 * (1_u64 << (attempt - 1).min(4)));
            tokio::time::sleep(backoff).await;
        }
        if !ended {
            let _ = app.emit(
                "ab://session-state",
                serde_json::json!({
                    "session_id": session_id,
                    "game_id": "",
                    "status": "FAILED",
                    "step": 0,
                    "phase": "GROWTH",
                    "reason": "WS_RECONNECT_EXHAUSTED"
                }),
            );
        }
    });
}

async fn run_once(
    app: &AppHandle,
    session_id: &str,
    url: &str,
    token: &str,
    frames: &Arc<Mutex<SessionFrames>>,
    ended: &mut bool,
) -> Result<(), String> {
    let mut request = url
        .to_string()
        .into_client_request()
        .map_err(|e| e.to_string())?;
    if !token.is_empty() {
        let value = format!("Bearer {token}");
        request.headers_mut().insert(
            "Authorization",
            value
                .parse()
                .map_err(|_| "invalid bearer token".to_string())?,
        );
    }
    let (mut stream, _) = tokio_tungstenite::connect_async(request)
        .await
        .map_err(|e| e.to_string())?;
    while let Some(message) = stream.next().await {
        let message = message.map_err(|e| e.to_string())?;
        match message {
            Message::Text(text) => {
                let payload: Value =
                    serde_json::from_str(text.as_str()).map_err(|e| e.to_string())?;
                let kind = payload.get("type").and_then(Value::as_str).unwrap_or("");
                match kind {
                    "session_state" => {
                        let _ = app.emit(
                            "ab://session-state",
                            serde_json::json!({
                                "session_id": payload.get("session_id").and_then(Value::as_str).unwrap_or(session_id),
                                "game_id": payload.get("game_id").and_then(Value::as_str).unwrap_or(""),
                                "status": payload.get("status").and_then(Value::as_str).unwrap_or("RUNNING"),
                                "step": payload.get("step").and_then(Value::as_u64).unwrap_or(0),
                                "phase": payload.get("phase").and_then(Value::as_str).unwrap_or("GROWTH"),
                                "reason": null
                            }),
                        );
                    }
                    "step_progress" => {
                        let _ = app.emit(
                            "ab://session-progress",
                            serde_json::json!({
                                "session_id": session_id,
                                "step": payload.get("step").and_then(Value::as_u64).unwrap_or(0),
                                "phase": payload.get("phase").and_then(Value::as_str).unwrap_or("GROWTH"),
                                "sims_done": payload.get("sims_done").and_then(Value::as_u64).unwrap_or(0),
                                "sims_total": payload.get("sims_total").and_then(Value::as_u64).unwrap_or(0),
                                "cache_hits": payload.get("cache_hits").and_then(Value::as_u64).unwrap_or(0),
                                "elapsed_ms": payload.get("elapsed_ms").and_then(Value::as_f64).unwrap_or(0.0)
                            }),
                        );
                    }
                    "session_ended" => {
                        *ended = true;
                        let _ = app.emit(
                            "ab://session-state",
                            serde_json::json!({
                                "session_id": session_id,
                                "game_id": "",
                                "status": payload.get("status").and_then(Value::as_str).unwrap_or("COMPLETED"),
                                "step": payload.get("total_steps").and_then(Value::as_u64).unwrap_or(0),
                                "phase": "REFINEMENT",
                                "reason": payload.get("reason").and_then(Value::as_str).unwrap_or("")
                            }),
                        );
                    }
                    "error" => {
                        let _ = app.emit(
                            "ab://session-state",
                            serde_json::json!({
                                "session_id": session_id,
                                "status": "FAILED",
                                "code": payload.get("code"),
                                "message": payload.get("message")
                            }),
                        );
                    }
                    _ => {}
                }
            }
            Message::Binary(bytes) => {
                let packet = WirePacket::decode(bytes.as_ref()).map_err(|e| e.to_string())?;
                if packet.payload_type != payload_type::STEP_FRAME {
                    continue;
                }
                let metadata: Value =
                    serde_json::from_str(&packet.header_json).map_err(|e| e.to_string())?;
                let step = metadata.get("step").and_then(Value::as_u64).unwrap_or(0) as u32;
                let options = MeshOptions::default();
                match frame_from_wire(&packet, &options) {
                    Ok(mesh) => {
                        let mut cache = frames.lock().map_err(|e| e.to_string())?;
                        cache.insert(session_id, step, mesh);
                        drop(cache);
                        let _ = app.emit(
                            "ab://session-step-ready",
                            serde_json::json!({
                                "session_id": session_id,
                                "step": step,
                                "phase": metadata.get("phase").and_then(Value::as_str).unwrap_or("GROWTH"),
                                "value": metadata.get("value").and_then(Value::as_f64).unwrap_or(0.0),
                                "volume_fraction": metadata.get("volume_fraction").and_then(Value::as_f64).unwrap_or(0.0),
                                "compliance_fem": metadata.get("compliance_fem"),
                                "is_connected": metadata.get("is_connected").and_then(Value::as_bool).unwrap_or(false),
                                "refutation": metadata.get("mcts_stats").and_then(|s| s.get("refutation")).and_then(Value::as_bool).unwrap_or(false)
                            }),
                        );
                    }
                    Err(error) => {
                        eprintln!("ws_bridge {session_id} step {step}: {error}");
                    }
                }
            }
            Message::Close(_) => break,
            Message::Ping(payload) => {
                let _ = stream.send(Message::Pong(payload)).await;
            }
            _ => {}
        }
    }
    Ok(())
}
