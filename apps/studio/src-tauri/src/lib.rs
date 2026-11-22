#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

pub mod commands;
pub mod contracts;
pub mod state;
mod ws_bridge;

use std::time::Duration;

use tauri::{Emitter, Manager};

use crate::state::AppState;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let config_dir = app
                .path()
                .app_config_dir()
                .unwrap_or_else(|_| std::env::temp_dir().join("alphabuilder-studio"));
            app.manage(AppState::new(config_dir));
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    let state = handle.state::<AppState>();
                    let payload = {
                        let Ok(active) = state.active.lock() else {
                            continue;
                        };
                        active.as_ref().map(|connection| {
                            serde_json::json!({
                                "profile_id": connection.profile.id,
                                "connected": true,
                                "rtt_ms": connection.rtt_ms,
                                "capabilities": connection.capabilities,
                                "active_model": connection.capabilities.get("active_model").cloned(),
                                "error": null
                            })
                        })
                    };
                    if let Some(payload) = payload {
                        let _ = handle.emit("ab://compute-status", payload);
                    }
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.state::<AppState>();
                let process = state
                    .sidecar
                    .lock()
                    .ok()
                    .and_then(|mut slot| slot.take());
                if let Some(mut process) = process {
                    let _ = process.child.kill();
                    let _ = process.child.wait();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::datasets::list_local_databases,
            commands::datasets::open_database_file,
            commands::datasets::list_entries,
            commands::datasets::get_entry_metadata,
            commands::datasets::get_frame_mesh,
            commands::datasets::prefetch_frames,
            commands::datasets::compute_ab_comparison,
            commands::studio::studio_analyze_local,
            commands::studio::studio_build_mesh,
            commands::compute::compute_list_profiles,
            commands::compute::compute_save_profile,
            commands::compute::compute_delete_profile,
            commands::compute::compute_activate_profile,
            commands::compute::compute_sidecar_control,
            commands::compute::compute_list_models,
            commands::compute::compute_load_model,
            commands::compute::compute_unload_model,
            commands::compute::compute_infer_xray,
            commands::compute::compute_evaluate_fem,
            commands::compute::compute_harvest_status,
            commands::compute::session_start,
            commands::compute::session_send_command,
            commands::compute::session_get_step_mesh,
            commands::export::export_geometry,
            commands::export::export_png_bytes,
        ])
        .run(tauri::generate_context!())
        .expect("error while running AlphaBuilder Studio");
}
