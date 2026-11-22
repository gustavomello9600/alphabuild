//! Global Tauri state: profiles, LRU ABMESH cache, sidecar and sessions.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::{Arc, Mutex};

use ab_core::sqlite_reader::DatabaseInfo;
use serde_json::Value;

use crate::contracts::{ComputeProfile, SessionHandle};

pub const CACHE_LIMIT_BYTES: usize = 256 * 1024 * 1024;

pub struct MeshCache {
    map: HashMap<String, Arc<Vec<u8>>>,
    order: VecDeque<String>,
    total_bytes: usize,
    limit_bytes: usize,
}

impl MeshCache {
    pub fn new(limit_bytes: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            total_bytes: 0,
            limit_bytes,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<Arc<Vec<u8>>> {
        let value = self.map.get(key).cloned();
        if value.is_some() {
            self.order.retain(|item| item != key);
            self.order.push_back(key.to_string());
        }
        value
    }

    pub fn insert(&mut self, key: String, value: Vec<u8>) {
        let size = value.len();
        if size > self.limit_bytes {
            return;
        }
        if let Some(previous) = self.map.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(previous.len());
            self.order.retain(|item| item != &key);
        }
        while self.total_bytes + size > self.limit_bytes {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(removed) = self.map.remove(&oldest) {
                self.total_bytes = self.total_bytes.saturating_sub(removed.len());
            }
        }
        self.total_bytes += size;
        self.order.push_back(key.clone());
        self.map.insert(key, Arc::new(value));
    }
}

pub struct ActiveConnection {
    pub profile: ComputeProfile,
    pub client: reqwest::Client,
    pub capabilities: Value,
    pub rtt_ms: f64,
}

impl ActiveConnection {
    pub fn base_url(&self) -> String {
        self.profile.base_url.trim_end_matches('/').to_string()
    }

    pub fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url(), path);
        let builder = self.client.request(method, url);
        if self.profile.api_token.is_empty() {
            builder
        } else {
            builder.bearer_auth(&self.profile.api_token)
        }
    }
}

pub struct SidecarProcess {
    pub child: Child,
    pub port: u16,
}

#[derive(Default)]
pub struct SessionFrames {
    pub frames: HashMap<String, HashMap<u32, Arc<Vec<u8>>>>,
}

impl SessionFrames {
    pub fn insert(&mut self, session_id: &str, step: u32, bytes: Vec<u8>) {
        self.frames
            .entry(session_id.to_string())
            .or_default()
            .insert(step, Arc::new(bytes));
    }

    pub fn get(&self, session_id: &str, step: u32) -> Option<Arc<Vec<u8>>> {
        self.frames
            .get(session_id)
            .and_then(|map| map.get(&step))
            .cloned()
    }
}

pub struct AppState {
    pub config_dir: PathBuf,
    pub databases: Mutex<HashMap<String, DatabaseInfo>>,
    pub profiles: Mutex<Vec<ComputeProfile>>,
    pub active: Mutex<Option<ActiveConnection>>,
    pub cache: Arc<Mutex<MeshCache>>,
    pub sidecar: Mutex<Option<SidecarProcess>>,
    pub sessions: Mutex<HashMap<String, SessionHandle>>,
    pub session_frames: Arc<Mutex<SessionFrames>>,
}

impl AppState {
    pub fn new(config_dir: PathBuf) -> Self {
        let profiles = load_profiles(&config_dir);
        Self {
            config_dir,
            databases: Mutex::new(HashMap::new()),
            profiles: Mutex::new(profiles),
            active: Mutex::new(None),
            cache: Arc::new(Mutex::new(MeshCache::new(CACHE_LIMIT_BYTES))),
            sidecar: Mutex::new(None),
            sessions: Mutex::new(HashMap::new()),
            session_frames: Arc::new(Mutex::new(SessionFrames::default())),
        }
    }

    pub fn save_profiles(&self) -> Result<(), String> {
        let profiles = self.profiles.lock().map_err(|e| e.to_string())?.clone();
        save_profiles(&self.config_dir, &profiles)
    }

    pub fn active_connection(&self) -> Result<ActiveConnection, String> {
        self.active
            .lock()
            .map_err(|e| e.to_string())?
            .as_ref()
            .map(|conn| ActiveConnection {
                profile: conn.profile.clone(),
                client: conn.client.clone(),
                capabilities: conn.capabilities.clone(),
                rtt_ms: conn.rtt_ms,
            })
            .ok_or_else(|| "Nenhum perfil de compute ativo. Abra o Compute Popover.".to_string())
    }
}

pub fn default_profiles() -> Vec<ComputeProfile> {
    vec![
        ComputeProfile {
            id: "local-sidecar".to_string(),
            name: "Local Sidecar".to_string(),
            mode: "local_sidecar".to_string(),
            base_url: "http://127.0.0.1:8765".to_string(),
            api_token: String::new(),
            python_bin: Some("python3".to_string()),
            auto_start: Some(true),
        },
        ComputeProfile {
            id: "localhost-wsl2".to_string(),
            name: "Localhost / WSL2".to_string(),
            mode: "remote".to_string(),
            base_url: "http://127.0.0.1:8765".to_string(),
            api_token: String::new(),
            python_bin: None,
            auto_start: Some(false),
        },
        ComputeProfile {
            id: "remote-tailscale".to_string(),
            name: "Remoto Tailscale / HTTPS".to_string(),
            mode: "remote".to_string(),
            base_url: "http://100.64.0.1:8765".to_string(),
            api_token: String::new(),
            python_bin: None,
            auto_start: Some(false),
        },
    ]
}

pub fn load_profiles(config_dir: &Path) -> Vec<ComputeProfile> {
    let path = config_dir.join("profiles.json");
    if let Ok(text) = fs::read_to_string(&path) {
        if let Ok(profiles) = serde_json::from_str::<Vec<ComputeProfile>>(&text) {
            if !profiles.is_empty() {
                return profiles;
            }
        }
    }
    default_profiles()
}

pub fn save_profiles(config_dir: &Path, profiles: &[ComputeProfile]) -> Result<(), String> {
    fs::create_dir_all(config_dir).map_err(|e| e.to_string())?;
    let path = config_dir.join("profiles.json");
    let text = serde_json::to_string_pretty(profiles).map_err(|e| e.to_string())?;
    fs::write(&path, text).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

pub fn default_cache_key(
    db_id: &str,
    entry_id: &str,
    step: u32,
    options: &crate::contracts::MeshOptions,
) -> String {
    format!(
        "{db_id}|{entry_id}|{step}|{:.4}|{:.4}|{}|{}|{}",
        options.density_threshold,
        options.blueprint_threshold,
        options.use_rho_cont,
        options.include_diff_next,
        options.include_blueprint
    )
}
