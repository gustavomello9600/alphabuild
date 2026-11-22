//! Canonical snake_case contracts shared by commands and the WS bridge.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DbKind {
    HarvestV33,
    HarvestV2,
    Selfplay,
    LegacyV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceRef {
    pub origin: String,
    pub db_id: String,
    pub kind: DbKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadConfig {
    pub x: f64,
    pub y: f64,
    pub z_start: f64,
    pub z_end: f64,
    #[serde(default)]
    pub bc_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceVector {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub fx: f64,
    pub fy: f64,
    pub fz: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportVoxel {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub mask_x: bool,
    pub mask_y: bool,
    pub mask_z: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshOptions {
    #[serde(default = "default_threshold")]
    pub density_threshold: f32,
    #[serde(default = "default_threshold")]
    pub blueprint_threshold: f32,
    #[serde(default)]
    pub use_rho_cont: bool,
    #[serde(default = "default_true")]
    pub include_diff_next: bool,
    #[serde(default = "default_true")]
    pub include_blueprint: bool,
}

fn default_threshold() -> f32 {
    0.5
}

fn default_true() -> bool {
    true
}

impl Default for MeshOptions {
    fn default() -> Self {
        Self {
            density_threshold: 0.5,
            blueprint_threshold: 0.5,
            use_rho_cont: false,
            include_diff_next: true,
            include_blueprint: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntryListQuery {
    #[serde(default)]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
    pub search: Option<String>,
    pub strategy: Option<String>,
    pub bc_type: Option<String>,
    pub engine: Option<String>,
    pub version: Option<String>,
    #[serde(default)]
    pub connected_only: bool,
    pub sort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntrySummary {
    pub entry_id: String,
    pub kind: DbKind,
    pub total_steps: i64,
    pub steps_phase1: i64,
    pub steps_phase2: i64,
    pub final_reward: Option<f64>,
    pub final_compliance: Option<f64>,
    pub final_volume: Option<f64>,
    pub bc_type: Option<String>,
    pub strategy: Option<String>,
    pub neural_engine: Option<String>,
    pub checkpoint_version: Option<String>,
    pub seed: Option<i64>,
    pub is_connected: Option<bool>,
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryListPage {
    pub total_count: usize,
    pub items: Vec<EntrySummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryMetadata {
    pub entry_id: String,
    pub kind: DbKind,
    pub resolution: [u32; 3],
    pub bc_type: String,
    pub strategy: Option<String>,
    pub neural_engine: Option<String>,
    pub checkpoint_version: Option<String>,
    pub seed: Option<i64>,
    pub total_steps: i64,
    pub steps_phase1: i64,
    pub steps_phase2: i64,
    pub final_score: Option<f64>,
    pub final_compliance: Option<f64>,
    pub final_volume: Option<f64>,
    pub final_max_displacement: Option<f64>,
    pub real_scale_factor: f64,
    pub stiffness_e: f64,
    pub load_config: LoadConfig,
    pub has_blueprint: bool,
    pub has_rho_cont: bool,
    pub has_displacement: bool,
    pub supports: Vec<SupportVoxel>,
    pub forces: Vec<ForceVector>,
    pub phase_history: Vec<String>,
    pub value_history: Vec<f64>,
    pub volume_history: Vec<Option<f64>>,
    pub compliance_history: Vec<Option<f64>>,
    pub refutation_steps: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudioVoxelState {
    pub resolution: [u32; 3],
    pub bc_type: String,
    pub load_config: LoadConfig,
    pub custom_supports: Option<Vec<SupportVoxel>>,
    pub custom_forces: Option<Vec<ForceVector>>,
    pub solid_voxels: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalTopologyReport {
    pub is_connected: bool,
    pub n_islands: usize,
    pub loose_voxels: usize,
    pub volume_fraction: f64,
    pub connected_load_fraction: f64,
    pub island_penalty: f64,
    pub legal_add_count: usize,
    pub legal_remove_count: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PolicyEntry {
    pub flat_idx: u32,
    pub prob: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CachedInferOverlay {
    #[serde(default)]
    pub blueprint_probs: Option<Vec<f32>>,
    #[serde(default)]
    pub policy_add: Vec<PolicyEntry>,
    #[serde(default)]
    pub policy_remove: Vec<PolicyEntry>,
    #[serde(default)]
    pub value: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameRef {
    pub source: DataSourceRef,
    pub entry_id: String,
    pub step: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub iou_3d: f32,
    pub delta_compliance: Option<f64>,
    pub delta_compliance_pct: Option<f64>,
    pub delta_volume: Option<f64>,
    pub delta_reward: Option<f64>,
    pub delta_max_displacement: Option<f64>,
    pub delta_steps: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeProfile {
    pub id: String,
    pub name: String,
    pub mode: String,
    pub base_url: String,
    #[serde(default)]
    pub api_token: String,
    pub python_bin: Option<String>,
    pub auto_start: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeConnectionState {
    pub profile_id: String,
    pub connected: bool,
    pub rtt_ms: f64,
    pub capabilities: Option<serde_json::Value>,
    pub active_model: Option<serde_json::Value>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub model_id: String,
    pub name: String,
    pub size_mb: f64,
    pub modified_at: Option<String>,
    pub config_preview: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModelInfo {
    pub loaded: bool,
    pub model_id: Option<String>,
    pub architecture: Option<String>,
    pub feature_size: Option<i64>,
    pub in_channels: Option<i64>,
    pub has_blueprint_head: Option<bool>,
    pub device_resolved: Option<String>,
    pub epoch: Option<i64>,
    pub val_loss: Option<f64>,
    pub load_time_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelRequest {
    pub model_id: String,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default = "default_hint")]
    pub openvino_hint: String,
}

fn default_device() -> String {
    "auto".to_string()
}

fn default_hint() -> String {
    "LATENCY".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarStatus {
    pub running: bool,
    pub pid: Option<u32>,
    pub port: u16,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarvestClusterStatus {
    pub configured: bool,
    pub reachable: bool,
    pub supervisor_url: Option<String>,
    pub payload: Option<serde_json::Value>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferRequest {
    pub resolution: [u32; 3],
    pub bc_type: String,
    pub load_config: LoadConfig,
    pub custom_supports: Option<Vec<SupportVoxel>>,
    pub custom_forces: Option<Vec<ForceVector>>,
    #[serde(default)]
    pub solid_voxels: Vec<u32>,
    #[serde(default)]
    pub include_blueprint: bool,
    #[serde(default = "default_policy_mode")]
    pub policy_mode: String,
    #[serde(default = "default_temperature")]
    pub policy_temperature: f32,
    #[serde(default = "default_top_k")]
    pub policy_top_k: u32,
}

fn default_policy_mode() -> String {
    "masked_probs".to_string()
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_k() -> u32 {
    512
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FemEvalRequest {
    pub resolution: [u32; 3],
    pub bc_type: String,
    pub load_config: LoadConfig,
    pub custom_supports: Option<Vec<SupportVoxel>>,
    pub custom_forces: Option<Vec<ForceVector>>,
    #[serde(default)]
    pub solid_voxels: Vec<u32>,
    #[serde(default = "default_e")]
    pub e: f64,
    #[serde(default = "default_nu")]
    pub nu: f64,
    #[serde(default = "default_force")]
    pub force_magnitude: f64,
    #[serde(default = "default_true")]
    pub include_displacement_map: bool,
    #[serde(default)]
    pub include_displacement_vector: bool,
}

fn default_e() -> f64 {
    100.0
}

fn default_nu() -> f64 {
    0.25
}

fn default_force() -> f64 {
    -1.125
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCommandPayload {
    pub op: String,
    #[serde(default)]
    pub add_voxels: Vec<u32>,
    #[serde(default)]
    pub remove_voxels: Vec<u32>,
    pub load_config: Option<LoadConfig>,
    pub bc_type: Option<String>,
    pub phase1_mcts: Option<serde_json::Value>,
    pub phase2_mcts: Option<serde_json::Value>,
    pub use_guided_value: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartSessionRequest {
    pub kind: String,
    pub resolution: [u32; 3],
    pub bc_type: String,
    pub load_config: LoadConfig,
    #[serde(default)]
    pub initial_solid_voxels: Vec<u32>,
    pub resume_game_id: Option<String>,
    #[serde(default = "default_true")]
    pub persist_to_db: bool,
    pub db_path_id: Option<String>,
    #[serde(default)]
    pub start_paused: bool,
    #[serde(default = "default_true")]
    pub include_blueprint: bool,
    #[serde(default = "default_true")]
    pub allow_neural_only_phase2: bool,
    pub episode_config: Option<serde_json::Value>,
    pub phase1_mcts: Option<serde_json::Value>,
    pub phase2_mcts: Option<serde_json::Value>,
    pub strategy: Option<String>,
    pub target_vol_frac: Option<f64>,
    pub max_iter: Option<u32>,
    pub r_min: Option<f64>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHandle {
    pub session_id: String,
    pub game_id: String,
    pub status: String,
    pub fem_enabled: bool,
    pub ws_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportSource {
    Frame {
        source: DataSourceRef,
        entry_id: String,
        step: u32,
    },
    Studio {
        state: StudioVoxelState,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    pub ok: bool,
    pub bytes_written: usize,
    pub output_path: String,
}
