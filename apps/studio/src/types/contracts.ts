export type AppMode = "inspector" | "studio" | "compare";
export type Phase = "GROWTH" | "REFINEMENT";
export type BcType = "FULL_CLAMP" | "RAIL_XY" | "ROLLER_Y" | "ROLLER_Z" | "CUSTOM";
export type LensMode =
  | "structure"
  | "policy"
  | "mcts"
  | "refutation"
  | "decision"
  | "deformation"
  | "strain";

export type DbKind = "harvest_v33" | "harvest_v2" | "selfplay" | "legacy_v1";

export interface DataSourceRef {
  origin: "local" | "remote";
  db_id: string;
  kind: DbKind;
}

export interface LoadConfig {
  x: number;
  y: number;
  z_start: number;
  z_end: number;
  bc_type?: BcType;
}

export interface ForceVector {
  x: number;
  y: number;
  z: number;
  fx: number;
  fy: number;
  fz: number;
}

export interface SupportVoxel {
  x: number;
  y: number;
  z: number;
  mask_x: boolean;
  mask_y: boolean;
  mask_z: boolean;
}

export interface SelectedAction {
  channel: 0 | 1;
  x: number;
  y: number;
  z: number;
  visits: number;
  q_value: number;
  branch_index?: number;
}

export interface MctsStats {
  num_simulations: number;
  nodes_expanded: number;
  max_depth: number;
  cache_hits: number;
  top8_concentration: number;
  refutation: boolean;
}

export interface RewardV33Breakdown {
  compliance_score: number | null;
  volume_bonus: number | null;
  fem_score: number | null;
  displacement_penalty: number;
  island_penalty: number;
  connectivity_bonus: number;
  pbrs_delta?: number | null;
  total_reward: number;
  is_collapsed: boolean;
}

export interface IslandAnalysis {
  n_islands: number;
  loose_voxels: number;
  is_connected: boolean;
  connected_load_fraction: number;
  island_penalty: number;
}

export interface StepFrameMetadata {
  step: number;
  phase: Phase;
  value: number;
  value_target: number | null;
  fem_reward: number | null;
  compliance_fem: number | null;
  max_displacement: number | null;
  volume_fraction: number;
  is_connected: boolean;
  n_islands: number;
  loose_voxels: number;
  connected_load_fraction: number;
  island_penalty: number;
  reward_v33: RewardV33Breakdown;
  mcts_stats: MctsStats | null;
  selected_actions: SelectedAction[];
  inference_ms?: number | null;
  step_elapsed_ms?: number | null;
  reward_components?: Record<string, number | null>;
  fem_enabled?: boolean;
}

export interface EntryMetadata {
  entry_id: string;
  kind: DbKind;
  resolution: [number, number, number];
  bc_type: BcType;
  strategy: string | null;
  neural_engine: string | null;
  checkpoint_version: string | null;
  seed: number | null;
  total_steps: number;
  steps_phase1: number;
  steps_phase2: number;
  final_score: number | null;
  final_compliance: number | null;
  final_volume: number | null;
  final_max_displacement: number | null;
  real_scale_factor: number;
  stiffness_e: number;
  load_config: LoadConfig;
  has_blueprint: boolean;
  has_rho_cont: boolean;
  has_displacement: boolean;
  supports: SupportVoxel[];
  forces: ForceVector[];
  phase_history: Phase[];
  value_history: number[];
  volume_history: (number | null)[];
  compliance_history: (number | null)[];
  refutation_steps: number[];
}

export interface ComputeProfile {
  id: string;
  name: string;
  mode: "local_sidecar" | "remote";
  base_url: string;
  api_token: string;
  python_bin?: string;
  auto_start?: boolean;
}

export interface ComputeConnectionState {
  profile_id: string;
  connected: boolean;
  rtt_ms: number;
  capabilities: Capabilities | null;
  active_model: LoadedModelInfo | null;
  error: string | null;
}

export interface DatabaseDescriptor {
  db_id: string;
  name: string;
  path_display: string;
  kind: DbKind;
  item_count: number;
  total_steps: number;
  size_mb: number;
  has_legacy_pickle: boolean;
}

export interface EntryListQuery {
  limit: number;
  offset: number;
  search?: string;
  strategy?: string;
  bc_type?: BcType;
  engine?: string;
  version?: string;
  connected_only?: boolean;
  sort?: "recent" | "seed_asc" | "reward_desc" | "compliance_asc";
}

export interface EntrySummary {
  entry_id: string;
  kind: DbKind;
  total_steps: number;
  steps_phase1: number;
  steps_phase2: number;
  final_reward: number | null;
  final_compliance: number | null;
  final_volume: number | null;
  bc_type: BcType | null;
  strategy: string | null;
  neural_engine: string | null;
  checkpoint_version: string | null;
  seed: number | null;
  is_connected: boolean | null;
  created_at: string | null;
}

export interface EntryListPage {
  total_count: number;
  items: EntrySummary[];
}

export interface MeshOptions {
  density_threshold: number;
  blueprint_threshold: number;
  use_rho_cont: boolean;
  include_diff_next: boolean;
  include_blueprint: boolean;
}

export const DEFAULT_MESH_OPTIONS: MeshOptions = {
  density_threshold: 0.5,
  blueprint_threshold: 0.5,
  use_rho_cont: false,
  include_diff_next: true,
  include_blueprint: true,
};

export interface StudioVoxelState {
  resolution: [number, number, number];
  bc_type: BcType;
  load_config: LoadConfig;
  custom_supports: SupportVoxel[] | null;
  custom_forces: ForceVector[] | null;
  solid_voxels: number[];
}

export interface LocalTopologyReport {
  is_connected: boolean;
  n_islands: number;
  loose_voxels: number;
  volume_fraction: number;
  connected_load_fraction: number;
  island_penalty: number;
  legal_add_count: number;
  legal_remove_count: number;
}

export interface ComparisonMetrics {
  iou_3d: number;
  delta_compliance: number | null;
  delta_compliance_pct: number | null;
  delta_volume: number | null;
  delta_reward: number | null;
  delta_max_displacement: number | null;
  delta_steps: number;
}

export interface FrameRef {
  source: DataSourceRef;
  entry_id: string;
  step: number;
}

export interface DeviceInfo {
  id: string;
  available: boolean;
  name: string | null;
}

export interface FeatureFlags {
  has_torch: boolean;
  has_monai: boolean;
  has_openvino: boolean;
  has_fenics: boolean;
  has_fenitop_simp: boolean;
  has_harvest_supervisor: boolean;
}

export interface Capabilities {
  hostname: string;
  os: string;
  arch: string;
  devices: DeviceInfo[];
  features: FeatureFlags;
  limits: { max_voxels: number; max_concurrent_sessions: number };
  reward_spec: Record<string, string | number>;
  active_model: LoadedModelInfo | null;
}

export interface ModelSummary {
  model_id: string;
  name: string;
  size_mb: number;
  modified_at: string | null;
  config_preview: Record<string, unknown>;
}

export interface LoadedModelInfo {
  loaded: boolean;
  model_id: string | null;
  architecture: string | null;
  feature_size: number | null;
  in_channels: number | null;
  has_blueprint_head: boolean | null;
  device_resolved: string | null;
  epoch: number | null;
  val_loss: number | null;
  load_time_ms: number | null;
}

export interface SidecarStatus {
  running: boolean;
  pid: number | null;
  port: number;
  message: string;
}

export interface HarvestClusterStatus {
  configured: boolean;
  reachable: boolean;
  supervisor_url: string | null;
  payload: Record<string, unknown> | null;
  error: string | null;
}

export interface PolicyEntry {
  flat_idx: number;
  prob: number;
}

export interface CachedInferOverlay {
  blueprint_probs?: number[] | null;
  policy_add: PolicyEntry[];
  policy_remove: PolicyEntry[];
  value?: number | null;
}

export interface InferXRayRequest {
  resolution: [number, number, number];
  bc_type: BcType;
  load_config: LoadConfig;
  custom_supports: SupportVoxel[] | null;
  custom_forces: ForceVector[] | null;
  solid_voxels: number[];
  include_blueprint: boolean;
  policy_mode?: "masked_probs" | "raw";
  policy_temperature?: number;
  policy_top_k?: number;
}

export interface FemEvalXRayRequest {
  resolution: [number, number, number];
  bc_type: BcType;
  load_config: LoadConfig;
  custom_supports: SupportVoxel[] | null;
  custom_forces: ForceVector[] | null;
  solid_voxels: number[];
  E?: number;
  nu?: number;
  force_magnitude?: number;
  include_displacement_map?: boolean;
  include_displacement_vector?: boolean;
}

export interface StartSessionRequest {
  kind: "MCTS_LIVE" | "SIMP";
  resolution: [number, number, number];
  bc_type: BcType;
  load_config: LoadConfig;
  initial_solid_voxels: number[];
  resume_game_id?: string | null;
  persist_to_db?: boolean;
  db_path_id?: string | null;
  start_paused?: boolean;
  include_blueprint?: boolean;
  allow_neural_only_phase2?: boolean;
  episode_config?: Record<string, unknown> | null;
  phase1_mcts?: Record<string, unknown> | null;
  phase2_mcts?: Record<string, unknown> | null;
  strategy?: "FULL_DOMAIN" | "BEZIER";
  target_vol_frac?: number;
  max_iter?: number;
  r_min?: number;
  seed?: number;
}

export interface SessionHandle {
  session_id: string;
  game_id: string;
  status: string;
  fem_enabled: boolean;
  ws_url: string;
}

export interface SessionCommand {
  op: "pause" | "resume" | "step_once" | "edit_state" | "update_config" | "cancel";
  add_voxels?: number[];
  remove_voxels?: number[];
  load_config?: LoadConfig | null;
  bc_type?: BcType | null;
  phase1_mcts?: Record<string, unknown> | null;
  phase2_mcts?: Record<string, unknown> | null;
  use_guided_value?: boolean | null;
}

export interface ExportResult {
  ok: boolean;
  bytes_written: number;
  output_path: string;
}

export type ExportSource =
  | { kind: "frame"; source: DataSourceRef; entry_id: string; step: number }
  | { kind: "studio"; state: StudioVoxelState };

export interface ComputeStatusEvent {
  profile_id: string;
  connected: boolean;
  rtt_ms: number;
  capabilities: Capabilities | null;
  active_model: LoadedModelInfo | null;
  error: string | null;
}

export interface SessionStateEvent {
  session_id: string;
  game_id: string;
  status: string;
  step: number;
  phase: Phase;
  reason: string | null;
}

export interface SessionProgressEvent {
  session_id: string;
  step: number;
  phase: Phase;
  sims_done: number;
  sims_total: number;
  cache_hits: number;
  elapsed_ms: number;
}

export interface SessionStepReadyEvent {
  session_id: string;
  step: number;
  phase: Phase;
  value: number;
  volume_fraction: number;
  compliance_fem: number | null;
  is_connected: boolean;
  refutation: boolean;
}

export interface ParsedAbMesh {
  flags: number;
  stepIndex: number;
  dims: [number, number, number];
  dispP99: number;
  strainP99: number;
  positions: Float32Array;
  normals: Float32Array;
  aoAndScalars: Float32Array;
  voxelIds: Uint32Array;
  indices: Uint32Array;
  blueprintPositions: Float32Array;
  blueprintNormals: Float32Array;
  blueprintIndices: Uint32Array;
  policyAddRaw: DataView;
  policyAddCount: number;
  policyRemRaw: DataView;
  policyRemCount: number;
  mctsAddRaw: DataView;
  mctsAddCount: number;
  mctsRemRaw: DataView;
  mctsRemCount: number;
  diffAdded: Uint32Array;
  diffRemoved: Uint32Array;
  metadata: StepFrameMetadata;
}
