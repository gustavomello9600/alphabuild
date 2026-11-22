"""Pydantic v2 request/response models and the canonical error envelope."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from . import __version__

BcType = Literal["FULL_CLAMP", "RAIL_XY", "ROLLER_Y", "ROLLER_Z", "CUSTOM"]
DeviceType = Literal["auto", "cuda", "xpu", "mps", "npu", "cpu"]
PhaseType = Literal["GROWTH", "REFINEMENT"]
SessionStatus = Literal["CREATED", "RUNNING", "PAUSED", "COMPLETED", "FAILED", "CANCELLED"]
PayloadFormat = Literal["abwire01", "json"]

REWARD_SPEC: dict[str, Any] = {
    "version": "v3.3",
    "compliance_base": 1.15,
    "compliance_slope": 0.35,
    "compliance_min": -0.60,
    "compliance_max": 0.90,
    "volume_reference": 0.10,
    "volume_sensitivity": 2.50,
    "volume_bonus_max": 0.30,
    "volume_penalty_max": 0.60,
    "displacement_limit_raw": 128000.0,
    "real_scale_factor": 500000.0,
}

API_VERSION = "1.0.0"


class ApiError(Exception):
    """Domain error rendered through the standard snake_case envelope."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details

    def envelope(self, request_id: str) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "request_id": request_id,
            }
        }


def error_envelope(
    code: str, message: str, details: Optional[dict[str, Any]], request_id: str
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "request_id": request_id,
        }
    }


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadConfig(_StrictModel):
    x: float
    y: float
    z_start: float
    z_end: float
    bc_type: Optional[BcType] = None


class ForceVector(_StrictModel):
    x: int
    y: int
    z: int
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0


class SupportVoxel(_StrictModel):
    x: int
    y: int
    z: int
    mask_x: bool = False
    mask_y: bool = False
    mask_z: bool = False


class InferRequest(_StrictModel):
    resolution: list[int] = Field(min_length=3, max_length=3)
    bc_type: BcType = "FULL_CLAMP"
    load_config: LoadConfig
    custom_supports: Optional[list[SupportVoxel]] = None
    custom_forces: Optional[list[ForceVector]] = None
    solid_voxels: list[int] = Field(default_factory=list)
    density_f32_b64_zlib: Optional[str] = None
    include_blueprint: bool = True
    policy_mode: Literal["masked_probs", "raw"] = "masked_probs"
    policy_temperature: float = 1.0
    policy_top_k: int = 512
    response_format: PayloadFormat = "abwire01"


class FemEvalRequest(InferRequest):
    E: float = 100.0
    nu: float = 0.25
    force_magnitude: float = -1.125
    include_displacement_map: bool = True
    include_displacement_vector: bool = False


class SimpStartRequest(_StrictModel):
    resolution: list[int] = Field(min_length=3, max_length=3)
    bc_type: BcType = "FULL_CLAMP"
    load_config: LoadConfig
    strategy: Literal["FULL_DOMAIN", "BEZIER"] = "FULL_DOMAIN"
    target_vol_frac: float = 0.15
    max_iter: int = 80
    r_min: float = 1.5
    seed: int = 42


class MctsSettings(_StrictModel):
    num_simulations: int = 160
    batch_size: int = 8
    action_batch_size: int = 32
    c_puct: float = 1.2
    top_k_expansion: int = 64
    mixmax_beta: float = 0.75


class SimpSettings(_StrictModel):
    strategy: Literal["FULL_DOMAIN", "BEZIER"] = "FULL_DOMAIN"
    target_vol_frac: float = 0.15
    max_iter: int = 80
    r_min: float = 1.5
    seed: int = 42


class EpisodeSettings(_StrictModel):
    max_steps: int = 600
    max_phase1_steps: int = 200
    target_volume_fraction: float = 0.15
    use_guided_value: bool = True
    generation: int = 0
    guidance_decay_generations: int = 5


class SessionCreateRequest(_StrictModel):
    kind: Literal["MCTS_LIVE", "SIMP"] = "MCTS_LIVE"
    resolution: list[int] = Field(min_length=3, max_length=3)
    bc_type: BcType = "FULL_CLAMP"
    load_config: LoadConfig
    initial_solid_voxels: list[int] = Field(default_factory=list)
    resume_game_id: Optional[str] = None
    persist_to_db: bool = True
    db_path_id: Optional[str] = None
    start_paused: bool = False
    include_blueprint: bool = True
    allow_neural_only_phase2: bool = True
    episode_config: EpisodeSettings = Field(default_factory=EpisodeSettings)
    phase1_mcts: MctsSettings = Field(
        default_factory=lambda: MctsSettings(
            num_simulations=160,
            batch_size=8,
            action_batch_size=32,
            c_puct=1.2,
            mixmax_beta=0.75,
        )
    )
    phase2_mcts: MctsSettings = Field(
        default_factory=lambda: MctsSettings(
            num_simulations=160,
            batch_size=8,
            action_batch_size=8,
            c_puct=0.8,
            mixmax_beta=1.0,
        )
    )
    simp_config: Optional[SimpSettings] = None


class SessionCommandRequest(_StrictModel):
    op: Literal["pause", "resume", "step_once", "edit_state", "update_config", "cancel"]
    add_voxels: list[int] = Field(default_factory=list)
    remove_voxels: list[int] = Field(default_factory=list)
    load_config: Optional[LoadConfig] = None
    bc_type: Optional[BcType] = None
    phase1_mcts: Optional[MctsSettings] = None
    phase2_mcts: Optional[MctsSettings] = None
    use_guided_value: Optional[bool] = None


class LoadModelRequest(_StrictModel):
    model_id: str
    device: DeviceType = "auto"
    openvino_hint: str = "LATENCY"


class UnloadModelResponse(BaseModel):
    loaded: bool = False


class ModelSummary(BaseModel):
    model_id: str
    name: str
    size_mb: float
    modified_at: Optional[str] = None
    config_preview: dict[str, Any] = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    checkpoints_dir: str
    models: list[ModelSummary]


class LoadedModelInfo(BaseModel):
    loaded: bool
    model_id: Optional[str] = None
    architecture: Optional[str] = None
    feature_size: Optional[int] = None
    in_channels: Optional[int] = None
    has_blueprint_head: Optional[bool] = None
    device_resolved: Optional[str] = None
    epoch: Optional[int] = None
    val_loss: Optional[float] = None
    load_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    ok: bool = True
    service: str = "alphabuilder-engine"
    api_version: str = API_VERSION
    protocol: str = "ABWIRE01"
    code_version: str = "v3.3"
    uptime_sec: float
    auth_required: bool
    timestamp_utc: str


class DeviceInfo(BaseModel):
    id: str
    available: bool
    name: Optional[str] = None


class FeatureFlags(BaseModel):
    has_torch: bool
    has_monai: bool
    has_openvino: bool
    has_fenics: bool
    has_fenitop_simp: bool
    has_harvest_supervisor: bool


class LimitsInfo(BaseModel):
    max_voxels: int
    max_concurrent_sessions: int


class CapabilitiesResponse(BaseModel):
    hostname: str
    os: str
    arch: str
    devices: list[DeviceInfo]
    features: FeatureFlags
    limits: LimitsInfo
    reward_spec: dict[str, Any]
    active_model: Optional[LoadedModelInfo] = None


class RewardV33Breakdown(BaseModel):
    compliance_score: Optional[float] = None
    volume_bonus: Optional[float] = None
    fem_score: Optional[float] = None
    displacement_penalty: float = 0.0
    island_penalty: float = 0.0
    connectivity_bonus: float = 0.0
    pbrs_delta: Optional[float] = None
    total_reward: float
    is_collapsed: bool = False


class MctsStats(BaseModel):
    num_simulations: int
    nodes_expanded: int
    max_depth: int
    cache_hits: int
    top8_concentration: float
    refutation: bool


class SelectedAction(BaseModel):
    channel: int
    x: int
    y: int
    z: int
    visits: int
    q_value: float
    branch_index: Optional[int] = None


class StepFrameMetadata(BaseModel):
    step: int
    phase: PhaseType
    value: float
    value_target: Optional[float] = None
    fem_reward: Optional[float] = None
    compliance_fem: Optional[float] = None
    max_displacement: Optional[float] = None
    volume_fraction: float
    is_connected: bool
    n_islands: int
    loose_voxels: int
    connected_load_fraction: float
    island_penalty: float
    reward_v33: RewardV33Breakdown
    mcts_stats: Optional[MctsStats] = None
    selected_actions: list[SelectedAction] = Field(default_factory=list)
    step_elapsed_ms: Optional[float] = None
    inference_ms: Optional[float] = None


class SessionCreateResponse(BaseModel):
    session_id: str
    game_id: str
    status: SessionStatus
    fem_enabled: bool
    ws_url: str


class SessionEnded(BaseModel):
    type: Literal["session_ended"] = "session_ended"
    status: SessionStatus
    reason: str
    final_score: Optional[float] = None
    final_compliance: Optional[float] = None
    final_volume: Optional[float] = None
    total_steps: int = 0


class EntryListQuery(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    search: Optional[str] = None
    strategy: Optional[str] = None
    bc_type: Optional[BcType] = None
    engine: Optional[str] = None
    version: Optional[str] = None
    connected_only: bool = False
    sort: Literal["recent", "seed_asc", "reward_desc", "compliance_asc"] = "recent"
