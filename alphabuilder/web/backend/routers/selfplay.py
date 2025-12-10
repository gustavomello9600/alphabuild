from fastapi import APIRouter, HTTPException
from pathlib import Path
import sqlite3
import json
import numpy as np
from typing import Optional, List
from pydantic import BaseModel
import sys

try:
    from ..fastapi_utils import get_selfplay_db_path
except (ImportError, ValueError):
    from fastapi_utils import get_selfplay_db_path

# Add logic path to sys.path to import selfplay logic
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to import selfplay logic
try:
    from alphabuilder.src.logic.selfplay.storage import (
        list_games as sp_list_games,
        load_game as sp_load_game,
        load_all_game_steps,
        load_game_steps_range
    )
    SELFPLAY_AVAILABLE = True
except ImportError:
    SELFPLAY_AVAILABLE = False
    print("Warning: Self-play logic not available")

router = APIRouter()

# --- Pydantic Models ---

class SelfPlayGameSummary(BaseModel):
    game_id: str
    neural_engine: str
    checkpoint_version: str
    final_score: Optional[float]
    final_compliance: Optional[float]
    final_volume: Optional[float]
    total_steps: int
    created_at: str

class SelectedActionData(BaseModel):
    channel: int
    x: int
    y: int
    z: int
    visits: int
    q_value: float

class MCTSStatsData(BaseModel):
    num_simulations: int
    nodes_expanded: int
    max_depth: int
    cache_hits: int
    top8_concentration: float
    refutation: bool

class SelfPlayStepData(BaseModel):
    step: int
    phase: str
    tensor_shape: list[int]
    tensor_data: list[float]
    policy_add: list[float]
    policy_remove: list[float]
    mcts_visit_add: list[float]
    mcts_visit_remove: list[float]
    mcts_q_add: list[float]
    mcts_q_remove: list[float]
    selected_actions: list[SelectedActionData]
    value: float
    mcts_stats: MCTSStatsData
    n_islands: int = 1
    is_connected: bool = False
    volume_fraction: float = 0.0
    # Detailed reward fields
    compliance_fem: Optional[float] = None
    max_displacement: Optional[float] = None
    island_penalty: float = 0.0
    loose_voxels: int = 0
    reward_components: Optional[dict] = None


class SelfPlayGameData(BaseModel):
    game_id: str
    neural_engine: str
    checkpoint_version: str
    bc_type: str
    resolution: list[int]
    final_score: Optional[float]
    final_compliance: Optional[float]
    final_volume: Optional[float]
    total_steps: int
    steps: list[SelfPlayStepData]

class SelfPlayGameMetadata(BaseModel):
    """Metadata for a self-play game, for efficient loading."""
    game_id: str
    neural_engine: str
    checkpoint_version: str
    bc_type: str
    resolution: list[int]
    final_score: Optional[float]
    final_compliance: Optional[float]
    final_volume: Optional[float]
    total_steps: int
    value_history: Optional[list[float]] = None

# --- Endpoints ---

@router.get("/games", response_model=list[SelfPlayGameSummary])
def list_selfplay_games(
    engine: Optional[str] = None,
    version: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    deprecated: bool = False
):
    """List self-play games with optional filtering."""
    if not SELFPLAY_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-play module not available")
    
    db_path = get_selfplay_db_path(deprecated)
    if not db_path.exists():
        return []  # No games yet
    
    games = sp_list_games(
        db_path,
        engine=engine,
        version=version,
        limit=limit,
        offset=offset
    )
    
    return [
        SelfPlayGameSummary(
            game_id=g.game_id,
            neural_engine=g.neural_engine,
            checkpoint_version=g.checkpoint_version,
            final_score=g.final_score,
            final_compliance=g.final_compliance,
            final_volume=g.final_volume,
            total_steps=g.total_steps,
            created_at=g.created_at,
        )
        for g in games
    ]

@router.get("/games/{game_id}/metadata", response_model=SelfPlayGameMetadata)
def get_selfplay_game_metadata(game_id: str, deprecated: bool = False):
    """Get game metadata without steps - fast initial load."""
    if not SELFPLAY_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-play module not available")
    
    db_path = get_selfplay_db_path(deprecated)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Self-play database not found")
    
    game = sp_load_game(db_path, game_id)
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    
    # Get value history directly from DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT value FROM game_steps 
        WHERE game_id = ? ORDER BY step ASC
    """, (game_id,))
    value_rows = cursor.fetchall()
    value_history = [row[0] for row in value_rows]
    conn.close()

    return SelfPlayGameMetadata(
        game_id=game.game_id,
        neural_engine=game.neural_engine,
        checkpoint_version=game.checkpoint_version,
        bc_type=game.bc_type,
        resolution=list(game.resolution),
        final_score=game.final_score,
        final_compliance=game.final_compliance,
        final_volume=game.final_volume,
        total_steps=game.total_steps,
        value_history=value_history
    )

@router.get("/games/{game_id}/steps", response_model=list[SelfPlayStepData])
def get_selfplay_game_steps(
    game_id: str, 
    start: int = 0, 
    end: int = 50,
    deprecated: bool = False
):
    """Get range of steps for a self-play game."""
    if not SELFPLAY_AVAILABLE:
        raise HTTPException(status_code=501, detail="Self-play module not available")
    
    db_path = get_selfplay_db_path(deprecated)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Self-play database not found")
        
    game = sp_load_game(db_path, game_id)
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    steps = load_game_steps_range(db_path, game_id, game.resolution, start, end)
    
    step_data = []
    for s in steps:
        # Reconstruct full state tensor (7, D, H, W)
        state_tensor = np.concatenate([
            s.density[None],      # (1, D, H, W)
            game.bc_masks,        # (3, D, H, W)
            game.forces           # (3, D, H, W)
        ], axis=0)
        
        step_data.append(SelfPlayStepData(
            step=s.step,
            phase=s.phase.value,
            tensor_shape=list(state_tensor.shape),
            tensor_data=state_tensor.flatten().tolist(),
            policy_add=s.policy_add.flatten().tolist(),
            policy_remove=s.policy_remove.flatten().tolist(),
            mcts_visit_add=s.mcts_visit_add.flatten().tolist(),
            mcts_visit_remove=s.mcts_visit_remove.flatten().tolist(),
            mcts_q_add=s.mcts_q_add.flatten().tolist(),
            mcts_q_remove=s.mcts_q_remove.flatten().tolist(),
            selected_actions=[
                SelectedActionData(
                    channel=a.channel,
                    x=a.x, y=a.y, z=a.z,
                    visits=a.visits,
                    q_value=a.q_value
                )
                for a in s.selected_actions
            ],
            value=s.value,
            mcts_stats=MCTSStatsData(
                num_simulations=s.mcts_stats.num_simulations,
                nodes_expanded=s.mcts_stats.nodes_expanded,
                max_depth=s.mcts_stats.max_depth,
                cache_hits=s.mcts_stats.cache_hits,
                top8_concentration=s.mcts_stats.top8_concentration,
                refutation=s.mcts_stats.refutation,
            ),
            n_islands=getattr(s, 'n_islands', 1),
            is_connected=getattr(s, 'is_connected', False),
            volume_fraction=getattr(s, 'volume_fraction', 0.0),
            compliance_fem=getattr(s, 'compliance_fem', None),
            max_displacement=getattr(s, 'max_displacement', None),
            island_penalty=getattr(s, 'island_penalty', 0.0),
            loose_voxels=getattr(s, 'loose_voxels', 0),
            reward_components=getattr(s, 'reward_components', None) or None,
        ))
        
    return step_data
