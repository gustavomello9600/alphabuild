"""
Self-Play Storage Module for AlphaBuilder MCTS Games.

Stores complete game trajectories with detailed MCTS statistics for:
1. Frontend visualization (replay with visit counts, Q values)
2. Continued training (policy improvement targets)

Uses sparse encoding for MCTS data (similar to policy in storage.py).
"""

import sqlite3
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import directly from storage.py to avoid loading mcts module (which needs scipy)
import sys
_storage_path = Path(__file__).parent.parent / "storage.py"
if str(_storage_path.parent) not in sys.path:
    sys.path.insert(0, str(_storage_path.parent))

from storage import (
    serialize_array,
    deserialize_array,
    sparse_encode,
    sparse_decode,
    serialize_sparse,
    deserialize_sparse,
    Phase,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SelectedAction:
    """A single selected action from MCTS micro-batch."""
    channel: int      # 0=add, 1=remove
    x: int
    y: int
    z: int
    visits: int
    q_value: float


@dataclass
class MCTSStats:
    """Statistics from a single MCTS search."""
    num_simulations: int
    nodes_expanded: int
    max_depth: int
    cache_hits: int
    top8_concentration: float  # Fraction of visits in top-8 actions
    refutation: bool           # True if MCTS disagrees with raw policy


@dataclass
class GameInfo:
    """Game-level metadata (stored once per game)."""
    game_id: str
    neural_engine: str          # 'simple' | 'swin-unetr'
    checkpoint_version: str     # 'warmup', 'selfplay-v1', etc.
    bc_masks: np.ndarray        # (3, D, H, W) - mask_x, mask_y, mask_z
    forces: np.ndarray          # (3, D, H, W) - fx, fy, fz
    load_config: Dict[str, Any]
    bc_type: str
    resolution: Tuple[int, int, int]
    final_score: Optional[float] = None
    final_compliance: Optional[float] = None
    final_volume: Optional[float] = None
    total_steps: int = 0


@dataclass
class GameStep:
    """Single step record with full MCTS data."""
    game_id: str
    step: int
    phase: Phase
    density: np.ndarray                    # (D, H, W)
    policy_add: np.ndarray                 # (D, H, W) - raw network policy
    policy_remove: np.ndarray              # (D, H, W)
    mcts_visit_add: np.ndarray             # (D, H, W) - visit counts N
    mcts_visit_remove: np.ndarray          # (D, H, W)
    mcts_q_add: np.ndarray                 # (D, H, W) - mean Q values
    mcts_q_remove: np.ndarray              # (D, H, W)
    selected_actions: List[SelectedAction] # Top-8 micro-batch
    value: float                           # Value head estimate
    mcts_stats: MCTSStats
    # New fields for island analysis and FEM (v2 schema)
    n_islands: int = 1                     # Number of connected components
    loose_voxels: int = 0                  # Voxels not in main island
    is_connected: bool = False             # Main island connects support to load
    compliance_fem: Optional[float] = None # Real compliance from FEM (Phase 2 only)
    max_displacement: Optional[float] = None # Max displacement from FEM
    island_penalty: float = 0.0            # Penalty applied for disconnected islands
    volume_fraction: float = 0.0           # Pre-calculated volume fraction
    reward_components: Dict[str, float] = field(default_factory=dict) # Detailed reward breakdown


# =============================================================================
# Database Initialization & Migration
# =============================================================================

def migrate_db(conn: sqlite3.Connection) -> None:
    """
    Check for missing columns and migrate schema if necessary.
    """
    cursor = conn.cursor()
    
    # Check existing columns
    cursor.execute("PRAGMA table_info(game_steps)")
    columns = [row[1] for row in cursor.fetchall()]
    
    # List of columns to add if missing: (name, type, default)
    migrations = [
        ("reward_components_json", "TEXT", None),
        ("connected_load_fraction", "REAL", "0.0"),
        ("value_target", "REAL", None),
        ("final_reward", "REAL", None),
    ]
    
    for col_name, col_type, default in migrations:
        if col_name not in columns:
            print(f"Migrating database: Adding {col_name} column...")
            try:
                default_clause = f" DEFAULT {default}" if default else ""
                cursor.execute(f"ALTER TABLE game_steps ADD COLUMN {col_name} {col_type}{default_clause}")
                conn.commit()
                print(f"Migration of {col_name} successful.")
            except sqlite3.Error as e:
                print(f"Migration of {col_name} failed: {e}")

def initialize_selfplay_db(db_path: Path) -> None:
    """
    Create the self-play database with MCTS-aware schema.
    
    Args:
        db_path: Path to the SQLite database file.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    
    # Run migration check on every startup
    migrate_db(conn)
    
    cursor = conn.cursor()
    
    # Games table: metadata per game
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            neural_engine TEXT NOT NULL,
            checkpoint_version TEXT NOT NULL,
            bc_masks_blob BLOB NOT NULL,
            forces_blob BLOB NOT NULL,
            load_config TEXT NOT NULL,
            bc_type TEXT NOT NULL,
            resolution TEXT NOT NULL,
            final_score REAL,
            final_compliance REAL,
            final_volume REAL,
            total_steps INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Game steps table: MCTS data per step (all sparse encoded)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            phase TEXT NOT NULL,
            density_blob BLOB NOT NULL,
            policy_add_blob BLOB,
            policy_remove_blob BLOB,
            mcts_visit_add_blob BLOB,
            mcts_visit_remove_blob BLOB,
            mcts_q_add_blob BLOB,
            mcts_q_remove_blob BLOB,
            selected_actions_json TEXT,
            value REAL NOT NULL,
            mcts_stats_json TEXT,
            n_islands INTEGER DEFAULT 1,
            loose_voxels INTEGER DEFAULT 0,
            is_connected INTEGER DEFAULT 0,
            compliance_fem REAL,
            max_displacement REAL,
            island_penalty REAL DEFAULT 0.0,
            volume_fraction REAL DEFAULT 0.0,
            reward_components_json TEXT,
            connected_load_fraction REAL DEFAULT 0.0,
            value_target REAL,
            final_reward REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_id) REFERENCES games(game_id),
            UNIQUE(game_id, step)
        )
    """)
    
    # Indexes for efficient queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_game_steps_game 
        ON game_steps(game_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_engine 
        ON games(neural_engine)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_games_version 
        ON games(checkpoint_version)
    """)
    
    conn.commit()
    conn.close()


# =============================================================================
# Save Functions
# =============================================================================

def save_game(db_path: Path, game: GameInfo) -> None:
    """
    Save game metadata. Called once at start of each game.
    
    Args:
        db_path: Path to the SQLite database file.
        game: Game information to save.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO games 
            (game_id, neural_engine, checkpoint_version, bc_masks_blob, 
             forces_blob, load_config, bc_type, resolution,
             final_score, final_compliance, final_volume, total_steps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game.game_id,
            game.neural_engine,
            game.checkpoint_version,
            serialize_array(game.bc_masks),
            serialize_array(game.forces),
            json.dumps(game.load_config),
            game.bc_type,
            json.dumps(game.resolution),
            game.final_score,
            game.final_compliance,
            game.final_volume,
            game.total_steps
        ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving game: {e}")
    finally:
        conn.close()


def record_step(
    db_path: Path,
    state: Any, # Typed as EpisodeState
    result: Any, # Typed as SearchResult
    value: float,
    policy_add: np.ndarray,
    policy_remove: np.ndarray,
    island_analysis: Optional[Dict] = None,
    compliance_fem: Optional[float] = None,
    max_displacement: Optional[float] = None,
    island_penalty: float = 0.0,
    volume_fraction: float = 0.0,
    reward_components: Optional[Dict[str, float]] = None,
    executed_actions: Optional[List[Tuple[int, int, int, int]]] = None,
    connected_load_fraction: float = 0.0,
    value_target: Optional[float] = None
) -> None:
    """
    Record a single game step with all associated data.
    """
    conn = sqlite3.connect(db_path)
    
    # Compress arrays
    density_blob = serialize_array(state.density)
    
    # Sparse compress policies
    idx_add, val_add = sparse_encode(policy_add)
    idx_rem, val_rem = sparse_encode(policy_remove)
    
    p_add_blob = serialize_sparse(idx_add, val_add)
    p_rem_blob = serialize_sparse(idx_rem, val_rem)
    
    # Serialize MCTS data - FULL TREE AGGREGATION
    # Use collect_subtree_visits() to get visits from ALL levels of the tree,
    # not just the immediate children of the root.
    resolution = state.density.shape
    v_add = np.zeros(resolution, dtype=np.float32)
    v_rem = np.zeros(resolution, dtype=np.float32)
    
    # Get aggregated visits from entire subtree
    if hasattr(result, 'root') and result.root is not None:
        full_tree_visits = result.root.collect_subtree_visits()
    else:
        # Fallback to shallow distribution
        full_tree_visits = result.visit_distribution
    
    for action, visits in full_tree_visits.items():
        # Action is (channel, x, y, z) tuple
        channel, x, y, z = action
        if channel == 0:
            v_add[x, y, z] += visits  # Use += for aggregation
        else:
            v_rem[x, y, z] += visits
            
    idx_v_add, val_v_add = sparse_encode(v_add)
    idx_v_rem, val_v_rem = sparse_encode(v_rem)
    
    v_add_blob = serialize_sparse(idx_v_add, val_v_add)
    v_rem_blob = serialize_sparse(idx_v_rem, val_v_rem)
    
    # Selected actions (Executed Sequence vs Top Candidates)
    # If executed_actions is provided (PV Sequence), stick to that.
    # Otherwise fallback to result.actions (Top-K alternatives).
    actions_to_record = executed_actions if executed_actions is not None else result.actions
    
    actions_data = []
    
    # If using sequence, we need to trace stats from root
    current_node = result.root if hasattr(result, 'root') else None
    
    for a in actions_to_record:
        channel, x, y, z = a
        visits = 0
        q_value = 0.0
        
        # Try to find stats in the tree
        if current_node and a in current_node.children:
            child = current_node.children[a]
            visits = child.visit_count
            q_value = child.mean_value
            # Advance for next action in sequence (if it is a sequence)
            # Only if we are using executed_actions (sequence)
            if executed_actions is not None:
                current_node = child
        elif current_node is None and not executed_actions:
             # Fallback for result.actions (roots children)
             visits = result.visit_distribution.get(a, 0)
        
        actions_data.append({
             "channel": channel,
             "x": x, "y": y, "z": z,
             "visits": visits,
             "q_value": q_value
        })
    
    actions_json = json.dumps(actions_data)
    
    # MCTS Stats
    # Assuming result has stats dict or object
    stats_data = {
        "num_simulations": result.num_simulations,
        "nodes_expanded": 0, # TODO: extract from engine if available
        "max_depth": 0,
        "cache_hits": 0,
        "top8_concentration": 0.0,
        "refutation": False
    }
    stats_json = json.dumps(stats_data)

    # Reward Components
    reward_components_json = json.dumps(reward_components) if reward_components else None
    
    try:
        cursor = conn.cursor()
        
        # v3 Insert with comprehensive metrics
        cursor.execute("""
            INSERT INTO game_steps (
                game_id, step, phase, density_blob, 
                policy_add_blob, policy_remove_blob,
                mcts_visit_add_blob, mcts_visit_remove_blob,
                selected_actions_json, value, mcts_stats_json,
                n_islands, loose_voxels, is_connected,
                compliance_fem, max_displacement, island_penalty, volume_fraction,
                reward_components_json, connected_load_fraction, value_target
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.game_id, state.current_step, state.phase.value, density_blob,
            p_add_blob, p_rem_blob,
            v_add_blob, v_rem_blob,
            actions_json, value, stats_json,
            island_analysis['n_islands'] if island_analysis else 1,
            island_analysis['loose_voxels'] if island_analysis else 0,
            1 if (island_analysis and island_analysis['is_connected']) else 0,
            compliance_fem, max_displacement, island_penalty, volume_fraction,
            reward_components_json, connected_load_fraction, value_target
        ))
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving game step: {e}")
    finally:
        conn.close()


def update_game_final(
    db_path: Path, 
    game_id: str,
    final_score: float,
    final_compliance: Optional[float] = None,
    final_volume: Optional[float] = None,
    total_steps: Optional[int] = None
) -> None:
    """
    Update game with final results after completion.
    Also backfills final_reward to all steps of this episode.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Update games table
        cursor.execute("""
            UPDATE games 
            SET final_score = ?, final_compliance = ?, 
                final_volume = ?, total_steps = COALESCE(?, total_steps)
            WHERE game_id = ?
        """, (final_score, final_compliance, final_volume, total_steps, game_id))
        
        # Backfill final_reward to ALL steps of this episode
        cursor.execute("""
            UPDATE game_steps
            SET final_reward = ?
            WHERE game_id = ?
        """, (final_score, game_id))
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error updating game: {e}")
    finally:
        conn.close()


# =============================================================================
# Load Functions
# =============================================================================

def load_game(db_path: Path, game_id: str) -> Optional[GameInfo]:
    """Load game metadata by ID."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT neural_engine, checkpoint_version, bc_masks_blob, forces_blob,
               load_config, bc_type, resolution, final_score, final_compliance,
               final_volume, total_steps
        FROM games WHERE game_id = ?
    """, (game_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    return GameInfo(
        game_id=game_id,
        neural_engine=row[0],
        checkpoint_version=row[1],
        bc_masks=deserialize_array(row[2]),
        forces=deserialize_array(row[3]),
        load_config=json.loads(row[4]),
        bc_type=row[5],
        resolution=tuple(json.loads(row[6])),
        final_score=row[7],
        final_compliance=row[8],
        final_volume=row[9],
        total_steps=row[10] or 0
    )


def load_game_step(
    db_path: Path, 
    game_id: str, 
    step: int,
    resolution: Tuple[int, int, int]
) -> Optional[GameStep]:
    """Load a single step and decode sparse MCTS data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT phase, density_blob, policy_add_blob, policy_remove_blob,
               mcts_visit_add_blob, mcts_visit_remove_blob,
               mcts_q_add_blob, mcts_q_remove_blob,
               selected_actions_json, value, mcts_stats_json,
               n_islands, loose_voxels, is_connected, compliance_fem,
               max_displacement, island_penalty, volume_fraction
        FROM game_steps WHERE game_id = ? AND step = ?
    """, (game_id, step))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    # Decode sparse arrays
    def decode_sparse_or_zeros(blob: Optional[bytes]) -> np.ndarray:
        if blob is None:
            return np.zeros(resolution, dtype=np.float32)
        indices, values = deserialize_sparse(blob)
        return sparse_decode(indices, values, resolution)
    
    # Parse selected actions
    actions_data = json.loads(row[8]) if row[8] else []
    selected_actions = [
        SelectedAction(
            channel=a['channel'],
            x=a['x'], y=a['y'], z=a['z'],
            visits=a['visits'],
            q_value=a['q_value']
        )
        for a in actions_data
    ]
    
    # Parse MCTS stats
    stats_data = json.loads(row[10]) if row[10] else {}
    mcts_stats = MCTSStats(
        num_simulations=stats_data.get('num_simulations', 0),
        nodes_expanded=stats_data.get('nodes_expanded', 0),
        max_depth=stats_data.get('max_depth', 0),
        cache_hits=stats_data.get('cache_hits', 0),
        top8_concentration=stats_data.get('top8_concentration', 0.0),
        refutation=stats_data.get('refutation', False),
    )
    
    return GameStep(
        game_id=game_id,
        step=step,
        phase=Phase(row[0]),
        density=deserialize_array(row[1]),
        policy_add=decode_sparse_or_zeros(row[2]),
        policy_remove=decode_sparse_or_zeros(row[3]),
        mcts_visit_add=decode_sparse_or_zeros(row[4]),
        mcts_visit_remove=decode_sparse_or_zeros(row[5]),
        mcts_q_add=decode_sparse_or_zeros(row[6]),
        mcts_q_remove=decode_sparse_or_zeros(row[7]),
        selected_actions=selected_actions,
        value=row[9],
        mcts_stats=mcts_stats,
        n_islands=row[11] if row[11] is not None else 1,
        loose_voxels=row[12] if row[12] is not None else 0,
        is_connected=bool(row[13]) if row[13] is not None else False,
        compliance_fem=row[14],
        max_displacement=row[15],
        island_penalty=row[16] if row[16] is not None else 0.0,
        volume_fraction=row[17] if row[17] is not None else 0.0,
    )


def load_all_game_steps(
    db_path: Path, 
    game_id: str,
    resolution: Tuple[int, int, int]
) -> List[GameStep]:
    """Load all steps for a game (batch load for replay)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT step FROM game_steps 
        WHERE game_id = ? 
        ORDER BY step
    """, (game_id,))
    
    steps = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return [
        load_game_step(db_path, game_id, s, resolution)
        for s in steps
    ]


def load_game_steps_range(
    db_path: Path, 
    game_id: str,
    resolution: Tuple[int, int, int],
    start: int = 0,
    end: int = 50
) -> List[GameStep]:
    """Load a range of steps for a game (paginated load)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get only the steps in range
    limit = end - start
    cursor.execute("""
        SELECT phase, density_blob, policy_add_blob, policy_remove_blob,
               mcts_visit_add_blob, mcts_visit_remove_blob,
               mcts_q_add_blob, mcts_q_remove_blob,
               selected_actions_json, value, mcts_stats_json, step,
               n_islands, loose_voxels, is_connected, volume_fraction,
               reward_components_json, compliance_fem, island_penalty, connected_load_fraction
        FROM game_steps 
        WHERE game_id = ? 
        ORDER BY step
        LIMIT ? OFFSET ?
    """, (game_id, limit, start))
    
    rows = cursor.fetchall()
    conn.close()
    
    def decode_sparse_or_zeros(blob: Optional[bytes]) -> np.ndarray:
        if blob is None:
            return np.zeros(resolution, dtype=np.float32)
        indices, values = deserialize_sparse(blob)
        return sparse_decode(indices, values, resolution)
    
    result = []
    for row in rows:
        # Parse selected actions
        actions_data = json.loads(row[8]) if row[8] else []
        selected_actions = [
            SelectedAction(
                channel=a['channel'],
                x=a['x'], y=a['y'], z=a['z'],
                visits=a['visits'],
                q_value=a['q_value']
            )
            for a in actions_data
        ]
        
        # Parse MCTS stats
        stats_data = json.loads(row[10]) if row[10] else {}
        mcts_stats = MCTSStats(
            num_simulations=stats_data.get('num_simulations', 0),
            nodes_expanded=stats_data.get('nodes_expanded', 0),
            max_depth=stats_data.get('max_depth', 0),
            cache_hits=stats_data.get('cache_hits', 0),
            top8_concentration=stats_data.get('top8_concentration', 0.0),
            refutation=stats_data.get('refutation', False),
        )
        
        # Parse reward components (new)
        reward_components_raw = json.loads(row[16]) if row[16] else {}
        
        result.append(GameStep(
            game_id=game_id,
            step=row[11],
            phase=Phase(row[0]),
            density=deserialize_array(row[1]),
            policy_add=decode_sparse_or_zeros(row[2]),
            policy_remove=decode_sparse_or_zeros(row[3]),
            mcts_visit_add=decode_sparse_or_zeros(row[4]),
            mcts_visit_remove=decode_sparse_or_zeros(row[5]),
            mcts_q_add=decode_sparse_or_zeros(row[6]),
            mcts_q_remove=decode_sparse_or_zeros(row[7]),
            selected_actions=selected_actions,
            value=row[9],
            mcts_stats=mcts_stats,
            n_islands=row[12] if row[12] is not None else 1,
            loose_voxels=row[13] if row[13] is not None else 0,
            is_connected=bool(row[14]) if row[14] is not None else False,
            volume_fraction=row[15] if row[15] is not None else 0.0,
            reward_components=reward_components_raw,
            compliance_fem=row[17] if row[17] is not None else None,
            island_penalty=row[18] if row[18] is not None else 0.0,
        ))
    
    return result


# =============================================================================
# Query Functions
# =============================================================================

@dataclass
class GameSummary:
    """Summary info for game listing."""
    game_id: str
    neural_engine: str
    checkpoint_version: str
    final_score: Optional[float]
    final_compliance: Optional[float]
    final_volume: Optional[float]
    total_steps: int
    created_at: str


def list_games(
    db_path: Path,
    engine: Optional[str] = None,
    version: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[GameSummary]:
    """
    List games with optional filtering.
    
    Args:
        db_path: Path to database
        engine: Filter by neural engine ('simple', 'swin-unetr')
        version: Filter by checkpoint version
        limit: Max results
        offset: Pagination offset
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
        SELECT game_id, neural_engine, checkpoint_version,
               final_score, final_compliance, final_volume,
               total_steps, created_at
        FROM games
        WHERE 1=1
    """
    params = []
    
    if engine:
        query += " AND neural_engine = ?"
        params.append(engine)
    if version:
        query += " AND checkpoint_version = ?"
        params.append(version)
    
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        GameSummary(
            game_id=row[0],
            neural_engine=row[1],
            checkpoint_version=row[2],
            final_score=row[3],
            final_compliance=row[4],
            final_volume=row[5],
            total_steps=row[6] or 0,
            created_at=row[7]
        )
        for row in rows
    ]


def get_game_count(db_path: Path) -> int:
    """Get total number of games."""
    db_path = Path(db_path)
    if not db_path.exists():
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        count = 0
    
    conn.close()
    return count


def get_last_step(db_path: Path, game_id: str) -> Optional[int]:
    """
    Get the last recorded step number for a game.
    
    Returns None if no steps exist for the game.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT MAX(step) FROM game_steps WHERE game_id = ?
        """, (game_id,))
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def get_incomplete_games(db_path: Path, max_steps_limit: int = 600) -> List[GameSummary]:
    """
    List games that have recorded steps but haven't reached terminal state.
    
    A game is considered incomplete if:
    - It has at least one step recorded
    - Its total_steps < max_steps_limit (hasn't exhausted move budget)
    - Note: We can't easily detect terminal conditions without re-running physics,
      so we use a heuristic based on step count.
    
    Returns:
        List of incomplete games, sorted by created_at descending (most recent first)
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Find games that have steps but haven't reached max_steps
        cursor.execute("""
            SELECT g.game_id, g.neural_engine, g.checkpoint_version,
                   g.final_score, g.final_compliance, g.final_volume,
                   g.total_steps, g.created_at
            FROM games g
            WHERE g.total_steps > 0 
              AND g.total_steps < ?
              AND EXISTS (SELECT 1 FROM game_steps gs WHERE gs.game_id = g.game_id)
            ORDER BY g.created_at DESC
        """, (max_steps_limit,))
        
        rows = cursor.fetchall()
        return [
            GameSummary(
                game_id=row[0],
                neural_engine=row[1],
                checkpoint_version=row[2],
                final_score=row[3],
                final_compliance=row[4],
                final_volume=row[5],
                total_steps=row[6] or 0,
                created_at=row[7]
            )
            for row in rows
        ]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def generate_game_id() -> str:
    """Generate a unique game ID."""
    return str(uuid.uuid4())
