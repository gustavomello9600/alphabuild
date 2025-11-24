"""
Episode Runner for AlphaBuilder Training Data Generation.

Implements the two-phase execution pipeline:
1. Growth Phase: Connect loads to supports using greedy pathfinding
2. Refinement Phase: Random exploration with FEM validation
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass
import sys

# Import physics core
sys.path.append(str(Path(__file__).parent.parent.parent))
from alphabuilder.src.core.physics_model import (
    initialize_cantilever_context,
    PhysicalProperties,
    FEMContext
)
from alphabuilder.src.core.solver import solve_topology
from alphabuilder.src.logic.storage import (
    TrainingRecord,
    Phase,
    serialize_state,
    save_record,
    generate_episode_id
)


@dataclass
class EpisodeConfig:
    """Configuration for episode execution."""
    resolution: Tuple[int, int] = (32, 16)  # (nx, ny)
    max_refinement_steps: int = 100
    stagnation_threshold: float = 1e-4
    stagnation_patience: int = 20
    

def find_boundary_cells(topology: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find cells on the boundary of filled material.
    
    Args:
        topology: Binary topology matrix (ny, nx).
        
    Returns:
        List of (row, col) coordinates of boundary cells.
    """
    ny, nx = topology.shape
    boundary = []
    
    for i in range(ny):
        for j in range(nx):
            if topology[i, j] == 1:
                # Check if adjacent to empty space
                is_boundary = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ny and 0 <= nj < nx:
                        if topology[ni, nj] == 0:
                            is_boundary = True
                            break
                if is_boundary:
                    boundary.append((i, j))
                    
    return boundary


def is_connected_bfs(topology: np.ndarray, start_col: int = 0, end_col: int = -1) -> bool:
    """
    Check if there's a path from left edge to right edge using BFS.
    
    Args:
        topology: Binary topology matrix.
        start_col: Starting column (default: 0, left edge).
        end_col: Ending column (default: -1, right edge).
        
    Returns:
        True if connected, False otherwise.
    """
    ny, nx = topology.shape
    if end_col == -1:
        end_col = nx - 1
        
    # Find starting points on left edge
    start_points = [(i, start_col) for i in range(ny) if topology[i, start_col] == 1]
    if not start_points:
        return False
        
    # BFS
    visited = set()
    queue = start_points.copy()
    visited.update(start_points)
    
    while queue:
        i, j = queue.pop(0)
        
        # Check if we reached the right edge
        if j == end_col:
            return True
            
        # Explore neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx:
                if topology[ni, nj] == 1 and (ni, nj) not in visited:
                    visited.add((ni, nj))
                    queue.append((ni, nj))
                    
    return False


def phase1_growth(topology: np.ndarray, config: EpisodeConfig) -> np.ndarray:
    """
    Phase 1: Grow topology from left to right using greedy connection.
    
    Args:
        topology: Initial topology (should be empty or have left edge filled).
        config: Episode configuration.
        
    Returns:
        Connected topology.
    """
    ny, nx = config.resolution
    result = topology.copy()
    
    # Initialize left edge (support)
    result[:, 0] = 1
    
    # Greedy growth: connect to right edge
    current_col = 0
    target_row = ny // 2  # Middle of right edge
    
    # Simple straight line connection for now
    for col in range(nx):
        result[target_row, col] = 1
        
    return result


def phase2_refinement(
    topology: np.ndarray,
    ctx: FEMContext,
    props: PhysicalProperties,
    config: EpisodeConfig,
    db_path: Path,
    episode_id: str,
    rng: np.random.Generator
) -> Tuple[np.ndarray, List[float]]:
    """
    Phase 2: Refine topology using random exploration with FEM validation.
    
    Args:
        topology: Initial connected topology from Phase 1.
        ctx: Pre-initialized FEM context.
        props: Physical properties.
        config: Episode configuration.
        db_path: Database path for saving records.
        episode_id: Current episode ID.
        rng: Random number generator.
        
    Returns:
        Tuple of (final topology, fitness history).
    """
    current_topology = topology.copy()
    fitness_history = []
    stagnation_counter = 0
    best_fitness = -np.inf
    
    for step in range(config.max_refinement_steps):
        # Solve current topology
        result = solve_topology(current_topology, ctx, props)
        fitness = result.fitness
        fitness_history.append(fitness)
        
        # Save to database
        state_blob = serialize_state(current_topology)
        record = TrainingRecord(
            episode_id=episode_id,
            step=step,
            phase=Phase.REFINEMENT,
            state_blob=state_blob,
            fitness_score=fitness,
            valid_fem=result.valid,
            metadata={
                "max_displacement": result.max_displacement,
                "compliance": result.compliance
            }
        )
        save_record(db_path, record)
        
        # Check for stagnation
        if fitness > best_fitness + config.stagnation_threshold:
            best_fitness = fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        if stagnation_counter >= config.stagnation_patience:
            print(f"  Stagnation detected at step {step}, terminating episode.")
            break
            
        # Random action: ADD or REMOVE
        action = rng.choice(['add', 'remove'])
        
        if action == 'add':
            # Find empty cells adjacent to material
            candidates = []
            ny, nx = current_topology.shape
            for i in range(ny):
                for j in range(nx):
                    if current_topology[i, j] == 0:
                        # Check if adjacent to material
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < ny and 0 <= nj < nx:
                                if current_topology[ni, nj] == 1:
                                    candidates.append((i, j))
                                    break
            
            if candidates:
                i, j = candidates[rng.integers(0, len(candidates))]
                current_topology[i, j] = 1
                
        else:  # remove
            # Find material cells that can be removed without disconnecting
            candidates = []
            ny, nx = current_topology.shape
            for i in range(ny):
                for j in range(nx):
                    if current_topology[i, j] == 1 and j not in [0, nx-1]:  # Not on edges
                        # Test removal
                        test_topology = current_topology.copy()
                        test_topology[i, j] = 0
                        if is_connected_bfs(test_topology):
                            candidates.append((i, j))
            
            if candidates:
                i, j = candidates[rng.integers(0, len(candidates))]
                current_topology[i, j] = 0
                
    return current_topology, fitness_history


def run_episode(
    ctx: FEMContext,
    props: PhysicalProperties,
    db_path: Path,
    config: Optional[EpisodeConfig] = None,
    seed: Optional[int] = None
) -> str:
    """
    Run a complete episode (Growth + Refinement) and save to database.
    
    Args:
        ctx: Pre-initialized FEM context (reused across episodes).
        props: Physical properties.
        db_path: Path to SQLite database.
        config: Episode configuration (uses defaults if None).
        seed: Random seed for reproducibility.
        
    Returns:
        Episode ID.
    """
    if config is None:
        config = EpisodeConfig()
        
    rng = np.random.default_rng(seed)
    episode_id = generate_episode_id()
    
    print(f"Starting episode {episode_id[:8]}...")
    
    # Phase 1: Growth
    print("  Phase 1: Growth")
    ny, nx = config.resolution
    initial_topology = np.zeros((ny, nx), dtype=np.int32)
    connected_topology = phase1_growth(initial_topology, config)
    
    # Phase 2: Refinement
    print("  Phase 2: Refinement")
    final_topology, fitness_history = phase2_refinement(
        connected_topology,
        ctx,
        props,
        config,
        db_path,
        episode_id,
        rng
    )
    
    print(f"  Episode complete. Best fitness: {max(fitness_history):.6f}")
    
    return episode_id
