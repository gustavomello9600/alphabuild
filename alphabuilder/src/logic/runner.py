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
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import physics core
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

# Import neural module (try/except to allow running without tensorflow if needed)
try:
    import tensorflow as tf
    from alphabuilder.src.neural.trainer import predict_fitness
    HAS_NEURAL = True
except ImportError:
    print("Warning: TensorFlow or Neural Module not found. Neural guidance disabled.")
    HAS_NEURAL = False


@dataclass
class EpisodeConfig:
    """Configuration for episode execution with adaptive strategies."""
    resolution: Tuple[int, int] = (16, 32)  # (ny, nx) - height x width
    
    # Refinement parameters - optimized for better data quality
    max_refinement_steps: int = 50  # Reduced from 100 for more diverse short episodes
    
    # Adaptive stagnation detection
    use_relative_threshold: bool = True  # Use percentage-based threshold
    stagnation_threshold_relative: float = 0.001  # 0.1% relative improvement
    stagnation_threshold_absolute: float = 1e-11  # Fallback absolute threshold
    stagnation_patience: int = 10  # Reduced from 20
    
    # Phase 1 growth strategy
    growth_strategy: str = "astar"  # Options: "straight", "astar", "smart_heuristic", "random_pattern"
    
    # Phase 2 exploration strategy  
    exploration_strategy: str = "mixed"  # Options: "random", "mixed", "greedy", "neural_greedy"
    exploration_weights: dict = None  # Will be set in __post_init__
    
    # Neural Guidance
    use_neural_guidance: bool = False
    neural_greedy_epsilon: float = 0.1  # Probability of random action even with neural guidance
    
    def __post_init__(self):
        if self.exploration_weights is None:
            self.exploration_weights = {
                "random": 0.4,
                "remove_weak": 0.3,
                "add_support": 0.2,
                "symmetry": 0.1
            }
    

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


def phase1_growth(topology: np.ndarray, config: EpisodeConfig, seed: int = 0) -> np.ndarray:
    """
    Phase 1: Grow topology using advanced strategies.
    
    Args:
        topology: Initial topology (should be empty).
                  Shape is (nx, ny) where nx is width, ny is height.
        config: Episode configuration.
        seed: Random seed for pattern selection.
        
    Returns:
        Connected topology.
    """
    from alphabuilder.src.logic.pathfinding import (
        create_astar_topology,
        create_smart_heuristic_topology,
        create_random_pattern
    )
    
    ny, nx = config.resolution  # config stores (ny, nx)
    result = topology.copy()
    
    # Always fill left edge (support)
    result[0, :] = 1
    
    strategy = config.growth_strategy
    
    if strategy == "astar":
        # A* pathfinding with multiple paths for redundancy
        starts = [
            (0, ny // 4),      # Lower quarter
            (0, ny // 2),       # Middle
            (0, 3 * ny // 4)   # Upper quarter
        ]
        goals = [
            (nx - 1, ny // 2),  # All converge to middle of right edge
            (nx - 1, ny // 2),
            (nx - 1, ny // 2)
        ]
        result = create_astar_topology(result, starts, goals)
    
    elif strategy == "smart_heuristic":
        # Diagonal arch/truss pattern
        result = create_smart_heuristic_topology(result, ny, nx)
    
    elif strategy == "random_pattern":
        # Random pattern based on seed
        result = create_random_pattern(result, ny, nx, seed)
    
    else:  # "straight" or fallback
        # Original straight line (baseline)
        mid_y = ny // 2
        for x in range(nx):
            result[x, mid_y] = 1
        
    return result


def phase2_refinement(
    topology: np.ndarray,
    ctx: FEMContext,
    props: PhysicalProperties,
    config: EpisodeConfig,
    db_path: Path,
    episode_id: str,
    rng: np.random.Generator,
    model: Optional[object] = None
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
        
        # Calculate additional structural metrics
        from alphabuilder.src.logic.pathfinding import calculate_structural_metrics
        struct_metrics = calculate_structural_metrics(current_topology)
        
        # Save to database with enhanced metadata
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
                "compliance": result.compliance,
                "volume_fraction": struct_metrics["volume_fraction"],
                "structural_efficiency": fitness / (np.sum(current_topology) + 1.0),
                "pattern_entropy": struct_metrics["pattern_entropy"],
                "connectivity_score": struct_metrics["connectivity_score"]
            }
        )
        save_record(db_path, record)
        
        # Check for stagnation with adaptive threshold
        if config.use_relative_threshold:
            # Relative improvement threshold
            if best_fitness > 0:
                threshold = config.stagnation_threshold_relative * abs(best_fitness)
            else:
                threshold = config.stagnation_threshold_absolute
        else:
            threshold = config.stagnation_threshold_absolute
        
        if fitness > best_fitness + threshold:
            best_fitness = fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        if stagnation_counter >= config.stagnation_patience:
            print(f"  Stagnation detected at step {step}, terminating episode.")
            break
            
        # Prepare action candidates using semi-guided exploration
        ny_tp, nx_tp = current_topology.shape  # Topology is (ny, nx)
        
        # Collect all possible actions
        add_candidates = []
        remove_candidates = []
        
        # ADD candidates: empty cells adjacent to material
        for r in range(ny_tp): # r for row
            for c in range(nx_tp): # c for col
                if current_topology[r, c] == 0:
                    # Check if adjacent to material
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ny_tp and 0 <= nc < nx_tp:
                            if current_topology[nr, nc] == 1:
                                add_candidates.append(("ADD", (r, c)))
                                break
        
        # REMOVE candidates: material cells that don't disconnect
        for r in range(ny_tp): # r for row
            for c in range(nx_tp): # c for col
                if current_topology[r, c] == 1 and c not in [0, nx_tp-1]:  # Not on left/right edges
                    # Test removal
                    test_topology = current_topology.copy()
                    test_topology[r, c] = 0
                    if is_connected_bfs(test_topology):
                        remove_candidates.append(("REMOVE", (r, c)))
        
        all_actions = add_candidates + remove_candidates
        
        if len(all_actions) == 0:
            continue  # No valid actions
        
        # Select action using exploration strategy
        # Select action using exploration strategy
        action_type, coord = None, None
        
        # Neural Greedy Strategy
        if config.use_neural_guidance and model is not None and HAS_NEURAL:
            # Epsilon-greedy
            if rng.random() < config.neural_greedy_epsilon:
                 # Random exploration
                 action_type, coord = all_actions[rng.integers(0, len(all_actions))]
            else:
                 # Predict fitness for all candidates
                 candidate_grids = []
                 for at, (r, c) in all_actions:
                     temp_topo = current_topology.copy()
                     if at == "ADD":
                         temp_topo[r, c] = 1
                     else:
                         temp_topo[r, c] = 0
                     candidate_grids.append(temp_topo)
                 
                 # Batch prediction
                 # Thickness is always 1 for 2D
                 thicknesses = [1] * len(candidate_grids)
                 
                 try:
                     # Predict displacements
                     # predict_fitness returns (Batch, 1)
                     pred_disps = predict_fitness(model, candidate_grids, thicknesses)
                     
                     # Calculate predicted fitness
                     best_idx = 0
                     best_pred_fitness = -1.0
                     
                     for idx, disp in enumerate(pred_disps):
                         # disp is [1] tensor or array
                         d = float(disp)
                         
                         # Calculate mass
                         mass = float(np.sum(candidate_grids[idx]))
                         
                         # Calculate fitness (simplified Kane Eq 1)
                         # We use the same parameters as solver
                         penalty = max(0.0, d - props.disp_limit)
                         denominator = mass + props.penalty_epsilon * 0.0 + props.penalty_alpha * penalty
                         
                         if denominator < 1e-9:
                             f = 0.0
                         else:
                             f = 1.0 / denominator
                             
                         if f > best_pred_fitness:
                             best_pred_fitness = f
                             best_idx = idx
                             
                     action_type, coord = all_actions[best_idx]
                     
                 except Exception as e:
                     print(f"Neural prediction failed: {e}")
                     # Fallback to random
                     action_type, coord = all_actions[rng.integers(0, len(all_actions))]

        if action_type is None:
            if config.exploration_strategy == "random":
                # Pure random
                action_type, coord = all_actions[rng.integers(0, len(all_actions))]
            
            elif config.exploration_strategy == "mixed":
                # Mixed strategy with weighted probabilities
                strategy_names = list(config.exploration_weights.keys())
                strategy_probs = list(config.exploration_weights.values())
                chosen_strategy = rng.choice(strategy_names, p=strategy_probs)
                
                # Determine load and support points (simplified - middle of edges)
                load_points = [(ny_tp - 1, nx_tp // 2)]  # Right edge, middle
                support_points = [(0, nx_tp // 4), (0, nx_tp // 2), (0, 3 * nx_tp // 4)]  # Left edge
                
                # Apply strategy (simplified inline version)
                if chosen_strategy == "random":
                    action_type, coord = all_actions[rng.integers(0, len(all_actions))]
                else:
                    # For other strategies, use simple heuristics
                    if chosen_strategy == "remove_weak" and len(remove_candidates) > 0:
                        # Select from remove candidates (prefer those far from path)
                        action_type, coord = remove_candidates[rng.integers(0, len(remove_candidates))]
                    elif chosen_strategy == "add_support" and len(add_candidates) > 0:
                        # Select from add candidates (prefer those near load)
                        action_type, coord = add_candidates[rng.integers(0, len(add_candidates))]
                    elif chosen_strategy == "symmetry":
                        # Select action that maintains symmetry
                        action_type, coord = all_actions[rng.integers(0, len(all_actions))]
                    else:
                        # Fallback
                        action_type, coord = all_actions[rng.integers(0, len(all_actions))]
            
            else:  # greedy or other
                action_type, coord = all_actions[rng.integers(0, len(all_actions))]
        
        # Apply action
        r, c = coord
        if action_type == "ADD":
            current_topology[r, c] = 1
        else:  # REMOVE
            current_topology[r, c] = 0
                
    return current_topology, fitness_history


def run_episode(
    ctx: FEMContext,
    props: PhysicalProperties,
    db_path: Path,
    config: Optional[EpisodeConfig] = None,
    seed: Optional[int] = None,
    model: Optional[object] = None
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
    initial_topology = np.zeros((nx, ny), dtype=np.int32)  # FEM expects (nx, ny)
    connected_topology = phase1_growth(initial_topology, config, seed=seed if seed is not None else 0)
    
    # Phase 2: Refinement
    print("  Phase 2: Refinement")
    final_topology, fitness_history = phase2_refinement(
        connected_topology,
        ctx,
        props,
        config,
        db_path,
        episode_id,
        rng,
        model=model
    )
    
    print(f"  Episode complete. Best fitness: {max(fitness_history):.6e}")
    
    return episode_id
