import numpy as np
from pathlib import Path
from typing import Optional
from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties
from alphabuilder.src.core.solver import solve_topology_3d
from alphabuilder.src.logic.game_rules import GameState
from alphabuilder.src.logic.mcts import MCTSAgent
from dataclasses import dataclass

@dataclass
class EpisodeConfig:
    resolution: tuple = (64, 32, 32)
    max_refinement_steps: int = 100
    stagnation_threshold: float = 1e-4
    stagnation_patience: int = 20
    growth_strategy: str = "random"
    exploration_strategy: str = "random"

def run_episode_v1_1(
    db_path: Path,
    max_steps: int = 20,
    model = None,
    resolution: tuple = (64, 32, 32),
    config: Optional[EpisodeConfig] = None,
    ctx = None,
    props = None,
    seed: int = None
):
    """
    Run a v1.1 Episode:
    1. Init 3D Context
    2. Loop (MCTS -> Action -> Physics -> Reward)
    """
    if config:
        resolution = config.resolution
        max_steps = config.max_refinement_steps
        
    if ctx is None:
        print(f"Initializing 3D Physics Context with resolution {resolution}...")
        ctx = initialize_cantilever_context(resolution=resolution)
        
    if props is None:
        props = PhysicalProperties()
    
    # Init State (Empty 5D Tensor)
    # (5, D, H, W)
    D, H, W = resolution
    tensor = np.zeros((5, D, H, W), dtype=np.float32)
    
    # Set BCs (Left Wall Support)
    tensor[1, :, :, 0] = 1.0
    
    # Set Load (Tip)
    # Middle Depth, Middle Height, Right Edge
    tensor[3, D//2, H//2, W-1] = -1.0
    
    state = GameState(tensor=tensor, phase='GROWTH', step_count=0)
    
    agent = MCTSAgent(model=model, num_simulations=10) # Low sim for speed
    
    print("Starting Game Loop...")
    for step in range(max_steps):
        print(f"Step {step}: Phase={state.phase}, Density={np.mean(state.density):.4f}")
        
        # 1. MCTS Select Action
        action_type, coord = agent.search(state)
        print(f"  Action: {action_type} at {coord}")
        
        # 2. Apply Action
        new_tensor = state.tensor.copy()
        d, h, w = coord
        if action_type == 'ADD':
            new_tensor[0, d, h, w] = 1
        elif action_type == 'REMOVE':
            new_tensor[0, d, h, w] = 0
            
        # 3. Physics Solve (Oracle)
        # Only solve if material changed
        sim_result = solve_topology_3d(new_tensor, ctx, props)
        print(f"  Physics: Compliance={sim_result.compliance:.4f}, MaxDisp={sim_result.max_displacement:.4f}")
        
        # 4. Update State
        # Check Phase Transition logic (Simplified)
        new_phase = state.phase
        if state.phase == 'GROWTH' and sim_result.max_displacement < 1e9: # Connected?
             # If physics solves with reasonable displacement, we are connected!
             new_phase = 'REFINEMENT'
             
        state = GameState(tensor=new_tensor, phase=new_phase, step_count=step+1)
        
    print("Episode Complete.")
    
    # Save to DB
    from alphabuilder.src.logic.storage import save_record, TrainingRecord, Phase, serialize_state, generate_episode_id
    
    episode_id = generate_episode_id()
    state_blob = serialize_state(state.tensor)
    
    # Map phase string to Enum
    phase_enum = Phase.GROWTH if state.phase == 'GROWTH' else Phase.REFINEMENT
    
    record = TrainingRecord(
        episode_id=episode_id,
        step=step,
        phase=phase_enum,
        state_blob=state_blob,
        fitness_score=sim_result.fitness, # Use fitness from result
        valid_fem=sim_result.valid,
        metadata={
            "compliance": sim_result.compliance,
            "max_disp": sim_result.max_displacement,
            "steps": step
        }
    )
    
    save_record(db_path, record)
    print(f"Saved episode to {db_path}")

if __name__ == "__main__":
    # Smoke Test
    run_episode_v1_1(Path("dummy.db"))
