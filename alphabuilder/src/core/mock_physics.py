import numpy as np
from dataclasses import dataclass

@dataclass
class PhysicalProperties:
    E: float = 1.0
    nu: float = 0.3
    rho: float = 1.0
    disp_limit: float = 100.0
    penalty_epsilon: float = 0.1
    penalty_alpha: float = 10.0

@dataclass
class FEMContext:
    domain: object = None
    V: object = None
    bc: object = None
    problem: object = None
    u_sol: object = None
    dof_map: object = None
    material_field: object = None

@dataclass
class SimulationResult:
    fitness: float
    max_displacement: float
    compliance: float
    valid: bool
    displacement_array: np.ndarray

def initialize_cantilever_context(resolution=(64, 32, 32)) -> FEMContext:
    print("MOCK: Initializing Cantilever Context")
    return FEMContext()

def solve_topology_3d(tensor_state: np.ndarray, ctx: FEMContext, props: PhysicalProperties) -> SimulationResult:
    # Mock Solver
    # Calculate volume
    density = tensor_state[0]
    vol = np.mean(density)
    
    # Mock Compliance: Inverse of Volume (More material = Stiffer)
    # Add random noise to simulate complexity
    compliance = 1.0 / (vol + 1e-6)
    max_disp = compliance * 10.0
    
    return SimulationResult(
        fitness=vol, # Mock fitness
        max_displacement=max_disp,
        compliance=compliance,
        valid=True,
        displacement_array=np.array([0.0])
    )
