from dataclasses import dataclass
from typing import Tuple, FrozenSet, Literal
import numpy as np

# Primitive Types
Coord = Tuple[int, int]
ActionType = Literal['ADD', 'REMOVE']
PhaseType = Literal['GROWTH', 'REFINEMENT']

@dataclass(frozen=True)
class GameAction:
    """Atomic game action."""
    type: ActionType
    coord: Coord

@dataclass(frozen=True)
class DesignState:
    """
    Complete and immutable system state.
    Uses dataclass(frozen=True) to be hashable and lightweight.
    """
    grid: np.ndarray              # Binary Matrix (Read-only)
    supports: Tuple[Coord, ...]   # Fixed coordinates
    loads: Tuple[Coord, ...]      # Load coordinates
    phase: PhaseType              # Current phase
    
    # Graph Cache (Derived metadata)
    is_connected: bool            # Global Connectivity
    volume: int                   # Material count
    perimeter: FrozenSet[Coord]   # Valid expansion boundary

    def __hash__(self):
        return hash(self.grid.tobytes())

@dataclass(frozen=True)
class SimulationRecord:
    """DTO for DB persistence."""
    episode_id: str
    step: int
    state_bytes: bytes
    fitness: float
    is_valid: bool
