"""
Configuration and Constants for Data Harvest.
"""
from dataclasses import dataclass
from typing import Dict, Any

# =========== SIMPConfig ===========
@dataclass
class SIMPConfig:
    """Configuration for SIMP optimization via FEniTop."""
    vol_frac: float = 0.15
    max_iter: int = 120
    r_min: float = 1.5
    adaptive_penal: bool = True
    load_config: Dict[str, Any] = None
    debug_log_path: str = None

# --- Constants for Value Normalization (Spec 4.2) ---
# Estimated from typical compliance ranges (C ~ 10 to 10000)
# S_raw = -log(C + ε) - α·Vol, where α controls volume penalty
# Calibrated to give volume 20% more impact than compliance
LOG_SQUASH_ALPHA = 12.0  # Volume penalty coefficient
LOG_SQUASH_MU = -6.65    # Estimated mean of log(C) distribution
LOG_SQUASH_SIGMA = 2.0   # Estimated std of log(C) distribution
LOG_SQUASH_EPSILON = 1e-9

# --- Policy Quality Settings (v3.1 Quality Improvements) ---
# Window size for REFINEMENT phase (compare t with t+k)
REFINEMENT_WINDOW_SIZE = 5  # Compare frames 5 steps apart
REFINEMENT_SLIDE_STEP = 1  # Default: 1 (sliding window for more data)

# Binarization settings for REFINEMENT targets
BINARIZE_REFINEMENT_TARGETS = True
BINARIZE_THRESHOLD = 0.005  # Absolute threshold for significant change
BINARIZE_PERCENTILE = 90    # Alternative: use top N% of changes

# Target normalization: scale ADD/REMOVE to comparable magnitudes
NORMALIZE_TARGETS = True

# --- Boundary Condition Types (Spec 2.1) ---
BC_TYPES = ['FULL_CLAMP', 'RAIL_XY']
