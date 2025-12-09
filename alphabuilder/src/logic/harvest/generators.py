"""
Generators for Initial Structures and Load Configurations.
"""
import random
import numpy as np
from typing import Tuple, Dict, Any, List

def quadratic_bezier(p0, p1, p2, t):
    """Calculate point on quadratic Bézier curve at t."""
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def generate_bezier_structure(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any]
) -> np.ndarray:
    """
    Generate a procedural structure using Bézier curves with rectangular sections.
    Uses triangular distribution biased toward smaller section sizes.
    """
    nx, ny, nz = resolution
    # Safe Background: 0.15 to allow sensitivity flow (Vanishing Gradient Fix)
    voxel_grid = np.full(resolution, 0.15, dtype=np.float32)
    
    # 1. Determine Number of Curves
    num_curves = random.randint(2, 4)
    
    # Load Point (Target)
    target_x = load_config['x']
    target_y = load_config['y']
    target_z_center = (load_config['z_start'] + load_config['z_end']) / 2.0
    
    p2 = np.array([target_x, target_y, target_z_center])
    
    # Track final section positions for anchor placement
    final_positions = []
    final_sections = []
    
    for _ in range(num_curves):
        # 2. Start Point (Wall X=0)
        start_y = random.uniform(0, ny-1)
        start_z = random.uniform(0, nz-1)
        p0 = np.array([0.0, start_y, start_z])
        
        # 3. Control Point (Intermediate) - reduced noise for less curvature
        midpoint = (p0 + p2) / 2.0
        noise = np.random.normal(0, 3.0, size=3)  # Reduced from 5.0
        p1 = midpoint + noise
        p1[0] = np.clip(p1[0], 0, nx-1)
        p1[1] = np.clip(p1[1], 0, ny-1)
        p1[2] = np.clip(p1[2], 0, nz-1)
        
        # 4. Rasterize Curve
        num_steps = 100
        t_values = np.linspace(0, 1, num_steps)
        
        # Section dimensions with BIAS toward smaller values using triangular distribution
        w_tip = random.triangular(2, 2.5, 6)    # Bias toward 2-3
        w_base = random.triangular(w_tip, w_tip + 0.5, 8)  # Slightly larger than tip
        h_tip = random.triangular(2, 3, 12)     # Bias toward 2-4, max reduced to 12
        h_base = random.triangular(h_tip, h_tip + 2, min(h_tip + 10, 20))  # Controlled growth
        
        for i, t in enumerate(t_values):
            # Current point on curve
            p = quadratic_bezier(p0, p1, p2, t)
            cx, cy, cz = p
            
            # Current section size (interpolated)
            w_curr = w_base + (w_tip - w_base) * t
            h_curr = h_base + (h_tip - h_base) * t
            
            # Section centered on curve point
            y_min = int(cy - h_curr / 2)
            y_max = int(cy + h_curr / 2)
            z_min = int(cz - w_curr / 2)
            z_max = int(cz + w_curr / 2)
            x_curr = int(cx)
            
            # Clip to bounds
            y_min = max(0, y_min)
            y_max = min(ny, y_max)
            z_min = max(0, z_min)
            z_max = min(nz, z_max)
            x_curr = max(0, min(nx-1, x_curr))
            
            # Fill voxels (thin slice in X for connectivity)
            x_min = max(0, x_curr)
            x_max = min(nx, x_curr + 2)
            
            voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max] = 1.0
            
            # Track final position (last step of each curve)
            if i == len(t_values) - 1:
                final_positions.append((cx, cy, cz))
                final_sections.append((w_tip, h_tip))
            
    # Ensure Wall Connection (X=0) - small base plate
    mid_y, mid_z = ny//2, nz//2
    voxel_grid[0:2, mid_y-3:mid_y+3, mid_z-2:mid_z+2] = 1.0

    # Load Anchor: Must cover ALL final Bezier curve endpoints (bounding box approach)
    if final_positions:
        # Calculate bounding box of all final positions
        all_y = [p[1] for p in final_positions]
        all_z = [p[2] for p in final_positions]
        
        min_y, max_y = min(all_y), max(all_y)
        min_z, max_z = min(all_z), max(all_z)
        
        # Add padding based on section sizes
        max_w = max([s[0] for s in final_sections])
        max_h = max([s[1] for s in final_sections])
        
        # Anchor covers the bounding box with padding
        anchor_depth = 3
        y_padding = max(2, int(max_h / 2) + 1)
        z_padding = max(2, int(max_w / 2) + 1)
        
        x_s = max(0, nx - anchor_depth)
        x_e = nx
        y_s = max(0, int(min_y - y_padding))
        y_e = min(ny, int(max_y + y_padding))
        z_s = max(0, int(min_z - z_padding))
        z_e = min(nz, int(max_z + z_padding))
    else:
        # Fallback to load config
        anchor_depth = 3
        x_s = max(0, nx - anchor_depth)
        x_e = nx
        y_s = max(0, int(load_config['y'] - 4))
        y_e = min(ny, int(load_config['y'] + 4))
        z_center = (load_config['z_start'] + load_config['z_end']) / 2.0
        z_s = max(0, int(z_center - 3))
        z_e = min(nz, int(z_center + 3))
    
    voxel_grid[x_s:x_e, y_s:y_e, z_s:z_e] = 1.0
        
    return voxel_grid

def generate_seeded_cantilever(
    resolution: Tuple[int, int, int],
    load_config: Dict[str, Any]
) -> np.ndarray:
    """
    Generate a seeded initialization for Full Domain strategy.
    Creates a gray background with a solid bar connecting load to support.
    """
    nx, ny, nz = resolution
    
    # 1. Gray Background (0.35)
    grid = np.full(resolution, 0.35, dtype=np.float32)
    
    # 2. Define Seed Bar (Load -> Support)
    ly = load_config['y']
    lz_center = (load_config['z_start'] + load_config['z_end']) / 2
    
    # Point A: Load (at free end)
    p1 = np.array([nx - 1, ly, lz_center])
    
    # Point B: Support Center (X=0)
    p0 = np.array([0.0, ny / 2.0, nz / 2.0])
    
    # Rasterize Line
    num_steps = int(np.linalg.norm(p1 - p0)) * 2
    t_values = np.linspace(0, 1, num_steps)
    
    thickness = 2 # Radius of the bar
    
    for t in t_values:
        p = p0 + (p1 - p0) * t
        cx, cy, cz = int(p[0]), int(p[1]), int(p[2])
        
        # Draw sphere/box at p
        x_min, x_max = max(0, cx-thickness), min(nx, cx+thickness+1)
        y_min, y_max = max(0, cy-thickness), min(ny, cy+thickness+1)
        z_min, z_max = max(0, cz-thickness), min(nz, cz+thickness+1)
        
        grid[x_min:x_max, y_min:y_max, z_min:z_max] = 1.0
    
    # 3. Load Anchor at free end (ensures connection with 2x2 load region)
    anchor_depth = 4
    anchor_margin = 3
    
    x_s = max(0, nx - anchor_depth)
    x_e = nx
    y_s = max(0, int(ly - anchor_margin))
    y_e = min(ny, int(ly + anchor_margin + 1))
    z_s = max(0, int(lz_center - anchor_margin))
    z_e = min(nz, int(lz_center + anchor_margin + 1))
    
    grid[x_s:x_e, y_s:y_e, z_s:z_e] = 1.0
        
    return grid

def generate_random_load_config(resolution):
    """
    Generate load configuration for cantilever beam (v3.1).
    
    Load position constraints:
    - X: Random in range (L/2, L-1] to ensure load is in the "free" half
    - Y: Random around center with ±25% variation
    - Z: Random around center with ±25% variation
    """
    nx, ny, nz = resolution
    
    # Load X is random in range (L/2, L-1] - ensures X > L/2
    min_load_x = nx // 2 + 1  # Minimum is L/2 + 1
    max_load_x = nx - 1       # Maximum is L - 1 (last valid index)
    load_x = random.randint(min_load_x, max_load_x)
    
    # Center the load region in Y and Z with some randomness
    load_y = ny // 2 + random.randint(-ny//4, ny//4)
    load_y = max(2, min(ny-3, load_y))  # Keep within bounds
    
    load_z_center = nz // 2 + random.randint(-nz//4, nz//4)
    load_z_center = max(2, min(nz-3, load_z_center))
    
    # 2x2 load region
    half_width = 1
    load_z_start = load_z_center - half_width
    load_z_end = load_z_center + half_width
    
    # Random BC type for dataset diversity (Spec 2.1)
    bc_roll = random.random()
    if bc_roll < 0.70:
        bc_type = 'FULL_CLAMP'
    else:
        bc_type = 'RAIL_XY'
    
    return {
        'x': load_x,
        'y': load_y,
        'z_start': load_z_start,
        'z_end': load_z_end,
        'bc_type': bc_type,
    }

def generate_bc_masks(resolution: tuple, bc_type: str) -> tuple:
    """
    Generate support masks based on boundary condition type.
    """
    nx, ny, nz = resolution
    mask_x = np.zeros(resolution, dtype=np.float32)
    mask_y = np.zeros(resolution, dtype=np.float32)
    mask_z = np.zeros(resolution, dtype=np.float32)
    
    if bc_type == 'FULL_CLAMP':
        # All DOFs fixed at X=0 (full cantilever)
        mask_x[0, :, :] = 1.0
        mask_y[0, :, :] = 1.0
        mask_z[0, :, :] = 1.0
    elif bc_type == 'RAIL_XY':
        # X and Y fixed at X=0 (allows Z sliding - "rail" constraint)
        mask_x[0, :, :] = 1.0
        mask_y[0, :, :] = 1.0
        # mask_z remains zero
    else:
        # Default to full clamp
        mask_x[0, :, :] = 1.0
        mask_y[0, :, :] = 1.0
        mask_z[0, :, :] = 1.0
    
    return mask_x, mask_y, mask_z
