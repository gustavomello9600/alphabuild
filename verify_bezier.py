import numpy as np
import random
import scipy.ndimage
from alphabuilder.src.logic.simp_generator import SIMPConfig

def quadratic_bezier(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def generate_bezier_structure(resolution, load_config):
    nx, ny, nz = resolution
    voxel_grid = np.full(resolution, 0.01, dtype=np.float32)
    
    num_curves = random.randint(3, 5)
    target_x = load_config['x']
    target_y = load_config['y']
    target_z_center = (load_config['z_start'] + load_config['z_end']) / 2.0
    p2 = np.array([target_x, target_y, target_z_center])
    
    for _ in range(num_curves):
        start_y = random.uniform(0, ny-1)
        start_z = random.uniform(0, nz-1)
        p0 = np.array([0.0, start_y, start_z])
        
        midpoint = (p0 + p2) / 2.0
        noise = np.random.normal(0, 5.0, size=3)
        p1 = midpoint + noise
        p1[0] = np.clip(p1[0], 0, nx-1)
        p1[1] = np.clip(p1[1], 0, ny-1)
        p1[2] = np.clip(p1[2], 0, nz-1)
        
        num_steps = 100
        t_values = np.linspace(0, 1, num_steps)
        
        w_base = random.uniform(8, 14)
        h_base = random.uniform(12, 24)
        w_tip = random.uniform(6, 10) 
        h_tip = random.uniform(6, 10)
        
        for t in t_values:
            p = quadratic_bezier(p0, p1, p2, t)
            cx, cy, cz = p
            
            w_curr = w_base + (w_tip - w_base) * t
            h_curr = h_base + (h_tip - h_base) * t
            
            y_max = int(cy + 1)
            y_min = int(cy - h_curr + 1)
            z_min = int(cz - w_curr / 2)
            z_max = int(cz + w_curr / 2)
            x_curr = int(cx)
            
            y_min = max(0, y_min)
            y_max = min(ny, y_max)
            z_min = max(0, z_min)
            z_max = min(nz, z_max)
            x_curr = max(0, min(nx-1, x_curr))
            
            x_min = max(0, x_curr)
            x_max = min(nx, x_curr + 2)
            
            voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max] = 1.0
            
    if np.sum(voxel_grid[0, :, :]) == 0:
        mid_y, mid_z = ny//2, nz//2
        voxel_grid[0:2, mid_y-4:mid_y+4, mid_z-4:mid_z+4] = 1.0
        
    return voxel_grid

def check_connectivity(grid):
    binary = grid > 0.5
    labeled, n_components = scipy.ndimage.label(binary)
    if n_components == 0: return False, 0
    
    # Check if X=0 is connected to X=end
    start_labels = np.unique(labeled[0, :, :])
    start_labels = start_labels[start_labels > 0]
    
    # Assuming load is at some X > 0. Let's check max X.
    end_x = np.max(np.where(binary)[0])
    end_labels = np.unique(labeled[end_x, :, :])
    end_labels = end_labels[end_labels > 0]
    
    common = np.intersect1d(start_labels, end_labels)
    return len(common) > 0, n_components

# Test
resolution = (64, 32, 8)
load_config = {'x': 63, 'y': 16, 'z_start': 2, 'z_end': 6}

print("Running 100 connectivity tests...")
failures = 0
for i in range(100):
    grid = generate_bezier_structure(resolution, load_config)
    connected, n_comp = check_connectivity(grid)
    if not connected:
        failures += 1
        print(f"Run {i}: Disconnected! (Components: {n_comp})")

print(f"Failures: {failures}/100")
