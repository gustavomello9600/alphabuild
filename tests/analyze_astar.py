"""
Analyze A* Backbone Structure

Check what the A* pathfinder is actually creating.
"""

import numpy as np
from alphabuilder.src.core.physics_model import initialize_cantilever_context, PhysicalProperties
from alphabuilder.src.logic.runner_astar import run_episode_astar

def analyze_astar_backbone():
    print("="*80)
    print("A* Backbone Structure Analysis")
    print("="*80)
    
    resolution = (64, 32, 8)
    ctx = initialize_cantilever_context(resolution=resolution)
    props = PhysicalProperties()
    
    # Build A* backbone
    state = run_episode_astar(ctx, props, resolution=resolution)
    
    # Analyze structure
    density = state.density
    D, H, W = density.shape
    
    print(f"\n📊 Backbone Statistics:")
    print(f"  Resolution: {D}x{H}x{W} = {D*H*W} total voxels")
    print(f"  Solid voxels: {np.sum(density)}")
    print(f"  Volume fraction: {np.mean(density):.4f} ({np.mean(density)*100:.2f}%)")
    print(f"  Min density: {np.min(density):.4f}")
    print(f"  Max density: {np.max(density):.4f}")
    
    # Check cross-section at different points
    print(f"\n🔍 Cross-Section Analysis:")
    for x_pos, label in [(0, "Left (support)"), (D//2, "Middle"), (D-1, "Right (load)")]:
        cross_section = density[x_pos, :, :]
        solid_count = np.sum(cross_section)
        total_count = H * W
        print(f"  {label:20s}: {solid_count}/{total_count} voxels ({solid_count/total_count*100:.1f}%)")
    
    # Check if it's just a thin line
    print(f"\n📏 Path Thickness:")
    # For each x-position, count how many (h,w) positions have material
    for x_pos in [0, D//4, D//2, 3*D//4, D-1]:
        cross_section = density[x_pos, :, :]
        solid_positions = np.argwhere(cross_section > 0)
        if len(solid_positions) > 0:
            # Calculate bounding box
            h_min, w_min = solid_positions.min(axis=0)
            h_max, w_max = solid_positions.max(axis=0)
            h_span = h_max - h_min + 1
            w_span = w_max - w_min + 1
            print(f"  x={x_pos:2d}: {len(solid_positions)} voxels, span: {h_span}x{w_span}")
    
    print(f"\n{'='*80}\n")
    
    return state

if __name__ == "__main__":
    analyze_astar_backbone()
