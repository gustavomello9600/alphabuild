export type Phase = 'GROWTH' | 'REFINEMENT';

/**
 * Tensor for 3D voxel representation.
 * 
 * v3.1 Channel Layout (7 channels):
 * - 0: Density (ρ) - Material state (0.0 to 1.0)
 * - 1: Mask X - 1.0 if displacement u_x is fixed (support)
 * - 2: Mask Y - 1.0 if displacement u_y is fixed (support)
 * - 3: Mask Z - 1.0 if displacement u_z is fixed (support)
 * - 4: Force X (Fx) - Normalized force component
 * - 5: Force Y (Fy) - Normalized force component
 * - 6: Force Z (Fz) - Normalized force component
 * 
 * Legacy (5 channels):
 * - 0: Density, 1: Support, 2: (unused), 3: Fy, 4: (unused)
 */
export interface Tensor5D {
    shape: [number, number, number, number]; // [Channels, Depth, Height, Width]
    data: Float32Array; // Flattened data in C-order (Channel-major)
}

export interface SimulationMetrics {
    compliance: number;
    max_displacement: number;
    volume_fraction: number;
}

export interface GameState {
    episode_id: string;
    step: number;
    phase: Phase;
    tensor: Tensor5D;
    fitness_score: number;
    valid_fem: boolean;
    metadata: SimulationMetrics;
    policy_heatmap?: { add: Float32Array, remove: Float32Array }; // Optional: Flattened 3D arrays
    value_confidence?: number; // 0.0 - 1.0
}

export interface Project {
    id: string;
    name: string;
    thumbnail_url?: string;
    last_modified: string;
    status: 'IN_PROGRESS' | 'COMPLETED' | 'FAILED';
    episode_id: string;
}

export interface Action {
    type: 'ADD' | 'REMOVE';
    x: number;
    y: number;
    z: number;
    value_estimate?: number;
    visit_count?: number;
}

export interface RewardComponents {
    base_reward: number;
    connectivity_bonus?: number;
    island_penalty?: number;
    loose_penalty?: number;
    fem_reward?: number;
    volume_penalty?: number;
    validity_penalty?: number;
    total: number;
    // New metrics [v3.1]
    n_islands?: number;
    loose_voxels?: number;
    disconnected_volume_fraction?: number;
    connected_load_fraction?: number;
}
