export type Phase = 'GROWTH' | 'REFINEMENT';

export interface Tensor5D {
    shape: [number, number, number, number]; // [Channels, Depth, Height, Width]
    data: Float32Array; // Flattened data
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
    policy_heatmap?: Float32Array; // Optional: Flattened 3D array for visualization
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
