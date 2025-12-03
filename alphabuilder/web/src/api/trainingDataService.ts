/**
 * Training Data Service
 * 
 * Communicates with the FastAPI backend to fetch training data from SQLite databases.
 */

const API_BASE = 'http://localhost:8000';

// --- Types ---
export interface DatabaseInfo {
    id: string;
    name: string;
    path: string;
    episode_count: number;
    total_steps: number;
    size_mb: number;
}

export interface EpisodeSummary {
    episode_id: string;
    steps_phase1: number;
    steps_phase2: number;
    total_steps: number;
    final_reward: number | null;
    final_compliance: number | null;
    final_volume_fraction: number | null;
}

export interface Frame {
    step: number;
    phase: string;
    tensor_shape: number[];
    tensor_data: number[];
    fitness_score: number;
    compliance: number | null;
    volume_fraction: number | null;
    policy_add: number[] | null;
    policy_remove: number[] | null;
}

export interface EpisodeData {
    episode_id: string;
    frames: Frame[];
}

// --- API Functions ---

/**
 * Fetch all available databases
 */
export async function fetchDatabases(): Promise<DatabaseInfo[]> {
    const response = await fetch(`${API_BASE}/databases`);
    if (!response.ok) {
        throw new Error('Failed to fetch databases');
    }
    return response.json();
}

/**
 * Fetch all episodes in a database
 */
export async function fetchEpisodes(dbId: string): Promise<EpisodeSummary[]> {
    const response = await fetch(`${API_BASE}/databases/${dbId}/episodes`);
    if (!response.ok) {
        throw new Error(`Failed to fetch episodes for database ${dbId}`);
    }
    return response.json();
}

/**
 * Fetch full episode data for replay
 */
export async function fetchEpisodeData(dbId: string, episodeId: string): Promise<EpisodeData> {
    const response = await fetch(`${API_BASE}/databases/${dbId}/episodes/${episodeId}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch episode ${episodeId}`);
    }
    return response.json();
}

// --- Replay Service Class ---

type ReplayCallback = (state: ReplayState) => void;

export interface ReplayState {
    episode_id: string;
    step: number;
    phase: 'GROWTH' | 'REFINEMENT';
    tensor: {
        shape: [number, number, number, number];
        data: Float32Array;
    };
    fitness_score: number;
    valid_fem: boolean;
    metadata: {
        compliance: number;
        max_displacement: number;
        volume_fraction: number;
    };
    value_confidence: number;
    policy_heatmap?: {
        add: Float32Array;
        remove: Float32Array;
    };
}

export class TrainingDataReplayService {
    private subscribers: ReplayCallback[] = [];
    private intervalId: ReturnType<typeof setInterval> | null = null;
    private currentStepIndex = 0;
    private frames: ReplayState[] = [];
    private isPlaying = false;
    private maxCompliance = 1.0;

    /**
     * Load episode data and prepare for replay
     */
    async loadEpisode(dbId: string, episodeId: string): Promise<void> {
        this.stop();
        
        const data = await fetchEpisodeData(dbId, episodeId);
        this.frames = this.processFrames(data);
        
        // Calculate max compliance for scaling
        this.maxCompliance = Math.max(
            1.0,
            ...this.frames.map(f => f.metadata.compliance || 0)
        );
        
        this.currentStepIndex = 0;
        if (this.frames.length > 0) {
            this.notifySubscribers(this.frames[0]);
        }
    }

    private processFrames(data: EpisodeData): ReplayState[] {
        return data.frames.map(frame => {
            // Convert tensor data to Float32Array
            const tensorData = new Float32Array(frame.tensor_data);
            const shape = frame.tensor_shape as [number, number, number, number];
            
            // Process policy heatmap if available
            // Note: policy_add or policy_remove may be null/empty arrays
            let policyHeatmap = undefined;
            if (frame.policy_add || frame.policy_remove) {
                const spatialSize = shape[1] * shape[2] * shape[3]; // D * H * W
                policyHeatmap = {
                    add: frame.policy_add 
                        ? new Float32Array(frame.policy_add) 
                        : new Float32Array(spatialSize),
                    remove: frame.policy_remove 
                        ? new Float32Array(frame.policy_remove) 
                        : new Float32Array(spatialSize),
                };
            }

            return {
                episode_id: data.episode_id,
                step: frame.step,
                phase: frame.phase as 'GROWTH' | 'REFINEMENT',
                tensor: {
                    shape,
                    data: tensorData,
                },
                fitness_score: frame.fitness_score,
                valid_fem: true,
                metadata: {
                    compliance: frame.compliance || 0,
                    max_displacement: 0,
                    volume_fraction: frame.volume_fraction || 0,
                },
                value_confidence: frame.fitness_score,
                policy_heatmap: policyHeatmap,
            };
        });
    }

    play(): void {
        if (this.isPlaying || this.frames.length === 0) return;
        
        // If at end, restart
        if (this.currentStepIndex >= this.frames.length - 1) {
            this.currentStepIndex = 0;
        }
        
        this.isPlaying = true;
        this.intervalId = setInterval(() => {
            if (this.currentStepIndex >= this.frames.length - 1) {
                this.pause();
                return;
            }
            
            this.currentStepIndex++;
            this.notifySubscribers(this.frames[this.currentStepIndex]);
        }, 500);
    }

    pause(): void {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.isPlaying = false;
        if (this.frames.length > 0) {
            this.notifySubscribers(this.frames[this.currentStepIndex]);
        }
    }

    togglePlay(): void {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    stop(): void {
        this.pause();
        this.frames = [];
        this.currentStepIndex = 0;
    }

    seekToStep(index: number): void {
        if (index >= 0 && index < this.frames.length) {
            this.currentStepIndex = index;
            this.notifySubscribers(this.frames[index]);
        }
    }

    stepForward(): void {
        if (this.currentStepIndex < this.frames.length - 1) {
            this.seekToStep(this.currentStepIndex + 1);
        }
    }

    stepBackward(): void {
        if (this.currentStepIndex > 0) {
            this.seekToStep(this.currentStepIndex - 1);
        }
    }

    subscribe(callback: ReplayCallback): () => void {
        this.subscribers.push(callback);
        return () => {
            this.subscribers = this.subscribers.filter(s => s !== callback);
        };
    }

    private notifySubscribers(state: ReplayState): void {
        this.subscribers.forEach(cb => cb(state));
    }

    getState() {
        return {
            episodeId: this.frames.length > 0 ? this.frames[0].episode_id : null,
            stepsLoaded: this.frames.length,
            currentStep: this.currentStepIndex,
            isPlaying: this.isPlaying,
            maxCompliance: this.maxCompliance,
        };
    }
}

// Singleton instance
export const trainingDataReplayService = new TrainingDataReplayService();


