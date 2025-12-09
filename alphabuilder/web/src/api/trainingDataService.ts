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

export interface EpisodeMetadata {
    episode_id: string;
    steps_phase1: number;
    steps_phase2: number;
    total_steps: number;
    final_reward: number | null;
    final_compliance: number | null;
    final_volume_fraction: number | null;
    bc_type?: string;
    strategy?: string;
    resolution?: number[];
    bc_masks?: number[];
    forces?: number[];
    fitness_history?: number[];
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
    try {
        const response = await fetch(`${API_BASE}/databases?t=${Date.now()}`);
        if (!response.ok) throw new Error(`Erro do servidor (${response.status})`);
        return response.json();
    } catch (err) {
        throw new Error('Não foi possível conectar ao backend.');
    }
}

/**
 * Fetch all episodes in a database
 */
export async function fetchEpisodes(dbId: string): Promise<EpisodeSummary[]> {
    try {
        const response = await fetch(`${API_BASE}/databases/${dbId}/episodes?t=${Date.now()}`);
        if (!response.ok) throw new Error(`Erro ao buscar episódios (${response.status})`);
        return response.json();
    } catch (err) {
        throw new Error('Não foi possível conectar ao backend.');
    }
}

/**
 * Fetch episode metadata
 */
export async function fetchEpisodeMetadata(dbId: string, episodeId: string): Promise<EpisodeMetadata> {
    const response = await fetch(`${API_BASE}/databases/${dbId}/episodes/${episodeId}/metadata?t=${Date.now()}`);
    if (!response.ok) throw new Error(`Erro ao buscar metadados (${response.status})`);
    return response.json();
}

/**
 * Fetch paginated episode frames
 */
export async function fetchEpisodeFrames(dbId: string, episodeId: string, start: number, end: number): Promise<Frame[]> {
    const response = await fetch(`${API_BASE}/databases/${dbId}/episodes/${episodeId}/frames?start=${start}&end=${end}&t=${Date.now()}`);
    if (!response.ok) throw new Error(`Erro ao buscar frames (${response.status})`);
    return response.json();
}

/**
 * Fetch full episode data for replay (Legacy)
 */
export async function fetchEpisodeData(dbId: string, episodeId: string): Promise<EpisodeData> {
    const response = await fetch(`${API_BASE}/databases/${dbId}/episodes/${episodeId}?t=${Date.now()}`);
    if (!response.ok) throw new Error(`Erro ao buscar episódio (${response.status})`);
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

    // State
    private dbId: string | null = null;
    private episodeId: string | null = null;
    private metadata: EpisodeMetadata | null = null;

    // Pagination / Buffering
    private framesMap: Map<number, ReplayState> = new Map();
    private currentStepIndex = 0;
    private isPlaying = false;

    // Constants
    private readonly FETCH_CHUNK_SIZE = 25;
    private readonly PREFETCH_THRESHOLD = 5;

    /**
     * Load episode metadata and initial chunk
     */
    async loadEpisode(dbId: string, episodeId: string): Promise<void> {
        this.stop();
        this.dbId = dbId;
        this.episodeId = episodeId;

        try {
            // 1. Fetch Metadata
            this.metadata = await fetchEpisodeMetadata(dbId, episodeId);

            // 2. Fetch Initial Chunk
            await this.loadChunk(0, this.FETCH_CHUNK_SIZE);

            this.currentStepIndex = 0;

            if (this.framesMap.has(0)) {
                this.notifySubscribers(this.framesMap.get(0)!);
            }
        } catch (err) {
            console.error("Error loading episode:", err);
            throw err;
        }
    }

    private async loadChunk(start: number, count: number): Promise<void> {
        if (!this.dbId || !this.episodeId) return;

        const end = start + count;
        // console.log(`Fetching chunk ${start} - ${end}`);

        const frames = await fetchEpisodeFrames(this.dbId, this.episodeId, start, end);
        const processed = this.processFrames(frames);

        // Add to map
        processed.forEach(f => {
            this.framesMap.set(f.step, f);
        });

        this.pruneCache();
    }

    private pruneCache() {
        // Keep a window around currentStepIndex
        const KEEP_WINDOW = 100; // Increased window to avoid frequent refetching if seeking nearby
        const minStep = this.currentStepIndex - KEEP_WINDOW;
        const maxStep = this.currentStepIndex + KEEP_WINDOW;

        for (const step of this.framesMap.keys()) {
            if (step < minStep || step > maxStep) {
                this.framesMap.delete(step);
            }
        }
    }

    private processFrames(frames: Frame[]): ReplayState[] {
        return frames.map(frame => {
            const tensorData = new Float32Array(frame.tensor_data);
            const shape = frame.tensor_shape as [number, number, number, number];

            let policyHeatmap = undefined;
            if (frame.policy_add || frame.policy_remove) {
                const spatialSize = shape[1] * shape[2] * shape[3];
                policyHeatmap = {
                    add: frame.policy_add ? new Float32Array(frame.policy_add) : new Float32Array(spatialSize),
                    remove: frame.policy_remove ? new Float32Array(frame.policy_remove) : new Float32Array(spatialSize),
                };
            }

            return {
                episode_id: this.episodeId!,
                step: frame.step,
                phase: frame.phase as 'GROWTH' | 'REFINEMENT',
                tensor: { shape, data: tensorData },
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
        if (this.isPlaying || !this.metadata) return;

        // If at end, restart
        if (this.currentStepIndex >= this.metadata.total_steps - 1) {
            this.currentStepIndex = 0;
            this.seekToStep(0); // This will handle fetching if needed
        }

        this.isPlaying = true;
        this.intervalId = setInterval(async () => {
            if (!this.metadata) return;

            // Check if next step is valid
            if (this.currentStepIndex >= this.metadata.total_steps - 1) {
                this.pause();
                return;
            }

            this.currentStepIndex++;

            // Check buffering / prefetch
            if (!this.framesMap.has(this.currentStepIndex)) {
                // Buffer underrun! Pause and load.
                // console.log("Buffer underrun, pausing to load...");
                this.pause();
                await this.loadChunk(this.currentStepIndex, this.FETCH_CHUNK_SIZE);
                this.play(); // Resume
                return;
            }

            // Prefetch if needed
            const nextChunkStart = this.currentStepIndex + this.PREFETCH_THRESHOLD;
            if (nextChunkStart < this.metadata.total_steps && !this.framesMap.has(nextChunkStart)) {
                this.loadChunk(nextChunkStart, this.FETCH_CHUNK_SIZE).catch(console.error);
            }

            this.notifySubscribers(this.framesMap.get(this.currentStepIndex)!);
        }, 500); // 2 FPS
    }

    pause(): void {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.isPlaying = false;
        if (this.framesMap.has(this.currentStepIndex)) {
            this.notifySubscribers(this.framesMap.get(this.currentStepIndex)!);
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
        this.framesMap.clear();
        this.currentStepIndex = 0;
        this.metadata = null;
        this.dbId = null;
        this.episodeId = null;
    }

    async seekToStep(index: number): Promise<void> {
        if (!this.metadata || index < 0 || index >= this.metadata.total_steps) return;

        this.currentStepIndex = index;

        if (this.framesMap.has(index)) {
            this.notifySubscribers(this.framesMap.get(index)!);
        } else {
            // Need to fetch
            this.pause(); // Pause while seeking if not buffered
            // console.log("Seeking to unbuffered step...");
            await this.loadChunk(index, this.FETCH_CHUNK_SIZE);
            if (this.framesMap.has(index)) {
                this.notifySubscribers(this.framesMap.get(index)!);
            }
        }
    }

    stepForward(): void {
        if (!this.metadata) return;
        if (this.currentStepIndex < this.metadata.total_steps - 1) {
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
            episodeId: this.episodeId,
            stepsLoaded: this.metadata ? this.metadata.total_steps : 0, // Used for progress bar max
            currentStep: this.currentStepIndex,
            isPlaying: this.isPlaying,
            maxCompliance: 1.0, // Deprecated/Unused scaling
        };
    }

    /**
     * Get fitness history for graph
     */
    getFitnessHistory(): number[] {
        return this.metadata?.fitness_history || [];
    }

    // Deprecated but kept for compatibility - returns ONLY buffered frames
    getAllFrames(): ReplayState[] {
        return Array.from(this.framesMap.values()).sort((a, b) => a.step - b.step);
    }
}

// Singleton instance
export const trainingDataReplayService = new TrainingDataReplayService();
