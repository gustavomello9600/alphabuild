/**
 * Training Data Service
 * 
 * Communicates with the FastAPI backend to fetch training data from SQLite databases.
 */

const API_BASE = 'http://localhost:8000';

import type { Action, RewardComponents } from './types';

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
    action_sequence?: Action[] | null;
    reward_components?: RewardComponents | null;
    displacement_map?: string; // Base64 encoded float32 array
}

export interface EpisodeData {
    episode_id: string;
    frames: Frame[];
}

// --- API Functions ---

export async function fetchDatabases(): Promise<DatabaseInfo[]> {
    try {
        const response = await fetch(`${API_BASE}/databases?t=${Date.now()}`);
        if (!response.ok) throw new Error(`Erro do servidor (${response.status})`);
        return response.json();
    } catch (err) {
        throw new Error('Não foi possível conectar ao backend.');
    }
}

export async function fetchEpisodes(dbId: string): Promise<EpisodeSummary[]> {
    try {
        const response = await fetch(`${API_BASE}/databases/${dbId}/episodes?t=${Date.now()}`);
        if (!response.ok) throw new Error(`Erro ao buscar episódios (${response.status})`);
        return response.json();
    } catch (err) {
        throw new Error('Não foi possível conectar ao backend.');
    }
}

export async function fetchEpisodeMetadata(dbId: string, episodeId: string): Promise<EpisodeMetadata> {
    const response = await fetch(`${API_BASE}/databases/${dbId}/episodes/${episodeId}/metadata?t=${Date.now()}`);
    if (!response.ok) throw new Error(`Erro ao buscar metadados (${response.status})`);
    return response.json();
}

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

import ReplayWorker from '../workers/replay.worker?worker';

type ReplayCallback = (state: ReplayState) => void;

export interface ReplayState {
    episode_id: string;
    step: number;
    phase: 'GROWTH' | 'REFINEMENT';
    tensor: {
        shape: [number, number, number, number];
        data: Float32Array;
    };
    fitness_score: number; // retained for compat, map to value
    value?: number; // compat with GameReplayState
    valid_fem: boolean; // default true
    metadata?: {
        compliance: number;
        max_displacement: number;
        volume_fraction: number;
    };
    value_confidence: number;
    policy_heatmap?: {
        add: Float32Array;
        remove: Float32Array;
    };
    // Map to GameReplayState fields for component compat
    policy?: {
        add: Float32Array;
        remove: Float32Array;
    };
    action_sequence?: Action[] | null;
    reward_components?: RewardComponents | null;
    displacement_map?: Float32Array;

    // Visuals from Worker
    visuals?: {
        opaqueMatrix: Float32Array;
        opaqueColor: Float32Array;
        overlayMatrix: Float32Array;
        overlayColor: Float32Array;
        mctsMatrix?: Float32Array;
        mctsColor?: Float32Array;
    };

    // MCTS Stats (Empty for training usually)
    mcts_stats?: any;
    selected_actions?: any[];
}

export class TrainingDataReplayService {
    private subscribers: ReplayCallback[] = [];
    private intervalId: ReturnType<typeof setInterval> | null = null;
    private worker: Worker;

    // State
    private dbId: string | null = null;
    private episodeId: string | null = null;
    private metadata: EpisodeMetadata | null = null;

    // Pagination / Buffering
    private framesMap: Map<number, ReplayState> = new Map();
    private currentStepIndex = 0;
    private isPlaying = false;

    // Constants
    private readonly FETCH_CHUNK_SIZE = 50;
    private readonly PREFETCH_THRESHOLD = 20;

    constructor() {
        this.worker = new ReplayWorker();
        this.worker.onmessage = this.handleWorkerMessage.bind(this);
    }

    private handleWorkerMessage(e: MessageEvent) {
        const { type, payload } = e.data;
        if (type === 'ERROR') {
            console.error('ReplayWorker Error:', payload);
            return;
        }
        if (type === 'CHUNK_LOADED') {
            const steps = payload as ReplayState[];
            console.log(`[Service] Received CHUNK_LOADED: ${steps.length} steps. Indices: ${steps.map(s => s.step).join(', ')}`);
            steps.forEach(s => {
                this.framesMap.set(s.step, s);
            });
            this.pruneCache();

            // Notify if we are waiting for current frame
            if (this.framesMap.has(this.currentStepIndex)) {
                this.notifySubscribers(this.framesMap.get(this.currentStepIndex)!);
            }
        }
    }

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

            // 2. Fetch Initial Chunk via Worker
            this.requestChunk(0, this.FETCH_CHUNK_SIZE);

            // Wait for initial chunk to populate
            await this.waitForFirstChunk();

            // Set current step to the first available step (likely 1 for Training Data)
            const steps = Array.from(this.framesMap.keys());
            if (steps.length > 0) {
                this.currentStepIndex = Math.min(...steps);
                console.log(`[Service] Initialized at step ${this.currentStepIndex}`);
            } else {
                this.currentStepIndex = 0; // Fallback
            }

            await this.waitForFrame(this.currentStepIndex);

        } catch (err) {
            console.error("Error loading episode:", err);
            throw err;
        }
    }

    private async waitForFirstChunk(timeout = 10000): Promise<void> {
        return new Promise((resolve, reject) => {
            if (this.framesMap.size > 0) {
                resolve();
                return;
            }

            const checkInterval = setInterval(() => {
                if (this.framesMap.size > 0) {
                    clearInterval(checkInterval);
                    resolve();
                }
            }, 100);

            setTimeout(() => {
                clearInterval(checkInterval);
                reject(new Error("Timeout waiting for first chunk"));
            }, timeout);
        });
    }

    private async waitForFrame(step: number, timeout = 10000): Promise<void> {
        return new Promise((resolve, reject) => {
            if (this.framesMap.has(step)) {
                resolve();
                return;
            }

            const checkInterval = setInterval(() => {
                if (this.framesMap.has(step)) {
                    clearInterval(checkInterval);
                    resolve();
                }
            }, 100);

            setTimeout(() => {
                clearInterval(checkInterval);
                reject(new Error(`Timeout waiting for frame ${step}`));
            }, timeout);
        });
    }

    private requestChunk(start: number, count: number): void {
        if (!this.dbId || !this.episodeId) return;
        const end = start + count;

        if (this.framesMap.has(start)) return;

        this.worker.postMessage({
            type: 'LOAD_CHUNK',
            payload: {
                source: 'training',
                apiBase: API_BASE,
                dbId: this.dbId,
                episodeId: this.episodeId,
                start,
                end,
                staticData: {
                    bc_masks: this.metadata?.bc_masks,
                    forces: this.metadata?.forces
                }
            }
        });
    }

    private pruneCache() {
        const KEEP_WINDOW = 200;
        const minStep = this.currentStepIndex - KEEP_WINDOW;
        const maxStep = this.currentStepIndex + KEEP_WINDOW;

        for (const step of this.framesMap.keys()) {
            if (step < minStep || step > maxStep) {
                this.framesMap.delete(step);
            }
        }
    }

    play(): void {
        if (this.isPlaying || !this.metadata) return;

        if (this.currentStepIndex >= this.metadata.total_steps - 1) {
            this.currentStepIndex = 0;
            this.seekToStep(0);
        }

        this.isPlaying = true;
        this.intervalId = setInterval(() => {
            if (!this.metadata) return;

            if (this.currentStepIndex >= this.metadata.total_steps - 1) {
                this.pause();
                return;
            }

            this.currentStepIndex++;

            if (!this.framesMap.has(this.currentStepIndex)) {
                this.pause();
                this.requestChunk(this.currentStepIndex, this.FETCH_CHUNK_SIZE);
                this.waitForFrame(this.currentStepIndex).then(() => this.play()).catch(() => { });
                return;
            }

            const nextChunkStart = this.currentStepIndex + this.PREFETCH_THRESHOLD;
            if (nextChunkStart < this.metadata.total_steps && !this.framesMap.has(nextChunkStart)) {
                this.requestChunk(nextChunkStart, this.FETCH_CHUNK_SIZE);
            }

            this.notifySubscribers(this.framesMap.get(this.currentStepIndex)!);
        }, 100);
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
            this.pause();
            this.requestChunk(index, this.FETCH_CHUNK_SIZE);
            try {
                await this.waitForFrame(index);
                this.notifySubscribers(this.framesMap.get(index)!);
            } catch (e) {
                // Ignore timeout
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
            stepsLoaded: this.metadata ? this.metadata.total_steps : 0,
            currentStep: this.currentStepIndex,
            isPlaying: this.isPlaying,
            maxCompliance: 1.0,
        };
    }

    getFitnessHistory(): number[] {
        return this.metadata?.fitness_history || [];
    }

    getAllFrames(): ReplayState[] {
        return Array.from(this.framesMap.values()).sort((a, b) => a.step - b.step);
    }

    public getFrame(step: number): ReplayState | undefined {
        return this.framesMap.get(step);
    }
}

// Singleton instance
export const trainingDataReplayService = new TrainingDataReplayService();
