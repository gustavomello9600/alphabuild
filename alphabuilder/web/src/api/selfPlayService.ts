/**
 * Self-Play Service
 * 
 * Communicates with the FastAPI backend to fetch self-play game data with MCTS statistics.
 */

const API_BASE = 'http://localhost:8000';

// --- Types ---
export interface GameSummary {
    game_id: string;
    neural_engine: string;
    checkpoint_version: string;
    final_score: number | null;
    final_compliance: number | null;
    final_volume: number | null;
    total_steps: number;
    initial_cantilever_problem_real_scale_factor: number;
    created_at: string;
}

export interface SelectedAction {
    channel: number;
    x: number;
    y: number;
    z: number;
    visits: number;
    q_value: number;
}

export interface MCTSStats {
    num_simulations: number;
    nodes_expanded: number;
    max_depth: number;
    cache_hits: number;
    top8_concentration: number;
    refutation: boolean;
}

import type { RewardComponents } from './types';

export interface GameStep {
    step: number;
    phase: string;
    tensor_shape: number[];
    tensor_data: number[];
    policy_add: number[];
    policy_remove: number[];
    mcts_visit_add: number[];
    mcts_visit_remove: number[];
    mcts_q_add: number[];
    mcts_q_remove: number[];
    selected_actions: SelectedAction[];
    value: number;
    mcts_stats: MCTSStats;
    n_islands?: number;
    is_connected?: boolean;
    volume_fraction?: number;
    // Reward details
    compliance_fem?: number;
    max_displacement?: number;
    island_penalty?: number;
    loose_voxels?: number;
    reward_components?: RewardComponents | null;
    displacement_map?: Float32Array; // [NEW]
}



export interface GameData {
    game_id: string;
    neural_engine: string;
    checkpoint_version: string;
    bc_type: string;
    resolution: number[];
    final_score: number | null;
    final_compliance: number | null;
    final_volume: number | null;
    total_steps: number;
    initial_cantilever_problem_real_scale_factor: number;
    steps: GameStep[];
}

export interface GameMetadata {
    game_id: string;
    neural_engine: string;
    checkpoint_version: string;
    bc_type: string;
    resolution: number[];
    final_score: number | null;
    final_compliance: number | null;
    final_volume: number | null;
    total_steps: number;
    initial_cantilever_problem_real_scale_factor: number;
    bc_masks: number[];
    forces: number[];
    value_history?: number[];
}

// --- API Functions ---

/**
 * Fetch all self-play games with optional filtering
 */
export async function fetchGames(
    engine?: string,
    version?: string,
    limit: number = 50,
    deprecated: boolean = false
): Promise<GameSummary[]> {
    try {
        const params = new URLSearchParams();
        if (engine) params.append('engine', engine);
        if (version) params.append('version', version);
        params.append('limit', limit.toString());
        if (deprecated) params.append('deprecated', 'true');
        params.append('t', Date.now().toString());

        const response = await fetch(`${API_BASE}/selfplay/games?${params}`);
        if (!response.ok) throw new Error(`Erro do servidor (${response.status})`);
        return response.json();
    } catch (err) {
        throw new Error('Não foi possível conectar ao backend.');
    }
}

/**
 * Fetch full game data (Legacy)
 */
export async function fetchGameData(gameId: string, deprecated: boolean = false): Promise<GameData> {
    const params = new URLSearchParams();
    if (deprecated) params.append('deprecated', 'true');
    params.append('t', Date.now().toString());

    const response = await fetch(`${API_BASE}/selfplay/games/${gameId}?${params}`);
    if (!response.ok) throw new Error(`Erro ao buscar jogo (${response.status})`);
    return response.json();
}

/**
 * Fetch game metadata
 */
export async function fetchGameMetadata(gameId: string, deprecated: boolean = false): Promise<GameMetadata> {
    const params = new URLSearchParams();
    if (deprecated) params.append('deprecated', 'true');
    params.append('t', Date.now().toString());

    const response = await fetch(`${API_BASE}/selfplay/games/${gameId}/metadata?${params}`);
    if (!response.ok) throw new Error(`Erro ao buscar metadados (${response.status})`);
    return response.json();
}

/**
 * Fetch paginated game steps
 */
export async function fetchGameSteps(
    gameId: string,
    start: number = 0,
    end: number = 50,
    deprecated: boolean = false
): Promise<GameStep[]> {
    const params = new URLSearchParams();
    params.append('start', start.toString());
    params.append('end', end.toString());
    if (deprecated) params.append('deprecated', 'true');
    params.append('t', Date.now().toString());

    const response = await fetch(`${API_BASE}/selfplay/games/${gameId}/steps?${params}`);
    if (!response.ok) throw new Error(`Erro ao buscar steps (${response.status})`);
    return response.json();
}

// --- Replay Service Class ---

type ReplayCallback = (state: GameReplayState) => void;

export interface GameReplayState {
    game_id: string;
    step: number;
    phase: 'GROWTH' | 'REFINEMENT';
    tensor: {
        shape: [number, number, number, number];
        data: Float32Array;
    };
    value: number;
    policy: {
        add: Float32Array;
        remove: Float32Array;
    };
    mcts: {
        visit_add: Float32Array;
        visit_remove: Float32Array;
    };
    visuals?: {
        opaqueMatrix: Float32Array;
        opaqueColor: Float32Array;
        overlayMatrix: Float32Array;
        overlayColor: Float32Array;
        mctsMatrix?: Float32Array;
        mctsColor?: Float32Array;
    };
    selected_actions: SelectedAction[];
    mcts_stats: MCTSStats;
    // Added for simpler UI consumption
    volume_fraction: number;
    is_connected: boolean;
    n_islands: number;
    compliance_fem?: number;
    max_displacement?: number;
    island_penalty?: number;
    loose_voxels?: number;
    connected_load_fraction?: number;
    reward_components?: RewardComponents | null;
    displacement_map?: Float32Array; // [NEW]
}


// Worker Setup
import ReplayWorker from '../workers/replay.worker?worker';

export class SelfPlayReplayService {
    private subscribers: ReplayCallback[] = [];
    private intervalId: ReturnType<typeof setInterval> | null = null;
    private worker: Worker;

    // State
    private gameId: string | null = null;
    private metadata: GameMetadata | null = null;
    private deprecated: boolean = false;

    // Pagination / Buffering
    private framesMap: Map<number, GameReplayState> = new Map();
    private currentStepIndex = 0;
    private isPlaying = false;

    // Constants
    private readonly FETCH_CHUNK_SIZE = 10; // Reduced to avoid timeout on visual computation
    private readonly PREFETCH_THRESHOLD = 5;

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
            const steps = payload as GameReplayState[];
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
     * Load game data
     */
    async loadGame(gameId: string, deprecated: boolean = false): Promise<void> {
        this.stop();
        this.gameId = gameId;
        this.deprecated = deprecated;

        try {
            // 1. Fetch Metadata (Keep on main thread for now as it's small JSON)
            this.metadata = await fetchGameMetadata(gameId, deprecated);

            // 2. Fetch Initial Chunk via Worker
            this.requestChunk(0, this.FETCH_CHUNK_SIZE);

            this.currentStepIndex = 0;
            // Wait a bit? Or just let the worker callback update call us.
            // We can resolve immediately to unblock UI, loading state handled by UI?
            // Existing UI waits for load() promise.
            // We should probably wait for at least first frame.

            await this.waitForFrame(0);

        } catch (err) {
            console.error("Error loading game:", err);
            throw err;
        }
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
                reject(new Error("Timeout waiting for frame"));
            }, timeout);
        });
    }

    private requestChunk(start: number, count: number): void {
        if (!this.gameId) return;
        const end = start + count;

        // Check if we already have these?
        // Simple check: if we have the start, maybe skip.
        if (this.framesMap.has(start)) return;

        this.worker.postMessage({
            type: 'LOAD_CHUNK',
            payload: {
                gameId: this.gameId,
                start,
                end,
                isDeprecated: this.deprecated,
                apiBase: 'http://localhost:8000' // Hardcoded for now or get from env
            }
        });
    }

    private pruneCache() {
        const KEEP_WINDOW = 200; // Keep more frames in memory now that they are binary efficient?
        // Actually Float32Arrays are still big.
        const minStep = this.currentStepIndex - KEEP_WINDOW;
        const maxStep = this.currentStepIndex + KEEP_WINDOW;

        for (const step of this.framesMap.keys()) {
            if (step < minStep || step > maxStep) {
                this.framesMap.delete(step);
            }
        }
    }

    // ProcessSteps logic moved to Worker (binary parsing)

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

            // Check buffering
            if (!this.framesMap.has(this.currentStepIndex)) {
                // Buffer underrun
                this.pause();
                this.requestChunk(this.currentStepIndex, this.FETCH_CHUNK_SIZE);
                // Try resuming after a bit?
                this.waitForFrame(this.currentStepIndex).then(() => this.play()).catch(() => { });
                return;
            }

            // Prefetch
            const nextChunkStart = this.currentStepIndex + this.PREFETCH_THRESHOLD;
            if (nextChunkStart < this.metadata.total_steps && !this.framesMap.has(nextChunkStart)) {
                this.requestChunk(nextChunkStart, this.FETCH_CHUNK_SIZE);
            }

            this.notifySubscribers(this.framesMap.get(this.currentStepIndex)!);
        }, 100); // 100ms = 10fps, faster playback possible now?
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
        this.gameId = null;
    }

    async seekToStep(index: number): Promise<void> {
        if (!this.metadata || index < 0 || index >= this.metadata.total_steps) return;

        this.currentStepIndex = index;

        if (this.framesMap.has(index)) {
            this.notifySubscribers(this.framesMap.get(index)!);
        } else {
            this.pause();
            this.requestChunk(index, this.FETCH_CHUNK_SIZE);
            // We don't await here to keep UI responsive, but we could show loading
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

    private notifySubscribers(state: GameReplayState): void {
        this.subscribers.forEach(cb => cb(state));
    }

    getState() {
        return {
            gameId: this.gameId,
            stepsLoaded: this.metadata ? this.metadata.total_steps : 0,
            currentStep: this.currentStepIndex,
            isPlaying: this.isPlaying,
            metadata: this.metadata, // Replaces gameData (which was full)
            gameData: null, // Deprecated, kept for compat if needed but will be null
        };
    }

    /**
     * Get value history (from metadata)
     */
    getValueHistory(): number[] {
        return this.metadata?.value_history || [];
    }

    getCurrentFrame(): GameReplayState | null {
        return this.framesMap.get(this.currentStepIndex) || null;
    }

    public getFrame(step: number): GameReplayState | undefined {
        return this.framesMap.get(step);
    }
}

// Singleton instance
export const selfPlayReplayService = new SelfPlayReplayService();
