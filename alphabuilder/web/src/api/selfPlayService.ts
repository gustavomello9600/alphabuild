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
        q_add: Float32Array;
        q_remove: Float32Array;
    };
    selected_actions: SelectedAction[];
    mcts_stats: MCTSStats;
    // Added for simpler UI consumption
    volume_fraction: number;
    is_connected: boolean;
    n_islands: number;
    compliance_fem?: number;
    island_penalty?: number;
    loose_voxels?: number;
}

export class SelfPlayReplayService {
    private subscribers: ReplayCallback[] = [];
    private intervalId: ReturnType<typeof setInterval> | null = null;

    // State
    private gameId: string | null = null;
    private metadata: GameMetadata | null = null;
    private deprecated: boolean = false;

    // Pagination / Buffering
    private framesMap: Map<number, GameReplayState> = new Map();
    private currentStepIndex = 0;
    private isPlaying = false;

    // Constants
    private readonly FETCH_CHUNK_SIZE = 25;
    private readonly PREFETCH_THRESHOLD = 10;

    /**
     * Load game data
     */
    async loadGame(gameId: string, deprecated: boolean = false): Promise<void> {
        this.stop();
        this.gameId = gameId;
        this.deprecated = deprecated;

        try {
            // 1. Fetch Metadata
            this.metadata = await fetchGameMetadata(gameId, deprecated);

            // 2. Fetch Initial Chunk
            await this.loadChunk(0, this.FETCH_CHUNK_SIZE);

            this.currentStepIndex = 0;

            if (this.framesMap.has(0)) {
                this.notifySubscribers(this.framesMap.get(0)!);
            }
        } catch (err) {
            console.error("Error loading game:", err);
            throw err;
        }
    }

    private async loadChunk(start: number, count: number): Promise<void> {
        if (!this.gameId) return;

        const end = start + count;
        // console.log(`Fetching SP chunk ${start} - ${end}`);

        const steps = await fetchGameSteps(this.gameId, start, end, this.deprecated);
        const processed = this.processSteps(steps);

        processed.forEach(f => {
            this.framesMap.set(f.step, f);
        });

        this.pruneCache();
    }

    private pruneCache() {
        const KEEP_WINDOW = 100;
        const minStep = this.currentStepIndex - KEEP_WINDOW;
        const maxStep = this.currentStepIndex + KEEP_WINDOW;

        for (const step of this.framesMap.keys()) {
            if (step < minStep || step > maxStep) {
                this.framesMap.delete(step);
            }
        }
    }

    private processSteps(steps: GameStep[]): GameReplayState[] {
        return steps.map(step => {
            const tensorData = new Float32Array(step.tensor_data);
            const shape = step.tensor_shape as [number, number, number, number];

            return {
                game_id: this.gameId!,
                step: step.step,
                phase: step.phase as 'GROWTH' | 'REFINEMENT',
                tensor: { shape, data: tensorData },
                value: step.value,
                policy: {
                    add: new Float32Array(step.policy_add),
                    remove: new Float32Array(step.policy_remove),
                },
                mcts: {
                    visit_add: new Float32Array(step.mcts_visit_add),
                    visit_remove: new Float32Array(step.mcts_visit_remove),
                    q_add: new Float32Array(step.mcts_q_add),
                    q_remove: new Float32Array(step.mcts_q_remove),
                },
                selected_actions: step.selected_actions,
                mcts_stats: step.mcts_stats,
                volume_fraction: step.volume_fraction || 0,
                is_connected: step.is_connected || false,
                n_islands: step.n_islands || 1,
                compliance_fem: step.compliance_fem,
                island_penalty: step.island_penalty,
                loose_voxels: step.loose_voxels,
            };
        });
    }

    play(): void {
        if (this.isPlaying || !this.metadata) return;

        if (this.currentStepIndex >= this.metadata.total_steps - 1) {
            this.currentStepIndex = 0;
            this.seekToStep(0);
        }

        this.isPlaying = true;
        this.intervalId = setInterval(async () => {
            if (!this.metadata) return;

            if (this.currentStepIndex >= this.metadata.total_steps - 1) {
                this.pause();
                return;
            }

            this.currentStepIndex++;

            // Check buffering
            if (!this.framesMap.has(this.currentStepIndex)) {
                this.pause();
                await this.loadChunk(this.currentStepIndex, this.FETCH_CHUNK_SIZE);
                this.play();
                return;
            }

            // Prefetch
            const nextChunkStart = this.currentStepIndex + this.PREFETCH_THRESHOLD;
            if (nextChunkStart < this.metadata.total_steps && !this.framesMap.has(nextChunkStart)) {
                this.loadChunk(nextChunkStart, this.FETCH_CHUNK_SIZE).catch(console.error);
            }

            this.notifySubscribers(this.framesMap.get(this.currentStepIndex)!);
        }, 500);
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
}

// Singleton instance
export const selfPlayReplayService = new SelfPlayReplayService();
