import type { GameState, Phase, Project } from './types';
import mockEpisodeBezier from '../data/mock_episode_bezier.json';
import mockEpisodeFullDomain from '../data/mock_episode_fulldomain.json';

// Define interface for the JSON structure
interface MockEpisodeData {
    episode_id: string;
    load_config?: {
        x: number;
        y: number;
        z_start: number;
        z_end: number;
    };
    frames: Array<{
        step: number;
        phase: Phase;
        density?: number[][][];
        tensor?: {
            shape: number[];
            data: number[];
        };
        fitness: number;
        metadata: {
            compliance?: number;
            vol_frac?: number;
            volume_fraction?: number;
        };
        policy?: {
            add: number[][][];
            remove: number[][][];
        };
    }>;
}

export class MockService {
    private subscribers: ((state: GameState) => void)[] = [];
    private intervalId: any = null;
    private currentStepIndex = 0;
    private mockEpisode: GameState[] = [];
    private isPlaying: boolean = false;
    private maxCompliance: number = 1.0;

    // Simulate database of projects
    private projects: Project[] = [
        { id: 'bezier-run', name: 'Bezier Strategy', status: 'COMPLETED', thumbnail_url: '/placeholder.png', last_modified: '2025-11-29', episode_id: 'bezier-run' },
        { id: 'full-domain-run', name: 'Full Domain Strategy', status: 'COMPLETED', thumbnail_url: '/placeholder.png', last_modified: '2025-11-29', episode_id: 'full-domain-run' }
    ];

    constructor() {
        console.log("MockService Initialized");
    }

    async getProjects(): Promise<Project[]> {
        return new Promise(resolve => {
            setTimeout(() => resolve(this.projects), 500);
        });
    }

    async getProject(id: string): Promise<Project | undefined> {
        return new Promise(resolve => {
            setTimeout(() => resolve(this.projects.find(p => p.id === id)), 300);
        });
    }

    async startSimulation(id: string) {
        if (this.intervalId) clearInterval(this.intervalId);
        this.isPlaying = false;

        let data: MockEpisodeData;
        if (id === 'full-domain-run') {
            data = mockEpisodeFullDomain as unknown as MockEpisodeData;
        } else {
            // Default to Bezier - fetch from public folder
            try {
                const response = await fetch('/mock_episode.json');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                data = await response.json() as MockEpisodeData;
            } catch (e) {
                console.error("Failed to fetch mock_episode.json, falling back to static data", e);
                data = mockEpisodeBezier as unknown as MockEpisodeData;
            }
        }

        console.log(`[MockService] Loading episode data for ${id}. ID: ${data.episode_id}`);
        console.log(`[MockService] Data keys:`, Object.keys(data));
        if (data.frames) {
            console.log(`[MockService] Frames length: ${data.frames.length}`);
        } else {
            console.error(`[MockService] Frames is undefined!`);
        }

        try {
            this.mockEpisode = this.processData(data);

            // Calculate Max Compliance for the episode
            this.maxCompliance = 0;
            this.mockEpisode.forEach(step => {
                if (step.metadata.compliance && step.metadata.compliance > this.maxCompliance) {
                    this.maxCompliance = step.metadata.compliance;
                }
            });
            if (this.maxCompliance === 0) this.maxCompliance = 1.0; // Prevent division by zero

            console.log(`[MockService] Processed episode length: ${this.mockEpisode.length}, Max Compliance: ${this.maxCompliance}`);
            this.startLoop(id);
        } catch (e) {
            console.error("[MockService] Error processing data:", e);
        }
    }

    private processData(data: MockEpisodeData): GameState[] {
        const frames = data.frames;
        return frames.map((frame) => {
            let tensorData: Float32Array;
            let shape: [number, number, number, number];
            const CHANNELS = 5;

            // Check if we have pre-processed tensor data (New Format)
            if (frame.tensor) {
                const rawShape = frame.tensor.shape;
                const rawData = frame.tensor.data;
                tensorData = new Float32Array(rawData);
                shape = rawShape as [number, number, number, number];

            } else if (frame.density) {
                // Legacy Format: Construct Tensor from Density 3D Array
                const density3D = frame.density;
                const D = density3D.length;
                const H = density3D[0].length;
                const W = density3D[0][0].length;

                shape = [CHANNELS, D, H, W];
                tensorData = new Float32Array(CHANNELS * D * H * W).fill(0);

                // Fill Density (Channel 0)
                for (let d = 0; d < D; d++) {
                    for (let h = 0; h < H; h++) {
                        for (let w = 0; w < W; w++) {
                            const val = density3D[d][h][w];
                            const idx = 0 * (D * H * W) + d * (H * W) + h * W + w;
                            tensorData[idx] = val;

                            // Add Support (Channel 1) if X=0 (d=0)
                            if (d === 0) {
                                const sIdx = 1 * (D * H * W) + d * (H * W) + h * W + w;
                                tensorData[sIdx] = 1.0;
                            }
                        }
                    }
                }

                // Fill Load (Channel 3)
                if (data.load_config) {
                    const lc = data.load_config;
                    const lx = Math.min(lc.x, D - 1);
                    const ly = Math.min(lc.y, H - 1);
                    const lz_s = Math.max(0, lc.z_start);
                    const lz_e = Math.min(lc.z_end, W);

                    for (let z = lz_s; z < lz_e; z++) {
                        const lIdx = 3 * (D * H * W) + lx * (H * W) + ly * W + z;
                        tensorData[lIdx] = -1.0; // Fy = -1
                    }
                }
            } else {
                console.error("Frame missing both tensor and density data", frame);
                throw new Error("Invalid frame data: missing tensor/density");
            }

            // Parse Policy Heatmap
            let policyHeatmap = undefined;
            if (frame.policy) {
                // Need dimensions for flattening if not already provided
                // If we used tensor, we have shape. If density, we have D,H,W.
                const [, D, H, W] = shape;

                const flattenPolicy = (arr3d: number[][][]) => {
                    const flat = new Float32Array(D * H * W);
                    for (let d = 0; d < D; d++) {
                        for (let h = 0; h < H; h++) {
                            for (let w = 0; w < W; w++) {
                                const idx = d * (H * W) + h * W + w;
                                flat[idx] = arr3d[d][h][w];
                            }
                        }
                    }
                    return flat;
                };

                policyHeatmap = {
                    add: flattenPolicy(frame.policy.add),
                    remove: flattenPolicy(frame.policy.remove)
                };
            }

            // Handle metadata mapping
            const compliance = frame.metadata?.compliance ?? (frame as any).compliance ?? 0;
            const volFrac = frame.metadata?.volume_fraction ?? frame.metadata?.vol_frac ?? (frame as any).vol_frac ?? 0;

            // Handle fitness / value confidence
            const fitness = frame.fitness ?? (frame as any).fitness_score ?? 0;
            const valueConf = (frame as any).value_confidence ?? fitness;

            return {
                episode_id: data.episode_id,
                step: frame.step,
                phase: frame.phase,
                tensor: {
                    shape: shape,
                    data: tensorData
                },
                fitness_score: fitness,
                valid_fem: true,
                metadata: {
                    compliance: compliance,
                    max_displacement: 0,
                    volume_fraction: volFrac
                },
                value_confidence: valueConf,
                policy_heatmap: policyHeatmap
            };
        });
    }

    private startLoop(episodeId: string, resetIndex: boolean = true) {
        if (this.intervalId) clearInterval(this.intervalId);

        if (resetIndex) {
            this.currentStepIndex = 0;
        }
        console.log(`Starting simulation for ${episodeId} with ${this.mockEpisode.length} steps.`);
        this.isPlaying = true;

        this.intervalId = setInterval(() => {
            if (this.currentStepIndex >= this.mockEpisode.length - 1) {
                this.pause();
                return;
            }

            this.currentStepIndex++;
            const state = this.mockEpisode[this.currentStepIndex];
            this.notifySubscribers(state);
        }, 500); // 2 steps per second
    }

    stopSimulation() {
        if (this.intervalId) clearInterval(this.intervalId);
        this.isPlaying = false;
    }

    seekToStep(index: number) {
        if (index >= 0 && index < this.mockEpisode.length) {
            this.currentStepIndex = index;
            this.notifySubscribers(this.mockEpisode[index]);
        }
    }

    stepForward() {
        if (this.currentStepIndex < this.mockEpisode.length - 1) {
            this.seekToStep(this.currentStepIndex + 1);
        }
    }

    stepBackward() {
        if (this.currentStepIndex > 0) {
            this.seekToStep(this.currentStepIndex - 1);
        }
    }

    togglePlay() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.resume();
        }
    }

    pause() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.isPlaying = false;
        // Notify to update UI state
        if (this.mockEpisode.length > 0) {
            this.notifySubscribers(this.mockEpisode[this.currentStepIndex]);
        }
    }

    resume() {
        if (!this.intervalId && this.mockEpisode.length > 0) {
            // If at end, restart
            if (this.currentStepIndex >= this.mockEpisode.length - 1) {
                this.currentStepIndex = 0;
            }
            this.startLoop(this.mockEpisode[0].episode_id, false);
        }
    }

    subscribe(callback: (state: GameState) => void) {
        this.subscribers.push(callback);
        return () => {
            this.subscribers = this.subscribers.filter(s => s !== callback);
        };
    }

    private notifySubscribers(state: GameState) {
        this.subscribers.forEach(cb => cb(state));
    }

    getSimulationState() {
        return {
            episodeId: this.mockEpisode.length > 0 ? this.mockEpisode[0].episode_id : 'N/A',
            stepsLoaded: this.mockEpisode.length,
            currentStep: this.currentStepIndex,
            isPlaying: this.isPlaying,
            maxCompliance: this.maxCompliance,
            isRealRun: true
        };
    }

    // Deprecated alias for backward compatibility if needed, but we should use getSimulationState
    getDebugInfo() {
        return this.getSimulationState();
    }
}

export const mockService = new MockService();
