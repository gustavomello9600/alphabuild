import type { GameState, Phase, Tensor5D, Project } from './types';

// Mock Data Generators

const D = 16; // Reduced resolution for mock performance
const H = 16;
const W = 32;
const CHANNELS = 5;

const generateEmptyTensor = (): Tensor5D => {
    return {
        shape: [CHANNELS, D, H, W],
        data: new Float32Array(CHANNELS * D * H * W).fill(0)
    };
};

// Helper to set a voxel in the flattened tensor
const setVoxel = (tensor: Tensor5D, c: number, d: number, h: number, w: number, value: number) => {
    const index =
        c * (D * H * W) +
        d * (H * W) +
        h * W +
        w;
    tensor.data[index] = value;
};

// Generate a "fake" episode trace
const generateMockEpisode = (episodeId: string): GameState[] => {
    const steps: GameState[] = [];
    const totalSteps = 20;
    let currentPhase: Phase = 'GROWTH';
    const tensor = generateEmptyTensor();

    // Set BCs (Left Wall)
    for (let d = 0; d < D; d++) {
        for (let h = 0; h < H; h++) {
            setVoxel(tensor, 1, d, h, 0, 1); // Channel 1 = Support
        }
    }

    // Set Load (Tip)
    setVoxel(tensor, 3, D / 2, H / 2, W - 1, -1); // Channel 3 = Fy

    for (let i = 0; i < totalSteps; i++) {
        // Simulate Growth: Add voxels from left to right
        let currentW = 0;
        if (i < 10) {
            currentPhase = 'GROWTH';
            // Ensure we reach the end (W-1) by step 9
            const progress = (i + 1) / 10;
            currentW = Math.floor(progress * (W - 1));

            // Build a simple bar connecting to the load
            for (let w = 0; w <= currentW; w++) {
                for (let d = D / 2 - 1; d <= D / 2 + 1; d++) {
                    for (let h = H / 2 - 1; h <= H / 2 + 1; h++) {
                        setVoxel(tensor, 0, d, h, w, 1); // Channel 0 = Density
                    }
                }
            }
        } else {
            currentPhase = 'REFINEMENT';
            // Simulate Refinement: Remove some voxels randomly to "optimize"
            for (let k = 0; k < 5; k++) {
                const rd = Math.floor(Math.random() * D);
                const rh = Math.floor(Math.random() * H);
                const rw = Math.floor(Math.random() * W);
                setVoxel(tensor, 0, rd, rh, rw, 0);
            }
        }

        // --- Neural Mock Logic ---

        // Value Head: Sigmoid-like curve representing increasing confidence as structure forms
        // Starts low (0.1), rises during growth, plateaus high (0.9) during refinement
        const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
        const x = (i - 10) / 2; // Shift to center around step 10
        const valueConfidence = sigmoid(x);

        // Policy Head: Heatmap focused on the "Growth Front"
        // We create a probability cloud around the current tip of the beam
        const policyHeatmap = new Float32Array(D * H * W).fill(0);
        if (currentPhase === 'GROWTH') {
            const targetW = Math.min(currentW + 2, W - 1);
            for (let d = 0; d < D; d++) {
                for (let h = 0; h < H; h++) {
                    for (let w = 0; w < W; w++) {
                        // Gaussian blob around the tip
                        const dist = Math.sqrt(Math.pow(d - D / 2, 2) + Math.pow(h - H / 2, 2) + Math.pow(w - targetW, 2));
                        if (dist < 4) {
                            const prob = Math.exp(-dist);
                            const idx = d * (H * W) + h * W + w;
                            policyHeatmap[idx] = prob;
                        }
                    }
                }
            }
        }

        steps.push({
            episode_id: episodeId,
            step: i,
            phase: currentPhase,
            tensor: { ...tensor, data: new Float32Array(tensor.data) }, // Clone data
            fitness_score: valueConfidence, // Correlate fitness with value confidence
            valid_fem: true,
            metadata: {
                compliance: 100 - (i * 4.5), // Decreasing compliance (stiffer)
                max_displacement: 0.1 + (i * 0.005),
                volume_fraction: i < 10 ? (i / 10) * 0.3 : 0.3 - ((i - 10) / 10) * 0.05
            },
            value_confidence: valueConfidence,
            policy_heatmap: policyHeatmap
        });
    }

    return steps;
};

class MockService {
    private intervalId: any = null;
    private subscribers: ((state: GameState) => void)[] = [];
    private currentStepIndex = 0;
    private mockEpisode: GameState[] = [];

    constructor() {
        console.log("MockService Initialized");
    }

    startSimulation(episodeId: string) {
        if (this.intervalId) clearInterval(this.intervalId);

        this.mockEpisode = generateMockEpisode(episodeId);
        this.currentStepIndex = 0;

        console.log(`Starting simulation for ${episodeId} with ${this.mockEpisode.length} steps.`);

        this.intervalId = setInterval(() => {
            if (this.currentStepIndex >= this.mockEpisode.length) {
                clearInterval(this.intervalId);
                return;
            }

            const state = this.mockEpisode[this.currentStepIndex];
            this.notifySubscribers(state);
            this.currentStepIndex++;
        }, 1000); // 1 step per second
    }

    stopSimulation() {
        if (this.intervalId) clearInterval(this.intervalId);
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

    getProjects(): Project[] {
        return [
            { id: '1', name: 'Estudo de Viga em Balanço', last_modified: '2023-10-27', status: 'COMPLETED', episode_id: 'ep-001' },
            { id: '2', name: 'Otimização de Ponte', last_modified: '2023-10-26', status: 'IN_PROGRESS', episode_id: 'ep-002' },
            { id: '3', name: 'Viga MBB', last_modified: '2023-10-25', status: 'FAILED', episode_id: 'ep-003' },
        ];
    }
}

export const mockService = new MockService();
