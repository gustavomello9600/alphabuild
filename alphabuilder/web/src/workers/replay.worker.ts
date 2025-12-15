
// replay.worker.ts

/* eslint-disable no-restricted-globals */

interface WorkerMessage {
    type: 'LOAD_CHUNK';
    payload: {
        source: 'selfplay' | 'training';
        // Common
        start: number;
        end: number;
        apiBase: string;
        // SelfPlay specific
        gameId?: string;
        isDeprecated?: boolean;
        // Training specific
        dbId?: string;
        episodeId?: string;
        staticData?: {
            bc_masks?: number[];
            forces?: number[];
        };
    };
}

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
    if (e.data.type === 'LOAD_CHUNK') {
        const { source, apiBase, start, end } = e.data.payload;
        console.log(`[Worker] Received LOAD_CHUNK: ${source} ${start}-${end}`);
        try {
            let steps: any[] = [];

            if (source === 'selfplay') {
                const { gameId, isDeprecated } = e.data.payload;
                if (!gameId) throw new Error("GameID required for selfplay source");
                steps = await fetchBinarySteps(apiBase, gameId, start, end, !!isDeprecated);
            } else {
                const { dbId, episodeId, staticData } = e.data.payload;
                if (!dbId || !episodeId) throw new Error("DbId and EpisodeId required for training source");
                console.log(`[Worker] Fetching training steps for ${episodeId}`);
                steps = await fetchTrainingSteps(apiBase, dbId, episodeId, start, end, staticData);
                console.log(`[Worker] Fetched ${steps.length} steps`);
            }

            // Transfer buffers to main thread
            const buffers: Transferable[] = [];
            for (const s of steps) {
                if (s.tensor.data.buffer) buffers.push(s.tensor.data.buffer);

                if (s.policy?.add?.buffer) buffers.push(s.policy.add.buffer);
                if (s.policy?.remove?.buffer) buffers.push(s.policy.remove.buffer);

                if (s.mcts?.visit_add?.buffer) buffers.push(s.mcts.visit_add.buffer);
                if (s.mcts?.visit_remove?.buffer) buffers.push(s.mcts.visit_remove.buffer);

                if (s.visuals) {
                    if (s.visuals.opaqueMatrix.buffer) buffers.push(s.visuals.opaqueMatrix.buffer);
                    if (s.visuals.opaqueColor.buffer) buffers.push(s.visuals.opaqueColor.buffer);
                    if (s.visuals.overlayMatrix.buffer) buffers.push(s.visuals.overlayMatrix.buffer);
                    if (s.visuals.overlayColor.buffer) buffers.push(s.visuals.overlayColor.buffer);
                    if (s.visuals.mctsMatrix?.buffer) buffers.push(s.visuals.mctsMatrix.buffer);
                    if (s.visuals.mctsColor?.buffer) buffers.push(s.visuals.mctsColor.buffer);
                }

                if (s.displacement_map?.buffer) buffers.push(s.displacement_map.buffer);
            }

            // Remove duplicates just in case (though slicing creates unique buffers)
            const uniqueBuffers = Array.from(new Set(buffers));
            console.log(`[Worker] Posting CHUNK_LOADED with ${steps.length} steps`);
            self.postMessage({ type: 'CHUNK_LOADED', payload: steps }, uniqueBuffers as any);
        } catch (err: unknown) {
            const errorMessage = err instanceof Error ? err.message : 'Unknown error';
            console.error('[Worker] Error:', errorMessage);
            self.postMessage({ type: 'ERROR', payload: errorMessage });
        }
    }
};

async function fetchBinarySteps(
    apiBase: string,
    gameId: string,
    start: number,
    end: number,
    deprecated: boolean
) {
    const params = new URLSearchParams();
    params.append('start', start.toString());
    params.append('end', end.toString());
    if (deprecated) params.append('deprecated', 'true');
    params.append('t', Date.now().toString());

    const response = await fetch(`${apiBase}/selfplay/games/${gameId}/steps/binary?${params}`);
    if (!response.ok) throw new Error(`Fetch error: ${response.status}`);

    const buffer = await response.arrayBuffer();
    return parseBinaryStream(buffer);
}

// Fetch and Parse Training Data (JSON)
// Fetch and Parse Training Data (JSON)
async function fetchTrainingSteps(
    apiBase: string,
    dbId: string,
    episodeId: string,
    start: number,
    end: number,
    staticData?: { bc_masks?: number[], forces?: number[] }
) {
    const params = new URLSearchParams();
    params.append('start', start.toString());
    params.append('end', end.toString());
    params.append('t', Date.now().toString());

    const response = await fetch(`${apiBase}/databases/${dbId}/episodes/${episodeId}/frames?${params}`);
    if (!response.ok) throw new Error(`Fetch error: ${response.status}`);

    const frames = await response.json();
    return processTrainingFrames(frames, episodeId, staticData);
}

function processTrainingFrames(
    frames: any[],
    episodeId: string,
    staticData?: { bc_masks?: number[], forces?: number[] }
) {
    return frames.map(frame => {
        // frame.tensor_data only contains density (1 channel)
        const densityData = new Float32Array(frame.tensor_data);
        // frame.tensor_shape is [1, D, H, W]
        const [, D, H, W] = frame.tensor_shape as [number, number, number, number];
        const spatialSize = D * H * W;

        // Create full 7-channel tensor
        const fullTensorData = new Float32Array(7 * spatialSize);

        // Channel 0: Density
        fullTensorData.set(densityData, 0);

        // Channels 1-3: BC Masks
        if (staticData?.bc_masks && staticData.bc_masks.length === 3 * spatialSize) {
            fullTensorData.set(staticData.bc_masks, spatialSize); // Offset by 1 channel size
        }

        // Channels 4-6: Forces
        if (staticData?.forces && staticData.forces.length === 3 * spatialSize) {
            fullTensorData.set(staticData.forces, 4 * spatialSize); // Offset by 4 channel sizes
        }

        const shape: [number, number, number, number] = [7, D, H, W];

        const policyAdd = frame.policy_add ? new Float32Array(frame.policy_add) : new Float32Array(spatialSize);
        const policyRem = frame.policy_remove ? new Float32Array(frame.policy_remove) : new Float32Array(spatialSize);

        // MCTS Data not available in training frames usually
        const mctsAdd = new Float32Array(spatialSize);
        const mctsRem = new Float32Array(spatialSize);

        // Helper to parse base64 displacement
        let displacement_map = undefined;
        if (frame.displacement_map) {
            const binaryString = atob(frame.displacement_map);
            const len = binaryString.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) bytes[i] = binaryString.charCodeAt(i);
            displacement_map = new Float32Array(bytes.buffer);
        }

        // Compute Visuals
        const { opaqueMatrix, opaqueColor, overlayMatrix, overlayColor, mctsMatrix, mctsColor } = computeVisuals(
            fullTensorData,
            policyAdd, policyRem,
            mctsAdd, mctsRem,
            D, H, W
        );

        return {
            episode_id: episodeId,
            step: frame.step,
            phase: frame.phase,
            tensor: { shape, data: fullTensorData },
            policy: { add: policyAdd, remove: policyRem },
            mcts: { visit_add: mctsAdd, visit_remove: mctsRem },
            // Training specific fields mapped to common
            value: frame.fitness_score,
            value_confidence: frame.fitness_score, // Map fitness to confidence for HUD
            action_sequence: frame.action_sequence, // Keep passing this
            reward_components: frame.reward_components,
            displacement_map,

            // Stats (Mock or Empty)
            mcts_stats: {
                num_simulations: 0,
                nodes_expanded: 0,
                max_depth: 0,
                cache_hits: 0,
                top8_concentration: 0,
                refutation: false
            },

            // Visuals
            visuals: {
                opaqueMatrix,
                opaqueColor,
                overlayMatrix,
                overlayColor,
                mctsMatrix,
                mctsColor
            }
        };
    });
}

function parseBinaryStream(buffer: ArrayBuffer) {
    const view = new DataView(buffer);
    let offset = 0;
    const steps = [];

    while (offset < buffer.byteLength) {
        // Read Header (40 bytes)
        const stepIndex = view.getInt32(offset, true);
        const phaseCode = view.getInt32(offset + 4, true);
        const D = view.getInt32(offset + 8, true);
        const H = view.getInt32(offset + 12, true);
        const W = view.getInt32(offset + 16, true);
        const dataLength = view.getInt32(offset + 20, true);
        const numSimulations = view.getInt32(offset + 24, true);
        const fixedValue = view.getInt32(offset + 28, true);
        const fixedConnectivity = view.getInt32(offset + 32, true); // Slot 9: connectivity bonus
        const fixedVolumeFraction = view.getInt32(offset + 36, true); // Slot 10: volume fraction

        offset += 40;

        // Extract Data Arrays
        const spatialSize = D * H * W;
        const floatSize = 4;

        const tensorBytes = 7 * spatialSize * floatSize;
        const policyAudioBytes = spatialSize * floatSize;
        const mctsBytes = spatialSize * floatSize;

        const stepBodyStart = offset;

        const tensorData = new Float32Array(buffer.slice(stepBodyStart, stepBodyStart + tensorBytes));
        let cursor = stepBodyStart + tensorBytes;

        const policyAdd = new Float32Array(buffer.slice(cursor, cursor + policyAudioBytes));
        cursor += policyAudioBytes;

        const policyRem = new Float32Array(buffer.slice(cursor, cursor + policyAudioBytes));
        cursor += policyAudioBytes;

        const mctsAdd = new Float32Array(buffer.slice(cursor, cursor + mctsBytes));
        cursor += mctsBytes;

        const mctsRem = new Float32Array(buffer.slice(cursor, cursor + mctsBytes));
        cursor += mctsBytes;

        // Parse selected_actions
        const numActions = view.getInt32(cursor, true);
        cursor += 4;

        const selected_actions = [];
        for (let i = 0; i < numActions; i++) {
            const channel = view.getInt32(cursor, true);
            const x = view.getInt32(cursor + 4, true);
            const y = view.getInt32(cursor + 8, true);
            const z = view.getInt32(cursor + 12, true);
            const visits = view.getInt32(cursor + 16, true);
            const q_value = view.getFloat32(cursor + 20, true);
            cursor += 24; // 5 ints (20 bytes) + 1 float (4 bytes)

            selected_actions.push({ channel, x, y, z, visits, q_value });
        }

        // Parse extension block
        const n_islands = view.getInt32(cursor, true);
        const loose_voxels = view.getInt32(cursor + 4, true);
        const is_connected = view.getInt32(cursor + 8, true) === 1;
        const compliance_fem = view.getFloat32(cursor + 12, true);
        const island_penalty = view.getFloat32(cursor + 16, true);
        const max_displacement = view.getFloat32(cursor + 20, true);
        cursor += 24;

        // Parse reward_components
        const rcLength = view.getInt32(cursor, true);
        cursor += 4;

        let reward_components = undefined;
        if (rcLength > 0) {
            const decoder = new TextDecoder('utf-8');
            const jsonView = new Uint8Array(buffer, cursor, rcLength);
            const jsonStr = decoder.decode(jsonView);
            reward_components = JSON.parse(jsonStr);
            cursor += rcLength;
        }

        // Parse displacement_map
        const dispLength = view.getInt32(cursor, true);
        cursor += 4;

        let displacement_map = undefined;
        if (dispLength > 0) {
            displacement_map = new Float32Array(buffer.slice(cursor, cursor + dispLength));
            cursor += dispLength;
        }

        // Compute Visual Buffers immediately
        const { opaqueMatrix, opaqueColor, overlayMatrix, overlayColor, mctsMatrix, mctsColor } = computeVisuals(
            tensorData,
            policyAdd, policyRem,
            mctsAdd, mctsRem,
            D, H, W
        );

        steps.push({
            step: stepIndex,
            phase: phaseCode === 0 ? 'GROWTH' : 'REFINEMENT',
            tensor: { shape: [7, D, H, W], data: tensorData },
            policy: { add: policyAdd, remove: policyRem },
            mcts: {
                visit_add: mctsAdd,
                visit_remove: mctsRem,
            },
            selected_actions,
            value: fixedValue / 10000.0,
            connected_load_fraction: fixedConnectivity / 10000.0,
            volume_fraction: fixedVolumeFraction / 10000.0,
            n_islands,
            loose_voxels,
            is_connected,
            compliance_fem: compliance_fem || undefined,
            island_penalty: island_penalty || undefined,
            max_displacement: max_displacement || undefined,
            reward_components,
            displacement_map, // [NEW]
            mcts_stats: {
                num_simulations: numSimulations,
                nodes_expanded: 0,
                max_depth: 0,
                cache_hits: 0,
                top8_concentration: 0,
                refutation: false
            },
            // Visual Buffers
            visuals: {
                opaqueMatrix,
                opaqueColor,
                overlayMatrix,
                overlayColor,
                mctsMatrix,
                mctsColor
            }
        });

        offset = stepBodyStart + dataLength;
    }

    return steps;
}

// Visual Computation Helpers
function computeVisuals(
    tensor: Float32Array,
    pAdd: Float32Array, pRem: Float32Array,
    mAdd: Float32Array, mRem: Float32Array,
    D: number, H: number, W: number
) {
    // Estimations for buffer size
    const maxInstances = D * H * W; // worst case
    const opaqueMatrix = new Float32Array(maxInstances * 16);
    const opaqueColor = new Float32Array(maxInstances * 3);
    const overlayMatrix = new Float32Array(maxInstances * 16);
    const overlayColor = new Float32Array(maxInstances * 4); // RGB + Opacity

    // MCTS Buffers
    const mctsMatrix = new Float32Array(maxInstances * 16);
    const mctsColor = new Float32Array(maxInstances * 3);

    let oIdx = 0;
    let ovIdx = 0;
    let mIdx = 0;

    const spatial = D * H * W; // Used in other parts if revived, but for now clean up?
    // Constants removed to silence linter

    // Helper to set matrix at index
    const setMatrix = (out: Float32Array, idx: number, x: number, y: number, z: number, scale: number) => {
        const i = idx * 16;
        out[i] = scale; out[i + 1] = 0; out[i + 2] = 0; out[i + 3] = 0;
        out[i + 4] = 0; out[i + 5] = scale; out[i + 6] = 0; out[i + 7] = 0;
        out[i + 8] = 0; out[i + 9] = 0; out[i + 10] = scale; out[i + 11] = 0;
        out[i + 12] = x; out[i + 13] = y; out[i + 14] = z; out[i + 15] = 1;
    };

    // Helper to set color
    const setColor = (out: Float32Array, idx: number, r: number, g: number, b: number) => {
        const i = idx * 3;
        out[i] = r; out[i + 1] = g; out[i + 2] = b;
    };

    const setOverlayColor = (out: Float32Array, idx: number, r: number, g: number, b: number, a: number) => {
        const i = idx * 4;
        out[i] = r; out[i + 1] = g; out[i + 2] = b; out[i + 3] = a;
    };

    for (let d = 0; d < D; d++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const idx = d * (H * W) + h * W + w;
                const density = tensor[idx];
                const x = d - D / 2;
                const y = h + 0.5;
                const z = w - W / 2;

                // 1. Opaque Structure
                if (density > 0.5) {
                    setMatrix(opaqueMatrix, oIdx, x, y, z, 1.0);
                    // Gray: HSL(0,0, gray*0.95). 
                    const g = Math.max(0.2, density) * 0.95;
                    setColor(opaqueColor, oIdx, g, g, g);
                    oIdx++;
                }

                // 2. Overlay (Policy)
                const pAddVal = pAdd[idx];
                const pRemVal = pRem[idx];

                if (pAddVal > 0.01 && density <= 0.5) {
                    setMatrix(overlayMatrix, ovIdx, x, y, z, 1.0);
                    // #00FF9D -> 0.0, 1.0, 0.61
                    setOverlayColor(overlayColor, ovIdx, 0.0, 1.0, 0.61, Math.min(1, pAddVal * 2));
                    ovIdx++;
                } else if (pRemVal > 0.01 && density > 0.5) {
                    setMatrix(overlayMatrix, ovIdx, x, y, z, 1.0);
                    // #FF0055 -> 1.0, 0.0, 0.33
                    setOverlayColor(overlayColor, ovIdx, 1.0, 0.0, 0.33, Math.min(1, pRemVal * 2));
                    ovIdx++;
                }

                // MCTS Visuals
                if (mAdd[idx] > 0 && density <= 0.5) {
                    setMatrix(mctsMatrix, mIdx, x, y, z, 1.0);
                    setColor(mctsColor, mIdx, 0.0, 1.0, 0.6);
                    mIdx++;
                }

                if (mRem[idx] > 0 && density > 0.5) {
                    setMatrix(mctsMatrix, mIdx, x, y, z, 1.0);
                    setColor(mctsColor, mIdx, 1.0, 0.0, 0.33);
                    mIdx++;
                }

            }
        }
    }

    return {
        opaqueMatrix: opaqueMatrix.slice(0, oIdx * 16),
        opaqueColor: opaqueColor.slice(0, oIdx * 3),
        overlayMatrix: overlayMatrix.slice(0, ovIdx * 16),
        overlayColor: overlayColor.slice(0, ovIdx * 4),
        mctsMatrix: mctsMatrix.slice(0, mIdx * 16),
        mctsColor: mctsColor.slice(0, mIdx * 3),
    };
}
