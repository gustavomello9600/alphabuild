
// replay.worker.ts

/* eslint-disable no-restricted-globals */

interface WorkerMessage {
    type: 'LOAD_CHUNK';
    payload: {
        gameId: string;
        start: number;
        end: number;
        isDeprecated: boolean;
        apiBase: string; // Pass base URL since worker doesn't have env
    };
}

// 7-Channel structure
const CHANNEL = {
    DENSITY: 0,
    MASK_X: 1,
    MASK_Y: 2,
    MASK_Z: 3,
    FORCE_X: 4,
    FORCE_Y: 5,
    FORCE_Z: 6,
};

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
    if (e.data.type === 'LOAD_CHUNK') {
        const { gameId, start, end, isDeprecated, apiBase } = e.data.payload;
        try {
            const steps = await fetchBinarySteps(apiBase, gameId, start, end, isDeprecated);
            // Transfer buffers to main thread
            const buffers: Transferable[] = [];
            for (const s of steps) {
                if (s.tensor.data.buffer) buffers.push(s.tensor.data.buffer);

                if (s.policy.add.buffer) buffers.push(s.policy.add.buffer);
                if (s.policy.remove.buffer) buffers.push(s.policy.remove.buffer);

                if (s.mcts.visit_add.buffer) buffers.push(s.mcts.visit_add.buffer);
                if (s.mcts.visit_remove.buffer) buffers.push(s.mcts.visit_remove.buffer);

                if (s.visuals) {
                    if (s.visuals.opaqueMatrix.buffer) buffers.push(s.visuals.opaqueMatrix.buffer);
                    if (s.visuals.opaqueColor.buffer) buffers.push(s.visuals.opaqueColor.buffer);
                    if (s.visuals.overlayMatrix.buffer) buffers.push(s.visuals.overlayMatrix.buffer);
                    if (s.visuals.overlayColor.buffer) buffers.push(s.visuals.overlayColor.buffer);
                    if (s.visuals.mctsMatrix?.buffer) buffers.push(s.visuals.mctsMatrix.buffer);
                    if (s.visuals.mctsColor?.buffer) buffers.push(s.visuals.mctsColor.buffer);
                }
            }
            // Remove duplicates just in case (though slicing creates unique buffers)
            const uniqueBuffers = Array.from(new Set(buffers));
            self.postMessage({ type: 'CHUNK_LOADED', payload: steps }, uniqueBuffers);
        } catch (err: unknown) {
            const errorMessage = err instanceof Error ? err.message : 'Unknown error';
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
        // Total Float32 elements check
        // Tensor: 7 * D * H * W
        // Policy Add: D * H * W
        // Policy Rem: D * H * W
        // MCTS Add: D * H * W
        // MCTS Rem: D * H * W
        // Total per spatial: 7 + 1 + 1 + 1 + 1 = 11 floats per voxel
        // This is implicit in the data flow, we just slice by bytes

        // We slice the ArrayBuffer directly to create views
        // NOTE: These views share the underlying memory of 'buffer'.
        // When we transfer 'buffer' back, all these views become detached in the worker but valid in main.

        const spatialSize = D * H * W;
        const floatSize = 4;

        const tensorBytes = 7 * spatialSize * floatSize;
        const policyAudioBytes = spatialSize * floatSize;
        const mctsBytes = spatialSize * floatSize;

        // Create views using proper offsets
        // WARNING: DataView interaction is slow for bulk, use Float32Array on the buffer segment

        // Offset for this step's data body
        const stepBodyStart = offset;

        // We actually want to copy this out or just pass the shared buffer segment?
        // To allow Zero-Copy transfer of specific step data, strict isolation is needed.
        // But since we want to return a list of steps, we can parse them into objects holding TypedArrays.
        // If we want zero-copy transfer, we must transfer the underlying buffer.
        // But we have one big buffer for ALL steps.
        // Strategy: Create slice for each step's data to make them independent ArrayBuffers? 
        // No, that copies. 
        // Better Strategy: Return the ONE big buffer to main thread, and a list of pointer/offsets objects.
        // BUT main thread needs to put them into 'GameReplayState' objects.

        // Let's copy for now for simplicity of consumption in the App until we need extreme mem opt.
        // Actually, copying 50 steps x 1MB is 50MB copy. Not ideal but better than JSON.

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

        // Parse extension block: n_islands(i), loose_voxels(i), is_connected(i), compliance_fem(f), island_penalty(f)
        // Extension block is 20 bytes: 3 ints (12 bytes) + 2 floats (8 bytes)
        const n_islands = view.getInt32(cursor, true);
        const loose_voxels = view.getInt32(cursor + 4, true);
        const is_connected = view.getInt32(cursor + 8, true) === 1;
        const compliance_fem = view.getFloat32(cursor + 12, true);
        const island_penalty = view.getFloat32(cursor + 16, true);
        cursor += 20;

        // Parse reward_components (length int32 + bytes)
        const rcLength = view.getInt32(cursor, true);
        cursor += 4;

        let reward_components = undefined;
        if (rcLength > 0) {
            // Read bytes
            // Since we can't easily decode bytes to string in worker without TextDecoder (available?)
            // TextDecoder IS available in workers in modern browsers.
            const decoder = new TextDecoder('utf-8');
            // We need a view on the specific bytes
            const jsonView = new Uint8Array(buffer, cursor, rcLength);
            const jsonStr = decoder.decode(jsonView);
            reward_components = JSON.parse(jsonStr);
            cursor += rcLength;
        }

        // Compute Visual Buffers immediately
        const { opaqueMatrix, opaqueColor, overlayMatrix, overlayColor, mctsMatrix, mctsColor } = computeVisuals(
            tensorData,
            policyAdd, policyRem,
            mctsAdd, mctsRem,
            phaseCode,
            D, H, W,
            numSimulations
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
            reward_components, // [NEW]
            mcts_stats: {
                num_simulations: numSimulations,
                nodes_expanded: 0,
                max_depth: 0,
                cache_hits: 0,
                top8_concentration: 0,
                refutation: false
            },
            // Visual Buffers - NOW INCLUDING MCTS!
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
// Visual Computation Helpers
function computeVisuals(
    tensor: Float32Array,
    pAdd: Float32Array, pRem: Float32Array,
    mAdd: Float32Array, mRem: Float32Array,
    phase: number,
    D: number, H: number, W: number,
    totalSims: number
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

    const spatial = D * H * W;
    const CHANNEL_FORCE_X = 4;
    const CHANNEL_FORCE_Y = 5;
    const CHANNEL_FORCE_Z = 6;

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
                // Check loads
                const fx = tensor[CHANNEL_FORCE_X * spatial + idx];
                const fy = tensor[CHANNEL_FORCE_Y * spatial + idx];
                const fz = tensor[CHANNEL_FORCE_Z * spatial + idx];
                const hasLoad = Math.abs(fx) > 0.01 || Math.abs(fy) > 0.01 || Math.abs(fz) > 0.01;

                if (density > 0.5 || hasLoad) {
                    setMatrix(opaqueMatrix, oIdx, x, y, z, 1.0);

                    if (hasLoad) {
                        // #FF6B00 -> 1.0, 0.42, 0.0
                        setColor(opaqueColor, oIdx, 1.0, 0.42, 0.0);
                    } else {
                        // Gray: HSL(0,0, gray*0.95). 
                        const g = Math.max(0.2, density) * 0.95;
                        setColor(opaqueColor, oIdx, g, g, g);
                    }
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
                // Main thread can filter? No, main thread rendering is InstanceMesh, count is fixed by buffer.
                // We should filter here.

                // Optimization: Find Max first?
                // Iterating twice is okay for 260k elements in a worker.

                // Actually, let's just do one pass and normalized later? 
                // No, color needs normalization.
                // Let's assume we want to show significant visits. 
                // Any visit > 0 is significant? 
                // In large trees, we might have thousands.

                // Let's implement a dynamic threshold or just > 1 visit.

                if (mAdd[idx] > 0 && density <= 0.5) {
                    setMatrix(mctsMatrix, mIdx, x, y, z, 1.0);
                    // Green-ish wireframe.
                    // We store intensity in Alpha or just RGB?
                    // Wireframe material is Basic.
                    // Color: 1,1,1 implies white.
                    // We want Gradient?
                    // Let's store raw visit count in Alpha for shader? 
                    // Or just pre-compute specific color based on valid range (e.g. log scale).

                    // Simple: Green
                    setColor(mctsColor, mIdx, 0.0, 1.0, 0.6);
                    mIdx++;
                }

                if (mRem[idx] > 0 && density > 0.5) {
                    setMatrix(mctsMatrix, mIdx, x, y, z, 1.0);
                    // Red-ish wireframe
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
