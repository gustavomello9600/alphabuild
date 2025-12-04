import { useEffect, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Play,
    Pause,
    SkipForward,
    SkipBack,
    ArrowLeft,
    Brain,
    Layers,
    Loader,
    AlertCircle,
    Network,
    Sparkles,
} from 'lucide-react';
import {
    trainingDataReplayService,
    type ReplayState,
} from '../api/trainingDataService';
import * as THREE from 'three';

// --- 3D Components (Adapted from Workspace.tsx) ---
// 
// Tensor v3.1 Channel Layout (7 channels):
// 0: Density (ρ) - Material state (0.0 to 1.0)
// 1: Mask X - 1.0 if displacement u_x is fixed (support)
// 2: Mask Y - 1.0 if displacement u_y is fixed (support)
// 3: Mask Z - 1.0 if displacement u_z is fixed (support)
// 4: Force X (Fx) - Normalized force component
// 5: Force Y (Fy) - Normalized force component
// 6: Force Z (Fz) - Normalized force component

const CHANNEL = {
    DENSITY: 0,
    MASK_X: 1,
    MASK_Y: 2,
    MASK_Z: 3,
    FORCE_X: 4,
    FORCE_Y: 5,
    FORCE_Z: 6,
};

// Log-Squash constants (from config.py - Spec 4.2)
const LOG_SQUASH = {
    ALPHA: 12.0,      // Volume penalty coefficient
    MU: -6.65,        // Estimated mean of log(C) distribution
    SIGMA: 2.0,       // Estimated std of log(C) distribution
    EPSILON: 1e-9,
};

/**
 * Recover compliance from fitness score and volume fraction.
 * Inverse of the log-squash normalization:
 *   s_raw = -log(C + ε) - α·vol
 *   normalized = tanh((s_raw - μ) / σ)
 * 
 * Inverse:
 *   s_raw = arctanh(normalized) * σ + μ
 *   C = exp(-(s_raw + α·vol)) - ε
 */
function recoverCompliance(fitnessScore: number, volumeFraction: number): number {
    // Clamp fitness to avoid arctanh(±1) = ±∞
    const clampedFitness = Math.max(-0.999, Math.min(0.999, fitnessScore));

    // Inverse tanh: arctanh(x) = 0.5 * ln((1+x)/(1-x))
    const arctanh = 0.5 * Math.log((1 + clampedFitness) / (1 - clampedFitness));

    // Recover s_raw
    const sRaw = arctanh * LOG_SQUASH.SIGMA + LOG_SQUASH.MU;

    // Recover compliance: -log(C + ε) = s_raw + α·vol
    // => C = exp(-(s_raw + α·vol)) - ε
    const compliance = Math.exp(-(sRaw + LOG_SQUASH.ALPHA * volumeFraction)) - LOG_SQUASH.EPSILON;

    return Math.max(0, compliance); // Ensure non-negative
}

// Support type colors based on constraint combination
// Following design system colors for semantic meaning
const SUPPORT_COLORS = {
    FULL_CLAMP: '#00F0FF',  // Cyan - Fully fixed (XYZ)
    RAIL_XY: '#7000FF',     // Purple - Rail constraint (XY fixed, Z free)
    ROLLER_Y: '#00FF9D',    // Green - Roller (only Y fixed)
    PARTIAL: '#3B82F6',     // Blue - Other partial constraints
};

/**
 * Determine support type and color based on active masks
 */
function getSupportColor(maskX: number, maskY: number, maskZ: number): string | null {
    const hasX = maskX > 0.5;
    const hasY = maskY > 0.5;
    const hasZ = maskZ > 0.5;

    if (!hasX && !hasY && !hasZ) return null; // Not a support

    if (hasX && hasY && hasZ) {
        return SUPPORT_COLORS.FULL_CLAMP; // Full clamp (XYZ)
    } else if (hasX && hasY && !hasZ) {
        return SUPPORT_COLORS.RAIL_XY; // Rail XY (Z free)
    } else if (!hasX && hasY && !hasZ) {
        return SUPPORT_COLORS.ROLLER_Y; // Roller Y only
    } else {
        return SUPPORT_COLORS.PARTIAL; // Other combinations
    }
}

const LoadVector = ({ tensor }: { tensor: ReplayState['tensor'] | null }) => {
    if (!tensor) return null;

    const numChannels = tensor.shape[0];
    const [, D, H, W] = tensor.shape;
    const loadPoints: { pos: THREE.Vector3; dir: THREE.Vector3; magnitude: number }[] = [];

    // For 7-channel tensor: Forces are in channels 4, 5, 6
    // For 5-channel tensor (legacy): Force Y is in channel 3
    const is7Channel = numChannels >= 7;

    for (let d = 0; d < D; d++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const flatIdx = d * (H * W) + h * W + w;

                let fx = 0, fy = 0, fz = 0;

                if (is7Channel) {
                    // v3.1: Channels 4, 5, 6 are Fx, Fy, Fz
                    fx = tensor.data[CHANNEL.FORCE_X * (D * H * W) + flatIdx];
                    fy = tensor.data[CHANNEL.FORCE_Y * (D * H * W) + flatIdx];
                    fz = tensor.data[CHANNEL.FORCE_Z * (D * H * W) + flatIdx];
                } else {
                    // Legacy 5-channel: Channel 3 is Fy only
                    fy = tensor.data[3 * (D * H * W) + flatIdx];
                }

                const magnitude = Math.sqrt(fx * fx + fy * fy + fz * fz);

                if (magnitude > 0.01) {
                    const pos = new THREE.Vector3(d - D / 2, h + 0.5, w - W / 2);
                    const dir = new THREE.Vector3(fx, fy, fz).normalize();
                    loadPoints.push({ pos, dir, magnitude });
                }
            }
        }
    }

    if (loadPoints.length === 0) return null;

    const arrowLength = 5;

    return (
        <group>
            {loadPoints.map((lp, i) => {
                const origin = lp.pos.clone().sub(lp.dir.clone().multiplyScalar(arrowLength));
                return (
                    <group key={i}>
                        <arrowHelper args={[lp.dir, origin, arrowLength, 0xff0055, 1, 0.5]} />
                        <mesh position={lp.pos}>
                            <sphereGeometry args={[0.2, 8, 8]} />
                            <meshBasicMaterial color="#ff0055" />
                        </mesh>
                    </group>
                );
            })}
        </group>
    );
};

const VoxelGrid = ({
    tensor,
    heatmap,
    showHeatmap,
}: {
    tensor: ReplayState['tensor'] | null;
    heatmap?: { add: Float32Array; remove: Float32Array };
    showHeatmap: boolean;
}) => {
    // Separate refs for opaque (standard) and transparent (policy) meshes
    const opaqueRef = useRef<THREE.InstancedMesh>(null);
    const policyRef = useRef<THREE.InstancedMesh>(null);
    const opacityAttrRef = useRef<THREE.InstancedBufferAttribute | null>(null);
    const dummy = new THREE.Object3D();

    // Shader injection for per-instance opacity on policy mesh
    const policyMaterialRef = useRef<THREE.MeshStandardMaterial | null>(null);

    useEffect(() => {
        if (!policyRef.current) return;

        const material = policyRef.current.material as THREE.MeshStandardMaterial;
        policyMaterialRef.current = material;

        material.onBeforeCompile = (shader) => {
            shader.vertexShader = `
                attribute float instanceOpacity;
                varying float vInstanceOpacity;
                ${shader.vertexShader}
            `.replace(
                '#include <begin_vertex>',
                `
                #include <begin_vertex>
                vInstanceOpacity = instanceOpacity;
                `
            );

            shader.fragmentShader = `
                varying float vInstanceOpacity;
                ${shader.fragmentShader}
            `.replace(
                '#include <dithering_fragment>',
                `
                #include <dithering_fragment>
                gl_FragColor.a *= vInstanceOpacity;
                `
            );
        };
        material.needsUpdate = true;
    }, []);

    useEffect(() => {
        if (!opaqueRef.current || !policyRef.current || !tensor) return;

        const numChannels = tensor.shape[0];
        const [, D, H, W] = tensor.shape;
        const is7Channel = numChannels >= 7;

        let opaqueCount = 0;
        let policyCount = 0;

        // Initialize opacity attribute for policy mesh
        const maxPolicyInstances = 10000;
        if (!opacityAttrRef.current) {
            const opacityArray = new Float32Array(maxPolicyInstances).fill(1.0);
            opacityAttrRef.current = new THREE.InstancedBufferAttribute(opacityArray, 1);
            opacityAttrRef.current.setUsage(THREE.DynamicDrawUsage);
            policyRef.current.geometry.setAttribute('instanceOpacity', opacityAttrRef.current);
        }

        // Helper to get policy data (color + opacity)
        const getPolicyData = (idx: number): { color: THREE.Color; opacity: number } | null => {
            if (!showHeatmap || !heatmap) return null;

            const addVal = heatmap.add ? heatmap.add[idx] : 0;
            const removeVal = heatmap.remove ? heatmap.remove[idx] : 0;

            if (addVal < 0.01 && removeVal < 0.01) return null;

            // FIXED COLORS - only opacity varies
            if (addVal > removeVal) {
                return {
                    color: new THREE.Color('#00FF9D'),  // Fixed green
                    opacity: addVal  // Direct value as opacity (0-1)
                };
            } else {
                return {
                    color: new THREE.Color('#FF0055'),  // Fixed red
                    opacity: removeVal  // Direct value as opacity (0-1)
                };
            }
        };

        for (let d = 0; d < D; d++) {
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    const flatIdx = d * (H * W) + h * W + w;

                    // Channel 0: Density
                    const density = tensor.data[CHANNEL.DENSITY * (D * H * W) + flatIdx];

                    // Support detection and color based on constraint type
                    let supportColor: string | null = null;
                    let maskX = 0, maskY = 0, maskZ = 0;

                    if (is7Channel) {
                        maskX = tensor.data[CHANNEL.MASK_X * (D * H * W) + flatIdx];
                        maskY = tensor.data[CHANNEL.MASK_Y * (D * H * W) + flatIdx];
                        maskZ = tensor.data[CHANNEL.MASK_Z * (D * H * W) + flatIdx];
                        supportColor = getSupportColor(maskX, maskY, maskZ);
                    } else {
                        // Legacy 5-channel: Channel 1 is support (treat as full clamp)
                        const isSupport = tensor.data[1 * (D * H * W) + flatIdx] > 0.5;
                        if (isSupport) supportColor = SUPPORT_COLORS.FULL_CLAMP;
                    }

                    const isSupport = supportColor !== null;

                    // Load: Force magnitude > 0 (channels 4, 5, 6 for 7-channel, channel 3 for 5-channel)
                    let hasLoad = false;
                    if (is7Channel) {
                        const fx = tensor.data[CHANNEL.FORCE_X * (D * H * W) + flatIdx];
                        const fy = tensor.data[CHANNEL.FORCE_Y * (D * H * W) + flatIdx];
                        const fz = tensor.data[CHANNEL.FORCE_Z * (D * H * W) + flatIdx];
                        hasLoad = Math.abs(fx) > 0.01 || Math.abs(fy) > 0.01 || Math.abs(fz) > 0.01;
                    } else {
                        // Legacy: Channel 3 is Fy
                        hasLoad = Math.abs(tensor.data[3 * (D * H * W) + flatIdx]) > 0.01;
                    }

                    const isVisibleStandard = density > 0.1 || isSupport || hasLoad;

                    // Get policy data for this voxel
                    const addVal = (showHeatmap && heatmap?.add) ? heatmap.add[flatIdx] : 0;
                    const removeVal = (showHeatmap && heatmap?.remove) ? heatmap.remove[flatIdx] : 0;

                    // 1. Always render standard voxels (density, support, load)
                    if (isVisibleStandard) {
                        dummy.position.set(d - D / 2, h + 0.5, w - W / 2);
                        dummy.updateMatrix();
                        opaqueRef.current.setMatrixAt(opaqueCount, dummy.matrix);

                        if (supportColor) {
                            opaqueRef.current.setColorAt(opaqueCount, new THREE.Color(supportColor));
                        } else if (hasLoad) {
                            opaqueRef.current.setColorAt(opaqueCount, new THREE.Color('#FF0055'));
                        } else {
                            const val = Math.max(0.2, density);
                            const color = new THREE.Color().setHSL(0, 0, val * 0.95);
                            opaqueRef.current.setColorAt(opaqueCount, color);
                        }
                        opaqueCount++;
                    }

                    // 2. REMOVE channel: overlay on existing mass (voxel "glows" red)
                    if (showHeatmap && removeVal > 0.01 && isVisibleStandard) {
                        dummy.position.set(d - D / 2, h + 0.5, w - W / 2);
                        dummy.updateMatrix();
                        policyRef.current.setMatrixAt(policyCount, dummy.matrix);
                        policyRef.current.setColorAt(policyCount, new THREE.Color('#FF0055'));
                        if (opacityAttrRef.current) {
                            opacityAttrRef.current.setX(policyCount, removeVal);
                        }
                        policyCount++;
                    }

                    // 3. ADD channel: only where there's NO mass (new voxel suggestion)
                    if (showHeatmap && addVal > 0.01 && !isVisibleStandard) {
                        dummy.position.set(d - D / 2, h + 0.5, w - W / 2);
                        dummy.updateMatrix();
                        policyRef.current.setMatrixAt(policyCount, dummy.matrix);
                        policyRef.current.setColorAt(policyCount, new THREE.Color('#00FF9D'));
                        if (opacityAttrRef.current) {
                            opacityAttrRef.current.setX(policyCount, addVal);
                        }
                        policyCount++;
                    }
                }
            }
        }

        // Update opaque mesh
        opaqueRef.current.count = opaqueCount;
        opaqueRef.current.instanceMatrix.needsUpdate = true;
        if (opaqueRef.current.instanceColor) opaqueRef.current.instanceColor.needsUpdate = true;

        // Update policy mesh
        policyRef.current.count = policyCount;
        policyRef.current.instanceMatrix.needsUpdate = true;
        if (policyRef.current.instanceColor) policyRef.current.instanceColor.needsUpdate = true;
        if (opacityAttrRef.current) opacityAttrRef.current.needsUpdate = true;
    }, [tensor, heatmap, showHeatmap]);

    return (
        <group>
            {/* Opaque mesh for standard voxels - unchanged from original */}
            <instancedMesh ref={opaqueRef} args={[undefined, undefined, 20000]}>
                <boxGeometry args={[0.9, 0.9, 0.9]} />
                <meshStandardMaterial roughness={0.2} metalness={0.8} />
            </instancedMesh>

            {/* Transparent mesh for policy voxels only */}
            <instancedMesh ref={policyRef} args={[undefined, undefined, 10000]}>
                <boxGeometry args={[0.92, 0.92, 0.92]} />
                <meshStandardMaterial
                    roughness={0.3}
                    metalness={0.5}
                    transparent={true}
                    depthWrite={false}
                />
            </instancedMesh>
        </group>
    );
};

// --- UI Components ---

const ReplayHeader = ({
    episodeId,
    onBack,
}: {
    episodeId: string | null;
    onBack: () => void;
}) => (
    <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute left-8 top-8 z-10"
    >
        <div className="flex items-center gap-4 bg-matter/90 backdrop-blur-xl border border-white/10 rounded-xl p-4">
            <button
                onClick={onBack}
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-colors"
            >
                <ArrowLeft size={20} />
            </button>
            <div>
                <div className="text-xs text-white/40 uppercase tracking-wider mb-1">
                    Replay de Episódio
                </div>
                <div className="font-mono text-cyan text-sm">
                    {episodeId ? `${episodeId.slice(0, 20)}...` : 'Carregando...'}
                </div>
            </div>
        </div>
    </motion.div>
);

const ReplayControlPanel = ({
    state,
    simState,
    onPlayPause,
    onStepBack,
    onStepForward,
    onSeek,
}: {
    state: ReplayState | null;
    simState: ReturnType<typeof trainingDataReplayService.getState>;
    onPlayPause: () => void;
    onStepBack: () => void;
    onStepForward: () => void;
    onSeek: (step: number) => void;
}) => {
    const totalSteps = simState.stepsLoaded;
    const isPlaying = simState.isPlaying;
    const maxCompliance = simState.maxCompliance;

    return (
        <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            className="absolute right-0 top-0 bottom-0 w-80 bg-matter/90 backdrop-blur-xl border-l border-white/10 z-10 flex flex-col"
        >
            <div className="p-6 overflow-y-auto flex-1">
                <h3 className="text-sm font-mono text-white/40 uppercase mb-6 tracking-wider">
                    Controle do Replay
                </h3>

                {/* Phase Indicator */}
                <div className="mb-6">
                    <span className="text-xs text-white/40 uppercase mb-2 block">Fase Atual</span>
                    <span
                        className={`
                            inline-flex items-center gap-2 text-sm font-bold px-3 py-2 rounded-lg
                            ${state?.phase === 'GROWTH'
                                ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                                : 'bg-purple/20 text-purple border border-purple/30'
                            }
                        `}
                    >
                        {state?.phase === 'GROWTH' ? (
                            <Network size={16} />
                        ) : (
                            <Sparkles size={16} />
                        )}
                        {state?.phase === 'GROWTH' ? 'Conexão' : 'Refinamento'}
                    </span>
                </div>

                {/* Timeline */}
                <div className="mb-6 p-4 rounded-xl bg-black/30 border border-white/5">
                    <div className="flex justify-between items-center mb-3">
                        <span className="text-xs text-white/40">Progresso</span>
                        <span className="text-sm font-mono text-cyan font-bold">
                            {state?.step || 0} / {totalSteps}
                        </span>
                    </div>

                    {/* Custom Slider */}
                    <div className="relative h-2 bg-white/10 rounded-full overflow-hidden mb-3">
                        <motion.div
                            className="absolute h-full bg-gradient-to-r from-cyan to-purple rounded-full"
                            animate={{
                                width: `${totalSteps > 0 ? (simState.currentStep / (totalSteps - 1)) * 100 : 0}%`,
                            }}
                        />
                    </div>

                    <input
                        type="range"
                        min="0"
                        max={Math.max(0, totalSteps - 1)}
                        value={simState.currentStep}
                        onChange={(e) => onSeek(parseInt(e.target.value))}
                        className="w-full h-2 opacity-0 cursor-pointer absolute"
                        style={{ marginTop: '-1.5rem' }}
                    />

                    <div className="flex justify-between text-[10px] text-white/30 font-mono">
                        <span>Início</span>
                        <span>Fim</span>
                    </div>
                </div>

                {/* Metrics */}
                <div className="space-y-4 mb-8">
                    <div>
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-white/40">Compliance</span>
                            <span className="font-mono text-white">
                                {(() => {
                                    // Direct compliance if available
                                    if (state?.metadata.compliance) {
                                        return state.metadata.compliance.toFixed(2);
                                    }
                                    // Recover from fitness + volume
                                    if (state?.fitness_score !== undefined && state?.metadata.volume_fraction !== undefined) {
                                        const recovered = recoverCompliance(state.fitness_score, state.metadata.volume_fraction);
                                        return `~${recovered.toFixed(2)}`;
                                    }
                                    return 'N/A';
                                })()}
                            </span>
                        </div>
                        <div className="w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                            {(() => {
                                const compliance = state?.metadata.compliance
                                    || (state?.fitness_score !== undefined && state?.metadata.volume_fraction !== undefined
                                        ? recoverCompliance(state.fitness_score, state.metadata.volume_fraction)
                                        : null);

                                if (compliance) {
                                    return (
                                        <motion.div
                                            className="h-full bg-gradient-to-r from-magenta to-purple"
                                            animate={{
                                                width: `${Math.min(100, (compliance / maxCompliance) * 100)}%`,
                                            }}
                                        />
                                    );
                                }
                                return <div className="h-full bg-magenta/50 w-full animate-pulse" />;
                            })()}
                        </div>
                    </div>

                    <div>
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-white/40">Fração de Volume</span>
                            <span className="font-mono text-white">
                                {((state?.metadata.volume_fraction || 0) * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="w-full bg-white/5 h-1.5 rounded-full">
                            <motion.div
                                className="h-full bg-white/40 rounded-full"
                                animate={{
                                    width: `${(state?.metadata.volume_fraction || 0) * 100}%`,
                                }}
                            />
                        </div>
                    </div>

                    <div>
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-white/40">Fitness Score</span>
                            <span className="font-mono text-cyan">
                                {(state?.fitness_score || 0).toFixed(4)}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Playback Controls */}
                <div className="flex gap-2">
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={onStepBack}
                        disabled={isPlaying}
                        className={`
                            p-3 rounded-lg transition-colors
                            ${isPlaying
                                ? 'bg-white/5 text-white/20 cursor-not-allowed'
                                : 'bg-white/10 text-white hover:bg-white/20'
                            }
                        `}
                    >
                        <SkipBack size={18} />
                    </motion.button>

                    <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={onPlayPause}
                        className={`
                            flex-1 font-bold py-3 rounded-lg flex items-center justify-center gap-2 transition-all
                            ${!isPlaying
                                ? 'bg-gradient-to-r from-cyan to-cyan/80 text-black hover:shadow-[0_0_30px_rgba(0,240,255,0.3)]'
                                : 'bg-white/10 text-white hover:bg-white/20'
                            }
                        `}
                    >
                        {isPlaying ? (
                            <>
                                <Loader size={18} className="animate-spin" /> Reproduzindo...
                            </>
                        ) : (
                            <>
                                <Play size={18} fill="currentColor" /> Reproduzir
                            </>
                        )}
                    </motion.button>

                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={onPlayPause}
                        className={`
                            p-3 rounded-lg transition-colors
                            ${isPlaying
                                ? 'bg-magenta text-white hover:bg-magenta/80'
                                : 'bg-white/5 text-white/40 hover:bg-white/10'
                            }
                        `}
                    >
                        <Pause size={18} />
                    </motion.button>

                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={onStepForward}
                        disabled={isPlaying}
                        className={`
                            p-3 rounded-lg transition-colors
                            ${isPlaying
                                ? 'bg-white/5 text-white/20 cursor-not-allowed'
                                : 'bg-white/10 text-white hover:bg-white/20'
                            }
                        `}
                    >
                        <SkipForward size={18} />
                    </motion.button>
                </div>
            </div>
        </motion.div>
    );
};

const NeuralHUD = ({
    state,
    history,
    showHeatmap,
    setShowHeatmap,
}: {
    state: ReplayState | null;
    history: number[];
    showHeatmap: boolean;
    setShowHeatmap: (v: boolean) => void;
}) => {
    const [isOpen, setIsOpen] = useState(true);

    // Calculate scale that handles negative values
    const getScale = (values: number[]) => {
        if (values.length === 0) return { min: -1, max: 1, range: 2 };
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min;
        // Add padding to avoid edge cases
        const padding = range * 0.1 || 0.2;
        return {
            min: min - padding,
            max: max + padding,
            range: range + padding * 2,
        };
    };

    // Get color for value (positive = cyan, negative = magenta)
    const getValueColor = (value: number) => {
        if (value >= 0) {
            return 'from-cyan/30 to-cyan/80';
        } else {
            return 'from-magenta/30 to-magenta/80';
        }
    };

    return (
        <div className="absolute bottom-8 left-8 flex flex-col gap-2 pointer-events-none">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="bg-black/60 backdrop-blur border border-white/10 p-2 rounded-lg text-cyan hover:text-white self-start pointer-events-auto transition-colors"
            >
                <Brain size={20} />
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 20 }}
                        className="flex gap-4 items-end"
                    >
                        {/* Quality Estimate Graph */}
                        <div className="bg-black/60 backdrop-blur border border-white/10 p-4 rounded-xl w-64 pointer-events-auto">
                            <div className="flex items-center gap-2 mb-3 text-cyan">
                                <Brain size={16} />
                                <span className="text-xs font-bold uppercase tracking-wider">
                                    Estimativa de Qualidade
                                </span>
                            </div>
                            <div className="h-20 flex items-end gap-0.5 border-b border-white/10 pb-1 relative">
                                {(() => {
                                    const windowSize = 20;
                                    // Get last windowSize values (most recent at the end)
                                    const visibleHistory = history.length > 0
                                        ? history.slice(-windowSize)
                                        : [];

                                    if (visibleHistory.length === 0) {
                                        return Array.from({ length: windowSize }).map((_, i) => (
                                            <div key={i} className="flex-1" />
                                        ));
                                    }

                                    const scale = getScale(visibleHistory);
                                    const zeroLine = scale.min < 0 && scale.max > 0
                                        ? (-scale.min / scale.range) * 100
                                        : null;

                                    return (
                                        <>
                                            {/* Zero line indicator */}
                                            {zeroLine !== null && (
                                                <div
                                                    className="absolute left-0 right-0 h-px bg-white/20 pointer-events-none"
                                                    style={{ bottom: `${zeroLine}%` }}
                                                />
                                            )}
                                            {visibleHistory.map((value, i) => {
                                                // Normalize value to 0-100% based on scale
                                                const normalized = ((value - scale.min) / scale.range) * 100;
                                                const height = Math.max(2, Math.min(100, normalized)); // Min 2px for visibility
                                                const isPositive = value >= 0;

                                                return (
                                                    <div
                                                        key={i}
                                                        className={`flex-1 bg-gradient-to-t ${getValueColor(value)} rounded-t-sm transition-all duration-300`}
                                                        style={{ height: `${height}%` }}
                                                        title={`Step ${history.length - visibleHistory.length + i + 1}: ${value.toFixed(4)}`}
                                                    />
                                                );
                                            })}
                                            {/* Pad with empty slots if needed */}
                                            {Array.from({ length: windowSize - visibleHistory.length }).map((_, i) => (
                                                <div key={`pad-${i}`} className="flex-1" />
                                            ))}
                                        </>
                                    );
                                })()}
                            </div>
                            <div className="mt-2 flex justify-between font-mono text-xs">
                                {history.length > 0 ? (
                                    <>
                                        <span className="text-white/30">
                                            {(() => {
                                                const scale = getScale(history.slice(-20));
                                                return `${scale.min.toFixed(2)} - ${scale.max.toFixed(2)}`;
                                            })()}
                                        </span>
                                        <span className={state?.value_confidence && state.value_confidence >= 0 ? 'text-cyan' : 'text-magenta'}>
                                            {(state?.value_confidence || 0).toFixed(4)}
                                        </span>
                                    </>
                                ) : (
                                    <span className="text-white/30">Carregando...</span>
                                )}
                            </div>
                        </div>

                        {/* Policy Heatmap Toggle */}
                        <div className="bg-black/60 backdrop-blur border border-white/10 p-2 rounded-xl pointer-events-auto">
                            <button
                                onClick={() => setShowHeatmap(!showHeatmap)}
                                className={`
                                    flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all
                                    ${showHeatmap
                                        ? 'bg-cyan/20 text-cyan border border-cyan/50'
                                        : 'bg-white/10 text-white/60 hover:text-white hover:bg-white/20'
                                    }
                                `}
                            >
                                <Layers size={14} />
                                {showHeatmap ? 'Ocultar Policy' : 'Mostrar Policy'}
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

const LoadingOverlay = () => (
    <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-void/90 backdrop-blur flex flex-col items-center justify-center z-50"
    >
        <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            className="w-16 h-16 border-2 border-cyan/30 border-t-cyan rounded-full mb-6"
        />
        <p className="text-white/60 text-lg">Carregando episódio...</p>
    </motion.div>
);

const ErrorOverlay = ({ message, onBack }: { message: string; onBack: () => void }) => (
    <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="absolute inset-0 bg-void/90 backdrop-blur flex flex-col items-center justify-center z-50"
    >
        <div className="p-6 rounded-full bg-magenta/10 text-magenta mb-6">
            <AlertCircle size={48} />
        </div>
        <p className="text-white/60 text-lg mb-2">Erro ao carregar episódio</p>
        <p className="text-magenta/60 text-sm mb-6">{message}</p>
        <button
            onClick={onBack}
            className="px-6 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
        >
            Voltar
        </button>
    </motion.div>
);

// --- Main Component ---

export const EpisodeReplay = () => {
    const { dbId, episodeId } = useParams<{ dbId: string; episodeId: string }>();
    const navigate = useNavigate();

    const [replayState, setReplayState] = useState<ReplayState | null>(null);
    const [simState, setSimState] = useState(trainingDataReplayService.getState());
    const [history, setHistory] = useState<number[]>([]);
    const [showHeatmap, setShowHeatmap] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Load episode on mount
    useEffect(() => {
        if (dbId && episodeId) {
            loadEpisode();
        }

        return () => {
            trainingDataReplayService.stop();
        };
    }, [dbId, episodeId]);

    // Update history when step changes - ensure order by step index
    useEffect(() => {
        const allFramesData = trainingDataReplayService.getAllFrames();
        if (allFramesData.length > 0 && simState.currentStep >= 0 && simState.currentStep < allFramesData.length) {
            // Get current frame's step number
            const currentFrame = allFramesData[simState.currentStep];
            const currentStepNumber = currentFrame.step;

            // Get all frames up to current step, sorted by step number (ascending)
            // This ensures we show frames in order: oldest (left) to current (right)
            const framesUpToCurrent = allFramesData
                .filter(f => f.step <= currentStepNumber)
                .sort((a, b) => a.step - b.step);

            const historyValues = framesUpToCurrent.map(f => f.value_confidence || 0);
            setHistory(historyValues);
        }
    }, [simState.currentStep, simState.stepsLoaded, replayState]);

    const loadEpisode = async () => {
        if (!dbId || !episodeId) return;

        setLoading(true);
        setError(null);
        setHistory([]);

        try {
            const unsubscribe = trainingDataReplayService.subscribe((state) => {
                setReplayState(state);
                setSimState(trainingDataReplayService.getState());
            });

            await trainingDataReplayService.loadEpisode(dbId, episodeId);
            const serviceState = trainingDataReplayService.getState();
            setSimState(serviceState);

            // Build initial history from all loaded frames
            const allFramesData = trainingDataReplayService.getAllFrames();
            if (allFramesData.length > 0) {
                const currentFrame = allFramesData[serviceState.currentStep];
                const framesUpToCurrent = allFramesData
                    .filter(f => f.step <= currentFrame.step)
                    .sort((a, b) => a.step - b.step);
                const historyValues = framesUpToCurrent.map(f => f.value_confidence || 0);
                setHistory(historyValues);
            }

            setLoading(false);

            return unsubscribe;
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Erro desconhecido');
            setLoading(false);
        }
    };

    const handleBack = () => {
        trainingDataReplayService.stop();
        navigate(`/data/${dbId}`);
    };

    const handlePlayPause = () => {
        trainingDataReplayService.togglePlay();
        setSimState(trainingDataReplayService.getState());
    };

    const handleStepBack = () => {
        trainingDataReplayService.stepBackward();
        setSimState(trainingDataReplayService.getState());
    };

    const handleStepForward = () => {
        trainingDataReplayService.stepForward();
        setSimState(trainingDataReplayService.getState());
    };

    const handleSeek = (step: number) => {
        trainingDataReplayService.pause();
        trainingDataReplayService.seekToStep(step);
        setSimState(trainingDataReplayService.getState());
    };

    return (
        <div className="h-[calc(100vh-64px)] relative bg-void overflow-hidden">
            {/* 3D Canvas */}
            <div className="absolute inset-0">
                <Canvas camera={{ position: [50, 50, 50], fov: 45 }}>
                    <color attach="background" args={['#050505']} />
                    <fog attach="fog" args={['#050505', 50, 150]} />

                    <ambientLight intensity={1.2} />
                    <pointLight position={[10, 10, 10]} intensity={1.5} />
                    <directionalLight position={[-10, 20, -10]} intensity={0.8} />

                    <gridHelper args={[100, 100, '#1a1a1a', '#111111']} />

                    <VoxelGrid
                        tensor={replayState?.tensor || null}
                        heatmap={replayState?.policy_heatmap}
                        showHeatmap={showHeatmap}
                    />
                    <LoadVector tensor={replayState?.tensor || null} />

                    <OrbitControls makeDefault target={[0, 16, 0]} />
                    <GizmoHelper alignment="top-right" margin={[80, 80]}>
                        <GizmoViewport axisColors={['#FF0055', '#00FF9D', '#00F0FF']} labelColor="white" />
                    </GizmoHelper>
                </Canvas>
            </div>

            {/* UI Overlays - NO TOOLBAR as per requirement */}
            <ReplayHeader episodeId={episodeId || null} onBack={handleBack} />
            <ReplayControlPanel
                state={replayState}
                simState={simState}
                onPlayPause={handlePlayPause}
                onStepBack={handleStepBack}
                onStepForward={handleStepForward}
                onSeek={handleSeek}
            />
            <NeuralHUD
                state={replayState}
                history={history}
                showHeatmap={showHeatmap}
                setShowHeatmap={setShowHeatmap}
            />

            {/* Loading/Error States */}
            <AnimatePresence>
                {loading && <LoadingOverlay />}
                {error && <ErrorOverlay message={error} onBack={handleBack} />}
            </AnimatePresence>
        </div>
    );
};


