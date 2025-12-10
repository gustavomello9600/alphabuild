import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
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
    Box,
    GitBranch,
    Target,
    BarChart3,
    ChevronDown,
    ChevronUp,
    Network,
    Sparkles,
} from 'lucide-react';
import * as THREE from 'three';

import {
    selfPlayReplayService,
    type GameReplayState,
} from '../api/selfPlayService';
import { RewardBreakdown } from '../components/RewardBreakdown';

// =============================================================================
// View Modes
// =============================================================================

type ViewMode = 'structure' | 'policy' | 'mcts' | 'combined' | 'decision';

const VIEW_MODE_CONFIG: Record<ViewMode, { icon: typeof Box; label: string; shortcut: string }> = {
    structure: { icon: Box, label: 'Estrutura', shortcut: '1' },
    policy: { icon: Brain, label: 'Policy', shortcut: '2' },
    mcts: { icon: GitBranch, label: 'MCTS', shortcut: '3' },
    combined: { icon: Layers, label: 'Policy + MCTS', shortcut: '4' },
    decision: { icon: Target, label: 'Decisão', shortcut: '5' },
};

// =============================================================================
// 3D Components
// =============================================================================

// Channel indices for 7-channel tensor
const CHANNEL = {
    DENSITY: 0,
    MASK_X: 1,
    MASK_Y: 2,
    MASK_Z: 3,
    FORCE_X: 4,
    FORCE_Y: 5,
    FORCE_Z: 6,
};

// MCTS Wireframe component for a single voxel
const MCTSWireframe = ({
    position,
    intensity,
    isAdd
}: {
    position: THREE.Vector3;
    intensity: number;
    isAdd: boolean;
}) => {
    // Color interpolation: gray at 0, green/red at 1
    const baseColor = isAdd ? new THREE.Color('#00FF9D') : new THREE.Color('#FF0055');
    const grayColor = new THREE.Color('#666666');
    const color = grayColor.clone().lerp(baseColor, intensity);

    return (
        <lineSegments position={position}>
            <edgesGeometry args={[new THREE.BoxGeometry(0.95, 0.95, 0.95)]} />
            <lineBasicMaterial color={color} linewidth={2} />
        </lineSegments>
    );
};

const VoxelGridMCTS = ({
    step,
    viewMode,
}: {
    step: GameReplayState | null;
    viewMode: ViewMode;
}) => {
    const opaqueRef = useRef<THREE.InstancedMesh>(null);
    const overlayRef = useRef<THREE.InstancedMesh>(null);
    const opacityAttrRef = useRef<THREE.InstancedBufferAttribute | null>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);

    // Use state for wireframes to ensure render sync
    const [wireframes, setWireframes] = useState<Array<{ pos: THREE.Vector3, intensity: number, isAdd: boolean }>>([]);

    // Shader for per-instance opacity
    useEffect(() => {
        if (!overlayRef.current) return;
        const material = overlayRef.current.material as THREE.MeshStandardMaterial;
        material.onBeforeCompile = (shader) => {
            shader.vertexShader = `
                attribute float instanceOpacity;
                varying float vInstanceOpacity;
                ${shader.vertexShader}
            `.replace(
                '#include <begin_vertex>',
                `#include <begin_vertex>
                vInstanceOpacity = instanceOpacity;`
            );
            shader.fragmentShader = `
                varying float vInstanceOpacity;
                ${shader.fragmentShader}
            `.replace(
                '#include <dithering_fragment>',
                `#include <dithering_fragment>
                gl_FragColor.a *= vInstanceOpacity;`
            );
        };
        material.needsUpdate = true;
    }, []);

    useEffect(() => {
        if (!opaqueRef.current || !overlayRef.current || !step) return;

        const [C, D, H, W] = step.tensor.shape;
        const spatialSize = D * H * W;
        const tensorData = step.tensor.data;

        let opaqueCount = 0;
        let overlayCount = 0;
        const newWireframes: Array<{ pos: THREE.Vector3, intensity: number, isAdd: boolean }> = [];

        // Initialize opacity buffer
        const maxInstances = 20000;
        if (!opacityAttrRef.current) {
            const arr = new Float32Array(maxInstances).fill(1.0);
            opacityAttrRef.current = new THREE.InstancedBufferAttribute(arr, 1);
            opacityAttrRef.current.setUsage(THREE.DynamicDrawUsage);
            overlayRef.current.geometry.setAttribute('instanceOpacity', opacityAttrRef.current);
        }

        // Normalize MCTS visits for visualization
        const maxVisitAdd = Math.max(...Array.from(step.mcts.visit_add), 1);
        const maxVisitRem = Math.max(...Array.from(step.mcts.visit_remove), 1);

        // Pre-process selected actions for fast lookup
        // Map "x,y,z" -> Action
        const selectedActionMap = new Map<string, typeof step.selected_actions[0]>();
        if (viewMode === 'decision') {
            step.selected_actions.forEach(a => {
                selectedActionMap.set(`${a.x},${a.y},${a.z}`, a);
            });
        }

        for (let d = 0; d < D; d++) {
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    const idx = d * (H * W) + h * W + w;
                    // Density is channel 0
                    const density = tensorData[idx];
                    const pos = new THREE.Vector3(d - D / 2, h + 0.5, w - W / 2);

                    // Check for load (force channels)
                    let hasLoad = false;
                    if (C >= 7) {
                        const fx = tensorData[CHANNEL.FORCE_X * spatialSize + idx];
                        const fy = tensorData[CHANNEL.FORCE_Y * spatialSize + idx];
                        const fz = tensorData[CHANNEL.FORCE_Z * spatialSize + idx];
                        hasLoad = Math.abs(fx) > 0.01 || Math.abs(fy) > 0.01 || Math.abs(fz) > 0.01;
                    }

                    // Base structure (always render solid voxels + load voxels)
                    const isVisible = density > 0.5 || hasLoad;
                    if (isVisible) {
                        dummy.position.copy(pos);
                        dummy.updateMatrix();
                        opaqueRef.current.setMatrixAt(opaqueCount, dummy.matrix);

                        if (hasLoad) {
                            // Load voxels in magenta/orange
                            opaqueRef.current.setColorAt(opaqueCount, new THREE.Color('#FF6B00'));
                        } else {
                            // Gray color for structure
                            const gray = Math.max(0.2, density);
                            opaqueRef.current.setColorAt(opaqueCount, new THREE.Color().setHSL(0, 0, gray * 0.95));
                        }
                        opaqueCount++;
                    }

                    // Process based on view mode
                    if (viewMode === 'structure') continue;

                    const policyAdd = step.policy.add[idx];
                    const policyRem = step.policy.remove[idx];
                    const mctsAdd = step.mcts.visit_add[idx] / maxVisitAdd;
                    const mctsRem = step.mcts.visit_remove[idx] / maxVisitRem;

                    // MCTS modes use wireframe
                    if (viewMode === 'mcts' || viewMode === 'combined') {
                        // MCTS Add wireframe (green) - only on empty spaces
                        if (mctsAdd > 0.01 && density <= 0.5) {
                            newWireframes.push({ pos: pos.clone(), intensity: mctsAdd, isAdd: true });
                        }
                        // MCTS Remove wireframe (red) - only on filled spaces
                        if (mctsRem > 0.01 && density > 0.5) {
                            newWireframes.push({ pos: pos.clone(), intensity: mctsRem, isAdd: false });
                        }
                    }

                    // Policy overlay (filled cubes with transparency)
                    if (viewMode === 'policy' || viewMode === 'combined') {
                        if (policyAdd > 0.01 && density <= 0.5) {
                            dummy.position.copy(pos);
                            dummy.updateMatrix();
                            overlayRef.current!.setMatrixAt(overlayCount, dummy.matrix);
                            overlayRef.current!.setColorAt(overlayCount, new THREE.Color('#00FF9D'));
                            if (opacityAttrRef.current) {
                                opacityAttrRef.current.setX(overlayCount, Math.min(1, policyAdd * 2));
                            }
                            overlayCount++;
                        } else if (policyRem > 0.01 && density > 0.5) {
                            dummy.position.copy(pos);
                            dummy.updateMatrix();
                            overlayRef.current!.setMatrixAt(overlayCount, dummy.matrix);
                            overlayRef.current!.setColorAt(overlayCount, new THREE.Color('#FF0055'));
                            if (opacityAttrRef.current) {
                                opacityAttrRef.current.setX(overlayCount, Math.min(1, policyRem * 2));
                            }
                            overlayCount++;
                        }
                    }

                    // Decision mode (selected actions)
                    if (viewMode === 'decision') {
                        const key = `${d},${h},${w}`;
                        if (selectedActionMap.has(key)) {
                            const action = selectedActionMap.get(key)!;
                            dummy.position.copy(pos);
                            dummy.updateMatrix();
                            overlayRef.current!.setMatrixAt(overlayCount, dummy.matrix);
                            overlayRef.current!.setColorAt(
                                overlayCount,
                                new THREE.Color(action.channel === 0 ? '#00FF9D' : '#FF0055')
                            );
                            if (opacityAttrRef.current) {
                                opacityAttrRef.current.setX(overlayCount, 0.9);
                            }
                            overlayCount++;
                        }
                    }
                }
            }
        }

        // Update meshes
        opaqueRef.current.count = opaqueCount;
        opaqueRef.current.instanceMatrix.needsUpdate = true;
        if (opaqueRef.current.instanceColor) opaqueRef.current.instanceColor.needsUpdate = true;

        overlayRef.current.count = overlayCount;
        overlayRef.current.instanceMatrix.needsUpdate = true;
        if (overlayRef.current.instanceColor) overlayRef.current.instanceColor.needsUpdate = true;
        if (opacityAttrRef.current) opacityAttrRef.current.needsUpdate = true;

        // Update wireframes state
        setWireframes(newWireframes);
    }, [step, viewMode]);

    return (
        <group>
            <instancedMesh ref={opaqueRef} args={[undefined, undefined, 20000]}>
                <boxGeometry args={[0.9, 0.9, 0.9]} />
                <meshStandardMaterial roughness={0.2} metalness={0.8} />
            </instancedMesh>
            <instancedMesh ref={overlayRef} args={[undefined, undefined, 10000]}>
                <boxGeometry args={[0.92, 0.92, 0.92]} />
                <meshStandardMaterial
                    roughness={0.3}
                    metalness={0.5}
                    transparent={true}
                    depthWrite={false}
                />
            </instancedMesh>
            {/* MCTS Wireframes */}
            {wireframes.map((wf: { pos: THREE.Vector3, intensity: number, isAdd: boolean }, i: number) => (
                <MCTSWireframe key={i} position={wf.pos} intensity={wf.intensity} isAdd={wf.isAdd} />
            ))}
        </group>
    );
};

// Support colors helper
const SUPPORT_COLORS = {
    FULL_CLAMP: '#00F0FF',  // Cyan - Fully fixed (XYZ)
    RAIL_XY: '#7000FF',     // Purple - Rail constraint (XY fixed, Z free)
    ROLLER_Y: '#00FF9D',    // Green - Roller (only Y fixed)
    PARTIAL: '#3B82F6',     // Blue - Other partial constraints
};

function getSupportColor(maskX: number, maskY: number, maskZ: number): string | null {
    const hasX = maskX > 0.5;
    const hasY = maskY > 0.5;
    const hasZ = maskZ > 0.5;

    if (!hasX && !hasY && !hasZ) return null;

    if (hasX && hasY && hasZ) {
        return SUPPORT_COLORS.FULL_CLAMP;
    } else if (hasX && hasY && !hasZ) {
        return SUPPORT_COLORS.RAIL_XY;
    } else if (!hasX && hasY && !hasZ) {
        return SUPPORT_COLORS.ROLLER_Y;
    } else {
        return SUPPORT_COLORS.PARTIAL;
    }
}

const LoadVector = ({ step }: { step: GameReplayState | null }) => {
    if (!step) return null;

    const [C, D, H, W] = step.tensor.shape;
    if (C < 7) return null; // Need 7 channels

    const loadPoints: { pos: THREE.Vector3; dir: THREE.Vector3; magnitude: number }[] = [];
    const spatialSize = D * H * W;
    const tensorData = step.tensor.data;

    for (let d = 0; d < D; d++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                const flatIdx = d * (H * W) + h * W + w;

                const fx = tensorData[CHANNEL.FORCE_X * spatialSize + flatIdx];
                const fy = tensorData[CHANNEL.FORCE_Y * spatialSize + flatIdx];
                const fz = tensorData[CHANNEL.FORCE_Z * spatialSize + flatIdx];

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
                        <arrowHelper args={[lp.dir, origin, arrowLength, 0xff6b00, 1, 0.5]} />
                        <mesh position={lp.pos}>
                            <sphereGeometry args={[0.2, 8, 8]} />
                            <meshBasicMaterial color="#ff6b00" />
                        </mesh>
                    </group>
                );
            })}
        </group>
    );
};

const SupportVoxels = ({ step }: { step: GameReplayState | null }) => {
    const supportRef = useRef<THREE.InstancedMesh>(null);
    const dummy = new THREE.Object3D();

    useEffect(() => {
        if (!supportRef.current || !step) return;

        const [C, D, H, W] = step.tensor.shape;
        if (C < 7) return;

        const spatialSize = D * H * W;
        const tensorData = step.tensor.data;
        let count = 0;

        for (let d = 0; d < D; d++) {
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    const flatIdx = d * (H * W) + h * W + w;

                    const maskX = tensorData[CHANNEL.MASK_X * spatialSize + flatIdx];
                    const maskY = tensorData[CHANNEL.MASK_Y * spatialSize + flatIdx];
                    const maskZ = tensorData[CHANNEL.MASK_Z * spatialSize + flatIdx];

                    const color = getSupportColor(maskX, maskY, maskZ);
                    if (!color) continue;

                    const pos = new THREE.Vector3(d - D / 2, h + 0.5, w - W / 2);
                    dummy.position.copy(pos);
                    dummy.updateMatrix();
                    supportRef.current.setMatrixAt(count, dummy.matrix);
                    supportRef.current.setColorAt(count, new THREE.Color(color));
                    count++;
                }
            }
        }

        supportRef.current.count = count;
        supportRef.current.instanceMatrix.needsUpdate = true;
        if (supportRef.current.instanceColor) supportRef.current.instanceColor.needsUpdate = true;
    }, [step]);

    return (
        <instancedMesh ref={supportRef} args={[undefined, undefined, 1000]}>
            <boxGeometry args={[0.95, 0.95, 0.95]} />
            <meshStandardMaterial transparent opacity={0.6} roughness={0.4} metalness={0.3} />
        </instancedMesh>
    );
};

// =============================================================================
// UI Components
// =============================================================================

const ViewModeSwitcher = ({
    mode,
    setMode,
}: {
    mode: ViewMode;
    setMode: (m: ViewMode) => void;
}) => {
    // Keyboard shortcuts
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.key === '1') setMode('structure');
            if (e.key === '2') setMode('policy');
            if (e.key === '3') setMode('mcts');
            if (e.key === '4') setMode('combined');
            if (e.key === '5') setMode('decision');
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [setMode]);

    return (
        <div className="flex gap-1 bg-black/60 backdrop-blur border border-white/10 rounded-xl p-1">
            {(Object.entries(VIEW_MODE_CONFIG) as [ViewMode, typeof VIEW_MODE_CONFIG[ViewMode]][]).map(
                ([key, config]) => {
                    const Icon = config.icon;
                    const isActive = mode === key;
                    return (
                        <button
                            key={key}
                            onClick={() => setMode(key)}
                            className={`
                                p-2 rounded-lg transition-all relative group
                                ${isActive
                                    ? 'bg-cyan/20 text-cyan'
                                    : 'text-white/40 hover:text-white hover:bg-white/10'
                                }
                            `}
                            title={`${config.label} (${config.shortcut})`}
                        >
                            <Icon size={18} />
                            {/* Tooltip */}
                            <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 px-2 py-1 bg-matter border border-white/10 rounded text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none">
                                {config.label}
                                <span className="ml-1 text-cyan">{config.shortcut}</span>
                            </div>
                        </button>
                    );
                }
            )}
        </div>
    );
};

const MCTSStatsPanel = ({
    step,
    isOpen,
    toggle,
}: {
    step: GameReplayState | null;
    isOpen: boolean;
    toggle: () => void;
}) => {
    if (!step) return null;

    const { mcts_stats: stats, selected_actions } = step;

    return (
        <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="absolute left-8 top-32 z-10"
        >
            <button
                onClick={toggle}
                className="flex items-center gap-2 bg-black/60 backdrop-blur border border-white/10 rounded-lg px-3 py-2 text-sm text-white/60 hover:text-white transition-colors mb-2"
            >
                <BarChart3 size={16} />
                <span>Estatísticas MCTS</span>
                {isOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="bg-black/80 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden"
                    >
                        {/* Search Summary */}
                        <div className="p-4 border-b border-white/5">
                            <h4 className="text-xs font-mono text-white/40 uppercase mb-3">
                                Busca MCTS
                            </h4>
                            <div className="grid grid-cols-2 gap-3 text-sm">
                                <div>
                                    <span className="text-white/40">Simulações</span>
                                    <p className="text-white font-mono">{stats.num_simulations}</p>
                                </div>
                                <div>
                                    <span className="text-white/40">Nós</span>
                                    <p className="text-white font-mono">{stats.nodes_expanded}</p>
                                </div>
                                <div>
                                    <span className="text-white/40">Prof. Máx</span>
                                    <p className="text-white font-mono">{stats.max_depth}</p>
                                </div>
                                <div>
                                    <span className="text-white/40">Cache Hit</span>
                                    <p className="text-cyan font-mono">
                                        {((stats.cache_hits / (stats.num_simulations || 1)) * 100).toFixed(0)}%
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Quality Metrics */}
                        <div className="p-4 border-b border-white/5">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs text-white/40">Top-8 Concentration</span>
                                <span className="text-sm font-mono text-cyan">
                                    {(stats.top8_concentration * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="w-full bg-white/10 h-1.5 rounded-full">
                                <motion.div
                                    className="h-full bg-cyan rounded-full"
                                    animate={{ width: `${stats.top8_concentration * 100}%` }}
                                />
                            </div>
                            <div className="mt-3 flex items-center gap-2">
                                <span className="text-xs text-white/40">Refutação:</span>
                                <span className={`text-xs font-bold ${stats.refutation ? 'text-green-400' : 'text-white/40'}`}>
                                    {stats.refutation ? '✓ Rede ≠ MCTS' : '○ Alinhado'}
                                </span>
                            </div>
                        </div>

                        {/* Top Actions Table */}
                        <div className="p-4">
                            <h4 className="text-xs font-mono text-white/40 uppercase mb-3">
                                Micro-Batch ({selected_actions.length})
                            </h4>
                            <div className="space-y-1 max-h-40 overflow-y-auto">
                                {selected_actions.slice(0, 8).map((action, i) => (
                                    <div
                                        key={i}
                                        className="flex items-center justify-between py-1 px-2 rounded bg-white/5 hover:bg-white/10 text-xs"
                                    >
                                        <div className="flex items-center gap-2">
                                            <span className="w-4 text-white/40">{i + 1}</span>
                                            <span className={action.channel === 0 ? 'text-green-400' : 'text-red-400'}>
                                                {action.channel === 0 ? 'ADD' : 'REM'}
                                            </span>
                                            <span className="text-white/60 font-mono">
                                                ({action.x},{action.y},{action.z})
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <span className="text-white/40">N={action.visits}</span>
                                            <span className={action.q_value >= 0 ? 'text-cyan' : 'text-magenta'}>
                                                Q={action.q_value.toFixed(2)}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

// // Log-Squash constants
// const LOG_SQUASH = {
//    ALPHA: 12.0,
//    MU: -6.65,
//    SIGMA: 2.0,
// };

// const recoverCompliance = (fitness: number, volumeFraction: number): number => {
//    return Math.exp(-(fitness * LOG_SQUASH.SIGMA + LOG_SQUASH.MU + LOG_SQUASH.ALPHA * volumeFraction));
// };

const SimulationControls = ({
    currentStep,
    currentStepIndex,
    totalSteps,
    isPlaying,
    onPlayPause,
    onStepBack,
    onStepForward,
    onSeek,
}: {
    currentStep: GameReplayState | null;
    currentStepIndex: number;
    totalSteps: number;
    isPlaying: boolean;
    onPlayPause: () => void;
    onStepBack: () => void;
    onStepForward: () => void;
    onSeek: (step: number) => void;
}) => {
    const [localStepValue, setLocalStepValue] = useState((currentStepIndex + 1).toString());
    const [isEditing, setIsEditing] = useState(false);

    // Sync local state with prop only if not editing
    useEffect(() => {
        if (!isEditing) {
            setLocalStepValue((currentStepIndex + 1).toString());
        }
    }, [currentStepIndex, isEditing]);

    const handleCommitStep = (val: string) => {
        setIsEditing(false);
        let step = parseInt(val, 10);
        if (isNaN(step)) {
            setLocalStepValue((currentStepIndex + 1).toString());
            return;
        }
        step = Math.max(1, Math.min(totalSteps, step));
        onSeek(step - 1);
        setLocalStepValue(step.toString());
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            handleCommitStep(localStepValue);
            e.currentTarget.blur();
        }
    };

    return (
        <div className="absolute top-8 right-8 z-30 flex flex-col items-end gap-2 pointer-events-none">
            {/* Phase Badge */}
            <div className={`
                flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider backdrop-blur-md border shadow-lg pointer-events-auto
                ${currentStep?.phase === 'GROWTH'
                    ? 'bg-green-500/10 text-green-400 border-green-500/20'
                    : 'bg-purple/10 text-purple border-purple/20'
                }
            `}>
                {currentStep?.phase === 'GROWTH' ? <Network size={14} /> : <Sparkles size={14} />}
                {currentStep?.phase === 'GROWTH' ? 'Conexão (Growth)' : 'Refinamento (Fem)'}
            </div>

            {/* Controls Card */}
            <div className="bg-black/80 backdrop-blur-xl border border-white/10 rounded-2xl p-4 shadow-2xl min-w-[320px] pointer-events-auto">
                {/* Timeline */}
                <div className="flex justify-between items-center mb-2 px-1">
                    <span className="text-[10px] text-white/40 font-mono uppercase">Timeline</span>
                    <div className="flex items-center gap-1 text-xs font-mono">
                        <input
                            type="text"
                            inputMode="numeric"
                            pattern="[0-9]*"
                            value={localStepValue}
                            onChange={(e) => setLocalStepValue(e.target.value)}
                            onFocus={() => setIsEditing(true)}
                            onBlur={() => handleCommitStep(localStepValue)}
                            onKeyDown={handleKeyDown}
                            className="w-12 bg-transparent text-cyan font-bold text-right border-b border-white/10 focus:border-cyan focus:outline-none transition-colors"
                        />
                        <span className="text-white/30">/</span>
                        <span className="text-white/30">{totalSteps}</span>
                    </div>
                </div>
                <input
                    type="range"
                    min="0"
                    max={Math.max(0, totalSteps - 1)}
                    value={currentStepIndex}
                    onChange={(e) => onSeek(parseInt(e.target.value))}
                    className="w-full h-1.5 bg-white/10 rounded-full cursor-pointer accent-cyan appearance-none mb-4"
                />

                {/* Buttons */}
                <div className="flex gap-2">
                    <button
                        onClick={onStepBack}
                        disabled={isPlaying}
                        className={`p-2 rounded-xl transition-colors ${isPlaying ? 'bg-white/5 text-white/20' : 'bg-white/10 text-white hover:bg-white/20'}`}
                    >
                        <SkipBack size={18} />
                    </button>
                    <button
                        onClick={onPlayPause}
                        className={`flex-1 font-bold py-2 rounded-xl flex items-center justify-center gap-2 transition-all text-sm
                            ${!isPlaying
                                ? 'bg-cyan/20 text-cyan hover:bg-cyan/30 border border-cyan/20'
                                : 'bg-magenta/20 text-magenta hover:bg-magenta/30 border border-magenta/20'
                            }`}
                    >
                        {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        {isPlaying ? 'PAUSE' : 'PLAY'}
                    </button>
                    <button
                        onClick={onStepForward}
                        disabled={isPlaying}
                        className={`p-2 rounded-xl transition-colors ${isPlaying ? 'bg-white/5 text-white/20' : 'bg-white/10 text-white hover:bg-white/20'}`}
                    >
                        <SkipForward size={18} />
                    </button>
                </div>
            </div>
        </div>
    );
};

const RewardOverlay = ({ step }: { step: GameReplayState | null }) => {
    const [expanded, setExpanded] = useState(false);

    // Determine main value to show
    const rc = step?.reward_components;
    let rewardValue = 0;

    if (step?.phase === 'GROWTH') {
        const bonus = rc?.connectivity_bonus || 0;
        const valueHead = step?.value || 0;
        const islandPenalty = rc?.island_penalty || 0;
        rewardValue = valueHead + bonus - islandPenalty;
    } else {
        rewardValue = rc?.total ?? step?.value ?? 0;
    }

    const isPositive = rewardValue >= 0;

    return (
        <div className="absolute bottom-8 right-8 z-30 flex flex-col items-end pointer-events-none">
            <AnimatePresence>
                {expanded && (
                    <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20, scale: 0.95 }}
                        className="mb-3 w-[400px] pointer-events-auto"
                    >
                        <RewardBreakdown state={step} />
                    </motion.div>
                )}
            </AnimatePresence>

            <motion.button
                layout
                onClick={() => setExpanded(!expanded)}
                className={`
                    flex items-center gap-3 pl-5 pr-3 py-3 rounded-2xl shadow-2xl backdrop-blur-xl border transition-all group pointer-events-auto
                    ${expanded
                        ? 'bg-black/90 border-white/20'
                        : 'bg-black/60 border-white/10 hover:bg-black/80'
                    }
                `}
            >
                <div className="flex flex-col items-end">
                    <div className="text-[10px] text-white/40 uppercase font-bold tracking-wider mb-0.5">
                        Reward
                    </div>
                    <div className={`font-mono text-2xl font-bold leading-none ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                        {rewardValue > 0 ? '+' : ''}{rewardValue.toFixed(4)}
                    </div>
                </div>

                <div className={`
                    p-2 rounded-xl transition-colors ml-2
                    ${expanded ? 'bg-white/10 text-white' : 'bg-white/5 text-white/40 group-hover:text-white group-hover:bg-white/10'}
                `}>
                    {expanded ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
                </div>
            </motion.button>
        </div>
    );
};

const NeuralHUD = ({
    history,
    currentStep,
}: {
    history: number[];
    currentStep: number;
}) => {
    const [isOpen, setIsOpen] = useState(true);

    const currentValue = (history && currentStep >= 0 && currentStep < history.length)
        ? history[currentStep]
        : 0;

    const getValueColor = (value: number) => {
        return value >= 0 ? 'bg-cyan' : 'bg-magenta';
    };

    return (
        <div className="absolute bottom-8 left-8 flex flex-col gap-2 pointer-events-none">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="bg-black/60 backdrop-blur border border-white/10 p-2 rounded-lg text-cyan hover:text-white self-start pointer-events-auto transition-colors"
                title="Mostrar/Ocultar HUD Neural"
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
                        <div className="bg-black/60 backdrop-blur border border-white/10 p-4 rounded-xl w-64 pointer-events-auto">
                            <div className="flex items-center justify-between mb-3">
                                <div className="flex items-center gap-2 text-cyan">
                                    <Brain size={16} />
                                    <span className="text-xs font-bold uppercase tracking-wider">
                                        Qualidade
                                    </span>
                                </div>
                                <span className={`font-mono font-bold text-sm ${currentValue >= 0 ? 'text-cyan' : 'text-magenta'}`}>
                                    {currentValue > 0 ? '+' : ''}{currentValue.toFixed(3)}
                                </span>
                            </div>

                            {/* Chart Container */}
                            <div className="h-24 relative flex items-center border-l border-white/10 ml-6 pl-1">
                                {/* Y-Axis Labels */}
                                <div className="absolute left-0 top-0 bottom-0 -ml-7 flex flex-col justify-between text-[10px] text-white/30 font-mono py-0">
                                    <span>+1.0</span>
                                    <span> 0.0</span>
                                    <span>-1.0</span>
                                </div>

                                {/* Zero Axis Line */}
                                <div className="absolute left-0 right-0 top-1/2 h-px bg-white/20 z-0" />

                                {/* Bars Container */}
                                <div className="flex-1 h-full flex items-center gap-0.5 relative z-10">
                                    {(() => {
                                        const windowSize = 25;
                                        // Display last 'windowSize' steps up to currentStep
                                        const displayData = Array(windowSize).fill(null).map((_, i) => {
                                            const idx = currentStep - (windowSize - 1) + i;
                                            return (idx >= 0 && idx < history.length) ? history[idx] : null;
                                        });

                                        return displayData.map((v, i) => {
                                            if (v === null) return <div key={i} className="flex-1" />;

                                            const clampedV = Math.max(-1, Math.min(1, v));
                                            const heightPct = Math.abs(clampedV) * 50;
                                            const displayHeight = Math.max(heightPct, 2);

                                            return (
                                                <div key={i} className="flex-1 h-full relative group">
                                                    <div
                                                        className={`absolute w-full ${getValueColor(v)} opacity-80 rounded-[1px] transition-all duration-300`}
                                                        style={{
                                                            height: `${displayHeight}%`,
                                                            bottom: v >= 0 ? '50%' : undefined,
                                                            top: v < 0 ? '50%' : undefined,
                                                        }}
                                                    />
                                                </div>
                                            );
                                        });
                                    })()}
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

// =============================================================================
// Main Component
// =============================================================================

export const GameReplay = () => {
    const { gameId } = useParams<{ gameId: string }>();
    const navigate = useNavigate();

    // Parse deprecated flag
    const isDeprecated = new URLSearchParams(window.location.search).get('deprecated') === 'true';

    const [replayState, setReplayState] = useState<GameReplayState | null>(null);
    const [simState, setSimState] = useState(selfPlayReplayService.getState());
    const [history, setHistory] = useState<number[]>([]);
    const [viewMode, setViewMode] = useState<ViewMode>('structure');
    const [showStats, setShowStats] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Initial Load
    useEffect(() => {
        if (!gameId) return;

        const load = async () => {
            setLoading(true);
            try {
                // Subscribe first
                const unsubscribe = selfPlayReplayService.subscribe((state) => {
                    setReplayState(state);
                    setSimState(selfPlayReplayService.getState());
                });

                await selfPlayReplayService.loadGame(gameId, isDeprecated);

                // Update history from metadata
                const hist = selfPlayReplayService.getValueHistory();
                setHistory(hist);

                setSimState(selfPlayReplayService.getState());
                setLoading(false);
                return unsubscribe;
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Erro carreganod jogo');
                setLoading(false);
            }
        };

        load();
        // Unwrap promise if needed or just let effect cleanup
        return () => {
            selfPlayReplayService.stop();
        };
    }, [gameId, isDeprecated]);

    const handlePlayPause = () => {
        selfPlayReplayService.togglePlay();
        setSimState(selfPlayReplayService.getState());
    };

    const handleStepBack = () => {
        selfPlayReplayService.stepBackward();
        setSimState(selfPlayReplayService.getState());
    };

    const handleStepForward = () => {
        selfPlayReplayService.stepForward();
        setSimState(selfPlayReplayService.getState());
    };

    const handleSeek = (step: number) => {
        selfPlayReplayService.seekToStep(step);
        setSimState(selfPlayReplayService.getState());
    };

    // Sync state on change
    useEffect(() => {
        const unsubscribe = selfPlayReplayService.subscribe((state) => {
            setReplayState(state);
            setSimState(selfPlayReplayService.getState());
        });
        return unsubscribe;
    }, []);

    return (
        <div className="h-[calc(100vh-64px)] relative bg-void overflow-hidden">
            <div className="absolute inset-0">
                <Canvas camera={{ position: [50, 50, 50], fov: 45 }}>
                    <color attach="background" args={['#050505']} />
                    <fog attach="fog" args={['#050505', 50, 150]} />
                    <ambientLight intensity={1.2} />
                    <pointLight position={[10, 10, 10]} intensity={1.5} />
                    <directionalLight position={[-10, 20, -10]} intensity={0.8} />
                    <gridHelper args={[100, 100, '#1a1a1a', '#111111']} />

                    <VoxelGridMCTS step={replayState} viewMode={viewMode} />
                    <LoadVector step={replayState} />
                    <SupportVoxels step={replayState} />

                    <OrbitControls makeDefault target={[0, 16, 0]} />
                </Canvas>
            </div>

            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute left-8 top-8 z-10"
            >
                <div className="flex items-center gap-4 bg-matter/90 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                    <button
                        onClick={() => navigate('/games')}
                        className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-colors"
                    >
                        <ArrowLeft size={20} />
                    </button>
                    <div>
                        <div className="text-xs text-white/40 uppercase tracking-wider mb-1">
                            Replay de Partida
                        </div>
                        <div className="font-mono text-cyan text-sm">
                            {gameId ? `${gameId.slice(0, 20)}...` : 'Carregando...'}
                        </div>
                    </div>
                </div>
            </motion.div>

            <div className="absolute bottom-8 left-[340px] z-10">
                <ViewModeSwitcher mode={viewMode} setMode={setViewMode} />
            </div>

            <MCTSStatsPanel
                step={replayState}
                isOpen={showStats}
                toggle={() => setShowStats(!showStats)}
            />

            <NeuralHUD
                history={history}
                currentStep={simState.currentStep}
            />

            <SimulationControls
                currentStep={replayState}
                currentStepIndex={simState.currentStep}
                totalSteps={simState.stepsLoaded > 0 ? simState.stepsLoaded : 0}
                isPlaying={simState.isPlaying}
                onPlayPause={handlePlayPause}
                onStepBack={handleStepBack}
                onStepForward={handleStepForward}
                onSeek={handleSeek}
            />

            <RewardOverlay step={replayState} />

            <AnimatePresence>
                {(loading || error) && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 bg-void/90 backdrop-blur flex flex-col items-center justify-center z-50"
                    >
                        {error ? (
                            <>
                                <div className="text-red-400 mb-4">Erro ao carregar partida</div>
                                <p className="text-white/60 mb-4 text-sm">{error}</p>
                                <button
                                    onClick={() => navigate('/games')}
                                    className="px-4 py-2 bg-cyan/20 text-cyan rounded-lg hover:bg-cyan/30"
                                >
                                    Voltar para lista
                                </button>
                            </>
                        ) : (
                            <Loader size={32} className="animate-spin text-cyan" />
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
