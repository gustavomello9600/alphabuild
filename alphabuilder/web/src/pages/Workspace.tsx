import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { motion } from 'framer-motion';
import { Play, Pause, SkipForward, SkipBack, MousePointer, Box, Eraser, ArrowUp, Triangle, Layers, Brain } from 'lucide-react';
import { mockService } from '../api/mockService';
import type { GameState, Tensor5D } from '../api/types';
import * as THREE from 'three';

// --- 3D Components ---
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

const LoadVector = ({ tensor }: { tensor: Tensor5D | null }) => {
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
    showHeatmap
}: {
    tensor: Tensor5D | null,
    heatmap?: { add: Float32Array, remove: Float32Array },
    showHeatmap: boolean
}) => {
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const dummy = new THREE.Object3D();

    useEffect(() => {
        if (!meshRef.current || !tensor) return;

        const numChannels = tensor.shape[0];
        const [, D, H, W] = tensor.shape;
        const is7Channel = numChannels >= 7;
        let count = 0;

        // Helper for heatmap color
        const getHeatmapColor = (idx: number): THREE.Color | null => {
            if (!showHeatmap || !heatmap) return null;

            const addVal = heatmap.add ? heatmap.add[idx] : 0;
            const removeVal = heatmap.remove ? heatmap.remove[idx] : 0;

            if (addVal < 0.01 && removeVal < 0.01) return null;

            const color = new THREE.Color();
            if (addVal > removeVal) {
                // Add: Green (Project Identity)
                color.set('#00FF9D');
                const intensity = Math.min(1, addVal * 2.5);
                color.lerp(new THREE.Color('#333333'), 1 - intensity);
            } else {
                // Remove: Red/Pink (Project Identity)
                color.set('#FF0055');
                const intensity = Math.min(1, removeVal * 2.5);
                color.lerp(new THREE.Color('#333333'), 1 - intensity);
            }
            return color;
        };

        // Iterate through tensor
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

                    const heatmapColor = getHeatmapColor(flatIdx);

                    // Visibility Logic
                    const isVisibleStandard = density > 0.1 || isSupport || hasLoad;
                    const isVisibleHeatmap = !!heatmapColor;

                    if (isVisibleStandard || isVisibleHeatmap) {
                        dummy.position.set(d - D / 2, h + 0.5, w - W / 2);
                        dummy.updateMatrix();
                        meshRef.current.setMatrixAt(count, dummy.matrix);

                        // Color Logic
                        if (heatmapColor) {
                            meshRef.current.setColorAt(count, heatmapColor);
                        } else if (supportColor) {
                            // Color based on support constraint type
                            meshRef.current.setColorAt(count, new THREE.Color(supportColor));
                        } else if (hasLoad) {
                            // Magenta for loads (from design system)
                            meshRef.current.setColorAt(count, new THREE.Color('#FF0055'));
                            } else {
                                const val = Math.max(0.2, density);
                                const color = new THREE.Color().setHSL(0, 0, val * 0.95);
                                meshRef.current.setColorAt(count, color);
                        }

                        count++;
                    }
                }
            }
        }

        meshRef.current.count = count;
        meshRef.current.instanceMatrix.needsUpdate = true;
        if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;

    }, [tensor, heatmap, showHeatmap]);

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, 20000]}>
            <boxGeometry args={[0.9, 0.9, 0.9]} />
            <meshStandardMaterial roughness={0.2} metalness={0.8} />
        </instancedMesh>
    );
};

// --- UI Components ---

const Toolbar = () => {
    const [isOpen, setIsOpen] = useState(true);

    return (
        <div className="absolute left-8 top-8 flex flex-col gap-2 z-10">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="bg-matter/90 backdrop-blur border border-white/10 p-2 rounded-lg text-white/60 hover:text-white mb-2 self-start"
            >
                <Layers size={20} />
            </button>

            {isOpen && (
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex flex-col gap-2 bg-matter/90 backdrop-blur border border-white/10 p-2 rounded-lg"
                >
                    <button className="p-2 rounded bg-cyan/20 text-cyan hover:bg-cyan/30 transition-colors" title="Selecionar">
                        <MousePointer size={20} />
                    </button>
                    <button className="p-2 rounded text-white/60 hover:text-white hover:bg-white/10 transition-colors" title="Desenhar Voxel">
                        <Box size={20} />
                    </button>
                    <button className="p-2 rounded text-white/60 hover:text-white hover:bg-white/10 transition-colors" title="Apagar">
                        <Eraser size={20} />
                    </button>
                    <div className="h-px bg-white/10 my-1" />
                    <button className="p-2 rounded text-white/60 hover:text-white hover:bg-white/10 transition-colors" title="Aplicar Carga">
                        <ArrowUp size={20} className="text-red-500" />
                    </button>
                    <button className="p-2 rounded text-white/60 hover:text-white hover:bg-white/10 transition-colors" title="Adicionar Suporte">
                        <Triangle size={20} className="text-blue-500" />
                    </button>
                </motion.div>
            )}
        </div>
    );
};

const PropertiesPanel = ({ state }: { state: GameState | null }) => {
    const [isOpen, setIsOpen] = useState(true);
    // Use getSimulationState instead of getDebugInfo
    const simState = mockService.getSimulationState();
    const totalSteps = simState.stepsLoaded;
    const isPlaying = simState.isPlaying;
    const maxCompliance = simState.maxCompliance;

    return (
        <motion.div
            className="absolute right-0 top-0 bottom-0 bg-matter/90 backdrop-blur border-l border-white/10 z-10 flex flex-col"
            initial={{ width: 320 }}
            animate={{ width: isOpen ? 320 : 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
        >
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="absolute -left-8 top-1/2 -translate-y-1/2 bg-matter/90 border border-white/10 border-r-0 rounded-l p-1 text-white/60 hover:text-white"
            >
                {isOpen ? <SkipForward size={16} className="rotate-0" /> : <SkipForward size={16} className="rotate-180" />}
            </button>

            <div className="p-6 overflow-y-auto flex-1 w-80">
                <h3 className="text-sm font-mono text-white/40 uppercase mb-6">Controle da Simulação</h3>

                {/* Phase Indicator */}
                <div className="mb-4 flex items-center gap-2">
                    <span className="text-xs text-white/60 uppercase">Fase:</span>
                    <span className={`text-xs font-bold px-2 py-1 rounded ${state?.phase === 'GROWTH' ? 'bg-green-500/20 text-green-500' : 'bg-purple-500/20 text-purple-500'}`}>
                        {state?.phase === 'GROWTH' ? 'CRESCIMENTO (Fase 1)' : 'REFINAMENTO (Fase 2)'}
                    </span>
                </div>

                {/* Timeline */}
                <div className="mb-6 p-4 rounded bg-black/20 border border-white/5">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-xs text-white/60">Passo Atual</span>
                        <span className="text-xs font-bold text-cyan">{state?.step || 0} / {totalSteps}</span>
                    </div>

                    {/* Slider */}
                    <input
                        type="range"
                        min="0"
                        max={Math.max(0, totalSteps - 1)}
                        value={simState.currentStep}
                        onChange={(e) => {
                            mockService.pause();
                            mockService.seekToStep(parseInt(e.target.value));
                        }}
                        className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan mb-2"
                    />

                    <div className="flex justify-between mt-1">
                        <span className="text-[10px] text-white/30">Início</span>
                        <span className="text-[10px] text-white/30">Fim</span>
                    </div>
                </div>

                {/* Metrics */}
                <div className="space-y-4 mb-8">
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-white/60">Compliance (Rigidez)</span>
                            <span className={`font-mono ${!state?.metadata.compliance ? 'text-red-500 text-xs' : 'text-white'}`}>
                                {state?.metadata.compliance ? state.metadata.compliance.toFixed(2) : 'ESTRUTURA DESCONECTADA'}
                            </span>
                        </div>
                        <div className="w-full bg-white/5 h-1 rounded-full overflow-hidden">
                            {state?.metadata.compliance ? (
                                <motion.div
                                    className="h-full bg-magenta"
                                    animate={{ width: `${Math.min(100, (state.metadata.compliance / maxCompliance) * 100)}%` }}
                                />
                            ) : (
                                <div className="h-full bg-red-500/50 w-full" />
                            )}
                        </div>
                    </div>
                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-white/60">Volume %</span>
                            <span className="font-mono text-white">{((state?.metadata.volume_fraction || 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-white/5 h-1 rounded-full">
                            <motion.div className="h-full bg-white/40" animate={{ width: `${(state?.metadata.volume_fraction || 0) * 100}%` }} />
                        </div>
                    </div>
                </div>

                {/* Controls */}
                <div className="flex gap-2">
                    <button
                        onClick={() => mockService.stepBackward()}
                        disabled={isPlaying}
                        className={`p-2 rounded transition-colors ${isPlaying ? 'bg-white/5 text-white/20 cursor-not-allowed' : 'bg-white/10 text-white hover:bg-white/20'}`}
                        title="Passo Anterior"
                    >
                        <SkipBack size={16} />
                    </button>

                    <button
                        onClick={() => mockService.togglePlay()}
                        className={`flex-1 font-bold py-2 rounded flex items-center justify-center gap-2 transition-colors ${!isPlaying
                            ? 'bg-cyan text-black hover:bg-cyan/90' // Play is primary when paused
                            : 'bg-white/10 text-white hover:bg-white/20' // Play is secondary when playing
                            }`}
                    >
                        <Play size={16} /> {isPlaying ? 'Rodando...' : 'Simular'}
                    </button>

                    <button
                        onClick={() => mockService.togglePlay()}
                        className={`p-2 rounded transition-colors ${isPlaying
                            ? 'bg-red-500 text-white hover:bg-red-600' // Pause is primary when playing
                            : 'bg-white/5 text-white/40 hover:bg-white/10' // Pause is secondary when paused
                            }`}
                    >
                        <Pause size={16} />
                    </button>

                    <button
                        onClick={() => mockService.stepForward()}
                        disabled={isPlaying}
                        className={`p-2 rounded transition-colors ${isPlaying ? 'bg-white/5 text-white/20 cursor-not-allowed' : 'bg-white/10 text-white hover:bg-white/20'}`}
                        title="Próximo Passo"
                    >
                        <SkipForward size={16} />
                    </button>
                </div>
            </div>
        </motion.div>
    );
};

const NeuralHUD = ({
    state,
    history,
    showHeatmap,
    setShowHeatmap
}: {
    state: GameState | null,
    history: number[],
    showHeatmap: boolean,
    setShowHeatmap: (v: boolean) => void
}) => {
    const [isOpen, setIsOpen] = useState(true);

    return (
        <div className="absolute bottom-8 left-8 flex flex-col gap-2 pointer-events-none">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="bg-black/60 backdrop-blur border border-white/10 p-2 rounded-lg text-cyan hover:text-white self-start pointer-events-auto"
            >
                <Brain size={20} />
            </button>

            {isOpen && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-4 items-end"
                >
                    {/* Quality Estimate Graph */}
                    <div className="bg-black/60 backdrop-blur border border-white/10 p-4 rounded-lg w-64 pointer-events-auto">
                        <div className="flex items-center gap-2 mb-2 text-cyan">
                            <Brain size={16} />
                            <span className="text-xs font-bold uppercase tracking-wider">
                                Estimativa de Qualidade
                            </span>
                        </div>
                        <div className="h-24 flex items-end gap-1 border-b border-white/10 pb-1 relative">
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

                                // Calculate scale that handles negative values
                                const min = Math.min(...visibleHistory);
                                const max = Math.max(...visibleHistory);
                                const range = max - min;
                                const padding = range * 0.1 || 0.2;
                                const scale = {
                                    min: min - padding,
                                    max: max + padding,
                                    range: range + padding * 2,
                                };
                                
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
                                            const height = Math.max(2, Math.min(100, normalized));
                                            const isPositive = value >= 0;
                                            const colorClass = isPositive 
                                                ? 'bg-gradient-to-t from-cyan/30 to-cyan/80'
                                                : 'bg-gradient-to-t from-magenta/30 to-magenta/80';

                                            return (
                                                <div
                                                    key={i}
                                                    className={`flex-1 ${colorClass} rounded-t-sm transition-all duration-300`}
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
                                            const visibleHistory = history.slice(-20);
                                            const min = Math.min(...visibleHistory);
                                            const max = Math.max(...visibleHistory);
                                            const padding = (max - min) * 0.1 || 0.2;
                                            return `${(min - padding).toFixed(2)} - ${(max + padding).toFixed(2)}`;
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
                    <div className="bg-black/60 backdrop-blur border border-white/10 p-2 rounded-lg pointer-events-auto flex gap-2">
                        <button
                            onClick={() => setShowHeatmap(!showHeatmap)}
                            className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs transition-colors ${showHeatmap
                                ? 'bg-cyan/20 text-cyan border border-cyan/50'
                                : 'bg-white/10 text-white hover:bg-white/20'
                                }`}
                        >
                            <Layers size={14} /> {showHeatmap ? 'Ocultar policy' : 'Mostrar policy'}
                        </button>
                    </div>
                </motion.div>
            )}
        </div>
    );
};


export const Workspace = () => {
    const { id } = useParams();
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [history, setHistory] = useState<number[]>([]);
    const [showHeatmap, setShowHeatmap] = useState(false);

    useEffect(() => {
        if (id) {
            setHistory([]); // Reset history on new episode
            const unsubscribe = mockService.subscribe((state) => {
                setGameState(state);
                // Build history from all frames up to current step, sorted by step
                const simState = mockService.getSimulationState();
                if (simState.stepsLoaded > 0) {
                    const allFrames = mockService.getAllFrames();
                    if (allFrames.length > 0 && simState.currentStep < allFrames.length) {
                        // Get current frame's step number
                        const currentFrame = allFrames[simState.currentStep];
                        const currentStepNumber = currentFrame.step;
                        
                        // Get all frames up to current step, sorted by step (ascending)
                        const framesUpToCurrent = allFrames
                            .filter(f => f.step <= currentStepNumber)
                            .sort((a, b) => a.step - b.step);
                        
                        const historyValues = framesUpToCurrent.map(f => f.value_confidence || 0);
                        setHistory(historyValues);
                    }
                }
            });
            mockService.startSimulation(id);
            return () => {
                unsubscribe();
                mockService.stopSimulation();
            };
        }
    }, [id]);

    // Force re-render on interval to update UI state even if game state doesn't change (e.g. for play/pause toggle)
    // Actually, mockService notifies subscribers on pause/play, so we should be good.
    // However, we need to access non-state properties like isPlaying.
    // Let's use a timer to poll for UI updates or rely on the subscription.
    // Ideally, isPlaying should be part of the state or we should subscribe to it.
    // For now, we'll force a re-render every 100ms to keep UI snappy or just rely on the fact that 
    // mockService notifies on pause/play.

    // To ensure UI updates when isPlaying changes (which might not trigger a new GameState if paused),
    // we can add a local state for it or just rely on the fact that togglePlay calls notifySubscribers.
    // The current implementation of togglePlay in mockService calls notifySubscribers, so we are good.

    return (
        <div className="h-[calc(100vh-64px)] relative bg-void overflow-hidden">
            {/* 3D Canvas */}
            <div className="absolute inset-0">
                <Canvas camera={{ position: [50, 50, 50], fov: 45 }}>
                    <color attach="background" args={['#050505']} />
                    <fog attach="fog" args={['#050505', 50, 150]} />

                    {/* Improved Lighting */}
                    <ambientLight intensity={1.2} />
                    <pointLight position={[10, 10, 10]} intensity={1.5} />
                    <directionalLight position={[-10, 20, -10]} intensity={0.8} />

                    <gridHelper args={[100, 100, '#1a1a1a', '#111111']} />

                    <VoxelGrid
                        tensor={gameState?.tensor || null}
                        heatmap={gameState?.policy_heatmap}
                        showHeatmap={showHeatmap}
                    />
                    <LoadVector tensor={gameState?.tensor || null} />

                    <OrbitControls makeDefault target={[0, 16, 0]} />
                    <GizmoHelper alignment="top-right" margin={[80, 80]}>
                        <GizmoViewport axisColors={['#FF0055', '#00FF9D', '#00F0FF']} labelColor="white" />
                    </GizmoHelper>
                </Canvas>
            </div>

            {/* UI Overlays */}
            <Toolbar />
            <PropertiesPanel state={gameState} />
            <NeuralHUD
                state={gameState}
                history={history}
                showHeatmap={showHeatmap}
                setShowHeatmap={setShowHeatmap}
            />
        </div>
    );
};
