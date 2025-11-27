import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import { motion } from 'framer-motion';
import { Play, Pause, SkipForward, MousePointer, Box, Eraser, ArrowUp, Triangle, Layers, Brain } from 'lucide-react';
import { mockService } from '../api/mockService';
import type { GameState, Tensor5D } from '../api/types';
import * as THREE from 'three';

// --- 3D Components ---

const VoxelGrid = ({ tensor }: { tensor: Tensor5D | null }) => {
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const dummy = new THREE.Object3D();

    useEffect(() => {
        if (!meshRef.current || !tensor) return;

        const [, D, H, W] = tensor.shape;
        let count = 0;

        // Iterate through tensor to find active voxels
        // Channel 0 = Density, Channel 1 = Support, Channel 3 = Load
        for (let d = 0; d < D; d++) {
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    const index = 0 * (D * H * W) + d * (H * W) + h * W + w;
                    const density = tensor.data[index];

                    const supportIndex = 1 * (D * H * W) + d * (H * W) + h * W + w;
                    const isSupport = tensor.data[supportIndex] > 0.5;

                    const loadIndex = 3 * (D * H * W) + d * (H * W) + h * W + w;
                    const isLoad = tensor.data[loadIndex] !== 0;

                    if (density > 0.1 || isSupport || isLoad) {
                        dummy.position.set(w - W / 2, h - H / 2, d - D / 2);
                        dummy.updateMatrix();
                        meshRef.current.setMatrixAt(count, dummy.matrix);

                        // Color Logic
                        if (isSupport) {
                            meshRef.current.setColorAt(count, new THREE.Color('#00F0FF')); // Cyan
                        } else if (isLoad) {
                            meshRef.current.setColorAt(count, new THREE.Color('#FF0055')); // Magenta
                        } else {
                            // Titanium/Steel for better contrast against void background
                            meshRef.current.setColorAt(count, new THREE.Color('#E0E0E0'));
                        }

                        count++;
                    }
                }
            }
        }

        meshRef.current.count = count;
        meshRef.current.instanceMatrix.needsUpdate = true;
        if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;

    }, [tensor]);

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, 4000]}>
            <boxGeometry args={[0.9, 0.9, 0.9]} />
            <meshStandardMaterial />
        </instancedMesh>
    );
};

// --- UI Components ---

const Toolbar = () => (
    <div className="absolute left-8 top-8 flex flex-col gap-2 bg-matter/90 backdrop-blur border border-white/10 p-2 rounded-lg z-10">
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
            <ArrowUp size={20} className="text-magenta" />
        </button>
        <button className="p-2 rounded text-white/60 hover:text-white hover:bg-white/10 transition-colors" title="Adicionar Suporte">
            <Triangle size={20} className="text-cyan" />
        </button>
    </div>
);

const PropertiesPanel = ({ state }: { state: GameState | null }) => (
    <div className="absolute right-0 top-0 bottom-0 w-80 bg-matter/90 backdrop-blur border-l border-white/10 p-6 overflow-y-auto z-10">
        <h3 className="text-sm font-mono text-white/40 uppercase mb-6">Controle da Simulação</h3>

        {/* Phase Indicator */}
        <div className="mb-6 p-4 rounded bg-black/20 border border-white/5">
            <div className="flex justify-between items-center mb-2">
                <span className="text-xs text-white/60">Fase Atual</span>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${state?.phase === 'GROWTH' ? 'bg-green-500/20 text-green-400' : 'bg-purple-500/20 text-purple-400'}`}>
                    {state?.phase === 'GROWTH' ? 'CRESCIMENTO' : 'REFINAMENTO'}
                </span>
            </div>
            <div className="w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                <motion.div
                    className="h-full bg-cyan"
                    initial={{ width: 0 }}
                    animate={{ width: `${((state?.step || 0) / 20) * 100}%` }}
                />
            </div>
            <div className="flex justify-between mt-1">
                <span className="text-[10px] text-white/30">Passo {state?.step || 0}</span>
                <span className="text-[10px] text-white/30">Max 20</span>
            </div>
        </div>

        {/* Metrics */}
        <div className="space-y-4 mb-8">
            <div>
                <div className="flex justify-between text-sm mb-1">
                    <span className="text-white/60">Compliance (Rigidez)</span>
                    <span className="font-mono text-white">{state?.metadata.compliance.toFixed(2) || '---'}</span>
                </div>
                <div className="w-full bg-white/5 h-1 rounded-full">
                    <motion.div className="h-full bg-magenta" animate={{ width: `${100 - (state?.metadata.compliance || 0)}%` }} />
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
                onClick={() => state?.episode_id && mockService.startSimulation(state.episode_id)}
                className="flex-1 bg-cyan text-black font-bold py-2 rounded flex items-center justify-center gap-2 hover:bg-cyan/90"
            >
                <Play size={16} /> Iniciar
            </button>
            <button
                onClick={() => mockService.stopSimulation()}
                className="p-2 bg-white/5 text-white rounded hover:bg-white/10"
            >
                <Pause size={16} />
            </button>
            <button className="p-2 bg-white/5 text-white rounded hover:bg-white/10">
                <SkipForward size={16} />
            </button>
        </div>
    </div>
);

const NeuralHUD = ({ state }: { state: GameState | null }) => (
    <div className="absolute bottom-8 left-8 right-96 flex gap-4 items-end pointer-events-none">
        {/* Confidence Graph (Mock) */}
        <div className="bg-black/60 backdrop-blur border border-white/10 p-4 rounded-lg w-64 pointer-events-auto">
            <div className="flex items-center gap-2 mb-2 text-cyan">
                <Brain size={16} />
                <span className="text-xs font-bold uppercase">Confiança da Rede (Value)</span>
            </div>
            <div className="h-24 flex items-end gap-1">
                {Array.from({ length: 20 }).map((_, i) => (
                    <motion.div
                        key={i}
                        className="flex-1 bg-cyan/50 rounded-t-sm"
                        animate={{ height: `${20 + Math.random() * 60}%` }}
                        transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse", delay: i * 0.05 }}
                    />
                ))}
            </div>
            <div className="mt-2 text-right font-mono text-xs text-cyan">
                {(state?.value_confidence || 0).toFixed(4)}
            </div>
        </div>

        {/* Policy Heatmap Toggle */}
        <div className="bg-black/60 backdrop-blur border border-white/10 p-2 rounded-lg pointer-events-auto flex gap-2">
            <button className="flex items-center gap-2 px-3 py-1.5 rounded bg-white/10 text-white text-xs hover:bg-white/20 transition-colors">
                <Layers size={14} /> Visualizar Heatmap
            </button>
        </div>
    </div>
);

export const Workspace = () => {
    const { id } = useParams();
    const [gameState, setGameState] = useState<GameState | null>(null);

    useEffect(() => {
        if (id) {
            // Start listening to the mock stream
            const unsubscribe = mockService.subscribe((state) => {
                setGameState(state);
            });
            // Auto-start for demo
            mockService.startSimulation(id);
            return () => {
                unsubscribe();
                mockService.stopSimulation();
            };
        }
    }, [id]);

    return (
        <div className="h-[calc(100vh-64px)] relative bg-void overflow-hidden">
            {/* 3D Canvas */}
            <div className="absolute inset-0">
                <Canvas camera={{ position: [20, 20, 20], fov: 45 }}>
                    <color attach="background" args={['#050505']} />
                    <fog attach="fog" args={['#050505', 20, 60]} />
                    <ambientLight intensity={0.5} />
                    <pointLight position={[10, 10, 10]} intensity={1} />
                    <gridHelper args={[100, 100, '#1a1a1a', '#111111']} />

                    <VoxelGrid tensor={gameState?.tensor || null} />

                    <OrbitControls makeDefault />
                    <GizmoHelper alignment="top-right" margin={[80, 80]}>
                        <GizmoViewport axisColors={['#FF0055', '#00FF9D', '#00F0FF']} labelColor="white" />
                    </GizmoHelper>
                </Canvas>
            </div>

            {/* UI Overlays */}
            <Toolbar />
            <PropertiesPanel state={gameState} />
            <NeuralHUD state={gameState} />
        </div>
    );
};
