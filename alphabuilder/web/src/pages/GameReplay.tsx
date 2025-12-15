import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Loader, AlertCircle } from 'lucide-react';

import {
    selfPlayReplayService,
    type GameReplayState
} from '../api/selfPlayService';

// Shared Components
import { ViewModeToolbar } from '../components/replay/ViewModeToolbar';
import { SimulationControls } from '../components/replay/SimulationControls';
import { PolicyHUD } from '../components/replay/PolicyHUD';
import { MCTSStatsPanel } from '../components/replay/MCTSStats';
import { RewardCard } from '../components/replay/RewardCard';
import { VoxelGridMCTS, LoadVector, SupportVoxels } from '../components/replay/VoxelGrid';
import type { ViewMode } from '../components/replay/types';

// =============================================================================
// Game Replay Component
// =============================================================================

export const GameReplay = () => {
    const { gameId } = useParams<{ gameId: string }>();
    const navigate = useNavigate();

    const [replayState, setReplayState] = useState<GameReplayState | null>(null);
    const [nextReplayState, setNextReplayState] = useState<GameReplayState | null>(null);
    const [simState, setSimState] = useState(selfPlayReplayService.getState());
    const [history, setHistory] = useState<number[]>([]);
    const [viewMode, setViewMode] = useState<ViewMode>('structure');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [statsOpen, setStatsOpen] = useState(true);

    // Initial Load
    useEffect(() => {
        if (!gameId) return;

        const load = async () => {
            setLoading(true);
            setError(null);
            try {
                // Subscribe to store updates
                const unsubscribe = selfPlayReplayService.subscribe((state: GameReplayState) => {
                    setReplayState(state);
                    setSimState(selfPlayReplayService.getState());
                    setNextReplayState(selfPlayReplayService.getFrame(state.step + 1) || null);
                });

                // Load game data
                await selfPlayReplayService.loadGame(gameId);

                // Initialize history
                const fullHistory = selfPlayReplayService.getValueHistory();
                setHistory(fullHistory);

                // Force initial state update
                setSimState(selfPlayReplayService.getState());

                setLoading(false);
                return unsubscribe;
            } catch (err) {
                console.error('Failed to load game:', err);
                setError(err instanceof Error ? err.message : 'Erro ao carregar partida');
                setLoading(false);
            }
        };

        load();
        return () => {
            selfPlayReplayService.stop();
        };
    }, [gameId]);


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

    const handleBack = () => {
        selfPlayReplayService.stop();
        navigate('/');
    };

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

                    <VoxelGridMCTS step={replayState} nextStep={nextReplayState} viewMode={viewMode} />
                    <LoadVector step={replayState} />
                    <SupportVoxels step={replayState} />

                    <OrbitControls makeDefault target={[0, 16, 0]} />
                </Canvas>
            </div>

            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute left-8 top-8 z-10"
            >
                <div className="flex items-center gap-4 bg-matter/90 backdrop-blur-xl border border-white/10 rounded-xl p-4">
                    <button
                        onClick={handleBack}
                        className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-colors"
                    >
                        <ArrowLeft size={20} />
                    </button>
                    <div>
                        <div className="text-xs text-white/40 uppercase tracking-wider mb-1">
                            Replay de Partida
                        </div>
                        <div className="font-mono text-cyan text-sm">
                            {gameId ? `${gameId.slice(0, 8)}...` : 'Carregando...'}
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Tools */}
            <div className="absolute bottom-8 left-[340px] z-10 flex items-center gap-3">
                <ViewModeToolbar
                    mode={viewMode}
                    setMode={setViewMode}
                />
            </div>

            <PolicyHUD
                history={history}
                currentStep={simState.currentStep}
                viewMode={viewMode}
            />

            <MCTSStatsPanel
                step={replayState}
                isOpen={statsOpen}
                toggle={() => setStatsOpen(!statsOpen)}
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

            <RewardCard step={replayState} />

            <AnimatePresence>
                {(loading || error) && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 bg-void/90 backdrop-blur flex flex-col items-center justify-center z-50"
                    >
                        {error ? (
                            <div className="flex flex-col items-center">
                                <div className="p-6 rounded-full bg-magenta/10 text-magenta mb-6">
                                    <AlertCircle size={48} />
                                </div>
                                <p className="text-white/60 text-lg mb-2">Erro ao carregar partida</p>
                                <p className="text-magenta/60 text-sm mb-6">{error}</p>
                                <button
                                    onClick={handleBack}
                                    className="px-6 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
                                >
                                    Voltar
                                </button>
                            </div>
                        ) : (
                            <div className="flex flex-col items-center">
                                <Loader size={48} className="animate-spin text-cyan mb-4" />
                                <p className="text-white/60 text-lg">Carregando replay...</p>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
