import React, { useState, useEffect } from 'react';
import { Network, Sparkles, SkipBack, SkipForward, Play, Pause } from 'lucide-react';

interface SimulationControlsProps {
    currentStep: { phase: string } | null;
    currentStepIndex: number;
    totalSteps: number;
    isPlaying: boolean;
    onPlayPause: () => void;
    onStepBack: () => void;
    onStepForward: () => void;
    onSeek: (step: number) => void;
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
    currentStep,
    currentStepIndex,
    totalSteps,
    isPlaying,
    onPlayPause,
    onStepBack,
    onStepForward,
    onSeek,
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
