import React, { useState } from 'react';
import { Brain } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ViewMode } from './types';

interface PolicyHUDProps {
    history: number[];
    currentStep: number;
    // step: { value?: number; value_confidence?: number } | null; <-- Removed unused prop
    viewMode: ViewMode;
}

export const PolicyHUD: React.FC<PolicyHUDProps> = ({
    history,
    currentStep,
    // step, <-- Removed unused prop destructuring
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
