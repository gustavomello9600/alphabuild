import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronUp, ChevronDown } from 'lucide-react';
import { RewardBreakdown } from '../RewardBreakdown';

interface RewardCardProps {
    step: any;
    scaleFactor?: number;
}

export const RewardCard: React.FC<RewardCardProps> = ({ step, scaleFactor }) => {
    const [expanded, setExpanded] = useState(false);

    // Determine main value to show
    const rc = step?.reward_components;
    let rewardValue = 0;

    if (step?.phase === 'GROWTH') {
        const bonus = rc?.connectivity_bonus || 0;
        const valueHead = step?.value || 0;
        const islandPenalty = rc?.island_penalty || 0;
        rewardValue = Math.max(-1, Math.min(1, valueHead + bonus - islandPenalty));
    } else {
        // Phase 2: Show MCTS Mixed Value (Guided Value)
        // Formula: (1 - λ) * V_net + λ * S_FEM - Penalty
        const valueHead = step?.value || 0;
        const s_fem = rc?.fem_reward || valueHead; // Fallback if no FEM
        const islandPenalty = rc?.island_penalty || 0;

        // Calculate Lambda (Mixing Factor)
        const currentStep = step?.step || 0;
        const maxSteps = 600; // Assuming 600 from context or pass as prop
        const lambda = Math.min(1.0, currentStep / maxSteps);

        const mixed_value = (1 - lambda) * valueHead + lambda * s_fem;
        rewardValue = Math.max(-1, Math.min(1, mixed_value - islandPenalty));
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
                        <RewardBreakdown state={step} maxSteps={600} scaleFactor={scaleFactor} />
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
