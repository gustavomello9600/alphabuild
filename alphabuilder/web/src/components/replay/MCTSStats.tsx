import React from 'react';
import { BarChart3, ChevronUp, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface MCTSStatsPanelProps {
    step: {
        mcts_stats?: {
            num_simulations: number;
            nodes_expanded: number;
            max_depth: number;
            cache_hits: number;
            top8_concentration: number;
            refutation: boolean;
        };
        selected_actions?: Array<{
            channel: number;
            x: number;
            y: number;
            z: number;
            visits: number;
            q_value: number;
        }>;
    } | null;
    isOpen: boolean;
    toggle: () => void;
}

export const MCTSStatsPanel: React.FC<MCTSStatsPanelProps> = ({
    step,
    isOpen,
    toggle,
}) => {
    if (!step || !step.mcts_stats || !step.selected_actions) return null;

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
