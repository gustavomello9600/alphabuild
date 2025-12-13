import React from 'react';
import type { GameReplayState } from '../api/selfPlayService';
import type { ReplayState } from '../api/trainingDataService';

interface RewardBreakdownProps {
    state: GameReplayState | ReplayState | null;
}

// Backend constants (matching src/logic/selfplay/reward.py)
const CONSTANTS = {
    MU_SCORE: -6.65,
    SIGMA_SCORE: 2.0,
    ALPHA_VOL: 12.0,
    EPSILON: 1e-9,
    PENALTY_ISLAND: 0.02,
    PENALTY_LOOSE_SCALE: 0.1
};

export const RewardBreakdown: React.FC<RewardBreakdownProps> = ({ state }) => {
    if (!state) return <div className="text-gray-500 italic">No data</div>;

    const rc = state.reward_components;

    // Helper to render a math block
    const renderMathBlock = (label: string, expression: string, value: number, isTotal = false) => (
        <div className={`flex justify-between items-center ${isTotal ? 'mt-2 pt-2 border-t border-gray-600' : 'mt-1'}`}>
            <div className="flex flex-col">
                <span className={`text-[10px] uppercase ${isTotal ? 'text-gray-200 font-bold' : 'text-gray-500'}`}>{label}</span>
                <span className="text-[10px] text-gray-500 font-mono tracking-tight">{expression}</span>
            </div>
            <span className={`font-mono font-bold  ${value > 0 ? 'text-green-400' : value < 0 ? 'text-red-400' : 'text-gray-500'} ${isTotal ? 'text-sm' : 'text-xs'}`}>
                {value > 0 ? '+' : ''}{value.toFixed(4)}
            </span>
        </div>
    );

    const renderMetric = (label: string, value: string | number) => (
        <div className="flex justify-between text-[10px] text-gray-400">
            <span>{label}</span>
            <span className="font-mono text-gray-300">{value}</span>
        </div>
    );

    // --- PHASE 1: GROWTH (Connectivity Focus) ---
    if (state.phase === 'GROWTH') {
        const bonus = rc?.connectivity_bonus || 0;
        const fraction = rc?.connected_load_fraction ?? 0;
        const valueHead = state.value ?? 0;
        const islandPenalty = rc?.island_penalty ?? 0;
        const nIslands = rc?.n_islands ?? 1;
        const looseVoxels = rc?.loose_voxels ?? 0;

        return (
            <div className="bg-gray-900/50 p-3 rounded-lg border border-gray-700 font-sans shadow-lg backdrop-blur-sm">
                <div className="flex justify-between items-center mb-3 pb-2 border-b border-gray-700">
                    <span className="font-bold text-gray-200 text-sm">Growth Phase Reward</span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${fraction >= 0.99 ? 'bg-green-900/30 text-green-400 border border-green-700/50' : 'bg-blue-900/30 text-blue-400 border border-blue-700/50'}`}>
                        Target: Connectivity
                    </span>
                </div>

                {/* 1. Connectivity Progress */}
                <div className="mb-4">
                    <div className="flex justify-between text-[10px] text-gray-400 mb-1">
                        <span>Load Connection Progress</span>
                        <span className="font-mono">{Math.round(fraction * 100)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                        <div
                            className={`h-full transition-all duration-500 ${fraction >= 1.0 ? 'bg-green-500' : 'bg-blue-500'}`}
                            style={{ width: `${Math.min(100, fraction * 100)}%` }}
                        />
                    </div>
                    <p className="text-[9px] text-gray-600 mt-1 italic">
                        Bonus = 0.5 &times; Fraction + {fraction >= 0.999 ? '0.5 (Complete)' : '0.0 (Incomplete)'}
                    </p>
                </div>

                {/* 2. Calculation Flow */}
                <div className="space-y-2 bg-black/20 p-2 rounded border border-gray-800/50">
                    {/* Value Head (Neural Network Prediction) */}
                    {renderMathBlock("Value Head (V_net)", "Neural Network Prediction", valueHead)}

                    {/* Connectivity Bonus */}
                    {renderMathBlock("Connectivity Bonus (B_conn)", `0.5 × ${fraction.toFixed(2)} + ${fraction >= 0.99 ? '0.5' : '0'}`, bonus)}

                    {/* Island Penalties - always show details */}
                    <div className="mt-1 pt-1 border-t border-gray-800/50">
                        <div className="text-[9px] text-gray-500 uppercase font-bold mb-1">Island Penalties</div>
                        {renderMetric("Islands Detected", nIslands)}
                        {renderMetric("Loose Voxels", looseVoxels)}
                        {islandPenalty > 0 ? (
                            renderMathBlock("Penalty (P_ilhas)", `0.02×(${nIslands}-1) + 0.1×${looseVoxels}/V`, -islandPenalty)
                        ) : (
                            <div className="flex justify-between text-[10px] text-green-500/70 px-1">
                                <span>Total Penalty</span>
                                <span className="font-mono">0.0000 ✓</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Final Combined Value */}
                {renderMathBlock("Combined Value", "V_net + B_conn - P_ilhas", valueHead + bonus - islandPenalty, true)}

                <div className="mt-2 text-[9px] text-gray-500 text-center">
                    This is the guided value used for MCTS node selection
                </div>
            </div>
        );
    }

    // --- PHASE 2: REFINEMENT (Physics Focus) ---
    // If no reward components, show basic fallback
    if (!rc) return <div className="text-gray-500">Wait for physics...</div>;

    // Type-safe access
    let C = 0;
    let V = 0;

    if ('compliance_fem' in state && typeof state.compliance_fem === 'number') {
        C = state.compliance_fem;
    } else if ('metadata' in state && state.metadata && 'compliance' in state.metadata) {
        C = (state as any).metadata.compliance || 0;
    }

    if ('volume_fraction' in state && typeof state.volume_fraction === 'number') {
        V = state.volume_fraction;
    } else if ('metadata' in state && state.metadata && 'volume_fraction' in state.metadata) {
        V = (state as any).metadata.volume_fraction || 0;
    }

    const islands = rc.n_islands ?? 1;
    const loose = rc.loose_voxels ?? 0;
    const valueHead = state.value ?? 0;

    // Calculate basics for transparent display
    const raw_val = rc.fem_reward !== undefined ? (-Math.log(C + CONSTANTS.EPSILON) - CONSTANTS.ALPHA_VOL * V) : 0;

    return (
        <div className="bg-gray-900/50 p-3 rounded-lg border border-gray-700 font-sans shadow-lg backdrop-blur-sm">
            <div className="flex justify-between items-center mb-3 pb-2 border-b border-gray-700">
                <span className="font-bold text-gray-200 text-sm">Refinement Reward</span>
                <span className={`font-mono font-bold text-sm ${rc.total >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {rc.total.toFixed(4)}
                </span>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-3">
                {/* Inputs */}
                <div className="bg-black/20 p-2 rounded border border-gray-800/50">
                    <span className="text-[10px] text-gray-500 uppercase font-bold text-center block mb-1">Physics Inputs</span>
                    {renderMetric("Compliance (C)", `${C.toFixed(4)} J`)}
                    {renderMetric("Volume (V)", `${(V * 100).toFixed(1)}%`)}
                </div>
                {/* Topology */}
                <div className="bg-black/20 p-2 rounded border border-gray-800/50">
                    <span className="text-[10px] text-gray-500 uppercase font-bold text-center block mb-1">Topology</span>
                    {renderMetric("Islands", islands)}
                    {renderMetric("Loose Voxels", loose)}
                </div>
            </div>

            {/* Calculations Flow */}
            <div className="space-y-1">
                <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">Derivation Chain</div>

                {/* Value Head */}
                {renderMathBlock("0. Value Head (V_net)", "Neural Network Prediction", valueHead)}

                {/* 1. Raw Score */}
                {renderMathBlock("1. Raw Score (S_raw)", `-ln(${C.toFixed(2)}) - 12×${V.toFixed(2)}`, raw_val)}

                {/* 2. Normalization */}
                {renderMathBlock("2. Normalized (S_FEM)", `tanh((S_raw - 6.65) / 2.0)`, rc.fem_reward || 0)}

                {/* 3. Penalties */}
                {(rc.island_penalty || 0) > 0 ? (
                    renderMathBlock("3. Penalties (P_ilhas)", `Islands: ${islands}, Loose: ${loose}`, -(rc.island_penalty || 0))
                ) : (
                    <div className="flex justify-between text-[10px] text-green-500/50 px-1">
                        <span>3. Penalties</span>
                        <span>0.0000 ✓</span>
                    </div>
                )}

                {/* Final */}
                {renderMathBlock("Final Reward", "S_FEM - P_ilhas", (rc.fem_reward || 0) - (rc.island_penalty || 0), true)}
            </div>

            {/* Validity Warning */}
            {rc.validity_penalty && rc.validity_penalty < 0 && (
                <div className="mt-2 text-[10px] text-red-400 text-center border border-red-900/50 bg-red-900/20 rounded p-1">
                    ⚠️ Invalid Topology: {rc.validity_penalty}
                </div>
            )}
        </div>
    );
};
