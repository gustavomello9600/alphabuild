
import React from 'react';
import type { GameReplayState } from '../api/selfPlayService';

interface RewardBreakdownProps {
    state: GameReplayState | null;
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

    // Derived calculations
    const C = state.compliance_fem;
    const V = state.volume_fraction;

    // Only show detailed breakdown if we have Compliance (Phase 2 with connection)
    // OR if we are in Phase 1 (Growth) we show a different simplified view

    if (state.phase === 'GROWTH') {
        const value = state.value;
        return (
            <div className="bg-gray-900/50 p-2 rounded text-xs mt-2 border border-gray-700/50">
                <div className="flex justify-between items-center mb-1">
                    <span className="font-bold text-gray-300">Reward (Growth Phase)</span>
                    <span className={`font-mono font-bold ${value >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {value.toFixed(3)}
                    </span>
                </div>
                <div className="grid grid-cols-[1fr_auto] gap-x-2 text-gray-400 gap-y-1">
                    <span>Estimated Value (V_net)</span>
                    <span className="font-mono">{value.toFixed(3)}</span>

                    <span className="mt-1 text-gray-500 border-t border-gray-700 pt-1">Target</span>
                    <span className="mt-1 text-gray-500 border-t border-gray-700 pt-1 font-mono">Connection</span>
                </div>
            </div>
        );
    }

    // Phase 2 or generic Display
    // If we don't have C (e.g. disconnected), we can't show full calculation
    if (C === undefined || C === null) {
        return (
            <div className="bg-gray-900/50 p-2 rounded text-xs mt-2 border border-gray-700/50">
                <div className="flex justify-between items-center mb-1">
                    <span className="font-bold text-gray-300">Reward Breakdown</span>
                    <span className="font-mono text-gray-400">N/A (No Physics)</span>
                </div>
                <div className="text-amber-500/80">
                    Structure not valid for FEM analysis.
                </div>
            </div>
        );
    }

    // --- Calculation Logic ---

    // 1. Raw Score Parcels
    const score_compliance = -Math.log(C + CONSTANTS.EPSILON);
    const score_volume = -(CONSTANTS.ALPHA_VOL * V);
    const raw_score = score_compliance + score_volume;

    // 2. Normalization
    // R = tanh((raw - mu) / sigma)
    const norm_argument = (raw_score - CONSTANTS.MU_SCORE) / CONSTANTS.SIGMA_SCORE;
    const base_reward = Math.tanh(norm_argument);

    // 3. Penalties
    // These are SUBTRACTED from base_reward
    // In backend: adjusted = base - penalty. 
    // We want to show contributions, so if penalty is 0.05, contribution is -0.05
    const penalty_total = state.island_penalty || 0;
    const loose_voxels = state.loose_voxels || 0;
    const n_islands = state.n_islands || 1;

    // Total
    // Use the backend's final value to be authoritative, or calculate?
    // Let's calculate to match breakdown, but strict check against backend value
    const total_calculated = base_reward - penalty_total;
    // const isMismatch = Math.abs(total_calculated - state.value) > 0.001; 
    // Note: state.value might be clamped or originate from NN prediction if simulation didn't run FEM, 
    // but here we are assuming C exists so FEM ran.

    const renderRow = (label: string, valueStr: string, contrib: number, isSubtotal = false, isHeader = false) => (
        <React.Fragment key={label}>
            <div className={`${isSubtotal ? 'font-bold text-gray-300 pt-1 border-t border-gray-700' : 'text-gray-400'} ${isHeader ? 'text-gray-500 uppercase text-[10px] mt-2' : ''}`}>
                {label}
            </div>
            <div className={`font-mono text-right ${isSubtotal ? 'text-gray-300 pt-1 border-t border-gray-700' : 'text-gray-500'} ${isHeader ? 'mt-2' : ''}`}>
                {valueStr}
            </div>
            <div className={`font-mono text-right font-bold ${isSubtotal ? 'pt-1 border-t border-gray-700' : ''} ${contrib > 0 ? 'text-green-400' : contrib < 0 ? 'text-red-400' : 'text-gray-600'} ${isHeader ? 'mt-2' : ''}`}>
                {!isHeader && (contrib > 0 ? '+' : '')}{!isHeader && contrib.toFixed(3)}
            </div>
        </React.Fragment>
    );

    return (
        <div className="bg-gray-900/50 p-2 rounded text-xs mt-2 border border-gray-700/50 font-sans shadow-lg">
            <div className="flex justify-between items-center mb-2 pb-1 border-b border-gray-700">
                <span className="font-bold text-gray-200">Reward Dynamics</span>
                <span className={`font-mono font-bold text-sm ${total_calculated >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {total_calculated.toFixed(4)}
                </span>
            </div>

            <div className="grid grid-cols-[1fr_auto_auto] gap-x-3 gap-y-1">
                {/* Header */}
                <div className="text-[10px] text-gray-600 uppercase">Metric</div>
                <div className="text-[10px] text-gray-600 uppercase text-right">Value</div>
                <div className="text-[10px] text-gray-600 uppercase text-right">Impact</div>

                {/* Score Bruto Section */}
                <div className="col-span-3 text-gray-500 uppercase text-[10px] mt-1 font-bold">Raw Score Composition</div>

                {renderRow("Compliance (-ln C)", `C=${C.toFixed(4)}J`, score_compliance)}
                {renderRow(`Volume (-${CONSTANTS.ALPHA_VOL}·V)`, `V=${(V * 100).toFixed(1)}%`, score_volume)}
                {renderRow("Net Raw Score", "", raw_score, true)}

                {/* Normalization Section */}
                <div className="col-span-3 text-gray-500 uppercase text-[10px] mt-2 font-bold">Normalization (Tanh)</div>
                {renderRow("Centering (-μ)", `μ=${CONSTANTS.MU_SCORE}`, -CONSTANTS.MU_SCORE)}
                {renderRow("Scaling (1/σ)", `σ=${CONSTANTS.SIGMA_SCORE}`, 0)}
                {/* Note: Normalization is non-linear, so "contribution" column is tricky. 
                    We show the result of Tanh as the "Base Reward" subtotal. */}
                <div className="col-span-3 h-px bg-gray-800 my-0.5"></div>
                <div className="text-gray-300 font-bold">Base Reward</div>
                <div className="text-gray-500 font-mono text-right text-[10px] self-center">tanh((S-μ)/σ)</div>
                <div className={`font-mono text-right font-bold ${base_reward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {base_reward.toFixed(3)}
                </div>

                {/* Penalties Section */}
                {(n_islands > 1 || loose_voxels > 0) && (
                    <>
                        <div className="col-span-3 text-gray-500 uppercase text-[10px] mt-2 font-bold">Geometric Penalties</div>
                        {n_islands > 1 && renderRow("Islands", `${n_islands}`, -(n_islands - 1) * CONSTANTS.PENALTY_ISLAND)}
                        {loose_voxels > 0 && renderRow("Loose Voxels", `${loose_voxels}`, -(state.island_penalty || 0) + ((n_islands - 1) * CONSTANTS.PENALTY_ISLAND))}
                        {/* Loose penalty approx is total - island part. Or just show total penalty line */}
                    </>
                )}

                {/* Final Total */}
                <div className="col-span-3 h-px bg-gray-700 my-1"></div>
                <div className="font-bold text-white text-sm">TOTAL</div>
                <div></div>
                <div className={`font-mono font-bold text-sm text-right ${total_calculated >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {total_calculated > 0 ? '+' : ''}{total_calculated.toFixed(4)}
                </div>
            </div>

            {(state.is_connected) && (
                <div className="mt-2 text-[10px] text-green-500/70 text-center uppercase tracking-wider border border-green-900/30 bg-green-900/10 rounded py-0.5">
                    Structure Connected
                </div>
            )}
        </div>
    );
};
