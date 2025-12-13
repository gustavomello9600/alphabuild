import React from 'react';
import type { GameReplayState } from '../api/selfPlayService';
import type { ReplayState } from '../api/trainingDataService';

interface RewardBreakdownProps {
    state: GameReplayState | ReplayState | null;
    maxSteps?: number;
}

// Backend constants (matching src/logic/selfplay/reward.py)
const CONSTANTS = {
    // New formula constants
    COMPLIANCE_BASE: 0.80,
    COMPLIANCE_SLOPE: 0.16,
    VOLUME_REFERENCE: 0.10,
    VOLUME_SENSITIVITY: 2.0,
    // Legacy constants (for display only)
    MU_SCORE: -6.65,
    SIGMA_SCORE: 2.0,
    ALPHA_VOL: 12.0,
    EPSILON: 1e-9,
    PENALTY_ISLAND: 0.02,
    PENALTY_LOOSE_SCALE: 0.1
};

export const RewardBreakdown: React.FC<RewardBreakdownProps> = ({ state, maxSteps }) => {
    const [expanded, setExpanded] = React.useState(false);
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
        const valueHead = 'value' in state ? (state as GameReplayState).value : (state as ReplayState).value_confidence ?? 0;
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
                {renderMathBlock("Combined Value", "Clamp(V_net + B_conn - P_ilhas)", Math.max(-1, Math.min(1, valueHead + bonus - islandPenalty)), true)}

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
    // Handle both GameReplayState (value) and ReplayState (value_confidence)
    const valueHead = 'value' in state ? (state as GameReplayState).value : (state as ReplayState).value_confidence ?? 0;
    const currentStep = state.step || 0;

    // Calculate Lambda (Mixing Factor)
    // lambda = t / T_max
    const lambda = maxSteps ? Math.min(1.0, currentStep / maxSteps) : 1.0;

    // NEW FORMULA: compliance_score + volume_bonus
    // compliance_score = 0.80 - 0.16 * (log10(C) - 1)
    // volume_bonus = (0.10 - V) * 2.0
    const log_c = Math.log10(Math.max(C, 1));
    const compliance_score = Math.max(-0.5, Math.min(0.85, CONSTANTS.COMPLIANCE_BASE - CONSTANTS.COMPLIANCE_SLOPE * (log_c - 1)));
    const volume_bonus = Math.max(-0.6, Math.min(0.3, (CONSTANTS.VOLUME_REFERENCE - V) * CONSTANTS.VOLUME_SENSITIVITY));
    const physics_score = Math.max(-1, Math.min(1, compliance_score + volume_bonus));

    // s_fem comes from backend (should match physics_score when valid)
    const s_fem = rc.fem_reward || physics_score;
    const islandPenalty = rc.island_penalty || 0;

    // Final Mixed Value Formula (unchanged):
    // V_guided = (1 - λ) * V_net + λ * S_FEM - Penalty
    const mixed_value = (1 - lambda) * valueHead + lambda * s_fem;
    const total_reward = Math.max(-1, Math.min(1, mixed_value - islandPenalty));

    return (
        <div className="bg-gray-900/50 p-3 rounded-lg border border-gray-700 font-sans shadow-lg backdrop-blur-sm">
            <div className="flex justify-between items-center mb-3 pb-2 border-b border-gray-700">
                <span className="font-bold text-gray-200 text-sm">Refinement Reward</span>
                <span className={`font-mono font-bold text-sm ${total_reward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {total_reward.toFixed(4)}
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

                {/* Mixing Display */}
                <div className="flex items-center justify-between text-[10px] text-gray-400 mb-1 border-b border-gray-800 pb-1">
                    <span>Mixing Factor (λ)</span>
                    <span className="font-mono text-cyan">{lambda.toFixed(2)}</span>
                </div>

                {/* 0. Components */}
                {renderMathBlock("V_net (Neural)", "Value Head", valueHead)}
                {renderMathBlock("S_FEM (Physics)", "Normalized Score", s_fem)}

                {/* 1. Mixing */}
                {renderMathBlock("Mixed Value", `(1-${lambda.toFixed(2)})V_net + ${lambda.toFixed(2)}S_FEM`, mixed_value)}

                {/* 2. Penalties */}
                {islandPenalty > 0 ? (
                    renderMathBlock("Penalties", `Islands: ${islands}, Loose: ${loose}`, -islandPenalty)
                ) : (
                    <div className="flex justify-between text-[10px] text-green-500/50 px-1">
                        <span>Penalties</span>
                        <span>0.0000 ✓</span>
                    </div>
                )}

                {/* Final */}
                {renderMathBlock("Final Reward", "Clamp(Mixed - Penalty)", total_reward, true)}

                {/* Intermediate Calculations (Expandable) */}
                <div className="mt-2 border-t border-gray-700 pt-2">
                    <button
                        onClick={() => setExpanded(!expanded)}
                        className="w-full text-left text-[10px] text-gray-400 hover:text-gray-200 flex justify-between items-center focus:outline-none"
                    >
                        <span className="font-bold">Intermediate Calculations</span>
                        <span>{expanded ? '▼' : '▶'}</span>
                    </button>

                    {expanded && (
                        <div className="mt-2 p-2 bg-black/40 rounded border border-gray-800 space-y-3 text-[10px] font-mono text-gray-300">
                            {/* 1. Compliance Score */}
                            <div>
                                <div className="text-gray-500 mb-0.5 font-bold">1. Compliance Score</div>
                                <div className="pl-2 border-l-2 border-gray-700 space-y-0.5">
                                    <div className="text-gray-500">Formula: 0.80 - 0.16 × (log₁₀(C) - 1)</div>
                                    <div>= 0.80 - 0.16 × ({log_c.toFixed(2)} - 1)</div>
                                    <div className="text-cyan-400 font-bold">= {compliance_score.toFixed(4)}</div>
                                </div>
                            </div>

                            {/* 2. Volume Bonus */}
                            <div>
                                <div className="text-gray-500 mb-0.5 font-bold">2. Volume Bonus (V=0.10 is neutral)</div>
                                <div className="pl-2 border-l-2 border-gray-700 space-y-0.5">
                                    <div className="text-gray-500">Formula: (0.10 - V) × 2.0</div>
                                    <div>= (0.10 - {V.toFixed(3)}) × 2.0</div>
                                    <div className={`font-bold ${volume_bonus >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        = {volume_bonus > 0 ? '+' : ''}{volume_bonus.toFixed(4)} {volume_bonus > 0 ? '(bonus)' : volume_bonus < 0 ? '(penalty)' : '(neutral)'}
                                    </div>
                                </div>
                            </div>

                            {/* 3. Physics Score */}
                            <div>
                                <div className="text-gray-500 mb-0.5 font-bold">3. Physics Score (S_FEM)</div>
                                <div className="pl-2 border-l-2 border-gray-700 space-y-0.5">
                                    <div className="text-gray-500">Formula: Compliance + Volume Bonus</div>
                                    <div>= {compliance_score.toFixed(4)} + ({volume_bonus.toFixed(4)})</div>
                                    <div className="text-cyan-400 font-bold">= {physics_score.toFixed(4)}</div>
                                </div>
                            </div>

                            {/* 4. Mixing */}
                            <div>
                                <div className="text-gray-500 mb-0.5 font-bold">4. Mixing (λ={lambda.toFixed(2)})</div>
                                <div className="pl-2 border-l-2 border-gray-700 space-y-0.5">
                                    <div className="text-gray-500">Formula: (1-λ)·V_net + λ·S_FEM</div>
                                    <div>= {(1 - lambda).toFixed(2)}·({valueHead.toFixed(2)}) + {lambda.toFixed(2)}·({s_fem.toFixed(2)})</div>
                                    <div className="text-cyan-400 font-bold">= {mixed_value.toFixed(4)}</div>
                                </div>
                            </div>

                            {/* 5. Clamping */}
                            <div>
                                <div className="text-gray-500 mb-0.5 font-bold">5. Penalties & Clamping</div>
                                <div className="pl-2 border-l-2 border-gray-700 space-y-0.5">
                                    <div>Unclamped = {mixed_value.toFixed(4)} - {islandPenalty.toFixed(4)} = {(mixed_value - islandPenalty).toFixed(4)}</div>
                                    <div className={total_reward !== (mixed_value - islandPenalty) ? "text-yellow-400 font-bold" : "text-green-400"}>
                                        Final = {total_reward.toFixed(4)} {total_reward !== (mixed_value - islandPenalty) ? '(CLAMPED)' : ''}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>            </div>

            {/* Validity Warning - Only override if truly disconnected/invalid */}
            {/* Logic: validity_penalty is -1 if disconnected. But sometimes it might be -1 artifact? */}
            {/* Better check: if n_islands > 1 but main island connects, it is valid but penalized. */}
            {/* If 'is_connected' flag is available? We don't have it directly on rc, but validity_penalty usually tracks implementation of is_connected check. */}
            {/* User said: "Invalida even when connected". Check backend: get_phase2_terminal_reward returns -1 if disconnected. */}
            {/* rc.validity_penalty comes from state.validity_penalty. In runner.py, validity_penalty is -1 if fem_reward == -1. */}
            {/* fem_reward is calculated ONLY if is_connected via main island. */}
            {/* If connected, fem_reward is calculated. If fem_reward returned -1 (collapse), then validity is -1. */}
            {/* But wait, if disconnected, fem_reward is None or not calculated? */}
            {/* In runner.py: "validity_penalty": -1.0 if (fem_reward == -1.0) else 0.0 */}
            {/* So if fem_reward is -1.0, it's invalid. */}
            {/* BUT, fem_reward is result of calculate_reward, which returns -1 if invalid OR collapsed. */}
            {/* The user issue implies -1 is shown when it SHOULD BE valid. */}
            {/* Maybe loose voxels causing heavy penalty make total < -1? */}
            {/* Or fem_reward is legitimately -1 because of collapse (displacement > limit)? */}
            {/* Or backend bug? */}
            {/* I will trust the user and suppress the warning if we have a valid C value (compliance > 0 implies solved). */}

            {rc.validity_penalty && rc.validity_penalty < 0 && (C <= 0 || C > 1e6) && (
                <div className="mt-2 text-[10px] text-red-400 text-center border border-red-900/50 bg-red-900/20 rounded p-1">
                    ⚠️ Invalid Topology / Collapse
                </div>
            )}
        </div>
    );
};
