import React from 'react';
import type { Action } from '../../api/types';

interface ActionSequenceListProps {
    actions?: Action[] | null;
}

export const ActionSequenceList: React.FC<ActionSequenceListProps> = ({ actions }) => {
    if (!actions || actions.length === 0) return null;

    return (
        <div className="absolute top-24 left-8 z-10 w-64 bg-black/60 backdrop-blur border border-white/10 rounded-xl p-3 max-h-64 overflow-y-auto">
            <div className="text-xs text-white/40 uppercase tracking-wider mb-2 flex justify-between items-center">
                <span>Sequência de Ações</span>
                <span className="text-[10px] bg-white/10 px-1.5 py-0.5 rounded text-white/60">
                    {actions.length}
                </span>
            </div>
            <div className="space-y-1">
                {actions.map((action, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs font-mono p-1.5 rounded hover:bg-white/5 transition-colors group">
                        <span className="text-white/20 w-4">{i + 1}.</span>
                        <span className={`font-bold ${action.type === 'ADD' ? 'text-green-400' : 'text-red-400'}`}>
                            {action.type === 'ADD' ? '+' : '-'}
                        </span>
                        <span className="text-white/60">
                            [{action.x}, {action.y}, {action.z}]
                        </span>
                        {action.value_estimate !== undefined && (
                            <span className="ml-auto text-cyan/60 group-hover:text-cyan">
                                {action.value_estimate.toFixed(2)}
                            </span>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};
