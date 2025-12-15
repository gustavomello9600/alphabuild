import React, { useEffect } from 'react';
import { VIEW_MODE_CONFIG, type ViewMode } from './types';

interface ViewModeToolbarProps {
    mode: ViewMode;
    setMode: (m: ViewMode) => void;
    showMCTS?: boolean;
}

export const ViewModeToolbar: React.FC<ViewModeToolbarProps> = ({ mode, setMode, showMCTS = true }) => {
    // Keyboard shortcuts
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            // Ignore if input/textarea is focused
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

            if (e.key === '1') setMode('structure');
            if (e.key === '2') setMode('policy');
            if (showMCTS) {
                if (e.key === '3') setMode('mcts');
                if (e.key === '4') setMode('combined');
            }
            if (e.key === '5') setMode('decision');
            if (e.key === 'd' || e.key === 'D') setMode('deformation');
            if (e.key === 's' || e.key === 'S') setMode('strain');
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [setMode, showMCTS]);

    const visibleModes = Object.entries(VIEW_MODE_CONFIG).filter(([key]) => {
        if (!showMCTS && (key === 'mcts' || key === 'combined')) return false;
        return true;
    });

    return (
        <div className="flex gap-1 bg-black/60 backdrop-blur border border-white/10 rounded-xl p-1 pointer-events-auto">
            {(visibleModes as [ViewMode, typeof VIEW_MODE_CONFIG[ViewMode]][]).map(
                ([key, config]) => {
                    const Icon = config.icon;
                    const isActive = mode === key;
                    return (
                        <button
                            key={key}
                            onClick={() => setMode(key)}
                            className={`
                                p-2 rounded-lg transition-all relative group
                                ${isActive
                                    ? 'bg-cyan/20 text-cyan'
                                    : 'text-white/40 hover:text-white hover:bg-white/10'
                                }
                            `}
                            title={`${config.label} (${config.shortcut})`}
                        >
                            <Icon size={18} />
                            {/* Tooltip */}
                            <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 px-2 py-1 bg-matter border border-white/10 rounded text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
                                {config.label}
                                <span className="ml-1 text-cyan">{config.shortcut}</span>
                            </div>
                        </button>
                    );
                }
            )}
        </div>
    );
};
