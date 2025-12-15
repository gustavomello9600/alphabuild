import { Box, Brain, GitBranch, Layers, Target, Move, TrendingUp } from 'lucide-react';

export type ViewMode = 'structure' | 'policy' | 'mcts' | 'combined' | 'decision' | 'deformation' | 'strain';

export const VIEW_MODE_CONFIG: Record<ViewMode, { icon: any; label: string; shortcut: string }> = {
    structure: { icon: Box, label: 'Estrutura', shortcut: '1' },
    policy: { icon: Brain, label: 'Policy', shortcut: '2' },
    mcts: { icon: GitBranch, label: 'MCTS', shortcut: '3' },
    combined: { icon: Layers, label: 'Policy + MCTS', shortcut: '4' },
    decision: { icon: Target, label: 'Decisão', shortcut: '5' },
    deformation: { icon: Move, label: 'Deslocamento', shortcut: 'D' },
    strain: { icon: TrendingUp, label: 'Deformação', shortcut: 'S' },
};

// Channel indices for 7-channel tensor
export const CHANNEL = {
    DENSITY: 0,
    MASK_X: 1,
    MASK_Y: 2,
    MASK_Z: 3,
    FORCE_X: 4,
    FORCE_Y: 5,
    FORCE_Z: 6,
};

// Support colors helper
export const SUPPORT_COLORS = {
    FULL_CLAMP: '#9D4EDD',  // Vivid purple
    RAIL_XY: '#7B2CBF',     // Deep purple
    ROLLER_Y: '#5A189A',    // Dark purple
    PARTIAL: '#3C096C',     // Very dark purple
};

export function getSupportColor(maskX: number, maskY: number, maskZ: number): string | null {
    const hasX = maskX > 0.5;
    const hasY = maskY > 0.5;
    const hasZ = maskZ > 0.5;

    if (!hasX && !hasY && !hasZ) return null;

    if (hasX && hasY && hasZ) {
        return SUPPORT_COLORS.FULL_CLAMP;
    } else if (hasX && hasY && !hasZ) {
        return SUPPORT_COLORS.RAIL_XY;
    } else if (!hasX && hasY && !hasZ) {
        return SUPPORT_COLORS.ROLLER_Y;
    } else {
        return SUPPORT_COLORS.PARTIAL;
    }
}
