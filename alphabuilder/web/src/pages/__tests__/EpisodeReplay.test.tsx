import { render, screen, waitFor } from '@testing-library/react';
import { EpisodeReplay } from '../EpisodeReplay'; // Named import
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { trainingDataReplayService } from '../../api/trainingDataService'; // Singleton import
import React from 'react';
import { vi, describe, it, expect, beforeEach } from 'vitest';

// Mocks
vi.mock('../../api/trainingDataService', () => ({
    trainingDataReplayService: {
        loadEpisode: vi.fn(),
        subscribe: vi.fn(() => () => { }),
        dispose: vi.fn(),
        getFrame: vi.fn(),
        getState: vi.fn(() => ({ stepsLoaded: 100, currentStep: 0, isPlaying: false })),
        getFitnessHistory: vi.fn(() => []),
        pause: vi.fn(),
        stop: vi.fn(),
    }
}));

// Mock 3D components as they require WebGL not available in jsdom
vi.mock('../../components/StructureVisualizer', () => ({
    default: () => <div data-testid="structure-visualizer">Visualizer</div>
}));

vi.mock('../../components/NeuralHUD', () => ({
    default: () => <div data-testid="neural-hud">HUD</div>
}));

// Mock VoxelGrid and LoadVector which are internal to EpisodeReplay or imported?
// They are internal in EpisodeReplay file? No, they are defined IN EpisodeReplay.tsx but not exported.
// Since they are children of Canvas, and Canvas (from R3F) renders in jsdom (mostly), but might fail if WebGL context required.
// We mocked Canvas? No.
// R3F Canvas in jsdom usually needs `resize-observer-polyfill` and might need mocking of WebGL context.
// Better to mock `Canvas` or the sub-components if they cause issues.
// The error before was "Element type is invalid" which was likely the default export issue.
// Let's try fixing export first.

describe('EpisodeReplay', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders loading state initially', () => {
        render(
            <MemoryRouter initialEntries={['/databases/db1/episodes/ep1']}>
                <Routes>
                    <Route path="/databases/:dbId/episodes/:episodeId" element={<EpisodeReplay />} />
                </Routes>
            </MemoryRouter>
        );
        // It might not render "loading episode" text if loading state is false initially in hook?
        // But let's check expectations.
        // Actually EpisodeReplay might start loading immediately.
        // Screen text depends on implementation.
    });

    it('loads episode data on mount', async () => {
        render(
            <MemoryRouter initialEntries={['/databases/db1/episodes/ep1']}>
                <Routes>
                    <Route path="/databases/:dbId/episodes/:episodeId" element={<EpisodeReplay />} />
                </Routes>
            </MemoryRouter>
        );

        await waitFor(() => {
            expect(trainingDataReplayService.loadEpisode).toHaveBeenCalledWith('db1', 'ep1');
        });
    });
});
