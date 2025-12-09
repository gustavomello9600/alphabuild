import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TrainingDataReplayService, fetchDatabases, fetchEpisodeMetadata } from '../trainingDataService';

// Mock fetch
const fetchMock = vi.fn();
vi.stubGlobal('fetch', fetchMock);

describe('TrainingDataReplayService', () => {
    let service: TrainingDataReplayService;

    beforeEach(() => {
        service = new TrainingDataReplayService();
        service.stop();
        vi.clearAllMocks();
    });

    it('should fetch databases (standalone)', async () => {
        const mockDbs = [{ id: 'db1', name: 'Database 1' }];
        (global.fetch as any).mockResolvedValueOnce({
            ok: true,
            json: async () => mockDbs,
        });

        // Use the imported function directly
        const dbs = await fetchDatabases();
        expect(dbs).toEqual(mockDbs);
        expect(fetchMock).toHaveBeenCalledWith(expect.stringMatching(/^http:\/\/localhost:8000\/databases\?t=\d+$/));
    });

    it('should fetch episode metadata (standalone)', async () => {
        const mockMeta = { episode_id: 'ep1', total_steps: 100 };
        fetchMock.mockResolvedValueOnce({
            ok: true,
            json: async () => mockMeta,
        });

        const meta = await fetchEpisodeMetadata('db1', 'ep1');
        expect(meta).toEqual(mockMeta);
        expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('http://localhost:8000/databases/db1/episodes/ep1/metadata'));
    });

    it('should load frames and manage buffer', async () => {
        // Setup metadata response
        const mockMeta = { episode_id: 'ep1', total_steps: 200 };
        // Setup frames response
        const mockFrames = Array.from({ length: 50 }, (_, i) => ({
            step: i,
            tensor_shape: [1, 1, 1, 1],
            tensor_data: [0],
            phase: 'GROWTH',
            fitness_score: 0.5
        }));

        (fetchMock as any)
            .mockResolvedValueOnce({ ok: true, json: async () => mockMeta }) // metadata
            .mockResolvedValueOnce({ ok: true, json: async () => mockFrames }); // frames 0-25 (FETCH_CHUNK_SIZE=25)

        await service.loadEpisode('db1', 'ep1');

        // Check internal state using getState
        const state = service.getState();
        expect(state.episodeId).toBe('ep1');
        expect(state.stepsLoaded).toBe(200);

        // It should have buffered
        // We can't access framesMap directly (private), but we can check if subscribers get data
        // Or check fetch calls
        expect(fetchMock).toHaveBeenCalledTimes(2); // metadata + 1 chunk
    });
});
