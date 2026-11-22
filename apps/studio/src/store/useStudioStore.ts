import { create } from "zustand";
import type {
  BcType,
  CachedInferOverlay,
  EntryMetadata,
  LoadConfig,
  LocalTopologyReport,
  SessionHandle,
  StudioVoxelState,
} from "../types/contracts";
import { studioAnalyzeLocal } from "../lib/ipc";

export type StudioTool = "paint" | "erase" | "load" | "support";

interface StudioStore {
  resolution: [number, number, number];
  bcType: BcType;
  loadConfig: LoadConfig;
  solidVoxels: Set<number>;
  report: LocalTopologyReport | null;
  overlay: CachedInferOverlay | null;
  tool: StudioTool;
  brushSize: 1 | 3;
  liveXray: boolean;
  session: SessionHandle | null;
  sessionStatus: string;
  sessionStep: number;
  sessionProgress: number;
  sessionError: string | null;
  setResolution: (resolution: [number, number, number]) => void;
  setBcType: (bcType: BcType) => void;
  setLoadConfig: (config: LoadConfig) => void;
  setTool: (tool: StudioTool) => void;
  setBrushSize: (size: 1 | 3) => void;
  setLiveXray: (value: boolean) => void;
  setVoxel: (flatIdx: number, solid: boolean, brush?: 1 | 3) => void;
  clear: () => void;
  loadMetadata: (metadata: EntryMetadata) => void;
  setOverlay: (overlay: CachedInferOverlay | null) => void;
  analyze: () => Promise<void>;
  setSession: (session: SessionHandle | null) => void;
  setSessionStatus: (status: string, step?: number) => void;
  setSessionProgress: (value: number) => void;
  setSessionError: (error: string | null) => void;
  voxelState: () => StudioVoxelState;
}

const defaultLoadConfig: LoadConfig = {
  x: 31,
  y: 16,
  z_start: 3,
  z_end: 5,
  bc_type: "FULL_CLAMP",
};

export const useStudioStore = create<StudioStore>((set, get) => ({
  resolution: [64, 32, 8],
  bcType: "FULL_CLAMP",
  loadConfig: defaultLoadConfig,
  solidVoxels: new Set<number>(),
  report: null,
  overlay: null,
  tool: "paint",
  brushSize: 1,
  liveXray: false,
  session: null,
  sessionStatus: "IDLE",
  sessionStep: 0,
  sessionProgress: 0,
  sessionError: null,
  setResolution: (resolution) => set({ resolution, solidVoxels: new Set() }),
  setBcType: (bcType) => set({ bcType }),
  setLoadConfig: (loadConfig) => set({ loadConfig }),
  setTool: (tool) => set({ tool }),
  setBrushSize: (brushSize) => set({ brushSize }),
  setLiveXray: (liveXray) => set({ liveXray }),
  setVoxel: (flatIdx, solid, brush = get().brushSize) => {
    const [d, h, w] = get().resolution;
    const plane = h * w;
    const x = Math.floor(flatIdx / plane);
    const y = Math.floor((flatIdx % plane) / w);
    const z = flatIdx % w;
    const next = new Set(get().solidVoxels);
    const radius = brush === 3 ? 1 : 0;
    for (let dx = -radius; dx <= radius; dx += 1) {
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dz = -radius; dz <= radius; dz += 1) {
          const nx = x + dx;
          const ny = y + dy;
          const nz = z + dz;
          if (nx < 0 || ny < 0 || nz < 0 || nx >= d || ny >= h || nz >= w) continue;
          const idx = nx * plane + ny * w + nz;
          if (solid) next.add(idx);
          else next.delete(idx);
        }
      }
    }
    set({ solidVoxels: next });
  },
  clear: () => set({ solidVoxels: new Set(), report: null, overlay: null }),
  loadMetadata: (metadata) =>
    set({
      resolution: metadata.resolution,
      bcType: metadata.bc_type,
      loadConfig: metadata.load_config,
      solidVoxels: new Set(),
      report: null,
      overlay: null,
    }),
  setOverlay: (overlay) => set({ overlay }),
  analyze: async () => {
    try {
      const report = await studioAnalyzeLocal(get().voxelState());
      set({ report });
    } catch (error) {
      set({ sessionError: error instanceof Error ? error.message : String(error) });
    }
  },
  setSession: (session) => set({ session, sessionStatus: session?.status ?? "IDLE" }),
  setSessionStatus: (sessionStatus, step) =>
    set((state) => ({ sessionStatus, sessionStep: step ?? state.sessionStep })),
  setSessionProgress: (sessionProgress) => set({ sessionProgress }),
  setSessionError: (sessionError) => set({ sessionError }),
  voxelState: () => {
    const state = get();
    return {
      resolution: state.resolution,
      bc_type: state.bcType,
      load_config: state.loadConfig,
      custom_supports: null,
      custom_forces: null,
      solid_voxels: Array.from(state.solidVoxels).sort((a, b) => a - b),
    };
  },
}));
