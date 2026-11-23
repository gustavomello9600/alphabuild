import { create } from "zustand";
import type {
  BcType,
  CachedInferOverlay,
  EntryMetadata,
  ForceVector,
  LoadConfig,
  LocalTopologyReport,
  ParsedAbMesh,
  SessionHandle,
  StepFrameMetadata,
  StudioVoxelState,
  SupportVoxel,
} from "../types/contracts";
import { studioAnalyzeLocal } from "../lib/ipc";

export type StudioTool = "paint" | "erase" | "load" | "support";

export interface SessionHistoryEntry {
  step: number;
  metadata: StepFrameMetadata;
}

export interface StudioTopologyView {
  is_connected: boolean;
  n_islands: number;
  loose_voxels: number;
  volume_fraction: number;
  connected_load_fraction: number;
  legal_add_count: number | null;
  legal_remove_count: number | null;
  source: "session" | "local";
}

export function resolveStudioTopology(input: {
  session: SessionHandle | null;
  sessionFrame: ParsedAbMesh | null;
  sessionViewFrame: ParsedAbMesh | null;
  sessionStatus: string;
  sessionFollowLive: boolean;
  report: LocalTopologyReport | null;
  hasLocalEdits: boolean;
}): StudioTopologyView | null {
  const displayFrame = input.session ? (input.sessionViewFrame ?? input.sessionFrame) : null;
  const editingPausedLive =
    input.sessionStatus === "PAUSED" && input.sessionFollowLive && input.hasLocalEdits;
  if (displayFrame && !editingPausedLive) {
    const meta = displayFrame.metadata;
    return {
      source: "session",
      is_connected: meta.is_connected,
      n_islands: meta.n_islands,
      loose_voxels: meta.loose_voxels,
      volume_fraction: meta.volume_fraction,
      connected_load_fraction:
        meta.is_connected && meta.connected_load_fraction === 0 ? 1 : meta.connected_load_fraction,
      legal_add_count: meta.legal_add_count ?? null,
      legal_remove_count: meta.legal_remove_count ?? null,
    };
  }
  if (input.report) {
    return {
      source: "local",
      is_connected: input.report.is_connected,
      n_islands: input.report.n_islands,
      loose_voxels: input.report.loose_voxels,
      volume_fraction: input.report.volume_fraction,
      connected_load_fraction: input.report.connected_load_fraction,
      legal_add_count: input.report.legal_add_count,
      legal_remove_count: input.report.legal_remove_count,
    };
  }
  return null;
}

interface StudioStore {
  resolution: [number, number, number];
  bcType: BcType;
  loadConfig: LoadConfig;
  solidVoxels: Set<number>;
  customSupports: SupportVoxel[] | null;
  customForces: ForceVector[] | null;
  inferenceValue: number | null;
  inferenceMs: number | null;
  inferenceDevice: string | null;
  inferenceError: string | null;
  report: LocalTopologyReport | null;
  overlay: CachedInferOverlay | null;
  tool: StudioTool;
  brushSize: 1 | 3;
  liveXray: boolean;
  session: SessionHandle | null;
  sessionFrame: ParsedAbMesh | null;
  sessionHistory: SessionHistoryEntry[];
  sessionViewFrame: ParsedAbMesh | null;
  sessionViewStep: number | null;
  sessionFollowLive: boolean;
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
  setSolidVoxels: (voxels: Set<number>) => void;
  setSupportVoxel: (flatIdx: number, enabled: boolean) => void;
  setLoadFromVoxel: (x: number, y: number, z: number) => void;
  setInference: (value: number | null, ms: number | null, device: string | null) => void;
  setInferenceError: (error: string | null) => void;
  clear: () => void;
  loadMetadata: (metadata: EntryMetadata) => void;
  setOverlay: (overlay: CachedInferOverlay | null) => void;
  analyze: () => Promise<void>;
  setSession: (session: SessionHandle | null) => void;
  setSessionFrame: (frame: ParsedAbMesh | null) => void;
  appendSessionStep: (step: number, metadata: StepFrameMetadata) => void;
  setSessionViewFrame: (frame: ParsedAbMesh | null, step: number | null) => void;
  setSessionFollowLive: (value: boolean) => void;
  clearSessionHistory: () => void;
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
  customSupports: null,
  customForces: null,
  inferenceValue: null,
  inferenceMs: null,
  inferenceDevice: null,
  inferenceError: null,
  report: null,
  overlay: null,
  tool: "paint",
  brushSize: 1,
  liveXray: false,
  session: null,
  sessionFrame: null,
  sessionHistory: [],
  sessionViewFrame: null,
  sessionViewStep: null,
  sessionFollowLive: true,
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
  setSolidVoxels: (solidVoxels) => set({ solidVoxels, report: null }),
  setSupportVoxel: (flatIdx, enabled) => {
    const state = get();
    const dims = state.resolution;
    const plane = dims[1] * dims[2];
    const x = Math.floor(flatIdx / plane);
    const y = Math.floor((flatIdx % plane) / dims[2]);
    const z = flatIdx % dims[2];
    const existing = new Map((state.customSupports ?? []).map((item) => [`${item.x},${item.y},${item.z}`, item]));
    const key = `${x},${y},${z}`;
    if (enabled) existing.set(key, { x, y, z, mask_x: true, mask_y: true, mask_z: true });
    else existing.delete(key);
    set({
      customSupports: Array.from(existing.values()),
      bcType: "CUSTOM",
    });
  },
  setLoadFromVoxel: (x, y, z) => {
    const state = get();
    const [d, h, w] = state.resolution;
    set({
      loadConfig: {
        ...state.loadConfig,
        x: Math.min(d - 1, Math.max(0, Math.round(x))),
        y: Math.min(h - 1, Math.max(0, Math.round(y))),
        z_start: Math.min(w - 1, Math.max(0, Math.round(z))),
        z_end: Math.min(w - 1, Math.max(0, Math.round(z) + 1)),
      },
    });
  },
  setInference: (inferenceValue, inferenceMs, inferenceDevice) =>
    set({ inferenceValue, inferenceMs, inferenceDevice, inferenceError: null }),
  setInferenceError: (inferenceError) => set({ inferenceError }),
  clear: () => set({ solidVoxels: new Set(), report: null, overlay: null, customSupports: null, customForces: null, inferenceValue: null }),
  loadMetadata: (metadata) =>
    set({
      resolution: metadata.resolution,
      bcType: metadata.bc_type,
      loadConfig: metadata.load_config,
      solidVoxels: new Set(),
      customSupports: null,
      customForces: null,
      report: null,
      overlay: null,
      inferenceValue: null,
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
  setSession: (session) =>
    set((state) => {
      if (session?.session_id === state.session?.session_id) {
        return { session, sessionStatus: session?.status ?? "IDLE" };
      }
      return {
        session,
        sessionStatus: session?.status ?? "IDLE",
        sessionFrame: null,
        sessionHistory: [],
        sessionViewFrame: null,
        sessionViewStep: null,
        sessionFollowLive: true,
      };
    }),
  setSessionFrame: (sessionFrame) => set({ sessionFrame }),
  appendSessionStep: (step, metadata) =>
    set((state) => {
      const next = state.sessionHistory.filter((entry) => entry.step !== step);
      next.push({ step, metadata });
      next.sort((a, b) => a.step - b.step);
      return { sessionHistory: next };
    }),
  setSessionViewFrame: (sessionViewFrame, sessionViewStep) =>
    set({ sessionViewFrame, sessionViewStep }),
  setSessionFollowLive: (sessionFollowLive) => set({ sessionFollowLive }),
  clearSessionHistory: () =>
    set({
      sessionFrame: null,
      sessionHistory: [],
      sessionViewFrame: null,
      sessionViewStep: null,
      sessionFollowLive: true,
    }),
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
      custom_supports: state.customSupports,
      custom_forces: state.customForces,
      solid_voxels: Array.from(state.solidVoxels).sort((a, b) => a - b),
    };
  },
}));
