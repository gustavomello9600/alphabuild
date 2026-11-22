import { create } from "zustand";
import type {
  Capabilities,
  ComputeConnectionState,
  ComputeProfile,
  HarvestClusterStatus,
  LoadedModelInfo,
  ModelSummary,
  SidecarStatus,
} from "../types/contracts";
import {
  computeActivateProfile,
  computeDeleteProfile,
  computeHarvestStatus,
  computeListModels,
  computeListProfiles,
  computeLoadModel,
  computeSaveProfile,
  computeSidecarControl,
  computeUnloadModel,
} from "../lib/ipc";

interface ComputeStore {
  profiles: ComputeProfile[];
  activeProfileId: string | null;
  connection: ComputeConnectionState | null;
  capabilities: Capabilities | null;
  models: ModelSummary[];
  activeModel: LoadedModelInfo | null;
  sidecar: SidecarStatus | null;
  harvest: HarvestClusterStatus | null;
  loading: boolean;
  error: string | null;
  loadProfiles: () => Promise<void>;
  activateProfile: (profileId: string) => Promise<void>;
  saveProfile: (profile: ComputeProfile) => Promise<void>;
  deleteProfile: (profileId: string) => Promise<void>;
  startSidecar: (pythonBin: string | null, port: number) => Promise<void>;
  stopSidecar: (port: number) => Promise<void>;
  refreshSidecar: (port: number) => Promise<void>;
  loadModels: () => Promise<void>;
  loadModel: (modelId: string, device?: string) => Promise<void>;
  unloadModel: () => Promise<void>;
  refreshHarvest: (supervisorUrl?: string) => Promise<void>;
  setConnection: (connection: ComputeConnectionState) => void;
  setActiveModel: (model: LoadedModelInfo | null) => void;
}

export const useComputeStore = create<ComputeStore>((set, get) => ({
  profiles: [],
  activeProfileId: null,
  connection: null,
  capabilities: null,
  models: [],
  activeModel: null,
  sidecar: null,
  harvest: null,
  loading: false,
  error: null,
  loadProfiles: async () => {
    try {
      const profiles = await computeListProfiles();
      set({ profiles });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
    }
  },
  activateProfile: async (profileId) => {
    set({ loading: true, error: null });
    try {
      const connection = await computeActivateProfile(profileId);
      set({
        activeProfileId: profileId,
        connection,
        capabilities: connection.capabilities,
        activeModel: connection.active_model,
        error: connection.error,
        loading: false,
      });
    } catch (error) {
      set({
        loading: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  },
  saveProfile: async (profile) => {
    await computeSaveProfile(profile);
    await get().loadProfiles();
  },
  deleteProfile: async (profileId) => {
    await computeDeleteProfile(profileId);
    await get().loadProfiles();
  },
  startSidecar: async (pythonBin, port) => {
    set({ loading: true, error: null });
    try {
      const sidecar = await computeSidecarControl("start", pythonBin, port);
      set({ sidecar, loading: false });
    } catch (error) {
      set({ loading: false, error: error instanceof Error ? error.message : String(error) });
    }
  },
  stopSidecar: async (port) => {
    const sidecar = await computeSidecarControl("stop", null, port);
    set({ sidecar });
  },
  refreshSidecar: async (port) => {
    const sidecar = await computeSidecarControl("status", null, port);
    set({ sidecar });
  },
  loadModels: async () => {
    try {
      const models = await computeListModels();
      set({ models, error: null });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error) });
    }
  },
  loadModel: async (modelId, device = "auto") => {
    set({ loading: true, error: null });
    try {
      const activeModel = await computeLoadModel(modelId, device);
      set({ activeModel, loading: false });
    } catch (error) {
      set({
        loading: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  },
  unloadModel: async () => {
    await computeUnloadModel();
    set({ activeModel: null });
  },
  refreshHarvest: async (supervisorUrl) => {
    const harvest = await computeHarvestStatus(supervisorUrl);
    set({ harvest });
  },
  setConnection: (connection) =>
    set({
      connection,
      capabilities: connection.capabilities,
      activeModel: connection.active_model,
      activeProfileId: connection.profile_id,
      error: connection.error,
    }),
  setActiveModel: (activeModel) => set({ activeModel }),
}));
