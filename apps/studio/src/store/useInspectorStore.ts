import { create } from "zustand";
import type {
  DataSourceRef,
  EntryMetadata,
  LensMode,
  MeshOptions,
  ParsedAbMesh,
} from "../types/contracts";
import { DEFAULT_MESH_OPTIONS } from "../types/contracts";
import { clamp } from "../lib/formatters";

interface InspectorStore {
  source: DataSourceRef | null;
  metadata: EntryMetadata | null;
  entryId: string | null;
  step: number;
  playing: boolean;
  playbackSpeed: number;
  lens: LensMode;
  blueprintEnabled: boolean;
  blueprintThreshold: number;
  useRhoCont: boolean;
  clipX: number;
  clipY: number;
  clipZ: number;
  deformScale: number;
  meshOptions: MeshOptions;
  frame: ParsedAbMesh | null;
  loading: boolean;
  error: string | null;
  hoveredVoxelId: number | null;
  openEntry: (source: DataSourceRef, entryId: string, metadata: EntryMetadata) => void;
  setMetadata: (metadata: EntryMetadata) => void;
  setStep: (step: number) => void;
  setPlaying: (playing: boolean) => void;
  togglePlaying: () => void;
  setPlaybackSpeed: (speed: number) => void;
  setLens: (lens: LensMode) => void;
  toggleBlueprint: () => void;
  setBlueprintEnabled: (value: boolean) => void;
  setBlueprintThreshold: (value: number) => void;
  setUseRhoCont: (value: boolean) => void;
  setClip: (axis: "x" | "y" | "z", value: number) => void;
  setDeformScale: (value: number) => void;
  setMeshOptions: (options: Partial<MeshOptions>) => void;
  setFrame: (frame: ParsedAbMesh | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setHoveredVoxelId: (id: number | null) => void;
  closeEntry: () => void;
}

export const useInspectorStore = create<InspectorStore>((set, get) => ({
  source: null,
  metadata: null,
  entryId: null,
  step: 0,
  playing: false,
  playbackSpeed: 1,
  lens: "structure",
  blueprintEnabled: false,
  blueprintThreshold: 0.5,
  useRhoCont: false,
  clipX: 100,
  clipY: 100,
  clipZ: 100,
  deformScale: 0,
  meshOptions: { ...DEFAULT_MESH_OPTIONS },
  frame: null,
  loading: false,
  error: null,
  hoveredVoxelId: null,
  openEntry: (source, entryId, metadata) =>
    set({
      source,
      entryId,
      metadata,
      step: 0,
      playing: false,
      lens: "structure",
      // Blueprint always starts disabled; the user opts in explicitly.
      blueprintEnabled: false,
      useRhoCont: false,
      clipX: metadata.resolution[0],
      clipY: metadata.resolution[1],
      clipZ: metadata.resolution[2],
      deformScale: 0,
      frame: null,
      error: null,
      hoveredVoxelId: null,
      meshOptions: { ...DEFAULT_MESH_OPTIONS },
    }),
  setMetadata: (metadata) => set({ metadata }),
  setStep: (step) => {
    const metadata = get().metadata;
    const max = metadata ? Math.max(0, metadata.total_steps - 1) : 0;
    set({ step: clamp(Math.round(step), 0, max) });
  },
  setPlaying: (playing) => set({ playing }),
  togglePlaying: () => set((state) => ({ playing: !state.playing })),
  setPlaybackSpeed: (playbackSpeed) => set({ playbackSpeed }),
  setLens: (lens) =>
    set((state) => {
      if (
        (lens === "deformation" || lens === "strain") &&
        !state.metadata?.has_displacement
      ) {
        return { lens: state.lens };
      }
      return { lens };
    }),
  toggleBlueprint: () =>
    set((state) =>
      state.metadata?.has_blueprint
        ? { blueprintEnabled: !state.blueprintEnabled }
        : { blueprintEnabled: false },
    ),
  setBlueprintEnabled: (value) => set({ blueprintEnabled: value }),
  setBlueprintThreshold: (blueprintThreshold) =>
    set({ blueprintThreshold: clamp(blueprintThreshold, 0.1, 0.9) }),
  setUseRhoCont: (useRhoCont) => set({ useRhoCont }),
  setClip: (axis, value) => {
    const metadata = get().metadata;
    if (!metadata) return;
    const limits = {
      x: metadata.resolution[0],
      y: metadata.resolution[1],
      z: metadata.resolution[2],
    };
    const clipped = clamp(Math.round(value), 0, limits[axis]);
    if (axis === "x") set({ clipX: clipped });
    if (axis === "y") set({ clipY: clipped });
    if (axis === "z") set({ clipZ: clipped });
  },
  setDeformScale: (deformScale) => set({ deformScale: clamp(deformScale, 0, 20) }),
  setMeshOptions: (options) =>
    set((state) => ({ meshOptions: { ...state.meshOptions, ...options } })),
  setFrame: (frame) => set({ frame }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setHoveredVoxelId: (hoveredVoxelId) => set({ hoveredVoxelId }),
  closeEntry: () =>
    set({
      source: null,
      metadata: null,
      entryId: null,
      step: 0,
      playing: false,
      frame: null,
      error: null,
      hoveredVoxelId: null,
      blueprintEnabled: false,
    }),
}));
