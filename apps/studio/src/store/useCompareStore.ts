import { create } from "zustand";
import type {
  ComparisonMetrics,
  DataSourceRef,
  EntryMetadata,
  FrameRef,
} from "../types/contracts";
import { computeAbComparison } from "../lib/ipc";

export interface CompareSlot {
  source: DataSourceRef;
  entryId: string;
  metadata: EntryMetadata;
}

interface CompareStore {
  slotA: CompareSlot | null;
  slotB: CompareSlot | null;
  stepA: number;
  stepB: number;
  normalized: boolean;
  syncCamera: boolean;
  metrics: ComparisonMetrics | null;
  loading: boolean;
  error: string | null;
  setSlot: (slot: "a" | "b", value: CompareSlot | null) => void;
  setStep: (slot: "a" | "b", step: number) => void;
  setNormalized: (value: boolean) => void;
  setSyncCamera: (value: boolean) => void;
  compare: () => Promise<void>;
}

export const useCompareStore = create<CompareStore>((set, get) => ({
  slotA: null,
  slotB: null,
  stepA: 0,
  stepB: 0,
  normalized: false,
  syncCamera: true,
  metrics: null,
  loading: false,
  error: null,
  setSlot: (slot, value) => {
    const max = value ? Math.max(0, value.metadata.total_steps - 1) : 0;
    if (slot === "a") set({ slotA: value, stepA: Math.min(get().stepA, max), metrics: null });
    else set({ slotB: value, stepB: Math.min(get().stepB, max), metrics: null });
  },
  setStep: (slot, step) => {
    const current = slot === "a" ? get().slotA : get().slotB;
    const max = current ? Math.max(0, current.metadata.total_steps - 1) : 0;
    const clamped = Math.min(max, Math.max(0, Math.round(step)));
    if (slot === "a") set({ stepA: clamped });
    else set({ stepB: clamped });
  },
  setNormalized: (normalized) => set({ normalized }),
  setSyncCamera: (syncCamera) => set({ syncCamera }),
  compare: async () => {
    const state = get();
    if (!state.slotA || !state.slotB) {
      set({ error: "Carregue um episódio em cada slot para comparar." });
      return;
    }
    set({ loading: true, error: null });
    try {
      const a: FrameRef = {
        source: state.slotA.source,
        entry_id: state.slotA.entryId,
        step: state.stepA,
      };
      const b: FrameRef = {
        source: state.slotB.source,
        entry_id: state.slotB.entryId,
        step: state.stepB,
      };
      const metrics = await computeAbComparison(a, b);
      set({ metrics, loading: false });
    } catch (error) {
      set({
        loading: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  },
}));
