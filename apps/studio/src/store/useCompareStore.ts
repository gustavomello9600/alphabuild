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
  linkMode: "index" | "normalized";
  fraction: number;
  activeSlot: "a" | "b";
  metrics: ComparisonMetrics | null;
  loading: boolean;
  error: string | null;
  setSlot: (slot: "a" | "b", value: CompareSlot | null) => void;
  setStep: (slot: "a" | "b", step: number) => void;
  setLinkedStep: (value: number) => void;
  setFraction: (value: number) => void;
  setNormalized: (value: boolean) => void;
  setSyncCamera: (value: boolean) => void;
  setActiveSlot: (slot: "a" | "b") => void;
  compare: () => Promise<void>;
}

export const useCompareStore = create<CompareStore>((set, get) => ({
  slotA: null,
  slotB: null,
  stepA: 0,
  stepB: 0,
  normalized: false,
  syncCamera: true,
  linkMode: "index",
  fraction: 0,
  activeSlot: "a",
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
  setLinkedStep: (value) => {
    const state = get();
    const maxA = state.slotA ? Math.max(0, state.slotA.metadata.total_steps - 1) : 0;
    const maxB = state.slotB ? Math.max(0, state.slotB.metadata.total_steps - 1) : 0;
    if (state.linkMode === "normalized") {
      const fraction = Math.min(1, Math.max(0, value));
      set({
        fraction,
        stepA: Math.round(fraction * maxA),
        stepB: Math.round(fraction * maxB),
      });
    } else {
      const step = Math.min(Math.max(maxA, maxB), Math.max(0, Math.round(value)));
      set({ stepA: Math.min(step, maxA), stepB: Math.min(step, maxB) });
    }
  },
  setFraction: (value) => {
    const state = get();
    const fraction = Math.min(1, Math.max(0, value));
    const maxA = state.slotA ? Math.max(0, state.slotA.metadata.total_steps - 1) : 0;
    const maxB = state.slotB ? Math.max(0, state.slotB.metadata.total_steps - 1) : 0;
    set({ fraction, stepA: Math.round(fraction * maxA), stepB: Math.round(fraction * maxB) });
  },
  setNormalized: (normalized) =>
    set({ normalized, linkMode: normalized ? "normalized" : "index" }),
  setSyncCamera: (syncCamera) => set({ syncCamera }),
  setActiveSlot: (activeSlot) => set({ activeSlot }),
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
