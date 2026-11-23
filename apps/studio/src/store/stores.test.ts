import { beforeEach, describe, expect, it } from "vitest";
import type { DataSourceRef, EntryMetadata } from "../types/contracts";
import { useAppStore } from "./useAppStore";
import { useInspectorStore } from "./useInspectorStore";
import { useStudioStore } from "./useStudioStore";
import { useCompareStore } from "./useCompareStore";

const source: DataSourceRef = {
  origin: "local",
  db_id: "/tmp/test.db",
  kind: "harvest_v33",
};

const metadata: EntryMetadata = {
  entry_id: "episode-1",
  kind: "harvest_v33",
  resolution: [8, 4, 2],
  bc_type: "FULL_CLAMP",
  strategy: "FULL_DOMAIN",
  neural_engine: null,
  checkpoint_version: "v3.3",
  seed: 42,
  total_steps: 12,
  steps_phase1: 5,
  steps_phase2: 7,
  final_score: 0.5,
  final_compliance: 412.8,
  final_volume: 0.11,
  final_max_displacement: 41.2,
  real_scale_factor: 500_000,
  stiffness_e: 200e9,
  load_config: { x: 7, y: 2, z_start: 0, z_end: 2 },
  has_blueprint: true,
  has_rho_cont: true,
  has_displacement: true,
  supports: [{ x: 0, y: 1, z: 1, mask_x: true, mask_y: true, mask_z: true }],
  forces: [{ x: 7, y: 2, z: 1, fx: 0, fy: -1, fz: 0 }],
  phase_history: ["GROWTH", "GROWTH", "REFINEMENT"],
  value_history: [0.1, 0.2, 0.3],
  volume_history: [0.05, 0.08, 0.11],
  compliance_history: [null, null, 412.8],
  refutation_steps: [2],
};

beforeEach(() => {
  useAppStore.setState({ mode: "inspector", theme: "dark", libraryOpen: false });
  useInspectorStore.getState().closeEntry();
  useStudioStore.setState({ resolution: [64, 32, 8], solidVoxels: new Set() });
  useCompareStore.setState({
    slotA: null,
    slotB: null,
    stepA: 0,
    stepB: 0,
    metrics: null,
  });
});

describe("app store", () => {
  it("switches modes and theme", () => {
    useAppStore.getState().setMode("studio");
    expect(useAppStore.getState().mode).toBe("studio");
    useAppStore.getState().toggleTheme();
    expect(useAppStore.getState().theme).toBe("tcc-light");
  });
});

describe("inspector store", () => {
  it("opens an entry and clamps the step", () => {
    const store = useInspectorStore.getState();
    store.openEntry(source, "episode-1", metadata);
    expect(useInspectorStore.getState().step).toBe(0);
    useInspectorStore.getState().setStep(999);
    expect(useInspectorStore.getState().step).toBe(11);
    useInspectorStore.getState().setStep(-3);
    expect(useInspectorStore.getState().step).toBe(0);
  });

  it("keeps Blueprint off by default and never flips it on frame changes", () => {
    useInspectorStore.getState().openEntry(source, "episode-1", metadata);
    expect(useInspectorStore.getState().blueprintEnabled).toBe(false);
    useInspectorStore.getState().toggleBlueprint();
    expect(useInspectorStore.getState().blueprintEnabled).toBe(true);
    useInspectorStore.getState().openEntry(source, "episode-1", metadata);
    expect(useInspectorStore.getState().blueprintEnabled).toBe(false);
    useInspectorStore.getState().closeEntry();
    expect(useInspectorStore.getState().blueprintEnabled).toBe(false);
  });

  it("hides deformation and strain lenses when the DB has no FEM fields", () => {
    const trainingData = {
      ...metadata,
      has_displacement: false,
      has_rho_cont: false,
    };
    useInspectorStore.getState().openEntry(source, "warmup", trainingData);
    useInspectorStore.getState().setLens("deformation");
    expect(useInspectorStore.getState().lens).toBe("structure");
    useInspectorStore.getState().setLens("strain");
    expect(useInspectorStore.getState().lens).toBe("structure");
    useInspectorStore.getState().setLens("policy");
    expect(useInspectorStore.getState().lens).toBe("policy");
  });

  it("does not enable Blueprint for datasets without v_constructed", () => {
    useInspectorStore.getState().openEntry(source, "no-blueprint", {
      ...metadata,
      has_blueprint: false,
    });
    useInspectorStore.getState().toggleBlueprint();
    expect(useInspectorStore.getState().blueprintEnabled).toBe(false);
  });

  it("clamps clips and blueprint threshold", () => {
    const store = useInspectorStore.getState();
    store.openEntry(source, "episode-1", metadata);
    useInspectorStore.getState().setClip("x", 100);
    expect(useInspectorStore.getState().clipX).toBe(8);
    useInspectorStore.getState().setBlueprintThreshold(2);
    expect(useInspectorStore.getState().blueprintThreshold).toBe(0.9);
    useInspectorStore.getState().setLens("strain");
    expect(useInspectorStore.getState().lens).toBe("strain");
  });
});

describe("studio store", () => {
  it("paints a 3x3x3 brush and validates serialization", () => {
    useStudioStore.setState({ resolution: [4, 4, 4], solidVoxels: new Set() });
    useStudioStore.getState().setVoxel(2 * 16 + 2 * 4 + 2, true, 3);
    expect(useStudioStore.getState().solidVoxels.size).toBe(27);
    const state = useStudioStore.getState().voxelState();
    expect(state.solid_voxels).toHaveLength(27);
    expect(state.resolution).toEqual([4, 4, 4]);
    useStudioStore.getState().setVoxel(2 * 16 + 2 * 4 + 2, false, 1);
    expect(useStudioStore.getState().solidVoxels.size).toBe(26);
  });
});

describe("compare store", () => {
  it("clamps independent timeline steps", () => {
    useCompareStore.getState().setSlot("a", { source, entryId: "episode-1", metadata });
    useCompareStore.getState().setSlot("b", { source, entryId: "episode-1", metadata });
    useCompareStore.getState().setStep("a", 100);
    expect(useCompareStore.getState().stepA).toBe(11);
    useCompareStore.getState().setStep("b", 3);
    expect(useCompareStore.getState().stepB).toBe(3);
  });
});
