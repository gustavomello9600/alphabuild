import { useEffect, useRef } from "react";
import { parseAbMesh01, flatToCoord } from "../lib/abmesh";
import {
  computeInferXray,
  getFrameMesh,
  prefetchFrames,
  studioBuildMesh,
} from "../lib/ipc";
import { setActiveEngine } from "../lib/engineBus";
import { useAppStore } from "../store/useAppStore";
import { useComputeStore } from "../store/useComputeStore";
import { useInspectorStore } from "../store/useInspectorStore";
import { useStudioStore } from "../store/useStudioStore";
import { ViewportEngine } from "../engine3d/ViewportEngine";
import type {
  BcType,
  ForceVector,
  LoadConfig,
  StudioVoxelState,
  SupportVoxel,
} from "../types/contracts";

function supportPlane(
  state: StudioVoxelState,
  dims: [number, number, number],
  bcType: BcType,
): SupportVoxel[] {
  if (state.custom_supports && state.custom_supports.length > 0) {
    return state.custom_supports;
  }
  if (bcType === "CUSTOM") return [];
  const supports: SupportVoxel[] = [];
  const [, h, w] = dims;
  for (let y = 0; y < h; y += 1) {
    for (let z = 0; z < w; z += 1) {
      supports.push({
        x: 0,
        y,
        z,
        mask_x: true,
        mask_y: true,
        mask_z: bcType !== "RAIL_XY",
      });
    }
  }
  return supports;
}

function loadForces(
  state: StudioVoxelState,
  dims: [number, number, number],
): ForceVector[] {
  if (state.custom_forces && state.custom_forces.length > 0) return state.custom_forces;
  const [d, h, w] = dims;
  const config: LoadConfig = state.load_config;
  const lx = Math.min(d - 1, Math.max(0, Math.trunc(config.x)));
  const lzCenter = (config.z_start + config.z_end) / 2;
  const yMin = Math.max(0, Math.trunc(config.y - 1));
  const yMax = Math.min(h, Math.trunc(config.y + 1) + 1);
  const zMin = Math.max(0, Math.trunc(lzCenter - 1));
  const zMax = Math.min(w, Math.trunc(lzCenter + 1) + 1);
  const forces: ForceVector[] = [];
  for (let y = yMin; y < yMax; y += 1) {
    for (let z = zMin; z < zMax; z += 1) {
      forces.push({ x: lx, y, z, fx: 0, fy: -1, fz: 0 });
    }
  }
  return forces;
}

export function ViewportCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<ViewportEngine | null>(null);
  const requestRef = useRef(0);
  const versionRef = useRef(0);

  const mode = useAppStore((state) => state.mode);
  const theme = useAppStore((state) => state.theme);
  const figureMode = useAppStore((state) => state.figureMode);
  const source = useInspectorStore((state) => state.source);
  const entryId = useInspectorStore((state) => state.entryId);
  const metadata = useInspectorStore((state) => state.metadata);
  const step = useInspectorStore((state) => state.step);
  const lens = useInspectorStore((state) => state.lens);
  const blueprint = useInspectorStore((state) => state.blueprintEnabled);
  const blueprintThreshold = useInspectorStore((state) => state.blueprintThreshold);
  const useRhoCont = useInspectorStore((state) => state.useRhoCont);
  const clipX = useInspectorStore((state) => state.clipX);
  const clipY = useInspectorStore((state) => state.clipY);
  const clipZ = useInspectorStore((state) => state.clipZ);
  const deformScale = useInspectorStore((state) => state.deformScale);
  const meshOptions = useInspectorStore((state) => state.meshOptions);
  const setFrame = useInspectorStore((state) => state.setFrame);
  const setLoading = useInspectorStore((state) => state.setLoading);
  const setError = useInspectorStore((state) => state.setError);
  const setHoveredVoxelId = useInspectorStore((state) => state.setHoveredVoxelId);

  const studioResolution = useStudioStore((state) => state.resolution);
  const studioBc = useStudioStore((state) => state.bcType);
  const studioLoad = useStudioStore((state) => state.loadConfig);
  const studioSolid = useStudioStore((state) => state.solidVoxels);
  const studioSupports = useStudioStore((state) => state.customSupports);
  const studioForces = useStudioStore((state) => state.customForces);
  const studioOverlay = useStudioStore((state) => state.overlay);
  const sessionViewFrame = useStudioStore((state) => state.sessionViewFrame);
  const liveXray = useStudioStore((state) => state.liveXray);
  const sessionStatus = useStudioStore((state) => state.sessionStatus);
  const connected = useComputeStore((state) => state.connection?.connected ?? false);
  const activeModel = useComputeStore((state) => state.activeModel);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const engine = new ViewportEngine(canvas);
    engineRef.current = engine;
    setActiveEngine(engine);
    engine.setHoverHandler((pick) => setHoveredVoxelId(pick?.voxelId ?? null));
    return () => {
      engine.dispose();
      engineRef.current = null;
      setActiveEngine(null);
    };
  }, [setHoveredVoxelId]);

  useEffect(() => {
    engineRef.current?.setTheme(theme === "tcc-light");
  }, [theme]);

  useEffect(() => {
    engineRef.current?.setFigureMode(figureMode, theme === "tcc-light");
  }, [figureMode, theme]);

  useEffect(() => {
    engineRef.current?.setLens(lens);
  }, [lens]);

  useEffect(() => {
    engineRef.current?.setBlueprintVisible(blueprint);
  }, [blueprint]);

  useEffect(() => {
    engineRef.current?.setRhoContinuous(useRhoCont);
  }, [useRhoCont]);

  useEffect(() => {
    engineRef.current?.setClipPlanes(clipX, clipY, clipZ);
  }, [clipX, clipY, clipZ]);

  useEffect(() => {
    engineRef.current?.setDeformScale(deformScale);
  }, [deformScale]);

  useEffect(() => {
    if (mode !== "inspector") return;
    if (!metadata) return;
    engineRef.current?.setBoundaries(metadata.supports, metadata.forces, metadata.bc_type);
  }, [mode, metadata]);

  // Inspector playback fetch.
  useEffect(() => {
    if (mode !== "inspector") return;
    if (!source || !entryId || !metadata) {
      engineRef.current?.clearFrame();
      setFrame(null);
      return;
    }
    const request = ++requestRef.current;
    const options = {
      density_threshold: meshOptions.density_threshold,
      blueprint_threshold: blueprintThreshold,
      use_rho_cont: useRhoCont,
      include_diff_next: meshOptions.include_diff_next,
      include_blueprint: true,
    };
    setLoading(true);
    setError(null);
    const timeout = window.setTimeout(() => {
      void getFrameMesh(source, entryId, step, options)
        .then((buffer) => {
          if (requestRef.current !== request) return;
          const parsed = parseAbMesh01(buffer);
          engineRef.current?.applyFrame(parsed);
          engineRef.current?.setBoundaries(metadata.supports, metadata.forces, metadata.bc_type);
          setFrame(parsed);
          setLoading(false);
          void prefetchFrames(source, entryId, step + 1, 12, {
            ...options,
            include_diff_next: false,
          }).catch(() => {
            /* opportunistic */
          });
        })
        .catch((cause: unknown) => {
          if (requestRef.current !== request) return;
          setLoading(false);
          setError(cause instanceof Error ? cause.message : String(cause));
        });
    }, 16);
    return () => window.clearTimeout(timeout);
  }, [
    mode,
    source,
    entryId,
    metadata,
    step,
    meshOptions.density_threshold,
    meshOptions.include_diff_next,
    blueprintThreshold,
    useRhoCont,
    setFrame,
    setLoading,
    setError,
  ]);

  // Studio mesh rebuild.
  useEffect(() => {
    if (mode !== "studio") return;
    const version = ++versionRef.current;
    const store = useStudioStore.getState();
    const state = store.voxelState();
    const timer = window.setTimeout(() => {
      void studioBuildMesh(state, store.overlay)
        .then((buffer) => {
          if (versionRef.current !== version) return;
          engineRef.current?.applyFrame(parseAbMesh01(buffer));
        })
        .catch((cause: unknown) => {
          if (versionRef.current !== version) return;
          store.setInferenceError(
            cause instanceof Error ? cause.message : String(cause),
          );
        });
    }, 24);
    return () => window.clearTimeout(timer);
  }, [
    mode,
    studioResolution,
    studioBc,
    studioLoad,
    studioSolid,
    studioSupports,
    studioForces,
    studioOverlay,
  ]);

  useEffect(() => {
    if (mode !== "studio") return;
    const state = useStudioStore.getState().voxelState();
    const dims = studioResolution;
    engineRef.current?.setBoundaries(
      supportPlane(state, dims, studioBc),
      loadForces(state, dims),
      studioBc,
    );
  }, [mode, studioResolution, studioBc, studioLoad, studioSupports, studioForces]);

  useEffect(() => {
    if (mode !== "studio" || !sessionViewFrame) return;
    engineRef.current?.applyFrame(sessionViewFrame);
  }, [mode, sessionViewFrame]);

  useEffect(() => {
    if (mode !== "studio") return;
    const timer = window.setTimeout(() => {
      void useStudioStore.getState().analyze();
    }, 140);
    return () => window.clearTimeout(timer);
  }, [mode, studioSolid, studioBc, studioLoad, studioSupports, studioForces]);

  // Raio-X Live (120 ms debounce), only while connected and not mid-MCTS step.
  useEffect(() => {
    if (mode !== "studio" || !liveXray || !connected || !activeModel) return;
    if (sessionStatus === "RUNNING") return;
    const version = versionRef.current;
    const timer = window.setTimeout(() => {
      const store = useStudioStore.getState();
      const state = store.voxelState();
      void computeInferXray({
        resolution: state.resolution,
        bc_type: state.bc_type,
        load_config: state.load_config,
        custom_supports: state.custom_supports,
        custom_forces: state.custom_forces,
        solid_voxels: state.solid_voxels,
        include_blueprint: true,
        policy_mode: "masked_probs",
        policy_temperature: 1.0,
        policy_top_k: 512,
      })
        .then((buffer) => {
          if (versionRef.current !== version) return;
          const parsed = parseAbMesh01(buffer);
          engineRef.current?.applyFrame(parsed);
          store.setInference(
            parsed.metadata.value,
            parsed.metadata.inference_ms ?? null,
            (parsed.metadata as { device?: string }).device ?? null,
          );
        })
        .catch((cause: unknown) => {
          store.setInferenceError(cause instanceof Error ? cause.message : String(cause));
        });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [
    mode,
    liveXray,
    connected,
    activeModel,
    sessionStatus,
    studioSolid,
    studioBc,
    studioLoad,
    studioSupports,
    studioForces,
  ]);

  // Direct 3D manipulation: paint/erase/support/load drag with integer snapping.
  useEffect(() => {
    if (mode !== "studio") return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    let editStart: { x: number; y: number; moved: boolean } | null = null;
    let loadDragging = false;

    const applyAt = (event: PointerEvent) => {
      const engine = engineRef.current;
      if (!engine) return;
      const store = useStudioStore.getState();
      if (store.sessionStatus === "RUNNING") return;
      // Replay of an older live frame is read-only; return to the live edge to edit.
      if (store.session && !store.sessionFollowLive) return;
      if (store.tool === "load") {
        const pick = engine.pickVoxel(event);
        if (pick) {
          const [x, y, z] = flatToCoord(pick.voxelId, engine.getDims());
          store.setLoadFromVoxel(x, y, z);
        }
        return;
      }
      const pick = engine.pickVoxel(event);
      if (!pick) {
        if (store.tool === "paint") {
          const point = engine.pickFloor(event);
          if (point) {
            const [d, h, w] = engine.getDims();
            const x = Math.floor(point.x + d / 2);
            const z = Math.floor(point.z + w / 2);
            if (x >= 0 && x < d && z >= 0 && z < w) {
              store.setVoxel(x * h * w + z, true);
            }
          }
        }
        return;
      }
      if (store.tool === "support") {
        store.setSupportVoxel(pick.voxelId, true);
        return;
      }
      if (store.tool === "erase") {
        store.setVoxel(pick.voxelId, false);
        return;
      }
      const [x, y, z] = flatToCoord(pick.voxelId, engine.getDims());
      const [d, h, w] = engine.getDims();
      const nx = Math.min(d - 1, Math.max(0, x + Math.round(pick.normal.x)));
      const ny = Math.min(h - 1, Math.max(0, y + Math.round(pick.normal.y)));
      const nz = Math.min(w - 1, Math.max(0, z + Math.round(pick.normal.z)));
      store.setVoxel(nx * h * w + ny * w + nz, true);
    };

    const handleDown = (event: PointerEvent) => {
      if (event.button !== 0) return;
      const store = useStudioStore.getState();
      if (store.tool === "load") {
        // Load handles are intentionally draggable; they never create material.
        loadDragging = true;
        applyAt(event);
        return;
      }
      // Paint/erase/support are click-to-edit. A left drag belongs to
      // OrbitControls, so it must never mutate the voxel grid.
      editStart = { x: event.clientX, y: event.clientY, moved: false };
    };
    const handleMove = (event: PointerEvent) => {
      if (loadDragging) {
        applyAt(event);
        return;
      }
      if (!editStart) return;
      const dx = event.clientX - editStart.x;
      const dy = event.clientY - editStart.y;
      if (dx * dx + dy * dy > 16) editStart.moved = true;
    };
    const handleUp = (event: PointerEvent) => {
      if (loadDragging) {
        loadDragging = false;
        return;
      }
      const start = editStart;
      editStart = null;
      if (!start || start.moved) return;
      applyAt(event);
    };
    const handleCancel = () => {
      editStart = null;
      loadDragging = false;
    };
    canvas.addEventListener("pointerdown", handleDown);
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
    window.addEventListener("pointercancel", handleCancel);
    return () => {
      canvas.removeEventListener("pointerdown", handleDown);
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
      window.removeEventListener("pointercancel", handleCancel);
    };
  }, [mode]);

  return (
    <div className="viewport-root">
      <canvas ref={canvasRef} aria-label="Viewport 3D AlphaBuilder" />
    </div>
  );
}
