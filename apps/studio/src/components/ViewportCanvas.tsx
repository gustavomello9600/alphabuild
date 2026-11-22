import { useEffect, useRef } from "react";
import { parseAbMesh01 } from "../lib/abmesh";
import { getFrameMesh, prefetchFrames } from "../lib/ipc";
import { setActiveEngine } from "../lib/engineBus";
import { useAppStore } from "../store/useAppStore";
import { useInspectorStore } from "../store/useInspectorStore";
import { ViewportEngine } from "../engine3d/ViewportEngine";

export function ViewportCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<ViewportEngine | null>(null);
  const requestRef = useRef(0);

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
    const engine = engineRef.current;
    if (!engine) return;
    engine.setTheme(theme === "tcc-light");
  }, [theme]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.setFigureMode(figureMode, theme === "tcc-light");
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
    if (!metadata) return;
    engineRef.current?.setBoundaries(metadata.supports, metadata.forces, metadata.bc_type);
  }, [metadata]);

  useEffect(() => {
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
          setFrame(parsed);
          setLoading(false);
          const prefetchOptions = { ...options, include_diff_next: false };
          void prefetchFrames(source, entryId, step + 1, 12, prefetchOptions).catch(() => {
            /* prefetch is opportunistic */
          });
        })
        .catch((cause: unknown) => {
          if (requestRef.current !== request) return;
          setLoading(false);
          setError(cause instanceof Error ? cause.message : String(cause));
        });
    }, 40);
    return () => window.clearTimeout(timeout);
  }, [
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

  return (
    <div className="viewport-root">
      <canvas ref={canvasRef} aria-label="Viewport 3D AlphaBuilder" />
    </div>
  );
}
