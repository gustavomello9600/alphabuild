import { useEffect } from "react";
import { Database, FolderOpen } from "lucide-react";
import { TopBar } from "./components/TopBar";
import { ViewportCanvas } from "./components/ViewportCanvas";
import { LibraryDrawer } from "./components/LibraryDrawer";
import { LensToolbar } from "./components/LensToolbar";
import { TimelineDock } from "./components/TimelineDock";
import { LeftContextDock } from "./components/LeftContextDock";
import { RightTelemetryDock } from "./components/RightTelemetryDock";
import { VoxelTooltip } from "./components/VoxelTooltip";
import { ExportModal } from "./components/ExportModal";
import { ComputePopover } from "./components/ComputePopover";
import { CommandPalette } from "./components/CommandPalette";
import { StudioPanel } from "./components/StudioPanel";
import { CompareView } from "./components/CompareView";
import { useAppStore } from "./store/useAppStore";
import { useInspectorStore } from "./store/useInspectorStore";
import { useComputeStore } from "./store/useComputeStore";
import { useCompareStore } from "./store/useCompareStore";
import { useStudioStore } from "./store/useStudioStore";
import { parseAbMesh01 } from "./lib/abmesh";
import { getActiveEngine } from "./lib/engineBus";
import {
  isTauriRuntime,
  onComputeStatus,
  onSessionProgress,
  onSessionState,
  onSessionStepReady,
  sessionGetStepMesh,
} from "./lib/ipc";
import type { LensMode } from "./types/contracts";

const lensByKey: Record<string, LensMode> = {
  "1": "structure",
  "2": "policy",
  "3": "mcts",
  "4": "refutation",
  "5": "decision",
  "6": "deformation",
};

export default function App() {
  const mode = useAppStore((state) => state.mode);
  const theme = useAppStore((state) => state.theme);
  const figureMode = useAppStore((state) => state.figureMode);
  const setFigureMode = useAppStore((state) => state.setFigureMode);
  const setPaletteOpen = useAppStore((state) => state.setPaletteOpen);
  const setLibraryOpen = useAppStore((state) => state.setLibraryOpen);
  const setExportOpen = useAppStore((state) => state.setExportOpen);
  const setComputeOpen = useAppStore((state) => state.setComputeOpen);
  const metadata = useInspectorStore((state) => state.metadata);
  const setComputeConnection = useComputeStore((state) => state.setConnection);
  const loadProfiles = useComputeStore((state) => state.loadProfiles);
  const setSessionStatus = useStudioStore((state) => state.setSessionStatus);
  const setSessionProgress = useStudioStore((state) => state.setSessionProgress);
  const setSessionFrame = useStudioStore((state) => state.setSessionFrame);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    void loadProfiles();
    if (!isTauriRuntime()) return;
    const unlisteners: Array<Promise<() => void>> = [
      onComputeStatus((event) => setComputeConnection(event)),
      onSessionState((event) => setSessionStatus(event.status, event.step)),
      onSessionProgress((event) => {
        setSessionProgress(
          event.sims_total > 0 ? (event.sims_done / event.sims_total) * 100 : 0,
        );
      }),
      onSessionStepReady((event) => {
        setSessionStatus("RUNNING", event.step);
        setSessionProgress(100);
        void sessionGetStepMesh(event.session_id, event.step)
          .then((buffer) => {
            const parsed = parseAbMesh01(buffer);
            setSessionFrame(parsed);
            if (useAppStore.getState().mode === "studio") {
              getActiveEngine()?.applyFrame(parsed);
            }
          })
          .catch(() => {
            /* the Tauri bridge will retry via events */
          });
      }),
    ];
    return () => {
      void Promise.all(unlisteners).then((callbacks) => callbacks.forEach((fn) => fn()));
    };
  }, [
    loadProfiles,
    setComputeConnection,
    setSessionProgress,
    setSessionFrame,
    setSessionStatus,
  ]);

  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      const modifier = event.ctrlKey || event.metaKey;
      if (modifier && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen(true);
        return;
      }
      if (modifier) return;
      const inspector = useInspectorStore.getState();
      switch (event.key) {
        case "Escape":
          setPaletteOpen(false);
          setLibraryOpen(false);
          setExportOpen(false);
          setComputeOpen(false);
          break;
        case "l":
        case "L":
          setLibraryOpen(!useAppStore.getState().libraryOpen);
          break;
        case "b":
        case "B":
          inspector.toggleBlueprint();
          break;
        case "d":
        case "D":
          inspector.setLens("deformation");
          break;
        case "s":
        case "S":
          inspector.setLens("strain");
          break;
        case "p":
        case "P":
          setFigureMode(!useAppStore.getState().figureMode);
          break;
        case "e":
        case "E":
          setExportOpen(true);
          break;
        case " ":
          if (inspector.metadata) {
            event.preventDefault();
            inspector.togglePlaying();
          }
          break;
        case "ArrowLeft":
          if (inspector.metadata) {
            inspector.setStep(inspector.step - (event.shiftKey ? 10 : 1));
          }
          break;
        case "ArrowRight":
          if (inspector.metadata) {
            inspector.setStep(inspector.step + (event.shiftKey ? 10 : 1));
          }
          break;
        case "[":
          if (inspector.metadata) {
            const previous = [...inspector.metadata.refutation_steps]
              .reverse()
              .find((step) => step < inspector.step);
            if (previous !== undefined) inspector.setStep(previous);
          }
          break;
        case "]":
          if (inspector.metadata) {
            const next = inspector.metadata.refutation_steps.find(
              (step) => step > inspector.step,
            );
            if (next !== undefined) inspector.setStep(next);
          }
          break;
        case "r":
        case "R":
          getActiveEngine()?.resetCamera();
          break;
        case "v":
        case "V":
          getActiveEngine()?.cycleView();
          break;
        default: {
          const lens = lensByKey[event.key];
          if (lens) inspector.setLens(lens);
          if (event.key === "7") inspector.setLens("strain");
        }
      }
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [setComputeOpen, setExportOpen, setFigureMode, setLibraryOpen, setPaletteOpen]);

  function handleOpenDatabase() {
    setLibraryOpen(true);
  }

  const hasEntry = Boolean(metadata);

  return (
    <div className="app-shell" data-figure={figureMode ? "true" : "false"} data-mode={mode}>
      <ViewportCanvas />
      <TopBar />
      {mode === "inspector" ? <LeftContextDock /> : null}
      {mode === "studio" ? <LeftContextDock /> : null}
      {mode === "inspector" ? <RightTelemetryDock /> : null}
      {mode === "studio" ? <StudioPanel /> : null}
      {mode === "compare" ? <CompareView /> : null}
      {mode === "inspector" && hasEntry ? (
        <div className="bottom-dock">
          <LensToolbar />
          <TimelineDock />
        </div>
      ) : null}
      {mode === "inspector" && !hasEntry ? (
        <div className="center-empty">
          <div className="center-empty__card glass">
            <div className="center-empty__icon">
              <Database size={24} />
            </div>
            <div className="center-empty__title">Abra um banco v3.3</div>
            <div className="center-empty__text">
              Arraste um arquivo SQLite para a janela ou escolha um shard Harvest / Self-Play. O
              núcleo Rust decodifica <span className="mono">.npy</span> e{" "}
              <span className="mono">COO1</span> diretamente, sem Python e sem pickle.
            </div>
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                gap: 8,
                marginTop: 14,
              }}
            >
              <button
                type="button"
                className="pill-button pill-button--accent"
                onClick={() => setLibraryOpen(true)}
              >
                <FolderOpen size={13} />
                Biblioteca
              </button>
              <button type="button" className="pill-button" onClick={() => void handleOpenDatabase()}>
                Abrir .db
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {figureMode ? (
        <button
          type="button"
          className="pill-button glass"
          style={{
            position: "absolute",
            zIndex: 25,
            top: 14,
            right: 18,
            pointerEvents: "auto",
          }}
          onClick={() => setFigureMode(false)}
        >
          Sair do Modo TCC (P)
        </button>
      ) : null}
      <LibraryDrawer />
      <VoxelTooltip />
      <ExportModal />
      <ComputePopover />
      <CommandPalette />
      <CompareSync />
    </div>
  );
}

function CompareSync() {
  const setSlot = useCompareStore((state) => state.setSlot);
  const inspectorMetadata = useInspectorStore((state) => state.metadata);
  const source = useInspectorStore((state) => state.source);
  const entryId = useInspectorStore((state) => state.entryId);
  const mode = useAppStore((state) => state.mode);
  useEffect(() => {
    if (
      mode === "compare" &&
      source &&
      entryId &&
      inspectorMetadata &&
      !useCompareStore.getState().slotA
    ) {
      setSlot("a", { source, entryId, metadata: inspectorMetadata });
    }
  }, [mode, source, entryId, inspectorMetadata, setSlot]);
  return null;
}
