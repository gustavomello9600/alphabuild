import { useEffect, useMemo, useState } from "react";
import {
  Box,
  Camera,
  Cpu,
  Database,
  Download,
  Ghost,
  Library,
  Moon,
  PenTool,
  ScanSearch,
  Search,
  Sun,
} from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { useComputeStore } from "../store/useComputeStore";
import { useCompareStore } from "../store/useCompareStore";
import { useInspectorStore } from "../store/useInspectorStore";
import { getActiveEngine } from "../lib/engineBus";
import { isTauriRuntime, listLocalDatabases, openDatabaseFile } from "../lib/ipc";
import type { DatabaseDescriptor } from "../types/contracts";

interface Command {
  id: string;
  label: string;
  hint: string;
  icon: typeof Search;
  run: () => void;
}

export function CommandPalette() {
  const open = useAppStore((state) => state.paletteOpen);
  const setOpen = useAppStore((state) => state.setPaletteOpen);
  const setMode = useAppStore((state) => state.setMode);
  const toggleLibrary = useAppStore((state) => state.toggleLibrary);
  const setLibraryOpen = useAppStore((state) => state.setLibraryOpen);
  const setExportOpen = useAppStore((state) => state.setExportOpen);
  const toggleTheme = useAppStore((state) => state.toggleTheme);
  const setFigureMode = useAppStore((state) => state.setFigureMode);
  const figureMode = useAppStore((state) => state.figureMode);
  const metadata = useInspectorStore((state) => state.metadata);
  const setLens = useInspectorStore((state) => state.setLens);
  const setStep = useInspectorStore((state) => state.setStep);
  const setBlueprint = useInspectorStore((state) => state.toggleBlueprint);
  const setRho = useInspectorStore((state) => state.setUseRhoCont);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [databases, setDatabases] = useState<DatabaseDescriptor[]>([]);
  const profiles = useComputeStore((state) => state.profiles);
  const models = useComputeStore((state) => state.models);
  const activateProfile = useComputeStore((state) => state.activateProfile);
  const loadModel = useComputeStore((state) => state.loadModel);
  const setCompareSlot = useCompareStore((state) => state.setActiveSlot);

  useEffect(() => {
    if (!open || !isTauriRuntime()) return;
    void listLocalDatabases()
      .then(setDatabases)
      .catch(() => setDatabases([]));
    const compute = useComputeStore.getState();
    void compute.loadProfiles();
    if (compute.connection?.connected) void compute.loadModels();
  }, [open]);

  const commands = useMemo<Command[]>(() => {
    const list: Command[] = [
      {
        id: "mode-inspector",
        label: "Modo Inspetor",
        hint: "replay",
        icon: ScanSearch,
        run: () => setMode("inspector"),
      },
      {
        id: "mode-studio",
        label: "Modo Estúdio",
        hint: "CAD",
        icon: PenTool,
        run: () => setMode("studio"),
      },
      {
        id: "library",
        label: "Abrir Biblioteca",
        hint: "L",
        icon: Library,
        run: toggleLibrary,
      },
      {
        id: "export",
        label: "Exportar PNG / STL / OBJ / VTI",
        hint: "E",
        icon: Download,
        run: () => setExportOpen(true),
      },
      {
        id: "theme",
        label: "Alternar tema Studio / TCC",
        hint: "tema",
        icon: Moon,
        run: toggleTheme,
      },
      {
        id: "figure",
        label: "Modo Figura TCC",
        hint: "P",
        icon: Sun,
        run: () => setFigureMode(!figureMode),
      },
      {
        id: "camera-reset",
        label: "Resetar câmera isométrica",
        hint: "R",
        icon: Camera,
        run: () => getActiveEngine()?.resetCamera(),
      },
      {
        id: "camera-cycle",
        label: "Ciclar vista ortográfica",
        hint: "V",
        icon: Camera,
        run: () => getActiveEngine()?.cycleView(),
      },
      {
        id: "lens-structure",
        label: "Lente Estrutura",
        hint: "1",
        icon: Box,
        run: () => setLens("structure"),
      },
      {
        id: "lens-policy",
        label: "Lente Policy",
        hint: "2",
        icon: Ghost,
        run: () => setLens("policy"),
      },
      {
        id: "lens-deformation",
        label: "Lente Deslocamento",
        hint: "6",
        icon: Box,
        run: () => setLens("deformation"),
      },
      {
        id: "lens-strain",
        label: "Lente Tensão",
        hint: "7",
        icon: Box,
        run: () => setLens("strain"),
      },
    ];
    for (const database of databases) {
      list.push({
        id: `db-${database.db_id}`,
        label: `Abrir banco: ${database.name}`,
        hint: database.kind,
        icon: Database,
        run: () => {
          void openDatabaseFile(database.db_id)
            .then(() => {
              window.dispatchEvent(new CustomEvent("ab://library-refresh"));
              setLibraryOpen(true);
            })
            .catch(() => setLibraryOpen(true));
        },
      });
    }
    for (const profile of profiles) {
      list.push({
        id: `profile-${profile.id}`,
        label: `Ativar compute: ${profile.name}`,
        hint: profile.base_url,
        icon: Cpu,
        run: () => void activateProfile(profile.id),
      });
    }
    for (const model of models) {
      list.push({
        id: `model-${model.model_id}`,
        label: `Carregar checkpoint: ${model.model_id}`,
        hint: `${model.size_mb.toFixed(1)} MB`,
        icon: Cpu,
        run: () => void loadModel(model.model_id),
      });
    }
    list.push(
      {
        id: "compare-slot-a",
        label: "Comparador: carregar Slot A",
        hint: "A/B",
        icon: Box,
        run: () => {
          setMode("compare");
          setCompareSlot("a");
          setLibraryOpen(true);
        },
      },
      {
        id: "compare-slot-b",
        label: "Comparador: carregar Slot B",
        hint: "A/B",
        icon: Box,
        run: () => {
          setMode("compare");
          setCompareSlot("b");
          setLibraryOpen(true);
        },
      },
    );
    if (metadata) {
      list.push(
        {
          id: "first-step",
          label: "Ir ao primeiro passo",
          hint: "Home",
          icon: ScanSearch,
          run: () => setStep(0),
        },
        {
          id: "last-step",
          label: "Ir ao último passo",
          hint: "End",
          icon: ScanSearch,
          run: () => setStep(Math.max(0, metadata.total_steps - 1)),
        },
        {
          id: "blueprint",
          label: "Alternar Blueprint Ghost",
          hint: "B",
          icon: Ghost,
          run: setBlueprint,
        },
        {
          id: "rho",
          label: "Alternar ρ contínuo",
          hint: "rho",
          icon: Box,
          run: () => setRho(!useInspectorStore.getState().useRhoCont),
        },
      );
    }
    return list;
  }, [
    metadata,
    setMode,
    toggleLibrary,
    setExportOpen,
    toggleTheme,
    setFigureMode,
    figureMode,
    setLens,
    setStep,
    setBlueprint,
    setRho,
    databases,
    profiles,
    models,
    activateProfile,
    loadModel,
    setCompareSlot,
    setLibraryOpen,
  ]);

  const filtered = commands.filter((command) =>
    command.label.toLowerCase().includes(query.toLowerCase()),
  );

  useEffect(() => {
    setActive(0);
  }, [query, open]);

  useEffect(() => {
    if (!open) return;
    const handle = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        return;
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        setActive((value) => Math.min(filtered.length - 1, value + 1));
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setActive((value) => Math.max(0, value - 1));
      } else if (event.key === "Enter") {
        event.preventDefault();
        const selected = filtered[active];
        if (selected) {
          selected.run();
          setOpen(false);
          setQuery("");
        }
      }
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [open, filtered, active, setOpen]);

  if (!open) return null;

  return (
    <div
      className="palette-scrim"
      onClick={(event) => {
        if (event.target === event.currentTarget) setOpen(false);
      }}
    >
      <div className="command-palette glass">
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 9,
            padding: "12px 14px",
            borderBottom: "1px solid var(--hairline)",
          }}
        >
          <Search size={15} className="accent-cyan" />
          <input
            autoFocus
            className="field-input"
            style={{ border: "none", background: "transparent", height: 26 }}
            placeholder="Buscar comandos, lentes, modos, exportações…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <span className="badge">ESC</span>
        </div>
        <div className="command-list">
          {filtered.length === 0 ? (
            <div className="empty-card" style={{ margin: 8 }}>
              Nenhum comando corresponde a “{query}”.
            </div>
          ) : (
            filtered.map((command, index) => {
              const Icon = command.icon;
              return (
                <button
                  key={command.id}
                  type="button"
                  className="command-item"
                  data-active={index === active}
                  onMouseEnter={() => setActive(index)}
                  onClick={() => {
                    command.run();
                    setOpen(false);
                    setQuery("");
                  }}
                >
                  <span style={{ display: "flex", alignItems: "center", gap: 9 }}>
                    <Icon size={13} />
                    {command.label}
                  </span>
                  <span className="command-item__hint">{command.hint}</span>
                </button>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
