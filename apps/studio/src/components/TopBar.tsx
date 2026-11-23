import {
  Columns2,
  Command,
  Cpu,
  Download,
  Library,
  Moon,
  PenTool,
  ScanSearch,
  Sun,
} from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { useComputeStore } from "../store/useComputeStore";
import { WindowControls } from "./WindowControls";
import { BrandMark } from "./BrandMark";
import type { AppMode } from "../types/contracts";

const modes: Array<{ id: AppMode; label: string; icon: typeof ScanSearch }> = [
  { id: "inspector", label: "Inspetor", icon: ScanSearch },
  { id: "studio", label: "Estúdio", icon: PenTool },
  { id: "compare", label: "Comparador", icon: Columns2 },
];

export function TopBar() {
  const mode = useAppStore((state) => state.mode);
  const setMode = useAppStore((state) => state.setMode);
  const theme = useAppStore((state) => state.theme);
  const toggleTheme = useAppStore((state) => state.toggleTheme);
  const toggleLibrary = useAppStore((state) => state.toggleLibrary);
  const libraryOpen = useAppStore((state) => state.libraryOpen);
  const setPaletteOpen = useAppStore((state) => state.setPaletteOpen);
  const setExportOpen = useAppStore((state) => state.setExportOpen);
  const setComputeOpen = useAppStore((state) => state.setComputeOpen);
  const computeOpen = useAppStore((state) => state.computeOpen);
  const connection = useComputeStore((state) => state.connection);
  const activeModel = useComputeStore((state) => state.activeModel);
  const connected = connection?.connected ?? false;
  const rtt = connection?.rtt_ms;

  return (
    <header className="top-bar glass">
      <div className="top-bar__zone">
        <div className="brand">
          <BrandMark variant="studio" size={26} alt="AlphaBuilder Studio" />
          <span className="brand__name">
            Alpha<span className="brand__name-accent">Builder</span>
          </span>
          <span className="brand__tagline">3D Structural Optimization</span>
          <span className="brand__version">STUDIO v4.0.4</span>
        </div>
        <button
          type="button"
          className="pill-button"
          data-active={libraryOpen}
          onClick={toggleLibrary}
          title="Biblioteca (L)"
        >
          <Library size={13} />
          Biblioteca
        </button>
      </div>

      <nav className="mode-switch" aria-label="Modos">
        {modes.map((entry) => {
          const Icon = entry.icon;
          return (
            <button
              key={entry.id}
              type="button"
              data-active={mode === entry.id}
              onClick={() => setMode(entry.id)}
            >
              <Icon size={12} />
              {entry.label}
            </button>
          );
        })}
      </nav>

      <div className="top-bar__zone top-bar__zone--right">
        <WindowControls />
        <button
          type="button"
          className={connected ? "pill-button pill-button--accent" : "pill-button"}
          onClick={() => setComputeOpen(!computeOpen)}
          title="Compute & Inference"
        >
          <span className="status-dot" data-state={connected ? "on" : "off"} />
          <Cpu size={13} />
          <span className="hide-compact">
            {activeModel?.model_id
              ? activeModel.model_id.split("/").pop()
              : connected
                ? `${Math.round(rtt ?? 0)} ms`
                : "Compute"}
          </span>
          {activeModel ? <span className="badge" data-tone="accent">{activeModel.device_resolved}</span> : null}
        </button>
        <button
          type="button"
          className="icon-button"
          onClick={() => setPaletteOpen(true)}
          title="Command Palette (Ctrl+K)"
        >
          <Command size={14} />
        </button>
        <button
          type="button"
          className="icon-button"
          onClick={() => setExportOpen(true)}
          title="Exportar (E)"
        >
          <Download size={14} />
        </button>
        <button
          type="button"
          className="icon-button"
          onClick={toggleTheme}
          title={theme === "dark" ? "Tema claro TCC" : "Tema escuro Studio"}
        >
          {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
        </button>
      </div>
    </header>
  );
}
