import { useState } from "react";
import { Box, Camera, Download, FileCode2, X } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { useInspectorStore } from "../store/useInspectorStore";
import { exportGeometry, exportPngBytes, isTauriRuntime, pickSavePath } from "../lib/ipc";
import { getActiveEngine } from "../lib/engineBus";

type ExportFormat = "png" | "stl" | "obj" | "vti";

const formats: Array<{
  id: ExportFormat;
  label: string;
  detail: string;
  icon: typeof Box;
}> = [
  { id: "png", label: "PNG 4K", detail: "3840 × 2160 · figura TCC", icon: Camera },
  { id: "stl", label: "STL binário", detail: "malha watertight surface-culled", icon: Box },
  { id: "obj", label: "Wavefront OBJ", detail: "v / vn / faces i//i", icon: FileCode2 },
  { id: "vti", label: "VTK .vti", detail: "density · displacement · strain · blueprint", icon: Download },
];

export function ExportModal() {
  const open = useAppStore((state) => state.exportOpen);
  const setOpen = useAppStore((state) => state.setExportOpen);
  const mode = useAppStore((state) => state.mode);
  const source = useInspectorStore((state) => state.source);
  const entryId = useInspectorStore((state) => state.entryId);
  const step = useInspectorStore((state) => state.step);
  const [format, setFormat] = useState<ExportFormat>("png");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const hasData = mode === "studio" ? true : Boolean(source && entryId);

  async function handleExport() {
    if (!isTauriRuntime()) {
      setError("Exportação exige o aplicativo desktop Tauri.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      if (format === "png") {
        const engine = getActiveEngine();
        if (!engine) throw new Error("Viewport 3D ainda não está montado.");
        const dataUrl = engine.captureScreenshot(3840, 2160, false);
        const path = await pickSavePath(
          `alphabuilder_step_${step.toString().padStart(4, "0")}.png`,
          "PNG 4K",
          "png",
        );
        if (!path) return;
        const saved = await exportPngBytes(dataUrl, path);
        setResult(`${saved.bytes_written.toLocaleString("en-US")} bytes → ${saved.output_path}`);
        return;
      }
      if (!source || !entryId) throw new Error("Abra um episódio para exportar geometria.");
      const path = await pickSavePath(
        `alphabuilder_${entryId.slice(0, 8)}_${step.toString().padStart(4, "0")}.${format}`,
        format.toUpperCase(),
        format,
      );
      if (!path) return;
      const saved = await exportGeometry(
        { kind: "frame", source, entry_id: entryId, step },
        format,
        path,
      );
      setResult(`${saved.bytes_written.toLocaleString("en-US")} bytes → ${saved.output_path}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="modal-scrim"
      onClick={(event) => {
        if (event.target === event.currentTarget) setOpen(false);
      }}
    >
      <div className="modal-card glass">
        <div className="dock-header">
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Download size={14} className="accent-cyan" />
            <span className="dock-title">Exportação científica</span>
          </div>
          <button type="button" className="icon-button" onClick={() => setOpen(false)}>
            <X size={14} />
          </button>
        </div>
        <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 7 }}>
            {formats.map((entry) => {
              const Icon = entry.icon;
              return (
                <button
                  key={entry.id}
                  type="button"
                  className="db-row"
                  data-active={format === entry.id}
                  onClick={() => setFormat(entry.id)}
                >
                  <span className="db-row__title">
                    <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
                      <Icon size={13} className="accent-cyan" />
                      {entry.label}
                    </span>
                  </span>
                  <span className="db-row__meta">{entry.detail}</span>
                </button>
              );
            })}
          </div>
          <div className="empty-card" style={{ textAlign: "left", lineHeight: 1.55 }}>
            {mode === "studio"
              ? "O Estúdio exporta a escultura atual. A geometria é reconstruída no Rust a partir do grid esparso."
              : "A exportação usa a malha surface-culled do passo atual (sem reamostragem), com AO e atributos de deslocamento/tensão."}
          </div>
          {result ? (
            <div className="badge" data-tone="growth" style={{ height: "auto", padding: "6px 8px" }}>
              {result}
            </div>
          ) : null}
          {error ? (
            <div className="badge" data-tone="danger" style={{ height: "auto", padding: "6px 8px" }}>
              {error}
            </div>
          ) : null}
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 6 }}>
            <button type="button" className="pill-button" onClick={() => setOpen(false)}>
              Cancelar
            </button>
            <button
              type="button"
              className="pill-button pill-button--accent"
              onClick={() => void handleExport()}
              disabled={busy || !hasData}
            >
              {busy ? "Exportando…" : `Exportar ${format.toUpperCase()}`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
