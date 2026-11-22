import {
  Activity,
  Box,
  Eye,
  Ghost,
  GitCompareArrows,
  Maximize2,
  Sparkles,
  Target,
  Waves,
} from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { useInspectorStore } from "../store/useInspectorStore";
import type { LensMode } from "../types/contracts";

const lenses: Array<{ id: LensMode; key: string; label: string; icon: typeof Box }> = [
  { id: "structure", key: "1", label: "Estrutura", icon: Box },
  { id: "policy", key: "2", label: "Policy", icon: Sparkles },
  { id: "mcts", key: "3", label: "MCTS", icon: Target },
  { id: "refutation", key: "4", label: "Refutação", icon: GitCompareArrows },
  { id: "decision", key: "5", label: "Diff", icon: Eye },
  { id: "deformation", key: "6", label: "Desloc.", icon: Waves },
  { id: "strain", key: "7", label: "Tensão", icon: Activity },
];

export function LensToolbar() {
  const lens = useInspectorStore((state) => state.lens);
  const setLens = useInspectorStore((state) => state.setLens);
  const blueprint = useInspectorStore((state) => state.blueprintEnabled);
  const toggleBlueprint = useInspectorStore((state) => state.toggleBlueprint);
  const figureMode = useAppStore((state) => state.figureMode);
  const setFigureMode = useAppStore((state) => state.setFigureMode);
  const hasEntry = useInspectorStore((state) => state.entryId !== null);

  if (!hasEntry) return null;

  return (
    <div className="lens-bar glass" role="toolbar" aria-label="Lentes">
      {lenses.map((entry) => {
        const Icon = entry.icon;
        return (
          <button
            key={entry.id}
            type="button"
            className="lens-button"
            data-active={lens === entry.id}
            onClick={() => setLens(entry.id)}
            title={`${entry.label} (${entry.key})`}
          >
            <span className="lens-key">{entry.key}</span>
            <Icon size={11} />
            <span>{entry.label}</span>
          </button>
        );
      })}
      <span style={{ width: 1, height: 16, background: "var(--hairline)", margin: "0 3px" }} />
      <button
        type="button"
        className="lens-button"
        data-active={blueprint}
        onClick={toggleBlueprint}
        title="Blueprint / v_constructed (B)"
      >
        <span className="lens-key">B</span>
        <Ghost size={11} />
        <span>Blueprint</span>
      </button>
      <button
        type="button"
        className="lens-button"
        data-active={figureMode}
        onClick={() => setFigureMode(!figureMode)}
        title="Modo Figura TCC (P)"
      >
        <span className="lens-key">P</span>
        <Maximize2 size={11} />
        <span>TCC</span>
      </button>
    </div>
  );
}
