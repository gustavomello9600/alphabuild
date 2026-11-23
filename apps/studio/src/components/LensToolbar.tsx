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

const allLenses: Array<{ id: LensMode; key: string; label: string; icon: typeof Box }> = [
  { id: "structure", key: "1", label: "Estrutura", icon: Box },
  { id: "policy", key: "2", label: "Policy Head", icon: Sparkles },
  { id: "mcts", key: "3", label: "MCTS · visitas", icon: Target },
  { id: "refutation", key: "4", label: "Refutação MCTS×Policy", icon: GitCompareArrows },
  { id: "decision", key: "5", label: "Diff de decisão", icon: Eye },
  { id: "deformation", key: "6", label: "Deslocamento FEM", icon: Waves },
  { id: "strain", key: "7", label: "Tensão / strain", icon: Activity },
];

interface LensToolbarProps {
  forceVisible?: boolean;
  available?: { has_displacement: boolean; has_blueprint: boolean };
}

export function LensToolbar({ forceVisible, available }: LensToolbarProps = {}) {
  const lens = useInspectorStore((state) => state.lens);
  const setLens = useInspectorStore((state) => state.setLens);
  const metadata = useInspectorStore((state) => state.metadata);
  const blueprint = useInspectorStore((state) => state.blueprintEnabled);
  const toggleBlueprint = useInspectorStore((state) => state.toggleBlueprint);
  const figureMode = useAppStore((state) => state.figureMode);
  const setFigureMode = useAppStore((state) => state.setFigureMode);
  const entryId = useInspectorStore((state) => state.entryId);
  const hasEntry = forceVisible ?? entryId !== null;

  if (!hasEntry) return null;

  const hasDisplacement = available?.has_displacement ?? metadata?.has_displacement ?? false;
  const hasBlueprint = available?.has_blueprint ?? metadata?.has_blueprint ?? false;
  const lenses = allLenses.filter(
    (entry) =>
      hasDisplacement || (entry.id !== "deformation" && entry.id !== "strain"),
  );

  return (
    <div className="lens-bar glass" role="toolbar" aria-label="Lentes 3D">
      {lenses.map((entry) => {
        const Icon = entry.icon;
        const tooltip = `${entry.label} · tecla ${entry.key}`;
        return (
          <button
            key={entry.id}
            type="button"
            className="lens-button lens-button--icon"
            data-active={lens === entry.id}
            data-lens={entry.id}
            data-tooltip={tooltip}
            title={tooltip}
            aria-label={tooltip}
            aria-pressed={lens === entry.id}
            onClick={() => setLens(entry.id)}
          >
            <Icon size={15} strokeWidth={1.8} />
          </button>
        );
      })}
      <span className="lens-divider" />
      <button
        type="button"
        className="lens-button lens-button--icon"
        data-active={blueprint}
        data-lens="blueprint"
        data-tooltip={
          hasBlueprint
            ? `Blueprint / v_constructed · tecla B`
            : "Blueprint indisponível neste banco"
        }
        title={
          hasBlueprint
            ? "Blueprint / v_constructed (B)"
            : "Blueprint indisponível neste banco"
        }
        aria-label="Alternar Blueprint"
        aria-pressed={blueprint}
        disabled={!hasBlueprint}
        onClick={toggleBlueprint}
      >
        <Ghost size={15} strokeWidth={1.8} />
      </button>
      <button
        type="button"
        className="lens-button lens-button--icon"
        data-active={figureMode}
        data-lens="tcc"
        data-tooltip="Modo Figura TCC · tecla P"
        title="Modo Figura TCC (P)"
        aria-label="Modo Figura TCC"
        aria-pressed={figureMode}
        onClick={() => setFigureMode(!figureMode)}
      >
        <Maximize2 size={15} strokeWidth={1.8} />
      </button>
    </div>
  );
}
