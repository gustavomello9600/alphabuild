import { useEffect } from "react";
import { MousePointer2 } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { useStudioStore } from "../store/useStudioStore";

export function StudioPanel() {
  const setLiveXray = useStudioStore((state) => state.setLiveXray);
  const liveXray = useStudioStore((state) => state.liveXray);
  const setOpenLibrary = useAppStore((state) => state.setLibraryOpen);

  useEffect(() => {
    setLiveXray(liveXray);
  }, [liveXray, setLiveXray]);

  return (
    <section
      className="glass pointer-auto"
      style={{
        position: "absolute",
        zIndex: 14,
        left: 328,
        bottom: 118,
        width: 300,
        borderRadius: "var(--radius-lg)",
        padding: 12,
        display: "flex",
        flexDirection: "column",
        gap: 9,
      }}
    >
      <div className="panel-heading">
        <span>Estúdio generativo</span>
        <MousePointer2 size={12} />
      </div>
      <div className="empty-card" style={{ textAlign: "left", lineHeight: 1.55 }}>
        A malha esparsa é validada em Rust a cada edição. Abra um episódio para clonar o estado
        atual ou comece de um domínio vazio na Biblioteca.
      </div>
      <button
        type="button"
        className="pill-button pill-button--accent"
        onClick={() => setOpenLibrary(true)}
      >
        Abrir Biblioteca
      </button>
      <div className="mono muted" style={{ fontSize: 9.5 }}>
        Raio-X Neural e MCTS ao vivo são habilitados quando um Compute Engine está conectado.
      </div>
    </section>
  );
}
