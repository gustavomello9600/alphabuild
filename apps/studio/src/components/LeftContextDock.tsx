import { useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  CircleDot,
  Eraser,
  Ghost,
  Layers,
  Move3d,
  Paintbrush,
  ScanLine,
  SlidersHorizontal,
} from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { useInspectorStore } from "../store/useInspectorStore";
import { useStudioStore } from "../store/useStudioStore";

export function LeftContextDock() {
  const [collapsed, setCollapsed] = useState(true);
  const mode = useAppStore((state) => state.mode);
  const metadata = useInspectorStore((state) => state.metadata);
  const lens = useInspectorStore((state) => state.lens);
  const setLens = useInspectorStore((state) => state.setLens);
  const clipX = useInspectorStore((state) => state.clipX);
  const clipY = useInspectorStore((state) => state.clipY);
  const clipZ = useInspectorStore((state) => state.clipZ);
  const setClip = useInspectorStore((state) => state.setClip);
  const deformScale = useInspectorStore((state) => state.deformScale);
  const setDeformScale = useInspectorStore((state) => state.setDeformScale);
  const blueprint = useInspectorStore((state) => state.blueprintEnabled);
  const toggleBlueprint = useInspectorStore((state) => state.toggleBlueprint);
  const blueprintThreshold = useInspectorStore((state) => state.blueprintThreshold);
  const setBlueprintThreshold = useInspectorStore((state) => state.setBlueprintThreshold);
  const useRhoCont = useInspectorStore((state) => state.useRhoCont);
  const setUseRhoCont = useInspectorStore((state) => state.setUseRhoCont);

  return (
    <aside className="left-dock glass" data-collapsed={collapsed}>
      <div className="dock-header">
        {!collapsed ? (
          <span className="dock-title">{mode === "studio" ? "Escultura" : "Lentes & Cortes"}</span>
        ) : null}
        <button
          type="button"
          className="icon-button"
          onClick={() => setCollapsed(!collapsed)}
          title={collapsed ? "Expandir dock" : "Colapsar dock"}
          style={{ marginLeft: "auto" }}
        >
          {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      {!collapsed ? (
        <div className="left-dock__body scroll-thin">
          {mode === "studio" ? (
            <StudioTools />
          ) : !metadata ? (
            <div className="empty-card">
              Selecione um episódio na Biblioteca para habilitar lentes e planos de corte.
            </div>
          ) : (
            <>
              <section className="panel-section">
                <div className="panel-heading">
                  <span>Lente ativa</span>
                  <Layers size={12} />
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {(
                    [
                      ["structure", "Estrutura"],
                      ["policy", "Policy"],
                      ["mcts", "MCTS"],
                      ["refutation", "Refutação"],
                      ["decision", "Diff"],
                      ["deformation", "Deslocamento"],
                      ["strain", "Tensão"],
                    ] as const
                  ).map(([id, label]) => (
                    <button
                      key={id}
                      type="button"
                      className="pill-button"
                      data-active={lens === id}
                      onClick={() => setLens(id)}
                      style={{ height: 26, padding: "0 8px", fontSize: 10.5 }}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </section>

              <section className="panel-section">
                <div className="panel-heading">
                  <span>Mesa de cirurgia 3D</span>
                  <button
                    type="button"
                    className="pill-button"
                    style={{ height: 20, padding: "0 7px", fontSize: 9 }}
                    onClick={() => {
                      setClip("x", metadata.resolution[0]);
                      setClip("y", metadata.resolution[1]);
                      setClip("z", metadata.resolution[2]);
                    }}
                  >
                    Reset
                  </button>
                </div>
                <ClipSlider
                  axis="X"
                  value={clipX}
                  max={metadata.resolution[0]}
                  onChange={(value) => setClip("x", value)}
                />
                <ClipSlider
                  axis="Y"
                  value={clipY}
                  max={metadata.resolution[1]}
                  onChange={(value) => setClip("y", value)}
                />
                <ClipSlider
                  axis="Z"
                  value={clipZ}
                  max={metadata.resolution[2]}
                  onChange={(value) => setClip("z", value)}
                />
              </section>

              <section className="panel-section">
                <div className="panel-heading">
                  <span>Deslocamento</span>
                  <SlidersHorizontal size={12} />
                </div>
                <div className="slider-label">
                  <span>Exagero de deformação</span>
                  <span>{deformScale.toFixed(1)}×</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={20}
                  step={0.5}
                  value={deformScale}
                  onChange={(event) => setDeformScale(Number(event.target.value))}
                />
                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 7,
                    fontSize: 10.5,
                    color: "var(--text-secondary)",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={useRhoCont}
                    disabled={!metadata.has_rho_cont}
                    onChange={(event) => setUseRhoCont(event.target.checked)}
                  />
                  ρ contínuo (Harvest)
                </label>
              </section>

              <section className="panel-section">
                <div className="panel-heading">
                  <span>Blueprint Head</span>
                  <Ghost size={12} />
                </div>
                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 7,
                    fontSize: 10.5,
                    color: "var(--text-secondary)",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={blueprint}
                    disabled={!metadata.has_blueprint}
                    onChange={toggleBlueprint}
                  />
                  Camada ghost / v_constructed
                </label>
                <div className="slider-label">
                  <span>Threshold</span>
                  <span>{blueprintThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={blueprintThreshold}
                  onChange={(event) => setBlueprintThreshold(Number(event.target.value))}
                />
              </section>
            </>
          )}
        </div>
      ) : null}
    </aside>
  );
}

function ClipSlider(props: {
  axis: string;
  value: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="slider-label">
        <span>{props.axis}</span>
        <span>
          {props.value} / {props.max}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={props.max}
        step={1}
        value={props.value}
        onChange={(event) => props.onChange(Number(event.target.value))}
      />
    </div>
  );
}

function StudioTools() {
  const tool = useStudioStore((state) => state.tool);
  const setTool = useStudioStore((state) => state.setTool);
  const brush = useStudioStore((state) => state.brushSize);
  const setBrush = useStudioStore((state) => state.setBrushSize);
  const liveXray = useStudioStore((state) => state.liveXray);
  const setLiveXray = useStudioStore((state) => state.setLiveXray);
  const analyze = useStudioStore((state) => state.analyze);
  const report = useStudioStore((state) => state.report);
  return (
    <>
      <section className="panel-section">
        <div className="panel-heading">
          <span>Ferramenta</span>
          <Paintbrush size={12} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5 }}>
          <button
            type="button"
            className="pill-button"
            data-active={tool === "paint"}
            onClick={() => setTool("paint")}
          >
            <Paintbrush size={12} /> Adicionar
          </button>
          <button
            type="button"
            className="pill-button"
            data-active={tool === "erase"}
            onClick={() => setTool("erase")}
          >
            <Eraser size={12} /> Remover
          </button>
          <button
            type="button"
            className="pill-button"
            data-active={tool === "load"}
            onClick={() => setTool("load")}
          >
            <Move3d size={12} /> Carga
          </button>
          <button
            type="button"
            className="pill-button"
            data-active={tool === "support"}
            onClick={() => setTool("support")}
          >
            <CircleDot size={12} /> Suporte
          </button>
        </div>
        <div className="panel-heading">
          <span>Pincel</span>
        </div>
        <div style={{ display: "flex", gap: 5 }}>
          {[1, 3].map((size) => (
            <button
              key={size}
              type="button"
              className="pill-button"
              data-active={brush === size}
              onClick={() => setBrush(size as 1 | 3)}
            >
              {size}×{size}×{size}
            </button>
          ))}
        </div>
      </section>
      <section className="panel-section">
        <div className="panel-heading">
          <span>Raio-X Neural</span>
          <ScanLine size={12} />
        </div>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 7,
            fontSize: 10.5,
            color: "var(--text-secondary)",
          }}
        >
          <input
            type="checkbox"
            checked={liveXray}
            onChange={(event) => setLiveXray(event.target.checked)}
          />
          Inferência live (debounce 120 ms)
        </label>
        <button
          type="button"
          className="pill-button pill-button--accent"
          onClick={() => void analyze()}
        >
          Validar topologia local
        </button>
        {report ? (
          <div className="stat-grid">
            <Stat label="Ilhas" value={String(report.n_islands)} />
            <Stat label="Soltos" value={String(report.loose_voxels)} />
            <Stat label="Volume" value={`${(report.volume_fraction * 100).toFixed(1)}%`} />
            <Stat label="Legal +" value={String(report.legal_add_count)} />
          </div>
        ) : null}
      </section>
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-card">
      <div className="stat-card__label">{label}</div>
      <div className="stat-card__value">{value}</div>
    </div>
  );
}
