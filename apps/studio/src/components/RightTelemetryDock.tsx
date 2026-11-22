import { useState } from "react";
import {
  Activity,
  ChevronLeft,
  ChevronRight,
  Ghost,
  GitBranch,
  PanelRightClose,
} from "lucide-react";
import { useInspectorStore } from "../store/useInspectorStore";
import {
  formatCount,
  formatMaxDisplacement,
  formatMilliseconds,
  formatNumber,
  formatPercent,
  formatReward,
  formatSigned,
  formatVolume,
} from "../lib/formatters";
import { RewardBreakdown } from "./RewardBreakdown";

type Tab = "physics" | "mcts" | "blueprint";

export function RightTelemetryDock() {
  const [tab, setTab] = useState<Tab>("physics");
  const [collapsed, setCollapsed] = useState(false);
  const metadata = useInspectorStore((state) => state.metadata);
  const frame = useInspectorStore((state) => state.frame);
  const step = useInspectorStore((state) => state.step);
  const setStep = useInspectorStore((state) => state.setStep);
  const blueprint = useInspectorStore((state) => state.blueprintEnabled);
  const toggleBlueprint = useInspectorStore((state) => state.toggleBlueprint);
  const blueprintThreshold = useInspectorStore((state) => state.blueprintThreshold);
  const setBlueprintThreshold = useInspectorStore((state) => state.setBlueprintThreshold);

  if (!metadata) {
    return null;
  }

  if (collapsed) {
    return (
      <button
        type="button"
        className="right-dock__reopen pill-button glass"
        onClick={() => setCollapsed(false)}
        title="Abrir telemetria"
      >
        <Activity size={13} />
        Telemetria
        <ChevronLeft size={12} />
      </button>
    );
  }

  const meta = frame?.metadata;
  const phase = meta?.phase ?? metadata.phase_history[step] ?? "GROWTH";
  const reward = meta?.reward_v33 ?? null;
  const selected = meta?.selected_actions ?? [];
  const stats = meta?.mcts_stats ?? null;

  return (
    <aside className="right-dock glass">
      <div className="dock-header" style={{ paddingBottom: 8 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
          <span className="dock-title">Telemetria · passo {step}</span>
          <span
            className="mono"
            style={{
              fontSize: 10,
              color: "var(--text-secondary)",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {metadata.entry_id}
          </span>
        </div>
        <div style={{ display: "flex", gap: 3 }}>
          <span className="badge" data-tone={phase === "GROWTH" ? "growth" : "refinement"}>
            {phase}
          </span>
          <button
            type="button"
            className="icon-button"
            onClick={() => setCollapsed(true)}
            title="Colapsar telemetria"
          >
            <PanelRightClose size={13} />
          </button>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "0 10px" }}>
        <button
          type="button"
          className="icon-button"
          onClick={() => setStep(step - 1)}
          title="Passo anterior"
        >
          <ChevronLeft size={13} />
        </button>
        <div style={{ flex: 1 }}>
          <div className="slider-label">
            <span>
              {metadata.steps_phase1} F1 · {metadata.steps_phase2} F2
            </span>
            <span>
              {step + 1}/{metadata.total_steps}
            </span>
          </div>
        </div>
        <button
          type="button"
          className="icon-button"
          onClick={() => setStep(step + 1)}
          title="Próximo passo"
        >
          <ChevronRight size={13} />
        </button>
      </div>

      <div className="tabs">
        <button type="button" data-active={tab === "physics"} onClick={() => setTab("physics")}>
          Física & Reward
        </button>
        <button type="button" data-active={tab === "mcts"} onClick={() => setTab("mcts")}>
          MCTS & PV
        </button>
        <button type="button" data-active={tab === "blueprint"} onClick={() => setTab("blueprint")}>
          Blueprint
        </button>
      </div>

      <div className="right-dock__content scroll-thin">
        {tab === "physics" ? (
          <>
            <div className="stat-grid">
              <Stat label="Value" value={formatSigned(meta?.value, 4)} />
              <Stat label="Volume" value={formatVolume(meta?.volume_fraction)} />
              <Stat label="Compliance" value={formatNumber(meta?.compliance_fem, 2)} />
              <Stat
                label="Desloc. máx"
                value={formatMaxDisplacement(
                  meta?.max_displacement,
                  metadata.real_scale_factor,
                )}
              />
              <Stat label="Ilhas" value={formatCount(meta?.n_islands)} />
              <Stat label="Soltos" value={formatCount(meta?.loose_voxels)} />
              <Stat
                label="Carga conectada"
                value={formatPercent(meta?.connected_load_fraction)}
              />
              <Stat label="Inferência" value={formatMilliseconds(meta?.inference_ms)} />
            </div>
            <div className="panel-section">
              <div className="panel-heading">
                <span>Cadeia v3.3</span>
                <span className="mono">{phase === "GROWTH" ? "NEURAL-ONLY" : "FEM"}</span>
              </div>
              <RewardBreakdown
                breakdown={reward}
                islandPenalty={meta?.island_penalty}
                connectivityBonus={reward?.connectivity_bonus}
              />
            </div>
            <div className="panel-section">
              <div className="panel-heading">
                <span>Normalização científica</span>
              </div>
              <div className="stat-grid">
                <Stat label="disp p99" value={formatNumber(frame?.dispP99, 4)} />
                <Stat label="strain p99" value={formatNumber(frame?.strainP99, 4)} />
                <Stat label="Escala real" value={formatCount(metadata.real_scale_factor)} />
                <Stat label="E" value={`${(metadata.stiffness_e / 1e9).toFixed(0)} GPa`} />
              </div>
            </div>
          </>
        ) : null}

        {tab === "mcts" ? (
          <>
            <div className="stat-grid">
              <Stat label="Simulações" value={formatCount(stats?.num_simulations)} />
              <Stat label="Nós expandidos" value={formatCount(stats?.nodes_expanded)} />
              <Stat label="Profundidade" value={formatCount(stats?.max_depth)} />
              <Stat label="Cache hits" value={formatCount(stats?.cache_hits)} />
              <Stat
                label="Top-8 conc."
                value={formatPercent(stats?.top8_concentration)}
              />
              <Stat label="Refutação" value={stats?.refutation ? "SIM" : "não"} />
            </div>
            {stats?.refutation ? (
              <div className="reward-total" style={{ borderColor: "var(--accent-gold)" }}>
                <span className="reward-row__name" style={{ color: "var(--accent-gold)" }}>
                  MCTS corrigiu a Policy Head
                </span>
                <span className="badge" data-tone="warn">
                  REFUTATION
                </span>
              </div>
            ) : null}
            <div className="panel-section">
              <div className="panel-heading">
                <span>Principal Variation / micro-batch</span>
                <GitBranch size={12} />
              </div>
              {selected.length === 0 ? (
                <div className="empty-card">
                  Este passo não possui ações MCTS registradas (Harvest/FEniTop ou replay leve).
                </div>
              ) : (
                selected.map((action, index) => (
                  <div key={`${action.x}-${action.y}-${action.z}-${index}`} className="reward-row">
                    <span style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                      <span className="reward-row__name">
                        #{index + 1} · {action.channel === 0 ? "ADD" : "REMOVE"} (
                        {action.x}, {action.y}, {action.z})
                      </span>
                      <span className="reward-row__formula">
                        N={action.visits} · Q={formatSigned(action.q_value, 3)}
                      </span>
                    </span>
                    <span
                      className="badge"
                      data-tone={action.channel === 0 ? "growth" : "danger"}
                    >
                      {action.channel === 0 ? "ADD" : "REM"}
                    </span>
                  </div>
                ))
              )}
            </div>
          </>
        ) : null}

        {tab === "blueprint" ? (
          <>
            <div className="stat-grid">
              <Stat label="v_constructed" value={metadata.has_blueprint ? "presente" : "ausente"} />
              <Stat label="ρ contínuo" value={metadata.has_rho_cont ? "presente" : "ausente"} />
              <Stat label="Deslocamento" value={metadata.has_displacement ? "presente" : "ausente"} />
              <Stat label="Fonte" value={metadata.neural_engine ?? "Harvest"} />
            </div>
            <div className="panel-section">
              <div className="panel-heading">
                <span>Camada Ghost Fresnel</span>
                <Ghost size={12} />
              </div>
              <label className="reward-row" style={{ cursor: "pointer" }}>
                <span className="reward-row__name">Exibir Blueprint</span>
                <input
                  type="checkbox"
                  checked={blueprint}
                  disabled={!metadata.has_blueprint}
                  onChange={toggleBlueprint}
                />
              </label>
              <div className="slider-label">
                <span>Threshold de probabilidade</span>
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
              <div className="empty-card" style={{ textAlign: "left", lineHeight: 1.55 }}>
                Em Harvest v3.3, a camada projeta <span className="mono">v_constructed_blob</span>{" "}
                como alvo final. Em Self-Play, projeta{" "}
                <span className="mono">blueprint_probs &gt; threshold</span> da terceira cabeça.
              </div>
            </div>
          </>
        ) : null}

        <div className="panel-section">
          <div className="panel-heading">
            <span>Resumo final</span>
          </div>
          <div className="stat-grid">
            <Stat label="Score final" value={formatReward(metadata.final_score)} />
            <Stat label="Compliance final" value={formatNumber(metadata.final_compliance, 2)} />
            <Stat label="Volume final" value={formatVolume(metadata.final_volume)} />
            <Stat label="Seed" value={metadata.seed === null ? "—" : String(metadata.seed)} />
          </div>
        </div>
      </div>
    </aside>
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
