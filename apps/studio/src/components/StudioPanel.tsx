import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Box,
  Copy,
  FlaskConical,
  Grid3X3,
  Pause,
  Play,
  RefreshCw,
  ScanLine,
  StepForward,
  Square,
  Trash2,
  Wand2,
} from "lucide-react";
import { useComputeStore } from "../store/useComputeStore";
import { useInspectorStore } from "../store/useInspectorStore";
import { useStudioStore } from "../store/useStudioStore";
import type { BcType, LoadConfig, SupportVoxel } from "../types/contracts";
import { parseAbMesh01 } from "../lib/abmesh";
import { getActiveEngine } from "../lib/engineBus";
import {
  computeEvaluateFem,
  computeInferXray,
  sessionSendCommand,
  sessionStart,
} from "../lib/ipc";
import { formatMilliseconds, formatPercent, formatReward, formatSigned } from "../lib/formatters";
import { BrandMark } from "./BrandMark";

const RESOLUTIONS: Record<string, [number, number, number]> = {
  cantilever: [64, 32, 8],
  bridge: [64, 32, 8],
  empty: [64, 32, 8],
};

function cantileverSolid(dims: [number, number, number]): Set<number> {
  const [d, h, w] = dims;
  const solid = new Set<number>();
  const y0 = Math.floor(h / 2) - 1;
  const y1 = Math.floor(h / 2) + 2;
  const z0 = Math.floor(w / 2) - 1;
  const z1 = Math.floor(w / 2) + 2;
  for (let x = 0; x < d; x += 1) {
    for (let y = y0; y < y1; y += 1) {
      for (let z = z0; z < z1; z += 1) {
        solid.add(x * h * w + y * w + z);
      }
    }
  }
  return solid;
}

function bridgeSolid(dims: [number, number, number]): Set<number> {
  const [d, h, w] = dims;
  const solid = new Set<number>();
  const y0 = Math.max(1, Math.floor(h / 4));
  const y1 = y0 + 3;
  const z0 = Math.floor(w / 2) - 1;
  const z1 = Math.floor(w / 2) + 2;
  for (let x = 0; x < d; x += 1) {
    for (let y = y0; y < y1; y += 1) {
      for (let z = z0; z < z1; z += 1) {
        solid.add(x * h * w + y * w + z);
      }
    }
  }
  return solid;
}

function bridgeSupports(dims: [number, number, number]): SupportVoxel[] {
  const [d, h, w] = dims;
  const supports: SupportVoxel[] = [];
  for (const x of [0, d - 1]) {
    for (let y = 0; y < h; y += 1) {
      for (let z = 0; z < w; z += 1) {
        supports.push({ x, y, z, mask_x: true, mask_y: true, mask_z: true });
      }
    }
  }
  return supports;
}

function outsideSupports(dims: [number, number, number]): SupportVoxel[] {
  const [d, h, w] = dims;
  const supports: SupportVoxel[] = [];
  for (let y = 0; y < h; y += 1) {
    for (let z = 0; z < w; z += 1) {
      supports.push({ x: 0, y, z, mask_x: true, mask_y: true, mask_z: true });
      supports.push({ x: d - 1, y, z, mask_x: true, mask_y: true, mask_z: true });
    }
  }
  return supports;
}

export function StudioPanel() {
  const store = useStudioStore();
  const compute = useComputeStore();
  const metadata = useInspectorStore((state) => state.metadata);
  const inspectorFrame = useInspectorStore((state) => state.frame);
  const blueprint = useInspectorStore((state) => state.blueprintEnabled);
  const toggleBlueprint = useInspectorStore((state) => state.toggleBlueprint);
  const lastSent = useRef<Set<number>>(new Set());
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    if (
      store.sessionStatus !== "PAUSED" ||
      !store.sessionFrame ||
      !store.sessionFollowLive
    ) {
      return;
    }
    const next = new Set(store.sessionFrame.voxelIds);
    if (next.size > 0) {
      store.setSolidVoxels(next);
      lastSent.current = new Set(next);
    }
  }, [store.sessionStatus, store.sessionFrame]);

  function applyTemplate(kind: "cantilever" | "bridge" | "empty") {
    const dims = RESOLUTIONS[kind] ?? [64, 32, 8];
    const solid =
      kind === "cantilever"
        ? cantileverSolid(dims)
        : kind === "bridge"
          ? bridgeSolid(dims)
          : new Set<number>();
    const bcType: BcType = kind === "bridge" ? "CUSTOM" : "FULL_CLAMP";
    const loadConfig: LoadConfig = {
      x: kind === "bridge" ? Math.floor(dims[0] / 2) : dims[0] - 1,
      y: kind === "bridge" ? Math.max(2, Math.floor(dims[1] / 3)) : dims[1] / 2,
      z_start: Math.floor(dims[2] / 2) - 1,
      z_end: Math.floor(dims[2] / 2) + 1,
      bc_type: bcType,
    };
    useStudioStore.setState({
      resolution: dims,
      bcType,
      loadConfig,
      solidVoxels: solid,
      customSupports: kind === "bridge" ? bridgeSupports(dims) : null,
      customForces: null,
      overlay: null,
      session: null,
      sessionFrame: null,
      sessionStatus: "IDLE",
      inferenceValue: null,
      inferenceError: null,
      report: null,
    });
    lastSent.current = new Set(solid);
  }

  function cloneInspector() {
    if (!metadata || !inspectorFrame) {
      setLocalError("Abra um episódio no Inspetor antes de clonar o estado.");
      return;
    }
    const solid = new Set<number>(inspectorFrame.voxelIds);
    useStudioStore.setState({
      resolution: metadata.resolution,
      bcType: metadata.bc_type,
      loadConfig: metadata.load_config,
      solidVoxels: solid,
      customSupports: null,
      customForces: null,
      overlay: null,
      inferenceValue: null,
      inferenceError: null,
      report: null,
    });
    lastSent.current = new Set(solid);
  }

  async function inferNow() {
    if (!compute.connection?.connected || !compute.activeModel) {
      setLocalError("Conecte um Compute Engine e carregue um checkpoint para o Raio-X Neural.");
      return;
    }
    const state = store.voxelState();
    setBusy(true);
    setLocalError(null);
    try {
      const buffer = await computeInferXray({
        resolution: state.resolution,
        bc_type: state.bc_type,
        load_config: state.load_config,
        custom_supports: state.custom_supports,
        custom_forces: state.custom_forces,
        solid_voxels: state.solid_voxels,
        include_blueprint: true,
        policy_mode: "masked_probs",
        policy_temperature: 1.0,
        policy_top_k: 512,
      });
      const parsed = parseAbMesh01(buffer);
      getActiveEngine()?.applyFrame(parsed);
      store.setInference(
        parsed.metadata.value,
        parsed.metadata.inference_ms ?? null,
        (parsed.metadata as { device?: string }).device ?? null,
      );
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function evaluateFem() {
    const features = compute.capabilities?.features;
    if (!features?.has_fenics) {
      setLocalError("FEniCSx não está disponível neste servidor (capability has_fenics=false).");
      return;
    }
    const state = store.voxelState();
    setBusy(true);
    setLocalError(null);
    try {
      const buffer = await computeEvaluateFem({
        resolution: state.resolution,
        bc_type: state.bc_type,
        load_config: state.load_config,
        custom_supports: state.custom_supports,
        custom_forces: state.custom_forces,
        solid_voxels: state.solid_voxels,
        E: 100,
        nu: 0.25,
        force_magnitude: -1.125,
        include_displacement_map: true,
        include_displacement_vector: false,
      });
      const parsed = parseAbMesh01(buffer);
      getActiveEngine()?.applyFrame(parsed);
      store.setInference(
        parsed.metadata.reward_v33?.total_reward ?? null,
        null,
        "FEM",
      );
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function startLiveMcts() {
    if (!compute.connection?.connected || !compute.activeModel) {
      setLocalError("Conecte um Compute Engine e carregue um checkpoint para o MCTS ao vivo.");
      return;
    }
    const state = store.voxelState();
    setBusy(true);
    setLocalError(null);
    try {
      const handle = await sessionStart({
        kind: "MCTS_LIVE",
        resolution: state.resolution,
        bc_type: state.bc_type,
        load_config: state.load_config,
        initial_solid_voxels: state.solid_voxels,
        persist_to_db: false,
        start_paused: false,
        include_blueprint: true,
        allow_neural_only_phase2: true,
        episode_config: {
          max_steps: 600,
          max_phase1_steps: 200,
          target_volume_fraction: 0.15,
          use_guided_value: true,
          generation: 0,
          guidance_decay_generations: 5,
        },
        phase1_mcts: {
          num_simulations: 160,
          batch_size: 8,
          action_batch_size: 32,
          c_puct: 1.2,
          top_k_expansion: 64,
          mixmax_beta: 0.75,
        },
        phase2_mcts: {
          num_simulations: 160,
          batch_size: 8,
          action_batch_size: 8,
          c_puct: 0.8,
          top_k_expansion: 64,
          mixmax_beta: 1.0,
        },
      });
      lastSent.current = new Set(state.solid_voxels);
      store.clearSessionHistory();
      useInspectorStore.getState().setLens("structure");
      useInspectorStore.getState().setBlueprintEnabled(false);
      store.setSession(handle);
      store.setSessionStatus(handle.status, 0);
      store.setSessionProgress(0);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function startSimp() {
    const features = compute.capabilities?.features;
    if (!features?.has_fenitop_simp || !features.has_fenics) {
      setLocalError("FEniTop SIMP exige FEniCSx/fenitop no servidor.");
      return;
    }
    const state = store.voxelState();
    setBusy(true);
    setLocalError(null);
    try {
      const handle = await sessionStart({
        kind: "SIMP",
        resolution: state.resolution,
        bc_type: state.bc_type,
        load_config: state.load_config,
        initial_solid_voxels: state.solid_voxels,
        strategy: "FULL_DOMAIN",
        target_vol_frac: 0.15,
        max_iter: 80,
        r_min: 1.5,
        seed: 42,
      });
      lastSent.current = new Set(state.solid_voxels);
      store.clearSessionHistory();
      useInspectorStore.getState().setLens("structure");
      useInspectorStore.getState().setBlueprintEnabled(false);
      store.setSession(handle);
      store.setSessionStatus(handle.status, 0);
      store.setSessionProgress(0);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function send(op: "pause" | "resume" | "step_once" | "cancel") {
    if (!store.session) return;
    try {
      await sessionSendCommand(store.session.session_id, { op });
    } catch (error) {
      store.setSessionError(error instanceof Error ? error.message : String(error));
    }
  }

  async function applyEditAndContinue() {
    if (!store.session) return;
    const current = store.solidVoxels;
    const added = Array.from(current).filter((idx) => !lastSent.current.has(idx));
    const removed = Array.from(lastSent.current).filter((idx) => !current.has(idx));
    try {
      await sessionSendCommand(store.session.session_id, {
        op: "edit_state",
        add_voxels: added,
        remove_voxels: removed,
        load_config: store.loadConfig,
        bc_type: store.bcType,
      });
      lastSent.current = new Set(current);
      await sessionSendCommand(store.session.session_id, { op: "resume" });
    } catch (error) {
      store.setSessionError(error instanceof Error ? error.message : String(error));
    }
  }

  const session = store.session;
  const sessionMeta = (store.sessionViewFrame ?? store.sessionFrame)?.metadata;
  const running = store.sessionStatus === "RUNNING";
  const paused = store.sessionStatus === "PAUSED";

  return (
    <aside className="studio-panel glass">
      <div className="dock-header">
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <BrandMark variant="layered" size={19} alt="" />
          <span className="dock-title">Estúdio generativo</span>
        </div>
        <span className="badge" data-tone={store.sessionStatus === "IDLE" ? undefined : "accent"}>
          {store.sessionStatus}
        </span>
      </div>

      <div className="studio-panel__body scroll-thin">
        <section className="panel-section">
          <div className="panel-heading">
            <span>Templates</span>
            <Grid3X3 size={12} />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5 }}>
            <button type="button" className="pill-button" onClick={() => applyTemplate("cantilever")}>
              <Box size={12} /> Viga
            </button>
            <button type="button" className="pill-button" onClick={() => applyTemplate("bridge")}>
              <Wand2 size={12} /> Ponte
            </button>
            <button type="button" className="pill-button" onClick={() => applyTemplate("empty")}>
              <Trash2 size={12} /> Vazio
            </button>
            <button type="button" className="pill-button" onClick={cloneInspector}>
              <Copy size={12} /> Clonar
            </button>
          </div>
        </section>

        <section className="panel-section">
          <div className="panel-heading">
            <span>Suportes</span>
            <span className="mono">{store.customSupports?.length ?? "auto"}</span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {(["FULL_CLAMP", "RAIL_XY", "ROLLER_Y", "CUSTOM"] as BcType[]).map((bc) => (
              <button
                key={bc}
                type="button"
                className="pill-button"
                data-active={store.bcType === bc}
                onClick={() => {
                  useStudioStore.setState({ bcType: bc });
                  if (bc !== "CUSTOM") useStudioStore.setState({ customSupports: null });
                  if (bc === "CUSTOM" && !store.customSupports) {
                    useStudioStore.setState({ customSupports: outsideSupports(store.resolution) });
                  }
                }}
              >
                {bc}
              </button>
            ))}
          </div>
          <div className="mono muted" style={{ fontSize: 9.5 }}>
            Pincel de suporte: ferramenta “Support” no dock esquerdo (DOFs Ux/Uy/Uz).
          </div>
        </section>

        <section className="panel-section">
          <div className="panel-heading">
            <span>Carga externa</span>
            <span className="mono">(Fx,Fy,Fz)=(0,−1,0)</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 4 }}>
            <NumberField
              label="x"
              value={store.loadConfig.x}
              max={store.resolution[0] - 1}
              onChange={(value) => store.setLoadConfig({ ...store.loadConfig, x: value })}
            />
            <NumberField
              label="y"
              value={store.loadConfig.y}
              max={store.resolution[1] - 1}
              onChange={(value) => store.setLoadConfig({ ...store.loadConfig, y: value })}
            />
            <NumberField
              label="z0"
              value={store.loadConfig.z_start}
              max={store.resolution[2] - 1}
              onChange={(value) => store.setLoadConfig({ ...store.loadConfig, z_start: value })}
            />
            <NumberField
              label="z1"
              value={store.loadConfig.z_end}
              max={store.resolution[2] - 1}
              onChange={(value) => store.setLoadConfig({ ...store.loadConfig, z_end: value })}
            />
          </div>
          <div className="mono muted" style={{ fontSize: 9.5 }}>
            Ferramenta “Load”: arraste a alça laranja no viewport; snap inteiro automático.
          </div>
        </section>

        <section className="panel-section">
          <div className="panel-heading">
            <span>Raio-X Neural</span>
            <ScanLine size={12} />
          </div>
          <label className="reward-row" style={{ cursor: "pointer" }}>
            <span className="reward-row__name">Live (debounce 120 ms)</span>
            <input
              type="checkbox"
              checked={store.liveXray}
              onChange={(event) => store.setLiveXray(event.target.checked)}
            />
          </label>
          <label className="reward-row" style={{ cursor: "pointer" }}>
            <span className="reward-row__name">Blueprint ghost (B)</span>
            <input type="checkbox" checked={blueprint} onChange={toggleBlueprint} />
          </label>
          <div className="stat-grid">
            <Stat label="Value" value={formatSigned(store.inferenceValue, 4)} />
            <Stat label="Latência" value={formatMilliseconds(store.inferenceMs)} />
            <Stat label="Device" value={store.inferenceDevice ?? "—"} />
            <Stat label="Modelo" value={compute.activeModel?.model_id?.split("/").pop() ?? "—"} />
          </div>
          <button
            type="button"
            className="pill-button pill-button--accent"
            onClick={() => void inferNow()}
            disabled={busy}
          >
            <RefreshCw size={12} /> {busy ? "Inferindo…" : "Inferir agora"}
          </button>
          {store.inferenceError ? (
            <div className="badge" data-tone="danger" style={{ height: "auto", padding: "6px 8px" }}>
              {store.inferenceError}
            </div>
          ) : null}
        </section>

        <section className="panel-section">
          <div className="panel-heading">
            <span>Topologia Rust</span>
            <button type="button" className="icon-button" onClick={() => void store.analyze()}>
              <RefreshCw size={12} />
            </button>
          </div>
          {store.report ? (
            <>
              <div className="stat-grid">
                <Stat label="Ilhas" value={String(store.report.n_islands)} />
                <Stat label="Soltos" value={String(store.report.loose_voxels)} />
                <Stat label="Volume" value={formatPercent(store.report.volume_fraction)} />
                <Stat
                  label="Carga conectada"
                  value={formatPercent(store.report.connected_load_fraction)}
                />
                <Stat label="Legal +" value={String(store.report.legal_add_count)} />
                <Stat label="Legal −" value={String(store.report.legal_remove_count)} />
              </div>
              <span className="badge" data-tone={store.report.is_connected ? "growth" : "warn"}>
                {store.report.is_connected ? "CONECTADO" : "DESCONECTADO"}
              </span>
            </>
          ) : (
            <div className="empty-card">
              Pinte alguns voxels ou carregue um template. A topologia é recalculada em Rust.
            </div>
          )}
        </section>

        <section className="panel-section">
          <div className="panel-heading">
            <span>Física real</span>
            <FlaskConical size={12} />
          </div>
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
            <button
              type="button"
              className="pill-button"
              onClick={() => void evaluateFem()}
              disabled={busy || !compute.capabilities?.features.has_fenics}
              title="Requires FEniCSx"
            >
              <FlaskConical size={12} /> Avaliar FEM
            </button>
            <button
              type="button"
              className="pill-button"
              onClick={() => void startSimp()}
              disabled={!compute.capabilities?.features.has_fenitop_simp}
              title="Requires FEniTop SIMP"
            >
              <Grid3X3 size={12} /> Referência SIMP
            </button>
          </div>
        </section>

        <section className="panel-section">
          <div className="panel-heading">
            <span>Co-Design · MCTS ao vivo</span>
            <span className="badge" data-tone={running ? "growth" : paused ? "warn" : undefined}>
              {store.sessionStatus}
            </span>
          </div>
          {!session ? (
            <button
              type="button"
              className="pill-button pill-button--accent"
              onClick={() => void startLiveMcts()}
              disabled={busy}
            >
              <Play size={12} /> Otimizar com IA
            </button>
          ) : (
            <>
              <div className="slider-label">
                <span>
                  Passo {store.sessionStep} · {sessionMeta?.phase ?? "GROWTH"} ·{" "}
                  {sessionMeta?.mcts_stats?.num_simulations ?? 0} sims
                </span>
                <span>{store.sessionProgress.toFixed(0)}%</span>
              </div>
              <div
                style={{
                  height: 4,
                  borderRadius: 99,
                  background: "var(--hairline)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${store.sessionProgress}%`,
                    background: "var(--accent-cyan)",
                    transition: "width 200ms var(--ease-out)",
                  }}
                />
              </div>
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                {running ? (
                  <button type="button" className="pill-button" onClick={() => void send("pause")}>
                    <Pause size={12} /> Pausar
                  </button>
                ) : (
                  <button type="button" className="pill-button" onClick={() => void send("resume")}>
                    <Play size={12} /> Continuar
                  </button>
                )}
                <button
                  type="button"
                  className="pill-button"
                  onClick={() => void send("step_once")}
                  disabled={!paused}
                >
                  <StepForward size={12} /> 1 passo
                </button>
                <button
                  type="button"
                  className="pill-button pill-button--accent"
                  onClick={() => void applyEditAndContinue()}
                  disabled={!paused}
                >
                  Aplicar edição
                </button>
                <button type="button" className="pill-button" onClick={() => void send("cancel")}>
                  <Square size={12} /> Cancelar
                </button>
              </div>
              <div className="stat-grid">
                <Stat label="Score" value={formatReward(sessionMeta?.reward_v33?.total_reward)} />
                <Stat label="Value" value={formatSigned(sessionMeta?.value, 3)} />
                <Stat label="Volume" value={formatPercent(sessionMeta?.volume_fraction)} />
                <Stat
                  label="Refutação"
                  value={sessionMeta?.mcts_stats?.refutation ? "SIM" : "não"}
                />
              </div>
            </>
          )}
          {store.sessionError ? (
            <div className="badge" data-tone="danger" style={{ height: "auto", padding: "6px 8px" }}>
              {store.sessionError}
            </div>
          ) : null}
        </section>

        {localError ? (
          <div className="empty-card" style={{ borderColor: "var(--accent-magenta)" }}>
            <AlertTriangle size={14} style={{ color: "var(--accent-magenta)", marginBottom: 4 }} />
            {localError}
          </div>
        ) : null}

        <div className="mono muted" style={{ fontSize: 9.5, lineHeight: 1.5 }}>
          Edições pausam o MCTS: clique Pausar → esculpa/mova a carga → Aplicar edição → Continuar.
          A árvore e o cache são resetados no servidor.
        </div>
      </div>
    </aside>
  );
}

function NumberField(props: {
  label: string;
  value: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span className="mono muted" style={{ fontSize: 8.5 }}>
        {props.label}
      </span>
      <input
        className="field-input"
        type="number"
        min={0}
        max={props.max}
        step={1}
        value={Math.round(props.value)}
        onChange={(event) =>
          props.onChange(Math.max(0, Math.min(props.max, Number(event.target.value))))
        }
        style={{ height: 26, padding: "0 5px", fontSize: 10.5 }}
      />
    </label>
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
