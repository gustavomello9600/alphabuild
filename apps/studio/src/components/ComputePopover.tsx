import { useEffect, useState } from "react";
import {
  Activity,
  Check,
  Cpu,
  Play,
  RefreshCw,
  Server,
  Square,
  X,
} from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { useComputeStore } from "../store/useComputeStore";
import { formatMilliseconds } from "../lib/formatters";
import type { ComputeProfile } from "../types/contracts";

export function ComputePopover() {
  const open = useAppStore((state) => state.computeOpen);
  const setOpen = useAppStore((state) => state.setComputeOpen);
  const store = useComputeStore();
  const [harvestUrl, setHarvestUrl] = useState("");
  const [pythonBin, setPythonBin] = useState("python3");
  const [port, setPort] = useState(8765);
  const [draftName, setDraftName] = useState("");
  const [draftUrl, setDraftUrl] = useState("");
  const [draftToken, setDraftToken] = useState("");
  const [draftMode, setDraftMode] = useState<ComputeProfile["mode"]>("remote");
  const [draftPython, setDraftPython] = useState("python3");

  useEffect(() => {
    if (!open) return;
    void store.loadProfiles();
  }, [open]);

  useEffect(() => {
    const selected = store.profiles.find((profile) => profile.id === store.activeProfileId) ?? store.profiles[0];
    if (!selected) return;
    setDraftName(selected.name);
    setDraftUrl(selected.base_url);
    setDraftToken(selected.api_token ?? "");
    setDraftMode(selected.mode);
    setDraftPython(selected.python_bin ?? "python3");
  }, [store.profiles, store.activeProfileId]);

  useEffect(() => {
    if (!open || !store.connection?.connected) return;
    void store.loadModels();
  }, [open, store.connection?.connected]);

  if (!open) return null;

  return (
    <aside className="compute-popover glass">
      <div className="dock-header">
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <Cpu size={13} className="accent-cyan" />
          <span className="dock-title">Compute & Inference</span>
        </div>
        <button type="button" className="icon-button" onClick={() => setOpen(false)}>
          <X size={14} />
        </button>
      </div>
      <div style={{ padding: 10, display: "flex", flexDirection: "column", gap: 12 }}>
        <section className="panel-section">
          <div className="panel-heading">
            <span>Perfis</span>
            <span className="mono">{store.profiles.length}</span>
          </div>
          {store.profiles.map((profile) => (
            <button
              key={profile.id}
              type="button"
              className="db-row"
              data-active={store.activeProfileId === profile.id}
              onClick={() => void store.activateProfile(profile.id)}
            >
              <span className="db-row__title">
                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <Server size={12} className="accent-cyan" />
                  {profile.name}
                </span>
                {store.connection?.profile_id === profile.id && store.connection.connected ? (
                  <Check size={12} style={{ color: "var(--accent-emerald)" }} />
                ) : null}
              </span>
              <span className="db-row__meta">
                {profile.mode} · {profile.base_url}
              </span>
            </button>
          ))}
          {store.profiles.length === 0 ? (
            <div className="empty-card">
              Nenhum perfil persistido. O aplicativo desktop cria os três perfis padrão.
            </div>
          ) : null}
          <div className="divider" />
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div className="panel-heading">
              <span>Editar perfil</span>
              <span className="mono">{store.activeProfileId ?? "novo"}</span>
            </div>
            <input
              className="field-input"
              placeholder="Nome do perfil"
              value={draftName}
              onChange={(event) => setDraftName(event.target.value)}
            />
            <input
              className="field-input"
              placeholder="http://127.0.0.1:8765"
              value={draftUrl}
              onChange={(event) => setDraftUrl(event.target.value)}
            />
            <input
              className="field-input"
              type="password"
              placeholder="Bearer token (vazio em loopback)"
              value={draftToken}
              onChange={(event) => setDraftToken(event.target.value)}
            />
            <div style={{ display: "flex", gap: 4 }}>
              <select
                className="field-input"
                value={draftMode}
                onChange={(event) => setDraftMode(event.target.value as ComputeProfile["mode"])}
              >
                <option value="remote">remote</option>
                <option value="local_sidecar">local_sidecar</option>
              </select>
              <input
                className="field-input"
                placeholder="python3"
                value={draftPython}
                onChange={(event) => setDraftPython(event.target.value)}
              />
            </div>
            <button
              type="button"
              className="pill-button"
              onClick={() => {
                const id =
                  store.activeProfileId ??
                  `profile-${Date.now().toString(36)}`;
                const profile: ComputeProfile = {
                  id,
                  name: draftName || "Novo perfil",
                  mode: draftMode,
                  base_url: draftUrl,
                  api_token: draftToken,
                  python_bin: draftPython || undefined,
                };
                void store.saveProfile(profile);
              }}
            >
              Salvar perfil
            </button>
          </div>
        </section>

        {store.connection ? (
          <section className="panel-section">
            <div className="panel-heading">
              <span>Conexão</span>
              <span className="badge" data-tone={store.connection.connected ? "growth" : "danger"}>
                {store.connection.connected ? "ONLINE" : "OFFLINE"}
              </span>
            </div>
            <div className="stat-grid">
              <Stat label="RTT" value={formatMilliseconds(store.connection.rtt_ms)} />
              <Stat
                label="Device"
                value={store.activeModel?.device_resolved ?? store.capabilities?.devices.find((d) => d.available)?.id ?? "—"}
              />
            </div>
            {store.capabilities ? (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {Object.entries(store.capabilities.features)
                  .filter(([key]) => key.startsWith("has_"))
                  .map(([key, value]) => (
                    <span
                      key={key}
                      className="badge"
                      data-tone={value ? "accent" : undefined}
                      title={key}
                    >
                      {key.replace("has_", "")}
                    </span>
                  ))}
              </div>
            ) : null}
            {store.connection.error ? (
              <div className="empty-card" style={{ color: "var(--accent-magenta)" }}>
                {store.connection.error}
              </div>
            ) : null}
          </section>
        ) : null}

        <section className="panel-section">
          <div className="panel-heading">
            <span>Sidecar local</span>
            <span className="badge" data-tone={store.sidecar?.running ? "growth" : undefined}>
              {store.sidecar?.running ? "RUNNING" : "STOPPED"}
            </span>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <input
              className="field-input"
              style={{ width: 110 }}
              value={pythonBin}
              onChange={(event) => setPythonBin(event.target.value)}
              placeholder="python3"
            />
            <input
              className="field-input"
              style={{ width: 64 }}
              type="number"
              value={port}
              onChange={(event) => setPort(Number(event.target.value))}
            />
            <button
              type="button"
              className="pill-button pill-button--accent"
              onClick={() => void store.startSidecar(pythonBin, port)}
            >
              <Play size={12} /> Iniciar
            </button>
            <button
              type="button"
              className="pill-button"
              onClick={() => void store.stopSidecar(port)}
            >
              <Square size={12} /> Parar
            </button>
            <button
              type="button"
              className="icon-button"
              onClick={() => void store.refreshSidecar(port)}
            >
              <RefreshCw size={12} />
            </button>
          </div>
          <div className="mono" style={{ fontSize: 9.5, color: "var(--text-muted)" }}>
            {store.sidecar?.message ?? "python -m alphabuilder.server"}
          </div>
        </section>

        <section className="panel-section">
          <div className="panel-heading">
            <span>Checkpoints</span>
            <button
              type="button"
              className="icon-button"
              onClick={() => void store.loadModels()}
              title="Atualizar modelos"
            >
              <RefreshCw size={12} />
            </button>
          </div>
          {store.activeModel ? (
            <div className="reward-row">
              <span style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                <span className="reward-row__name">{store.activeModel.model_id}</span>
                <span className="reward-row__formula">
                  {store.activeModel.architecture} · {store.activeModel.device_resolved}
                </span>
              </span>
              <button
                type="button"
                className="pill-button"
                onClick={() => void store.unloadModel()}
              >
                Unload
              </button>
            </div>
          ) : null}
          {store.models.length === 0 ? (
            <div className="empty-card">Nenhum *.pt/*.pth listado no servidor.</div>
          ) : (
            store.models.slice(0, 8).map((model) => (
              <button
                key={model.model_id}
                type="button"
                className="entry-row"
                onClick={() => void store.loadModel(model.model_id)}
              >
                <span className="entry-row__id">{model.model_id}</span>
                <span className="entry-row__metrics">{model.size_mb.toFixed(1)} MB</span>
              </button>
            ))
          )}
        </section>

        <section className="panel-section">
          <div className="panel-heading">
            <span>Harvest Supervisor</span>
            <Activity size={12} />
          </div>
          <input
            className="field-input"
            placeholder="http://100.x.x.x:8450"
            value={harvestUrl}
            onChange={(event) => setHarvestUrl(event.target.value)}
          />
          <button
            type="button"
            className="pill-button"
            onClick={() => void store.refreshHarvest(harvestUrl || undefined)}
          >
            Consultar status (read-only)
          </button>
          {store.harvest ? (
            <div className="badge" data-tone={store.harvest.reachable ? "growth" : "warn"}>
              {store.harvest.configured
                ? store.harvest.reachable
                  ? "REACHABLE"
                  : store.harvest.error ?? "UNREACHABLE"
                : "NÃO CONFIGURADO"}
            </div>
          ) : null}
        </section>

        {store.error ? (
          <div className="empty-card" style={{ color: "var(--accent-magenta)" }}>
            {store.error}
          </div>
        ) : null}
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
