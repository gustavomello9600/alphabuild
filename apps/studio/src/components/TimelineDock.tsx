import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useInspectorStore } from "../store/useInspectorStore";
import { formatPercent, formatReward, formatVolume } from "../lib/formatters";

function buildPath(values: Array<number | null>, max: number, min: number): string {
  const points: string[] = [];
  const denominator = Math.max(1, values.length - 1);
  values.forEach((value, index) => {
    const x = (index / denominator) * 100;
    const normalized =
      value === null || !Number.isFinite(value) || max === min
        ? 0
        : (value - min) / (max - min);
    const y = 100 - Math.min(100, Math.max(0, normalized)) * 100;
    points.push(`${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`);
  });
  return points.join(" ");
}

export function TimelineDock() {
  const metadata = useInspectorStore((state) => state.metadata);
  const step = useInspectorStore((state) => state.step);
  const setStep = useInspectorStore((state) => state.setStep);
  const playing = useInspectorStore((state) => state.playing);
  const setPlaying = useInspectorStore((state) => state.setPlaying);
  const togglePlaying = useInspectorStore((state) => state.togglePlaying);
  const playbackSpeed = useInspectorStore((state) => state.playbackSpeed);
  const setPlaybackSpeed = useInspectorStore((state) => state.setPlaybackSpeed);
  const [hovered, setHovered] = useState<number | null>(null);

  const total = metadata?.total_steps ?? 0;
  const maxStep = Math.max(0, total - 1);
  const currentPhase = metadata?.phase_history[step] ?? "GROWTH";

  useEffect(() => {
    if (!playing || !metadata) return;
    if (step >= maxStep) {
      setPlaying(false);
      return;
    }
    const handle = window.setInterval(() => {
      const state = useInspectorStore.getState();
      if (state.step >= maxStep) {
        state.setPlaying(false);
        return;
      }
      state.setStep(state.step + 1);
    }, Math.max(40, 160 / Math.max(0.1, playbackSpeed)));
    return () => window.clearInterval(handle);
  }, [playing, metadata, maxStep, playbackSpeed, step, setPlaying]);

  const { valuePath, volumePath, growthPercent } = useMemo(() => {
    const values = metadata?.value_history ?? [];
    const volumes = metadata?.volume_history ?? [];
    const numericValues = values.filter((item): item is number => Number.isFinite(item));
    const minValue = Math.min(-1, ...numericValues);
    const maxValue = Math.max(1, ...numericValues);
    const numericVolumes = volumes.filter((item): item is number => item !== null);
    const maxVolume = Math.max(0.001, ...numericVolumes);
    const phases = metadata?.phase_history ?? [];
    const growth = phases.filter((phase) => phase === "GROWTH").length;
    return {
      valuePath: buildPath(values, maxValue, minValue),
      volumePath: buildPath(volumes, maxVolume, 0),
      growthPercent: phases.length ? (growth / phases.length) * 100 : 0,
    };
  }, [metadata]);

  if (!metadata) {
    return (
      <div className="timeline-card glass">
        <div className="empty-card">
          Abra um episódio Harvest ou uma partida Self-Play para ativar o filmstrip temporal.
        </div>
      </div>
    );
  }

  const percentage = maxStep === 0 ? 0 : (step / maxStep) * 100;
  const hoverIndex = hovered === null ? step : Math.min(maxStep, hovered);
  const hoverPhase = metadata.phase_history[hoverIndex] ?? "GROWTH";
  const hoverReward = metadata.value_history[hoverIndex] ?? 0;
  const hoverVolume = metadata.volume_history[hoverIndex] ?? null;
  const hoverCompliance = metadata.compliance_history[hoverIndex] ?? null;

  return (
    <div className="timeline-card glass">
      <div className="timeline-top">
        <div className="timeline-controls">
          <button
            type="button"
            className="icon-button"
            onClick={() => setStep(0)}
            title="Primeiro passo"
          >
            <SkipBack size={13} />
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => setStep(step - 1)}
            title="Passo anterior (←)"
          >
            <ChevronLeft size={14} />
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={togglePlaying}
            data-active={playing}
            title="Play/Pause (Espaço)"
          >
            {playing ? <Pause size={13} /> : <Play size={13} />}
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => setStep(step + 1)}
            title="Próximo passo (→)"
          >
            <ChevronRight size={14} />
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => setStep(maxStep)}
            title="Último passo"
          >
            <SkipForward size={13} />
          </button>
          <select
            className="field-input"
            style={{ width: 64, height: 26, fontSize: 10 }}
            value={playbackSpeed}
            onChange={(event) => setPlaybackSpeed(Number(event.target.value))}
            title="Velocidade"
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
            <option value={8}>8×</option>
          </select>
        </div>

        <div className="timeline-meta">
          <span className="badge" data-tone={currentPhase === "GROWTH" ? "growth" : "refinement"}>
            {currentPhase === "GROWTH" ? "FASE 1 · GROWTH" : "FASE 2 · REFINEMENT"}
          </span>
          <span>
            Passo {step} / {maxStep}
          </span>
          <span className="accent-cyan">R {formatReward(hoverReward)}</span>
          <span style={{ color: "var(--accent-gold)" }}>V {formatVolume(hoverVolume)}</span>
          <span>C {hoverCompliance === null ? "—" : hoverCompliance.toFixed(1)}</span>
        </div>
      </div>

      <div
        className="timeline-track"
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          if (rect.width <= 0) return;
          const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
          setHovered(Math.round(fraction * maxStep));
        }}
        onMouseLeave={() => setHovered(null)}
        title={`Passo ${hoverIndex} · ${hoverPhase} · R=${formatReward(hoverReward)} · C=${hoverCompliance ?? "—"} · V=${formatPercent(hoverVolume)}`}
      >
        {growthPercent > 0 ? (
          <div
            className="phase-band phase-band--growth"
            style={{ left: 0, width: `${growthPercent}%` }}
          />
        ) : null}
        {growthPercent < 100 ? (
          <div
            className="phase-band phase-band--refinement"
            style={{ left: `${growthPercent}%`, right: 0 }}
          />
        ) : null}
        <svg className="timeline-spark" viewBox="0 0 100 100" preserveAspectRatio="none">
          <path d={valuePath} fill="none" stroke="var(--accent-cyan)" strokeWidth={1.1} />
          <path
            d={volumePath}
            fill="none"
            stroke="var(--accent-gold)"
            strokeWidth={0.9}
            opacity={0.9}
          />
        </svg>
        {metadata.refutation_steps.map((refutation) => (
          <span
            key={refutation}
            className="timeline-marker"
            style={{ left: `${maxStep === 0 ? 0 : (refutation / maxStep) * 100}%` }}
            title={`Refutação MCTS no passo ${refutation}`}
          />
        ))}
        <span className="timeline-cursor" style={{ left: `${percentage}%` }} />
        <input
          type="range"
          className="timeline-scrub"
          min={0}
          max={Math.max(1, maxStep)}
          step={1}
          value={step}
          onChange={(event) => setStep(Number(event.target.value))}
          aria-label="Passo temporal"
        />
      </div>
    </div>
  );
}
