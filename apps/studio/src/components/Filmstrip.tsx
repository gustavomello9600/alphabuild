import { useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Pause, Play, Radio, SkipBack, SkipForward } from "lucide-react";
import type { Phase } from "../types/contracts";
import { formatPercent, formatReward, formatVolume } from "../lib/formatters";

export interface FilmstripData {
  total_steps: number;
  phase_history: Phase[];
  value_history: (number | null)[];
  volume_history: (number | null)[];
  compliance_history: (number | null)[];
  refutation_steps: number[];
}

interface FilmstripProps {
  data: FilmstripData;
  step: number;
  playing: boolean;
  speed: number;
  live?: boolean;
  followLive?: boolean;
  onStep: (step: number) => void;
  onTogglePlay: () => void;
  onSpeed: (speed: number) => void;
  onGoLive?: () => void;
}

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

/** Shared inspector/live-session filmstrip. */
export function Filmstrip({
  data,
  step,
  playing,
  speed,
  live = false,
  followLive = true,
  onStep,
  onTogglePlay,
  onSpeed,
  onGoLive,
}: FilmstripProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  const scrubbing = useRef(false);
  const maxStep = Math.max(0, data.total_steps - 1);
  const currentPhase = data.phase_history[step] ?? "GROWTH";
  const hoverIndex = hovered === null ? step : Math.min(maxStep, hovered);
  const hoverPhase = data.phase_history[hoverIndex] ?? "GROWTH";
  const hoverReward = data.value_history[hoverIndex] ?? 0;
  const hoverVolume = data.volume_history[hoverIndex] ?? null;
  const hoverCompliance = data.compliance_history[hoverIndex] ?? null;

  const { valuePath, volumePath, growthPercent } = useMemo(() => {
    const values = data.value_history;
    const volumes = data.volume_history;
    const numericValues = values.filter((item): item is number => Number.isFinite(item));
    const minValue = Math.min(-1, ...numericValues);
    const maxValue = Math.max(1, ...numericValues);
    const numericVolumes = volumes.filter((item): item is number => item !== null);
    const maxVolume = Math.max(0.001, ...numericVolumes);
    const growth = data.phase_history.filter((phase) => phase === "GROWTH").length;
    return {
      valuePath: buildPath(values, maxValue, minValue),
      volumePath: buildPath(volumes, maxVolume, 0),
      growthPercent: data.phase_history.length
        ? (growth / data.phase_history.length) * 100
        : 0,
    };
  }, [data]);

  function seekFromPointer(clientX: number, element: HTMLElement) {
    const rect = element.getBoundingClientRect();
    if (rect.width <= 0) return;
    const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    onStep(Math.round(fraction * maxStep));
  }

  const percentage = maxStep === 0 ? 0 : (step / maxStep) * 100;

  return (
    <div className="timeline-card glass">
      <div className="timeline-top">
        <div className="timeline-controls">
          <button type="button" className="icon-button" onClick={() => onStep(0)} title="Primeiro passo">
            <SkipBack size={13} />
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => onStep(step - 1)}
            title="Passo anterior (←)"
          >
            <ChevronLeft size={14} />
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={onTogglePlay}
            data-active={playing}
            title="Play/Pause (Espaço)"
          >
            {playing ? <Pause size={13} /> : <Play size={13} />}
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => onStep(step + 1)}
            title="Próximo passo (→)"
          >
            <ChevronRight size={14} />
          </button>
          <button type="button" className="icon-button" onClick={() => onStep(maxStep)} title="Último passo">
            <SkipForward size={13} />
          </button>
          <select
            className="field-input"
            style={{ width: 64, height: 26, fontSize: 10 }}
            value={speed}
            onChange={(event) => onSpeed(Number(event.target.value))}
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
          {live ? (
            <span className="badge" data-tone={followLive ? "accent" : "warn"}>
              <Radio size={9} />
              {followLive ? "AO VIVO" : "REPLAY"}
            </span>
          ) : null}
          <span>
            Passo {step} / {maxStep}
          </span>
          <span className="accent-cyan">R {formatReward(hoverReward)}</span>
          <span style={{ color: "var(--accent-gold)" }}>V {formatVolume(hoverVolume)}</span>
          <span>C {hoverCompliance === null ? "—" : hoverCompliance.toFixed(1)}</span>
          {live && !followLive && onGoLive ? (
            <button type="button" className="pill-button pill-button--accent" onClick={onGoLive}>
              <Radio size={11} /> Voltar ao vivo
            </button>
          ) : null}
        </div>
      </div>

      <div
        className="timeline-track"
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          scrubbing.current = true;
          seekFromPointer(event.clientX, event.currentTarget);
        }}
        onPointerMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          if (rect.width > 0) {
            const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
            setHovered(Math.round(fraction * maxStep));
          }
          if (scrubbing.current) seekFromPointer(event.clientX, event.currentTarget);
        }}
        onPointerUp={(event) => {
          scrubbing.current = false;
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }}
        onPointerCancel={() => {
          scrubbing.current = false;
        }}
        onPointerLeave={() => {
          if (!scrubbing.current) setHovered(null);
        }}
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
          <path d={volumePath} fill="none" stroke="var(--accent-gold)" strokeWidth={0.9} opacity={0.9} />
        </svg>
        {data.refutation_steps.map((refutation) => (
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
          onChange={(event) => onStep(Number(event.target.value))}
          aria-label="Passo temporal"
        />
      </div>
    </div>
  );
}
