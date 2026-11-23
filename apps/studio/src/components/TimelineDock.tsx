import { useEffect } from "react";
import { useInspectorStore } from "../store/useInspectorStore";
import { Filmstrip } from "./Filmstrip";

export function TimelineDock() {
  const metadata = useInspectorStore((state) => state.metadata);
  const step = useInspectorStore((state) => state.step);
  const setStep = useInspectorStore((state) => state.setStep);
  const playing = useInspectorStore((state) => state.playing);
  const setPlaying = useInspectorStore((state) => state.setPlaying);
  const togglePlaying = useInspectorStore((state) => state.togglePlaying);
  const playbackSpeed = useInspectorStore((state) => state.playbackSpeed);
  const setPlaybackSpeed = useInspectorStore((state) => state.setPlaybackSpeed);

  const maxStep = metadata ? Math.max(0, metadata.total_steps - 1) : 0;

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

  if (!metadata) {
    return (
      <div className="timeline-card glass">
        <div className="empty-card">
          Abra um episódio Harvest ou uma partida Self-Play para ativar o filmstrip temporal.
        </div>
      </div>
    );
  }

  return (
    <Filmstrip
      data={{
        total_steps: metadata.total_steps,
        phase_history: metadata.phase_history,
        value_history: metadata.value_history,
        volume_history: metadata.volume_history,
        compliance_history: metadata.compliance_history,
        refutation_steps: metadata.refutation_steps,
      }}
      step={step}
      playing={playing}
      speed={playbackSpeed}
      onStep={(value) => {
        setPlaying(false);
        setStep(value);
      }}
      onTogglePlay={togglePlaying}
      onSpeed={setPlaybackSpeed}
    />
  );
}
