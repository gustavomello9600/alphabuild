import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Filmstrip } from "./Filmstrip";
import { parseAbMesh01 } from "../lib/abmesh";
import { getActiveEngine } from "../lib/engineBus";
import { sessionGetStepMesh } from "../lib/ipc";
import { useStudioStore } from "../store/useStudioStore";
import type { Phase } from "../types/contracts";

export function StudioTimelineDock() {
  const session = useStudioStore((state) => state.session);
  const history = useStudioStore((state) => state.sessionHistory);
  const viewStep = useStudioStore((state) => state.sessionViewStep);
  const followLive = useStudioStore((state) => state.sessionFollowLive);
  const status = useStudioStore((state) => state.sessionStatus);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const requestRef = useRef(0);

  const maxStep = history.length > 0 ? (history[history.length - 1]?.step ?? 0) : 0;
  const selected = Math.min(maxStep, viewStep ?? maxStep);

  const data = useMemo(() => {
    const size = maxStep + 1;
    const phase_history: Phase[] = Array.from({ length: size }, () => "GROWTH");
    const value_history: (number | null)[] = Array.from({ length: size }, () => null);
    const volume_history: (number | null)[] = Array.from({ length: size }, () => null);
    const compliance_history: (number | null)[] = Array.from({ length: size }, () => null);
    const refutation_steps: number[] = [];
    for (const entry of history) {
      if (entry.step < 0 || entry.step >= size) continue;
      phase_history[entry.step] = entry.metadata.phase;
      value_history[entry.step] = entry.metadata.value;
      volume_history[entry.step] = entry.metadata.volume_fraction;
      compliance_history[entry.step] = entry.metadata.compliance_fem;
      if (entry.metadata.mcts_stats?.refutation) refutation_steps.push(entry.step);
    }
    return {
      total_steps: size,
      phase_history,
      value_history,
      volume_history,
      compliance_history,
      refutation_steps,
    };
  }, [history, maxStep]);

  const selectStep = useCallback(
    async (target: number) => {
      const store = useStudioStore.getState();
      if (!store.session) return;
      let clamped = Math.max(0, Math.min(maxStep, Math.round(target)));
      if (!store.sessionHistory.some((entry) => entry.step === clamped)) {
        const fallback = store.sessionHistory.reduce(
          (best, entry) =>
            Math.abs(entry.step - clamped) < Math.abs(best - clamped) ? entry.step : best,
          store.sessionHistory[0]?.step ?? -1,
        );
        if (fallback < 0) return;
        clamped = fallback;
      }
      const latest = clamped >= maxStep;
      const request = ++requestRef.current;

      if (latest && store.sessionFrame) {
        store.setSessionFollowLive(true);
        store.setSessionViewFrame(store.sessionFrame, clamped);
        getActiveEngine()?.applyFrame(store.sessionFrame);
        return;
      }
      store.setSessionFollowLive(false);
      try {
        const buffer = await sessionGetStepMesh(store.session.session_id, clamped);
        if (requestRef.current !== request) return;
        const parsed = parseAbMesh01(buffer);
        store.setSessionViewFrame(parsed, clamped);
        getActiveEngine()?.applyFrame(parsed);
      } catch (error) {
        store.setSessionError(error instanceof Error ? error.message : String(error));
      }
    },
    [maxStep],
  );

  useEffect(() => {
    if (!playing || history.length === 0) return;
    const handle = window.setInterval(() => {
      const store = useStudioStore.getState();
      const current = store.sessionViewStep ?? maxStep;
      if (current >= maxStep) {
        setPlaying(false);
        return;
      }
      void selectStep(current + 1);
    }, Math.max(40, 180 / Math.max(0.1, speed)));
    return () => window.clearInterval(handle);
  }, [playing, speed, history.length, maxStep, selectStep]);

  useEffect(() => {
    if (status === "IDLE" || status === "CANCELLED") setPlaying(false);
  }, [status]);

  if (!session || history.length === 0) {
    return (
      <div className="timeline-card glass">
        <div className="empty-card">
          {session
            ? "Sessão iniciada. O filmstrip e a sparkline serão construídos ao vivo a cada passo do MCTS."
            : "Inicie uma sessão de MCTS ao vivo no Estúdio para acompanhar a evolução em tempo real."}
        </div>
      </div>
    );
  }

  return (
    <Filmstrip
      data={data}
      step={selected}
      playing={playing}
      speed={speed}
      live
      followLive={followLive}
      onStep={(value) => {
        setPlaying(false);
        void selectStep(value);
      }}
      onTogglePlay={() => {
        if (!playing && selected >= maxStep) void selectStep(0);
        setPlaying((value) => !value);
      }}
      onSpeed={setSpeed}
      onGoLive={() => {
        setPlaying(false);
        void selectStep(maxStep);
      }}
    />
  );
}
