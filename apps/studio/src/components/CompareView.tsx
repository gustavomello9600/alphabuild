import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeftRight, Camera, Columns2, Download, FolderOpen, RefreshCw, X } from "lucide-react";
import { parseAbMesh01 } from "../lib/abmesh";
import { getEngine, registerEngine, unregisterEngine } from "../lib/engineBus";
import {
  exportPngBytes,
  getFrameMesh,
  isTauriRuntime,
  pickSavePath,
} from "../lib/ipc";
import { formatNumber, formatPercent, formatReward, formatSigned } from "../lib/formatters";
import { useAppStore } from "../store/useAppStore";
import { useCompareStore } from "../store/useCompareStore";
import { ViewportEngine } from "../engine3d/ViewportEngine";
import type { ParsedAbMesh } from "../types/contracts";

function loadImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Falha ao compor captura PNG"));
    image.src = source;
  });
}

export function CompareView() {
  const store = useCompareStore();
  const theme = useAppStore((state) => state.theme);
  const setLibraryOpen = useAppStore((state) => state.setLibraryOpen);
  const engineRefs = useRef<{ a: ViewportEngine | null; b: ViewportEngine | null }>({
    a: null,
    b: null,
  });
  const [engineTick, setEngineTick] = useState(0);
  const [split, setSplit] = useState(50);
  const [exporting, setExporting] = useState(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);
  const draggingRef = useRef(false);

  const onEngine = useCallback((slot: "a" | "b", engine: ViewportEngine | null) => {
    engineRefs.current[slot] = engine;
    setEngineTick((value) => value + 1);
  }, []);

  useEffect(() => {
    const a = engineRefs.current.a;
    const b = engineRefs.current.b;
    if (!a || !b) return;
    if (store.syncCamera) {
      a.syncWith(b);
      b.syncWith(a);
    } else {
      a.unsync();
      b.unsync();
    }
    return () => {
      a.unsync();
      b.unsync();
    };
  }, [engineTick, store.syncCamera]);

  useEffect(() => {
    if (!store.slotA || !store.slotB) return;
    const timer = window.setTimeout(() => {
      void store.compare();
    }, 180);
    return () => window.clearTimeout(timer);
  }, [store.slotA, store.slotB, store.stepA, store.stepB]);

  useEffect(() => {
    if (store.linkMode === "normalized") {
      const maxA = store.slotA ? Math.max(0, store.slotA.metadata.total_steps - 1) : 0;
      if (maxA > 0) store.setFraction(store.stepA / maxA);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.slotA?.metadata.total_steps, store.slotB?.metadata.total_steps]);

  useEffect(() => {
    const handleMove = (event: PointerEvent) => {
      if (!draggingRef.current) return;
      const fraction = Math.min(82, Math.max(18, (event.clientX / window.innerWidth) * 100));
      setSplit(fraction);
    };
    const handleUp = () => {
      draggingRef.current = false;
    };
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
    };
  }, []);

  const maxA = store.slotA ? Math.max(0, store.slotA.metadata.total_steps - 1) : 0;
  const maxB = store.slotB ? Math.max(0, store.slotB.metadata.total_steps - 1) : 0;
  const masterMax = store.linkMode === "normalized" ? 1000 : Math.max(maxA, maxB);
  const masterValue =
    store.linkMode === "normalized"
      ? Math.round(store.fraction * 1000)
      : Math.min(store.stepA, store.stepB);

  async function exportComposite() {
    const a = getEngine("compare-a");
    const b = getEngine("compare-b");
    if (!a || !b) {
      setExportMessage("Os dois viewports precisam estar montados.");
      return;
    }
    if (!isTauriRuntime()) {
      setExportMessage("Exportação exige o aplicativo desktop Tauri.");
      return;
    }
    setExporting(true);
    setExportMessage(null);
    try {
      const [left, right] = await Promise.all([
        loadImage(a.captureScreenshot(1920, 2160, false)),
        loadImage(b.captureScreenshot(1920, 2160, false)),
      ]);
      const canvas = document.createElement("canvas");
      canvas.width = 3840;
      canvas.height = 2160;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Canvas 2D indisponível para composição A/B");
      context.fillStyle = "#050608";
      context.fillRect(0, 0, 3840, 2160);
      context.drawImage(left, 0, 0, 1920, 2160);
      context.drawImage(right, 1920, 0, 1920, 2160);
      const dataUrl = canvas.toDataURL("image/png");
      const path = await pickSavePath("alphabuilder_compare_AB_4k.png", "PNG A/B 4K", "png");
      if (!path) return;
      const saved = await exportPngBytes(dataUrl, path);
      setExportMessage(`${saved.bytes_written.toLocaleString("en-US")} bytes → ${saved.output_path}`);
    } catch (error) {
      setExportMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setExporting(false);
    }
  }

  return (
    <section className="compare-root">
      <div
        className="compare-grid"
        style={{ gridTemplateColumns: `${split}% ${100 - split}%` }}
      >
        <ComparePane
          slot="a"
          onEngine={onEngine}
          theme={theme}
          onActivate={() => store.setActiveSlot("a")}
        />
        <ComparePane
          slot="b"
          onEngine={onEngine}
          theme={theme}
          onActivate={() => store.setActiveSlot("b")}
        />
      </div>
      <div
        className="compare-divider"
        style={{ left: `${split}%` }}
        onPointerDown={() => {
          draggingRef.current = true;
        }}
        title="Arraste para redimensionar"
      >
        <span />
      </div>

      <div className="compare-toolbar glass">
        <button
          type="button"
          className="pill-button"
          data-active={store.activeSlot === "a"}
          onClick={() => store.setActiveSlot("a")}
        >
          <Columns2 size={12} /> Slot A
        </button>
        <button
          type="button"
          className="pill-button"
          data-active={store.activeSlot === "b"}
          onClick={() => store.setActiveSlot("b")}
        >
          <Columns2 size={12} /> Slot B
        </button>
        <button type="button" className="pill-button" onClick={() => setLibraryOpen(true)}>
          <FolderOpen size={12} /> Carregar
        </button>
        <span className="divider" style={{ width: 1, height: 18, margin: "0 3px" }} />
        <button
          type="button"
          className="pill-button"
          data-active={store.linkMode === "index"}
          onClick={() => store.setNormalized(false)}
          title="Sincronizar por índice de passo"
        >
          <ArrowLeftRight size={12} /> Índice
        </button>
        <button
          type="button"
          className="pill-button"
          data-active={store.linkMode === "normalized"}
          onClick={() => store.setNormalized(true)}
          title="Sincronizar por fração 0–100%"
        >
          <RefreshCw size={12} /> Normalizado
        </button>
        <button
          type="button"
          className="pill-button"
          data-active={store.syncCamera}
          onClick={() => store.setSyncCamera(!store.syncCamera)}
        >
          <Camera size={12} /> Sync câmera
        </button>
        <button
          type="button"
          className="pill-button"
          onClick={() => {
            engineRefs.current.a?.resetCamera();
            engineRefs.current.b?.resetCamera();
          }}
        >
          Reset
        </button>
        <button
          type="button"
          className="pill-button pill-button--accent"
          onClick={() => void exportComposite()}
          disabled={exporting || !store.slotA || !store.slotB}
        >
          <Download size={12} /> PNG A/B 4K
        </button>
      </div>

      <div className="compare-delta glass">
        <div className="compare-delta__head">
          <div className="timeline-meta">
            <span className="badge" data-tone="accent">
              A/B SYNC
            </span>
            <span>Passo A {store.stepA}</span>
            <span>Passo B {store.stepB}</span>
            <span>{store.linkMode === "normalized" ? formatPercent(store.fraction) : "índice"}</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, maxWidth: 520 }}>
            <input
              type="range"
              min={0}
              max={Math.max(1, masterMax)}
              step={1}
              value={masterValue}
              onChange={(event) => {
                const value = Number(event.target.value);
                if (store.linkMode === "normalized") store.setFraction(value / 1000);
                else store.setLinkedStep(value);
              }}
              aria-label="Timeline sincronizada A/B"
            />
          </div>
        </div>
        {store.metrics ? (
          <div className="compare-metrics">
            <Metric label="IoU 3D" value={store.metrics.iou_3d.toFixed(4)} tone="accent" />
            <Metric
              label="Δ compliance"
              value={formatNumber(store.metrics.delta_compliance, 2)}
            />
            <Metric
              label="Δ compliance %"
              value={
                store.metrics.delta_compliance_pct === null
                  ? "—"
                  : `${formatSigned(store.metrics.delta_compliance_pct, 2)}%`
              }
            />
            <Metric label="Δ volume" value={formatPercent(store.metrics.delta_volume)} />
            <Metric label="Δ reward" value={formatSigned(store.metrics.delta_reward, 4)} />
            <Metric
              label="Δ desloc. máx"
              value={formatNumber(store.metrics.delta_max_displacement, 2)}
            />
            <Metric label="Δ passos" value={String(store.metrics.delta_steps)} />
          </div>
        ) : (
          <div className="timeline-meta" style={{ paddingTop: 6 }}>
            <span>
              Carregue dois episódios para calcular IoU 3D (|A∩B| / |A∪B|) e deltas exatos em Rust.
            </span>
          </div>
        )}
        {store.error ? <span className="badge" data-tone="danger">{store.error}</span> : null}
        {exportMessage ? (
          <span className="badge" data-tone="success" style={{ height: "auto", padding: "5px 7px" }}>
            {exportMessage}
          </span>
        ) : null}
      </div>
    </section>
  );
}

function ComparePane(props: {
  slot: "a" | "b";
  theme: "dark" | "tcc-light";
  onEngine: (slot: "a" | "b", engine: ViewportEngine | null) => void;
  onActivate: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<ViewportEngine | null>(null);
  const requestRef = useRef(0);
  const slot = useCompareStore((state) =>
    props.slot === "a" ? state.slotA : state.slotB,
  );
  const step = useCompareStore((state) => (props.slot === "a" ? state.stepA : state.stepB));
  const activeSlot = useCompareStore((state) => state.activeSlot);
  const [frame, setFrame] = useState<ParsedAbMesh | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const engine = new ViewportEngine(canvas);
    engineRef.current = engine;
    registerEngine(`compare-${props.slot}`, engine);
    props.onEngine(props.slot, engine);
    return () => {
      unregisterEngine(`compare-${props.slot}`);
      props.onEngine(props.slot, null);
      engine.dispose();
      engineRef.current = null;
    };
  }, [props.slot, props.onEngine]);

  useEffect(() => {
    engineRef.current?.setTheme(props.theme === "tcc-light");
  }, [props.theme]);

  useEffect(() => {
    if (!slot) {
      setFrame(null);
      engineRef.current?.clearFrame();
      return;
    }
    const request = ++requestRef.current;
    setLoading(true);
    setError(null);
    const timer = window.setTimeout(() => {
      void getFrameMesh(slot.source, slot.entryId, step, {
        density_threshold: 0.5,
        blueprint_threshold: 0.5,
        use_rho_cont: false,
        include_diff_next: false,
        include_blueprint: slot.metadata.has_blueprint,
      })
        .then((buffer) => {
          if (requestRef.current !== request) return;
          const parsed = parseAbMesh01(buffer);
          setFrame(parsed);
          setLoading(false);
          const engine = engineRef.current;
          if (engine) {
            engine.applyFrame(parsed);
            engine.setBoundaries(slot.metadata.supports, slot.metadata.forces, slot.metadata.bc_type);
            engine.setLens("structure");
            engine.setBlueprintVisible(false);
          }
        })
        .catch((cause: unknown) => {
          if (requestRef.current !== request) return;
          setLoading(false);
          setError(cause instanceof Error ? cause.message : String(cause));
        });
    }, 40);
    return () => window.clearTimeout(timer);
  }, [slot, step]);

  const meta = frame?.metadata;
  const phase = meta?.phase ?? (slot ? slot.metadata.phase_history[step] : null);

  return (
    <article
      className="compare-pane"
      data-active={activeSlot === props.slot}
      onClick={props.onActivate}
    >
      <canvas ref={canvasRef} />
      <header className="compare-pane__header glass">
        <span className="badge" data-tone={activeSlot === props.slot ? "accent" : undefined}>
          SLOT {props.slot.toUpperCase()}
        </span>
        {slot ? (
          <>
            <span className="mono" style={{ fontSize: 9.5 }}>
              {slot.entryId.slice(0, 18)}
            </span>
            <span className="badge" data-tone={phase === "GROWTH" ? "growth" : "refinement"}>
              {phase}
            </span>
            <span className="mono muted" style={{ fontSize: 9.5 }}>
              {step + 1}/{slot.metadata.total_steps}
            </span>
            <span className="mono accent-cyan" style={{ fontSize: 9.5 }}>
              R {formatReward(meta?.reward_v33?.total_reward ?? null)}
            </span>
          </>
        ) : (
          <span className="muted" style={{ fontSize: 10 }}>
            Slot vazio
          </span>
        )}
      </header>
      {!slot ? (
        <div className="compare-pane__empty">
          <Columns2 size={26} />
          <div>Selecione o Slot {props.slot.toUpperCase()} e carregue um episódio.</div>
        </div>
      ) : null}
      {loading ? <div className="compare-pane__loading">Decodificando ABMESH01…</div> : null}
      {error ? (
        <div className="compare-pane__error">
          <X size={14} /> {error}
        </div>
      ) : null}
    </article>
  );
}

function Metric(props: { label: string; value: string; tone?: "accent" }) {
  return (
    <div className="stat-card">
      <div className="stat-card__label">{props.label}</div>
      <div
        className="stat-card__value"
        style={props.tone === "accent" ? { color: "var(--accent-cyan)" } : undefined}
      >
        {props.value}
      </div>
    </div>
  );
}
