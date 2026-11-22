import { Columns2 } from "lucide-react";
import { useCompareStore } from "../store/useCompareStore";

export function CompareView() {
  const slotA = useCompareStore((state) => state.slotA);
  const slotB = useCompareStore((state) => state.slotB);
  const metrics = useCompareStore((state) => state.metrics);
  return (
    <section
      className="glass pointer-auto"
      style={{
        position: "absolute",
        zIndex: 14,
        right: 352,
        top: 68,
        width: 320,
        borderRadius: "var(--radius-lg)",
        padding: 12,
        display: "flex",
        flexDirection: "column",
        gap: 9,
      }}
    >
      <div className="panel-heading">
        <span>Arena A/B</span>
        <Columns2 size={12} />
      </div>
      <div className="empty-card" style={{ textAlign: "left", lineHeight: 1.55 }}>
        {slotA && slotB
          ? "Slots carregados. O comparador lado a lado será montado neste modo."
          : "Carregue um episódio no Slot A e outro no Slot B pela Biblioteca para sincronizar câmera e timeline."}
      </div>
      {metrics ? (
        <div className="stat-grid">
          <div className="stat-card">
            <div className="stat-card__label">IoU 3D</div>
            <div className="stat-card__value">{metrics.iou_3d.toFixed(4)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-card__label">Δ Reward</div>
            <div className="stat-card__value">
              {metrics.delta_reward === null ? "—" : metrics.delta_reward.toFixed(4)}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
