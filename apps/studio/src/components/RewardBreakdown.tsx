import type { RewardV33Breakdown } from "../types/contracts";
import { formatReward } from "../lib/formatters";

interface Props {
  breakdown: RewardV33Breakdown | null;
  islandPenalty?: number;
  connectivityBonus?: number;
  pbrsDelta?: number | null;
}

function row(
  name: string,
  formula: string,
  value: number | null | undefined,
  tone?: "pos" | "neg",
) {
  const sign = tone ?? (value !== null && value !== undefined && value < 0 ? "neg" : "pos");
  return (
    <div className="reward-row" data-sign={sign}>
      <span style={{ display: "flex", flexDirection: "column", gap: 1, minWidth: 0 }}>
        <span className="reward-row__name">{name}</span>
        <span className="reward-row__formula">{formula}</span>
      </span>
      <span
        className="reward-row__value"
        style={{
          color:
            value === null || value === undefined
              ? "var(--text-muted)"
              : value < 0
                ? "var(--accent-magenta)"
                : "var(--accent-emerald)",
        }}
      >
        {formatReward(value ?? null)}
      </span>
    </div>
  );
}

export function RewardBreakdown({ breakdown, islandPenalty, connectivityBonus, pbrsDelta }: Props) {
  if (!breakdown) {
    return (
      <div className="empty-card">
        Sem decomposição de reward neste passo. A Fase 2 com FEM produz a cadeia v3.3 completa.
      </div>
    );
  }
  const island = islandPenalty ?? breakdown.island_penalty;
  const connectivity = connectivityBonus ?? breakdown.connectivity_bonus;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      {row("Compliance", "S_comp(C) = 1.15 − 0.35·(log₁₀C − 1)", breakdown.compliance_score)}
      {row("Volume", "B_vol(V) = clip((0.10 − V)·2.50)", breakdown.volume_bonus)}
      {row("Física pura", "S_FEM = clip(S_comp + B_vol)", breakdown.fem_score)}
      {row("Deslocamento", "P_disp(U) até 0.50", -breakdown.displacement_penalty, "neg")}
      {row("Ilhas", "P_ilhas(n, loose)", -island, "neg")}
      {row("Conectividade", "Bônus de carga conectada", connectivity)}
      {pbrsDelta !== null && pbrsDelta !== undefined
        ? row("PBRS", "α·(Φ(s') − Φ(s))", pbrsDelta)
        : null}
      <div className="reward-total">
        <span style={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <span className="reward-row__name">Reward v3.3</span>
          <span className="reward-row__formula">R = clip(S_FEM − P_disp − P_ilhas)</span>
        </span>
        <span
          className="reward-total__value"
          style={{
            color: breakdown.total_reward < 0 ? "var(--accent-magenta)" : "var(--accent-cyan)",
          }}
        >
          {formatReward(breakdown.total_reward)}
        </span>
      </div>
      {breakdown.is_collapsed ? (
        <span className="badge" data-tone="danger">
          COLAPSO · reward −1.0
        </span>
      ) : null}
    </div>
  );
}
