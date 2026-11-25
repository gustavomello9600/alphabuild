#!/usr/bin/env python3
"""
Generate 5 publication-grade (300 DPI) study figures in `docs/figures/` for the
AlphaBuilder Master Study & Stage 6 Self-Play monograph, strictly grounded in live
Stage 5/5C empirical measurements, direct `dolfinx` diagnostics, and exact v3.3 physics/reward math.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphabuilder.src.logic.selfplay.reward import (
    COMPLIANCE_BASE,
    COMPLIANCE_MAX,
    COMPLIANCE_MIN,
    COMPLIANCE_SLOPE,
    VOLUME_BONUS_MAX,
    VOLUME_PENALTY_MAX,
    VOLUME_REFERENCE,
    VOLUME_SENSITIVITY,
)
from alphabuilder.src.logic.selfplay.scenarios_stage6 import CURRICULUM_SCHEDULE

OUT_DIR = REPO_ROOT / "docs" / "figures"
EVAL_JSON = REPO_ROOT / "data" / "analysis" / "eval_stage5_phys_uvit_results.json"
EVAL_EP30_JSON = REPO_ROOT / "data" / "analysis" / "eval_ep30_results.json"
EVAL_DB = REPO_ROOT / "data" / "eval_inference_v4_stage5.db"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.3,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "text.color": "#0F172A",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "savefig.dpi": 300,
    }
)

COLORS = {
    "phys": "#0F4C81",
    "ep30": "#D97706",
    "ep2": "#64748B",
    "simp_cont": "#059669",
    "simp_bin": "#DC2626",
    "witness": "#0EA5E9",
    "stage6": "#7C3AED",
    "muted": "#94A3B8",
    "grid": "#E2E8F0",
}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"Saved: {out}")


def build_fig01_compliance_gap_autopsy(eval_data: dict) -> None:
    """Fig 01: 3-panel compliance gap autopsy on S1..S6."""
    fig = plt.figure(figsize=(16.0, 5.5), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.30, 1.02, 0.95])

    # Panel A: Direct dolfinx compliance comparison on S1
    ax1 = fig.add_subplot(gs[0, 0])
    labels = [
        "Bloco Sólido 100% (ref. teórica)",
        "SIMP thr > 0.50 + Colar 1-voxel\n(V=10.35%, Score=+0.615)",
        "Treliça Binária 0/1 (fh=3 + pad)\n(V=15.37%, Score=+0.473)",
        "SIMP Contínuo FEniTop\n(V=12.83%, Score=+0.458)",
        "Treliça Binária 0/1 (patch exato)\n(V=15.26%, Score=+0.110)",
        "SIMP thr > 0.50 (patch exato)\n(V=10.15%, Score=+0.142)",
        "SIMP Binário Puro (DESCONECT.)\n(V=10.13%, Score=-1.000)",
        "Phys-UViT ep19 (Stage 5C)\n(V=8.89%, Score=-0.029)",
    ]
    comps = [33.68, 319.37, 356.42, 596.42, 3948.45, 7382.40, 21083.35, 24546.42]
    bar_colors = [
        "#475569",
        COLORS["witness"],
        COLORS["stage6"],
        COLORS["simp_cont"],
        "#F59E0B",
        "#EA580C",
        COLORS["simp_bin"],
        COLORS["phys"],
    ]
    y_pos = np.arange(len(labels))
    bars = ax1.barh(y_pos, comps, color=bar_colors, height=0.64, edgecolor="#1E293B", linewidth=0.5)
    ax1.set_xscale("log")
    ax1.set_xlim(18, 2.2e5)
    ax1.invert_yaxis()
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.2)
    ax1.axvline(596.42, color=COLORS["simp_cont"], linestyle="--", linewidth=1.3, alpha=0.85)
    ax1.grid(axis="x", color=COLORS["grid"], linestyle="-", linewidth=0.7)
    ax1.set_xlabel("Compliance Estrutural C [J] em S1 (dolfinx, escala log — menor é mais rígido)")
    ax1.set_title("A. Prova Empírica em S1: Binário 0/1 Supera SIMP Contínuo")

    for i, (b, c_val) in enumerate(zip(bars, comps)):
        # Keep numbers well clear of dashed vertical reference line at 596.4
        txt = f"{c_val:,.1f}" if c_val < 1000 else f"{c_val:,.0f}"
        if i == 1:
            ax1.text(28.0, b.get_y() + b.get_height() / 2.0, txt, va="center", ha="left", fontsize=8.0, fontweight="bold", color="white")
        else:
            ax1.text(c_val * 1.15, b.get_y() + b.get_height() / 2.0, txt, va="center", ha="left", fontsize=8.1, fontweight="bold", color="#0F172A")

    ax1.annotate(
        "Phys-UViT já ancora o colar\n(0% ganho com colar artificial)",
        xy=(24546.42, 7.0),
        xytext=(2400.0, 0.7),
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor=COLORS["phys"], lw=0.8),
        arrowprops=dict(arrowstyle="->", color=COLORS["phys"], lw=1.1, connectionstyle="arc3,rad=-0.22"),
    )

    # Panel B: Bending Moment of Inertia I_y across S1..S6
    ax2 = fig.add_subplot(gs[0, 1])
    scen_ids = ["S1", "S2", "S3", "S4", "S5", "S6"]
    iy_phys = [ep["geometry"]["inertia_y_norm"] for ep in eval_data["episodes"]]
    sb = eval_data["simp_baselines"]
    iy_simp = [sb[k]["geometry"]["inertia_y_norm"] for k in sb]

    x = np.arange(len(scen_ids))
    w = 0.36
    ax2.bar(x - w / 2, iy_phys, width=w, color=COLORS["phys"], label="Phys-UViT ep19", edgecolor="#0F172A", lw=0.5)
    ax2.bar(x + w / 2, iy_simp, width=w, color=COLORS["simp_cont"], label="SIMP Contínuo", edgecolor="#0F172A", lw=0.5)
    ax2.set_yscale("log")
    ax2.set_ylim(0.004, 1.45)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scen_ids, fontweight="bold")
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax2.set_ylabel("Momento de Inércia Normalizado I_y (log)")
    ax2.set_title("B. Déficit de Alavanca Flexional I_y (6x a 63x)")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.2)

    for xi, ip, isimp in zip(x, iy_phys, iy_simp):
        ratio = isimp / max(ip, 1e-9)
        ax2.text(xi, max(ip, isimp) * 1.18, f"{ratio:.1f}x", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=COLORS["simp_bin"])

    ax2.text(
        0.04,
        0.04,
        "Viga em flexão: C ~ 1 / (E * I_y) ~ 1 / h³\nDobrar altura h reduz compliance em ~8x!",
        transform=ax2.transAxes,
        fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEFCE8", edgecolor="#CA8A04", lw=0.8),
    )

    # Panel C: SIMP p=3 stiffness curve vs discrete 0/1
    ax3 = fig.add_subplot(gs[0, 2])
    rho = np.linspace(0.0, 1.0, 300)
    e_simp = 1e-6 + (1.0 - 1e-6) * (rho ** 3)
    ax3.plot(rho, e_simp * 100.0, color=COLORS["simp_cont"], lw=2.3, label=r"SIMP $E_{eff}(\rho) = \epsilon + (1-\epsilon)\rho^3$")
    ax3.step([0.0, 0.5, 1.0], [0.0, 0.0, 100.0], where="post", color=COLORS["phys"], lw=2.0, linestyle="--", label="Binário Discreto 0/1 (rho in {0,1})")

    ax3.axvspan(0.20, 0.65, color="#FDE68A", alpha=0.38, label="Zona Cinzenta SIMP (1% a 27% de E0)")
    ax3.scatter([0.50], [12.5], color=COLORS["simp_bin"], zorder=5, s=48)
    ax3.annotate(
        "rho = 0.50 -> apenas\n12.5% de rigidez!\nThreshold quebra carga",
        xy=(0.50, 12.5),
        xytext=(0.08, 48.0),
        fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor=COLORS["simp_bin"], lw=0.8),
        arrowprops=dict(arrowstyle="->", color=COLORS["simp_bin"], lw=1.0),
    )
    ax3.scatter([1.0], [100.0], color=COLORS["phys"], zorder=5, s=55)
    ax3.annotate(
        "Alvo Stage 6:\nvoxels puros 0/1 a 100% E0",
        xy=(1.0, 100.0),
        xytext=(0.40, 86.0),
        fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor=COLORS["phys"], lw=0.8),
        arrowprops=dict(arrowstyle="->", color=COLORS["phys"], lw=1.0),
    )

    ax3.set_xlim(-0.02, 1.03)
    ax3.set_ylim(-4, 108)
    ax3.set_xlabel("Densidade do Elemento rho")
    ax3.set_ylabel("Módulo Efetivo E_eff / E0 [%]")
    ax3.set_title("C. Penalização SIMP vs Sólido Discreto 0/1")
    ax3.grid(color=COLORS["grid"], linewidth=0.7)
    ax3.legend(loc="lower right", fontsize=7.5, frameon=True)

    fig.suptitle(
        "Autópsia do Gap de 41x no Compliance em S1..S6: Causa Geométrica (I_y) vs Físico-Discreta",
        fontsize=12.5,
        fontweight="bold",
    )
    _save(fig, "fig01_compliance_gap_autopsy.png")


def build_fig02_reward_landscape_and_overtaking_path(eval_data: dict) -> None:
    """Fig 02: Analytical v3.3 R(V, log10 C) surface & Stage 6 overtaking path."""
    fig = plt.figure(figsize=(15.4, 5.9), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.42, 0.98])

    ax1 = fig.add_subplot(gs[0, 0])
    vol_grid = np.linspace(0.035, 0.195, 240)
    logc_grid = np.linspace(1.9, 5.05, 240)
    VV, LL = np.meshgrid(vol_grid, logc_grid)
    score_c = np.clip(COMPLIANCE_BASE - COMPLIANCE_SLOPE * (LL - 1.0), COMPLIANCE_MIN, COMPLIANCE_MAX)
    score_v = np.clip((VOLUME_REFERENCE - VV) * VOLUME_SENSITIVITY, -VOLUME_PENALTY_MAX, VOLUME_BONUS_MAX)
    RR = np.clip(score_c + score_v, -1.0, 1.0)

    cmap = LinearSegmentedColormap.from_list(
        "ab_reward", ["#991B1B", "#F87171", "#FEF3C7", "#A7F3D0", "#059669", "#064E3B"]
    )
    cf = ax1.contourf(VV * 100.0, LL, RR, levels=np.linspace(-0.45, 0.85, 27), cmap=cmap, alpha=0.86)
    cs = ax1.contour(VV * 100.0, LL, RR, levels=[-0.1, 0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.7], colors="#1E293B", linewidths=0.75, alpha=0.75)
    ax1.clabel(cs, fmt="R=%.2f", fontsize=7.6, inline=True)
    fig.colorbar(cf, ax=ax1, label="Recompensa v3.3 R(C, V)", pad=0.02)

    if EVAL_EP30_JSON.exists():
        ep30 = _load_json(EVAL_EP30_JSON)
        v30 = [e["final_volume"] * 100.0 for e in ep30.get("episodes", [])]
        c30 = [np.log10(e["final_compliance"]) for e in ep30.get("episodes", [])]
        ax1.scatter(v30, c30, s=46, marker="s", color=COLORS["ep30"], edgecolor="#0F172A", lw=0.7, zorder=4, label="Simple ep30 (Stage 3)")

    ep_p = eval_data["episodes"]
    vp = [e["final_volume"] * 100.0 for e in ep_p]
    cp = [np.log10(e["final_compliance"]) for e in ep_p]
    ax1.scatter(vp, cp, s=72, marker="o", color=COLORS["phys"], edgecolor="#FFFFFF", lw=1.0, zorder=5, label="Phys-UViT ep19 (Stage 5C)")
    for i, (xv, yc) in enumerate(zip(vp, cp), start=1):
        dx_lbl = 0.20 if i != 2 else -0.45
        ax1.text(xv + dx_lbl, yc + 0.03, f"S{i}", fontsize=7.6, fontweight="bold", color="#0F172A")

    sb = eval_data["simp_baselines"]
    vs = [v["vol_cont"] * 100.0 for v in sb.values()]
    cs_log = [np.log10(v["comp_cont"]) for v in sb.values()]
    ax1.scatter(vs, cs_log, s=75, marker="^", color=COLORS["simp_cont"], edgecolor="#0F172A", lw=0.8, zorder=5, label="SIMP Contínuo FEniTop")

    ax1.scatter([15.37], [np.log10(356.42)], s=110, marker="*", color=COLORS["stage6"], edgecolor="#0F172A", lw=0.7, zorder=6, label="Treliça Binária 0/1 S1 (C=356, +0.473)")
    ax1.scatter([10.35], [np.log10(319.37)], s=105, marker="D", color=COLORS["witness"], edgecolor="#0F172A", lw=0.8, zorder=6, label="SIMP>0.5 + Colar S1 (C=319, +0.615)")
    ax1.scatter([14.41], [np.log10(6800.8)], s=65, marker="X", color=COLORS["simp_bin"], edgecolor="#0F172A", lw=0.7, zorder=5, label="SIMP Binário S6 (+0.048)")

    ax1.annotate(
        "",
        xy=(10.35, np.log10(325.0)),
        xytext=(float(np.mean(vp)), float(np.mean(cp))),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.45,head_length=0.7",
            color=COLORS["stage6"],
            lw=2.6,
            connectionstyle="arc3,rad=0.28",
        ),
        zorder=7,
    )
    ax1.text(
        4.1,
        2.92,
        "Rota Stage 6 Self-Play:\n+3% vol p/ alavanca I_y\n(-0.075 custo vol)\n+ 40x ganho rigidez\n(+0.56 ganho score!)",
        fontsize=8.0,
        fontweight="bold",
        color="#4C1D95",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#F5F3FF", edgecolor=COLORS["stage6"], lw=1.0),
    )

    ax1.set_xlim(3.5, 19.2)
    ax1.set_ylim(1.95, 5.05)
    ax1.set_xlabel("Fração de Volume Sólido V [%]")
    ax1.set_ylabel(r"$\log_{10}(\mathrm{Compliance}\; C)$ [J] (menor = mais rígido)")
    ax1.set_title("A. Superfície Analítica R(V, log10 C) e Rota de Ultrapassagem")
    ax1.legend(loc="upper right", fontsize=7.3, frameon=True, framealpha=0.94)

    # Panel B: Leverage law without any label clipping or box overlap
    ax2 = fig.add_subplot(gs[0, 1])
    dv_grid = np.linspace(0.0, 10.6, 120) / 100.0
    breakeven_factor = 10.0 ** ((VOLUME_SENSITIVITY * dv_grid) / COMPLIANCE_SLOPE)
    ax2.plot(dv_grid * 100.0, breakeven_factor, color="#0F172A", lw=2.2, linestyle="--", label="Linha de Empate (Delta R = 0)")
    ax2.fill_between(dv_grid * 100.0, breakeven_factor, 130.0, color="#D1FAE5", alpha=0.55, label="Região de Ganho Líquido (Delta R > 0)")
    ax2.fill_between(dv_grid * 100.0, 1.0, breakeven_factor, color="#FEE2E2", alpha=0.55, label="Região de Perda (Delta R < 0)")

    offsets = [(0.2, -5.5), (0.2, 4.0), (0.2, -4.5), (-1.4, 9.0), (0.2, 4.0), (-1.5, 6.0)]
    for i, ((ep, skey), (dx_o, dy_o)) in enumerate(zip(zip(ep_p, sb.keys()), offsets), start=1):
        dv_pct = (sb[skey]["vol_cont"] - ep["final_volume"]) * 100.0
        c_fac = ep["final_compliance"] / sb[skey]["comp_cont"]
        ax2.scatter([dv_pct], [c_fac], s=85, color=COLORS["phys"], edgecolor="#FFFFFF", lw=1.0, zorder=5)
        ax2.text(dv_pct + dx_o, c_fac + dy_o, f"S{i} ({c_fac:.0f}x)", fontsize=8.0, fontweight="bold")

    ax2.set_yscale("log")
    ax2.set_xlim(0.0, 10.6)
    ax2.set_ylim(1.0, 130.0)
    ax2.set_xlabel("Orçamento Extra de Volume Delta V = V_SIMP - V_Stage5 [%]")
    ax2.set_ylabel("Fator de Redução de Compliance C_Stage5 / C_SIMP")
    ax2.set_title("B. Lei de Alavancagem: 10^(7.14 * Delta V) vs Gap Real")
    ax2.grid(color=COLORS["grid"], linewidth=0.7)
    ax2.legend(loc="lower right", fontsize=7.9, frameon=True)
    ax2.text(
        0.03,
        0.30,
        "Taxa marginal de troca v3.3:\n"
        "|dR/dV| / |dR/d log10 C| = 2.5 / 0.35 = 7.14\n"
        "+4% vol para engrossar banzos\n"
        "custa -0.10, mas empataria com apenas\n"
        "1.93x menos C — ganho real ~30x–87x!",
        transform=ax2.transAxes,
        fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#FFFFFF", edgecolor=COLORS["simp_cont"], lw=0.9),
    )

    fig.suptitle(
        "Por Que o Self-Play Stage 6 Pode Investir Massa com Retorno Exponencial na Função de Recompensa v3.3",
        fontsize=12.5,
        fontweight="bold",
    )
    _save(fig, "fig02_reward_landscape_and_overtaking_path.png")


def build_fig03_ood_blindness_diagnosis() -> None:
    """Fig 03: Forensic OOD blindness diagnosis from eval_inference_v4_stage5.db."""
    fig = plt.figure(figsize=(16.2, 6.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.35, 0.88, 0.92], wspace=0.40, top=0.86, bottom=0.11, left=0.05, right=0.98)

    # Left column: 2x2 trajectories with clean manual spacing (zero constrained_layout clash)
    sub_gs = gs[0, 0].subgridspec(2, 2, wspace=0.30, hspace=0.42)
    outer_a = fig.add_subplot(gs[0, 0], frameon=False)
    outer_a.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    outer_a.set_title("A. Cegueira de Valor em Fase 2 (V_net cai, R_FEM sobe)", pad=18)

    scenarios = [
        ("S1_cantilever_long_center", "S1"),
        ("S3_cantilever_midspan_top", "S3"),
        ("S4_rail_xy_center", "S4"),
        ("S6_short_deep_clamp", "S6"),
    ]
    conn = sqlite3.connect(f"file:{EVAL_DB}?mode=ro", uri=True)
    try:
        for idx, (sc_id, short) in enumerate(scenarios):
            ax = fig.add_subplot(sub_gs[idx // 2, idx % 2])
            rows = conn.execute(
                "SELECT step, value, fem_reward FROM game_steps WHERE game_id = ? AND phase = 'REFINEMENT' ORDER BY step ASC",
                (f"phys_uvit_ep19__{sc_id}",),
            ).fetchall()
            steps = np.arange(len(rows))
            v_pred = np.array([r[1] for r in rows], dtype=float)
            r_fem = np.array([r[2] for r in rows], dtype=float)
            corr = float(np.corrcoef(v_pred, r_fem)[0, 1]) if np.std(v_pred) > 1e-8 and np.std(r_fem) > 1e-8 else 0.0

            ax.plot(steps, v_pred, color=COLORS["phys"], lw=2.0, label="V_net (previsto)")
            ax.plot(steps, r_fem, color=COLORS["simp_bin"], lw=2.0, linestyle="--", label="R_FEM (real)")
            ax.set_title(f"{short} (r = {corr:+.2f})", fontsize=9.1, fontweight="bold")
            ax.set_ylim(-0.25, 0.88)
            ax.grid(color=COLORS["grid"], linewidth=0.6)
            if idx // 2 == 1:
                ax.set_xlabel("Passo Fase 2", fontsize=8.1)
            if idx % 2 == 0:
                ax.set_ylabel("Score / Valor", fontsize=8.1)
            if idx == 1:
                ax.legend(loc="center right", fontsize=7.2, frameon=True)
    finally:
        conn.close()

    # Panel B: Concise labels so zero intrusion on Panel A or Panel C
    ax_b = fig.add_subplot(gs[0, 1])
    metrics_lbl = [
        "Veto P(conn)\nFase 1 -> 2\n(0 -> 1 exato)",
        "|Viés| log10(C)\n[décadas FEM]\n0.035 -> 1.103",
        "|Viés| Value\n[score v3.3]\n0.012 -> 0.463",
    ]
    in_dist = [1.000, 0.0352, 0.0117]
    ood_vals = [1.000, 1.103, 0.463]
    y_b = np.arange(len(metrics_lbl))
    h_b = 0.34
    ax_b.barh(y_b - h_b / 2, in_dist, height=h_b, color=COLORS["simp_cont"], label="Validação Stage 5 (In-Dist)")
    ax_b.barh(y_b + h_b / 2, ood_vals, height=h_b, color=COLORS["simp_bin"], label="Inferência Stage 5C (OOD)")
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(metrics_lbl, fontsize=8.1)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1.38)
    ax_b.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    ax_b.set_title("B. Calibração In-Dist vs OOD")
    ax_b.set_xlabel("Magnitude / Erro")
    ax_b.legend(loc="lower right", fontsize=7.6, frameon=True)

    for yi, iv, ov in zip(y_b, in_dist, ood_vals):
        ax_b.text(iv + 0.02, yi - h_b / 2, f"{iv:.3f}", va="center", fontsize=8.0, fontweight="bold", color="#065F46")
        ax_b.text(ov + 0.02, yi + h_b / 2, f"{ov:.3f}", va="center", fontsize=8.0, fontweight="bold", color="#991B1B")

    # Panel C: Concise labels & generous spacing
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.barh([0], [99.65], color=COLORS["phys"], height=0.42)
    ax_c.barh([0], [0.35], left=[99.65], color=COLORS["simp_bin"], height=0.42)
    ax_c.barh([1], [75.0], color=COLORS["stage6"], height=0.42)
    ax_c.barh([1], [25.0], left=[75.0], color="#A855F7", height=0.42)

    ax_c.text(49.5, 0.0, "1,435 ADD (99.65%)\nSó engrossa por fora!", ha="center", va="center", color="white", fontweight="bold", fontsize=8.2)
    ax_c.text(99.5, 0.36, "5 REM (0.35%)!", ha="right", va="bottom", color=COLORS["simp_bin"], fontweight="bold", fontsize=8.1)
    ax_c.text(37.5, 1.0, "ADD (~75%)", ha="center", va="center", color="white", fontweight="bold", fontsize=8.4)
    ax_c.text(87.5, 1.0, "REM >= 25%\n+ Probe SED", ha="center", va="center", color="white", fontweight="bold", fontsize=7.8)

    ax_c.text(
        0.5,
        0.04,
        "Orçamento Físico Fase 2:\n"
        "- Stage 5C: 30 passos x 8 vox = 240 edições (1.46% grid)\n"
        "- Stage 6: 64 passos (32x16 + 32x8) = 768 edições máx.\n"
        "- Probes SED contrafactuais destilam se R_probe > R_mcts",
        transform=ax_c.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.7,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#64748B", lw=0.8),
    )

    ax_c.set_yticks([0, 1])
    ax_c.set_yticklabels(["Stage 5C\n(ep19 OOD)", "Stage 6\n(Cota+SED)"], fontsize=8.3)
    ax_c.set_ylim(-0.55, 2.35)
    ax_c.set_xlim(0, 102)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Prior / Ações em Fase 2 [%]")
    ax_c.set_title("C. Paralisia de Remoção & Orçamento")
    ax_c.grid(axis="x", color=COLORS["grid"], linewidth=0.7)

    fig.suptitle(
        "Diagnóstico da Cegueira OOD no Stage 5C: Por Que o MCTS Ficou Preso Antes do Self-Play",
        fontsize=12.5,
        fontweight="bold",
        y=0.96,
    )
    _save(fig, "fig03_ood_blindness_diagnosis.png")


def _draw_domain_glyph(ax: plt.Axes, x0: float, y0: float, w: float, h: float, mode: str) -> None:
    """Draw a tiny schematic structural mechanics glyph inside Tier boxes."""
    ax.add_patch(patches.Rectangle((x0, y0), w, h, facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.8))
    if mode == "T0":
        ax.plot([x0, x0], [y0, y0 + h], color="#1E293B", lw=3.0)
        ax.annotate("", xy=(x0 + 0.85 * w, y0 + 0.25 * h), xytext=(x0 + 0.85 * w, y0 + 0.72 * h),
                    arrowprops=dict(arrowstyle="->", color=COLORS["simp_bin"], lw=1.6))
    elif mode == "T1":
        ax.plot([x0, x0], [y0, y0 + 0.28 * h], color="#1E293B", lw=3.0)
        ax.plot([x0, x0], [y0 + 0.72 * h, y0 + h], color="#1E293B", lw=3.0)
        ax.annotate("", xy=(x0 + 0.90 * w, y0 + 0.28 * h), xytext=(x0 + 0.70 * w, y0 + 0.72 * h),
                    arrowprops=dict(arrowstyle="->", color=COLORS["simp_bin"], lw=1.6))
    elif mode == "T2":
        ax.scatter([x0 + 0.08 * w, x0 + 0.92 * w], [y0 + 0.08 * h, y0 + 0.08 * h], marker="^", s=38, color="#1E293B", zorder=5)
        ax.add_patch(patches.Rectangle((x0 + 0.32 * w, y0 + 0.30 * h), 0.36 * w, 0.36 * h, facecolor="#CBD5E1", edgecolor="#475569", hatch="///", lw=0.8))
        ax.annotate("", xy=(x0 + 0.50 * w, y0 + 0.72 * h), xytext=(x0 + 0.50 * w, y0 + 0.98 * h),
                    arrowprops=dict(arrowstyle="->", color=COLORS["simp_bin"], lw=1.6))
    elif mode == "T3":
        ax.add_patch(patches.Rectangle((x0 + 0.08 * w, y0 + 0.10 * h), 0.52 * w, 0.76 * h, facecolor="#EDE9FE", edgecolor=COLORS["stage6"], lw=1.0))
        ax.add_patch(patches.Rectangle((x0 + 0.65 * w, y0 + 0.25 * h), 0.30 * w, 0.55 * h, facecolor="#DBEAFE", edgecolor=COLORS["phys"], lw=1.0))


def build_fig04_four_tier_curriculum_and_selfplay_loop() -> None:
    """Fig 04: 4-tier curriculum, Gen 1..5+ schedule, and closed-loop Self-Play flow."""
    fig = plt.figure(figsize=(16.0, 6.4), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.12, 0.96, 1.18])

    # Panel A: 4 concentric tiers schematic
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis("off")
    ax_a.set_title("A. Currículo em 4 Tiers Concêntricos (Stage 6)")

    tiers = [
        ("Tier 3 — Escala Multi-Res (3D RoPE)", "48x32x8 | 64x24x8 | 48x24x12 (0% padding)", "#F5F3FF", COLORS["stage6"], "T3", 7.6),
        ("Tier 2 — Shift Topológico + Keep-Out", "Ponte MBB (pino+rolete), L-wall, janelas proibidas", "#EFF6FF", COLORS["phys"], "T2", 5.2),
        ("Tier 1 — Shift Próximo de BCs e Cargas", "Dupla-aba, clamp parcial, cargas 3D inclinadas, 2 pts", "#ECFDF5", COLORS["simp_cont"], "T1", 2.8),
        ("Tier 0 — Âncora Canônica (64x32x8)", "FULL_CLAMP & RAIL_XY (S1..S6 estritamente excluídos)", "#FEF3C7", "#B45309", "T0", 0.4),
    ]
    for title_t, desc_t, bg, edge, mode, y_b in tiers:
        ax_a.add_patch(patches.FancyBboxPatch((0.2, y_b), 9.6, 2.0, boxstyle="round,pad=0.12", facecolor=bg, edgecolor=edge, lw=1.4))
        _draw_domain_glyph(ax_a, 0.45, y_b + 0.25, 2.2, 1.5, mode)
        ax_a.text(2.95, y_b + 1.32, title_t, fontsize=9.0, fontweight="bold", color=edge)
        ax_a.text(2.95, y_b + 0.62, desc_t, fontsize=7.9, color="#1E293B")

    # Panel B: Stacked bar schedule with clean top callout (zero overlap on white bar %)
    ax_b = fig.add_subplot(gs[0, 1])
    gens = sorted(CURRICULUM_SCHEDULE.keys())
    mat = np.array([CURRICULUM_SCHEDULE[g] for g in gens]) * 100.0
    t_labels = ["Tier 0 (Âncora)", "Tier 1 (Próximo)", "Tier 2 (Topologia)", "Tier 3 (Multi-Res)"]
    t_cols = ["#F59E0B", COLORS["simp_cont"], COLORS["phys"], COLORS["stage6"]]

    bottom = np.zeros(len(gens))
    for k in range(4):
        vals = mat[:, k]
        ax_b.bar(gens, vals, bottom=bottom, color=t_cols[k], edgecolor="#FFFFFF", lw=1.0, width=0.58, label=t_labels[k])
        for g_idx, v in enumerate(vals):
            if v >= 12.0:
                ax_b.text(gens[g_idx], bottom[g_idx] + v / 2.0, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        bottom += vals

    ax_b.set_xticks(gens)
    ax_b.set_xticklabels([f"Gen {g}" + ("+" if g == 5 else "") for g in gens], fontweight="bold")
    ax_b.set_ylim(0, 138)
    ax_b.set_ylabel("Fração de Episódios por Geração [%]")
    ax_b.set_title("B. Cronograma de Exposição & Portão Adaptativo")
    ax_b.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax_b.legend(loc="upper center", ncol=2, fontsize=7.5, frameon=True)
    ax_b.text(
        0.5,
        0.84,
        "Portão adaptativo (pacing gate): se conn Fase 1 < 65% em Tier t,\nparte do peso de Tier t+1/t+2 retorna para consolidar Tier t e Âncora.",
        transform=ax_b.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFBEB", edgecolor="#D97706", lw=0.9),
    )

    # Panel C: Closed-loop Stage 6 flow with generous left whitespace for feedback arrow
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_xlim(0, 10.5)
    ax_c.set_ylim(-0.3, 10.1)
    ax_c.axis("off")
    ax_c.set_title("C. Loop Fechado Self-Play + Mixed Replay Stage 6")

    boxes = [
        (8.9, "1. Phys-UViT 3D (Gen g)\nPolítica + Valor + log10(C) + Conn + SED", "#EFF6FF", COLORS["phys"]),
        (6.8, "2. MCTS (160 sims + PBRS + Cota REM >= 16)\nExploração Fase 1 + Batch 16->8 Fase 2", "#F5F3FF", COLORS["stage6"]),
        (4.7, "3. Oráculo fast_amg FEM (PyAMG + splu)\nCompliance exata + Mapa SED por elemento", "#ECFDF5", COLORS["simp_cont"]),
        (2.6, "4. Banco SQLite + Probes SED Contrafactuais\nDestila política apenas se R_probe > R_mcts", "#FEF3C7", "#B45309"),
        (0.5, "5. Mixed Replay (60% Self-Play + 40% Warmup)\nBucketed Resolution + D4xZ2 -> Gen g+1", "#FFF1F2", COLORS["simp_bin"]),
    ]
    for cy, txt, bg, edge in boxes:
        ax_c.add_patch(patches.FancyBboxPatch((1.65, cy - 0.65), 8.55, 1.35, boxstyle="round,pad=0.1", facecolor=bg, edgecolor=edge, lw=1.3))
        ax_c.text(5.92, cy + 0.02, txt, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0F172A")

    for y_top, y_bot in [(8.25, 7.52), (6.15, 5.42), (4.05, 3.32), (1.95, 1.22)]:
        ax_c.annotate("", xy=(5.92, y_bot), xytext=(5.92, y_top), arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.5))

    ax_c.annotate(
        "",
        xy=(1.55, 8.9),
        xytext=(1.55, 0.5),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["phys"], lw=2.2, connectionstyle="arc3,rad=0.22"),
    )
    ax_c.text(0.45, 4.7, "Ciclo Evolutivo\nGen g -> g+1", rotation=90, va="center", ha="center", fontsize=8.1, fontweight="bold", color=COLORS["phys"])

    fig.suptitle(
        "Arquitetura do Self-Play Stage 6: Expansão Controlada do Suporte Físico e Quebra de Pontos Cegos",
        fontsize=12.5,
        fontweight="bold",
    )
    _save(fig, "fig04_four_tier_curriculum_and_selfplay_loop.png")


def build_fig05_selfplay_projections_and_scaling_laws(eval_data: dict) -> None:
    """Fig 05: Grounded Stage 6 conservative/nominal/optimistic projections vs Gen 0."""
    fig = plt.figure(figsize=(15.6, 5.5), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.05, 1.05])

    gens = np.arange(0, 9)
    score_nom = np.array([0.076, 0.165, 0.265, 0.360, 0.435, 0.495, 0.540, 0.575, 0.600])
    score_cons = np.array([0.076, 0.130, 0.195, 0.270, 0.335, 0.395, 0.445, 0.485, 0.515])
    score_opt = np.array([0.076, 0.210, 0.330, 0.435, 0.515, 0.575, 0.620, 0.650, 0.670])

    simp_cont_scores = [v["reward_cont"] for v in eval_data["simp_baselines"].values()]
    mean_simp_cont = float(np.mean(simp_cont_scores))

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axhline(mean_simp_cont, color=COLORS["simp_cont"], lw=2.0, linestyle="--", label=f"SIMP Contínuo FEniTop (+{mean_simp_cont:.3f})")
    ax1.axhline(0.057, color=COLORS["ep30"], lw=1.4, linestyle=":", label="Simple ep30 Stage 3 (+0.057 medido)")
    ax1.axhline(0.048, color=COLORS["simp_bin"], lw=1.2, linestyle="-.", label="SIMP Binário S6 (+0.048; S1..S5 = -1.0)")

    ax1.fill_between(gens, score_cons, score_opt, color=COLORS["stage6"], alpha=0.18, label="Banda Projetada Stage 6 (Cons.-Otim.)")
    ax1.plot(gens, score_nom, color=COLORS["stage6"], lw=2.5, marker="o", ms=5, label="Projeção Nominal Phys-UViT Stage 6")
    ax1.scatter([0], [0.076], color=COLORS["phys"], s=95, zorder=6, edgecolor="#FFFFFF", lw=1.2, label="Gen 0 Empírico (ep19: +0.076)")

    ax1.axvline(5.3, color="#64748B", linestyle=":", lw=1.0)
    ax1.text(5.4, 0.26, "Gen 5-6:\ncruzamento sobre\nSIMP Contínuo", fontsize=7.9, fontweight="bold", color="#4C1D95")

    ax1.set_xlim(-0.3, 8.3)
    ax1.set_ylim(-0.04, 0.74)
    ax1.set_xlabel("Geração de Self-Play (Gen 0 = Warmup Stage 5 medido)")
    ax1.set_ylabel("Score v3.3 Médio em S1..S6")
    ax1.set_title("A. Evolução Projetada do Score v3.3 vs Baselines")
    ax1.grid(color=COLORS["grid"], linewidth=0.7)
    ax1.legend(loc="lower right", fontsize=7.2, frameon=True)

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    gmean_ratio0 = float(np.exp(np.mean([np.log(e["comp_ratio_vs_simp_cont"]) for e in eval_data["episodes"]])))
    ratio_nom = np.array([gmean_ratio0, 18.0, 8.5, 4.2, 2.2, 1.35, 0.96, 0.78, 0.68])
    ratio_cons = np.array([gmean_ratio0, 24.0, 13.5, 7.8, 4.4, 2.70, 1.85, 1.35, 1.08])
    ratio_opt = np.array([gmean_ratio0, 11.5, 4.8, 2.1, 1.15, 0.76, 0.61, 0.56, 0.54])

    ax2.fill_between(gens, ratio_opt, ratio_cons, color=COLORS["phys"], alpha=0.16, label="Banda Projetada (Otimista - Conservadora)")
    ax2.plot(gens, ratio_nom, color=COLORS["phys"], lw=2.5, marker="s", ms=4.8, label="Projeção Nominal C / C_SIMP_cont")
    ax2.scatter([0], [gmean_ratio0], color=COLORS["phys"], s=95, zorder=6)
    ax2.axhline(1.0, color=COLORS["simp_cont"], lw=2.0, linestyle="--", label="Empate Físico c/ SIMP Contínuo (1.0x)")
    ax2.axhline(0.54, color=COLORS["witness"], lw=1.5, linestyle=":", label="Testemunha Empírica Binária dolfinx (~0.54x)")

    ax2.set_yscale("log")
    ax2.set_xlim(-0.3, 8.3)
    ax2.set_ylim(0.32, 120.0)
    ax2.set_xlabel("Geração de Self-Play")
    ax2.set_ylabel("Razão C_rede / C_SIMP_cont (média geométrica, log)")
    ax2.set_title("B. Fechamento Multiplicativo do Gap de Compliance")
    ax2.grid(color=COLORS["grid"], linewidth=0.7)
    ax2.legend(loc="upper right", fontsize=7.3, frameon=True)

    # Panel C: Compute box placed cleanly in top-right open area below legend
    ax3 = fig.add_subplot(gs[0, 2])
    mae_logc = np.array([1.15, 0.72, 0.48, 0.34, 0.25, 0.19, 0.15, 0.13, 0.11])
    mae_val = np.array([0.46, 0.30, 0.21, 0.15, 0.11, 0.085, 0.070, 0.060, 0.052])
    ax3.plot(gens, mae_logc, color=COLORS["simp_bin"], lw=2.3, marker="o", ms=4.5, label="Erro OOD log10(C) [décadas]")
    ax3.plot(gens, mae_val, color=COLORS["ep30"], lw=2.3, marker="^", ms=5.0, label="Erro OOD Value Head [score]")
    ax3.axhline(0.0352, color=COLORS["simp_cont"], lw=1.2, linestyle="--", label="Piso In-Dist Val log10(C) MAE (0.035)")

    ax3.set_xlim(-0.3, 8.3)
    ax3.set_ylim(0.0, 1.38)
    ax3.set_xlabel("Geração de Self-Play (Kaggle 2xT4: ~1.8h / geração)")
    ax3.set_ylabel("MAE OOD Projetado (décadas / unidades de valor)")
    ax3.set_title("C. Convergência da Calibração Física OOD")
    ax3.grid(color=COLORS["grid"], linewidth=0.7)
    ax3.legend(loc="upper right", fontsize=7.3, frameon=True)

    ax3.text(
        0.34,
        0.52,
        "Escala de Compute (32 eps/gen, 160 sims):\n"
        "- 2x T4 (Kaggle): ~1.8h real/gen (~3.6 GPU-h/gen)\n"
        "  5 gerações ~ 9h wall-clock (1 sessão Kaggle)\n"
        "- 1x H100: ~3.5x mais rápido na rede (batch MCTS)\n"
        "  + multicore reduz fast_amg ~2x",
        transform=ax3.transAxes,
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#F8FAFC", edgecolor="#64748B", lw=0.8),
    )

    fig.suptitle(
        "Projeções Quantitativas Stage 6 (Conservadora / Nominal / Otimista) vs Ground-Truth Empírico",
        fontsize=12.5,
        fontweight="bold",
    )
    _save(fig, "fig05_selfplay_projections_and_scaling_laws.png")


def main() -> None:
    eval_data = _load_json(EVAL_JSON)
    build_fig01_compliance_gap_autopsy(eval_data)
    build_fig02_reward_landscape_and_overtaking_path(eval_data)
    build_fig03_ood_blindness_diagnosis()
    build_fig04_four_tier_curriculum_and_selfplay_loop()
    build_fig05_selfplay_projections_and_scaling_laws(eval_data)


if __name__ == "__main__":
    main()
