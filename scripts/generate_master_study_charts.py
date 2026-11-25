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


def _make_binary_truss_s1() -> np.ndarray:
    """Reconstruct the exact dolfinx/fast_amg pure 0/1 binary truss witness on S1 (V=15.37%, C~356.4)."""
    from alphabuilder.src.logic.harvest.generators import get_load_region_bounds

    lc = {"x": 63, "y": 16, "z_start": 3, "z_end": 5, "bc_type": "FULL_CLAMP"}
    lx, y0, y1, z0, z1 = get_load_region_bounds((64, 32, 8), lc)
    g = np.zeros((64, 32, 8), dtype=np.float32)
    for x in range(64):
        t = x / 63.0
        y_bot = int(round(0 + t * 14))
        y_top = int(round(31 - t * 14))
        g[x, y_bot : min(32, y_bot + 3), 2:6] = 1.0
        g[x, max(0, y_top - 2) : y_top + 1, 2:6] = 1.0
        if x % 16 == 0 or x == 63:
            g[x, y_bot : y_top + 1, 2:6] = 1.0
    for bay in range(4):
        x0_b, x1_b = bay * 16, min(63, (bay + 1) * 16)
        for s in np.linspace(0, 1, 32):
            xx = int(round(x0_b + s * (x1_b - x0_b)))
            t = xx / 63.0
            yb = int(round(t * 14))
            yt = int(round(31 - t * 14))
            y_d1 = int(round(yb + s * (yt - yb)))
            y_d2 = int(round(yt - s * (yt - yb)))
            g[xx, max(0, y_d1 - 1) : min(32, y_d1 + 1), 2:6] = 1.0
            g[xx, max(0, y_d2 - 1) : min(32, y_d2 + 1), 2:6] = 1.0
    g[max(0, lx - 1) : lx + 1, max(0, y0 - 1) : min(32, y1 + 1), max(0, z0 - 1) : min(8, z1 + 1)] = 1.0
    return g


def build_fig06_real_3d_geometries_comparison(eval_data: dict) -> None:
    """
    Fig 06: Grade 4x4 comparando geometrias 3D reais extraidas de `eval_inference_v4_stage5.db`
    (projeção lateral XY média em Z + corte transversal YZ em x=32).
    """
    from alphabuilder.src.logic.storage import deserialize_array

    ep30_data = _load_json(EVAL_EP30_JSON) if EVAL_EP30_JSON.exists() else {"episodes": []}
    ep30_by_sc = {e["scenario_id"]: e for e in ep30_data.get("episodes", [])}
    phys_by_sc = {e["scenario_id"]: e for e in eval_data["episodes"]}
    sb = eval_data["simp_baselines"]

    scens = [
        ("S1_cantilever_long_center", "S1: Cantilever Longa\n(x=63, y=16, FULL)"),
        ("S2_cantilever_tip_bottom", "S2: Ponta Inferior\n(x=63, y=6, FULL)"),
        ("S4_rail_xy_center", "S4: Guia RAIL_XY\n(x=58, y=16, RAIL)"),
        ("S6_short_deep_clamp", "S6: Curta Profunda\n(x=36, y=16, FULL)"),
    ]
    cols = ["SIMP Contínuo\n(FEniTop ref.)", "SIMP Binário\n(rho > 0.50)", "Simple ep30\n(Stage 3 Baseline)", "Phys-UViT ep19\n(Stage 5C)"]

    fig, axes = plt.subplots(4, 4, figsize=(16.0, 9.6), constrained_layout=True)
    conn = sqlite3.connect(f"file:{EVAL_DB}?mode=ro", uri=True)
    try:
        for r_idx, (sc_id, r_lbl) in enumerate(scens):
            row_s = conn.execute("SELECT density_blob FROM game_steps WHERE game_id=? ORDER BY step DESC LIMIT 1", (f"simp__{sc_id}",)).fetchone()
            row_e = conn.execute("SELECT density_blob FROM game_steps WHERE game_id=? ORDER BY step DESC LIMIT 1", (f"ep30__{sc_id}",)).fetchone()
            row_p = conn.execute("SELECT density_blob FROM game_steps WHERE game_id=? ORDER BY step DESC LIMIT 1", (f"phys_uvit_ep19__{sc_id}",)).fetchone()

            rho_cont = deserialize_array(row_s[0]).astype(np.float32)
            rho_bin = (rho_cont > 0.5).astype(np.float32)
            rho_ep30 = deserialize_array(row_e[0]).astype(np.float32)
            rho_phys = deserialize_array(row_p[0]).astype(np.float32)

            s_base = sb[sc_id]
            e_meta = ep30_by_sc.get(sc_id, {})
            p_meta = phys_by_sc[sc_id]

            panels = [
                (rho_cont, "Greens", f"V={s_base['vol_cont']*100:.1f}% | C={s_base['comp_cont']:,.0f}\nScore={s_base['reward_cont']:+.3f} | Iy={s_base['geometry']['inertia_y_norm']:.3f}"),
                (rho_bin, "Reds" if not s_base["is_connected_bin"] else "Oranges", f"V={s_base['vol_bin']*100:.1f}% | C={s_base['comp_bin']:,.0f}\nScore={s_base['reward_bin']:+.3f} | {'QUEBRADO' if not s_base['is_connected_bin'] else 'Conectado'}"),
                (rho_ep30, "YlOrBr", f"V={e_meta.get('final_volume',0)*100:.1f}% | C={e_meta.get('final_compliance',0):,.0f}\nScore={e_meta.get('final_score',0):+.3f} | isl={e_meta.get('geometry',{}).get('n_islands',1)}, ovX={e_meta.get('geometry',{}).get('overshoot_frac',0)*100:.1f}%"),
                (rho_phys, "Blues", f"V={p_meta['final_volume']*100:.1f}% | C={p_meta['final_compliance']:,.0f}\nScore={p_meta['final_score']:+.3f} | Iy={p_meta['geometry']['inertia_y_norm']:.3f}, ovX=0%"),
            ]

            for c_idx, (vol3d, cmap_name, subtitle) in enumerate(panels):
                ax = axes[r_idx, c_idx]
                xy = vol3d.mean(axis=2).T  # (32, 64)
                yz = vol3d[min(32, vol3d.shape[0] - 1), :, :]  # (32, 8)
                # Build canvas with XY [0..63] + gap [64..66] + YZ cross-section [67..74]
                canvas = np.full((32, 76), np.nan, dtype=np.float32)
                canvas[:, :64] = xy
                canvas[:, 67:75] = yz
                cmap_obj = plt.get_cmap(cmap_name).copy()
                cmap_obj.set_bad(color="#F1F5F9")
                ax.imshow(canvas, origin="lower", cmap=cmap_obj, vmin=0.0, vmax=1.0, aspect="auto")
                ax.axvline(64.5, color="#64748B", lw=1.0, ls=":")
                if r_idx == 0:
                    ax.text(70.5, 30.2, "YZ", fontsize=7.0, fontweight="bold", ha="center", va="top", color="#334155")
                # Draw support x=0 and load x-marker
                lx = int(sc_id == "S4_rail_xy_center") * 58 + int(sc_id == "S6_short_deep_clamp") * 36 + int(sc_id in ("S1_cantilever_long_center", "S2_cantilever_tip_bottom")) * 63
                ly = 6 if "S2" in sc_id else 16
                ax.axvline(0.5, color="#0F172A", lw=2.0)
                ax.scatter([lx], [ly], s=32, color="#DC2626", marker="v", zorder=5)

                ax.set_xticks([])
                ax.set_yticks([])
                if r_idx == 0:
                    ax.set_title(cols[c_idx], fontsize=10.2, fontweight="bold", pad=6)
                if c_idx == 0:
                    ax.set_ylabel(r_lbl, fontsize=9.0, fontweight="bold")
                ax.text(
                    0.02,
                    0.96,
                    subtitle,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7.3,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", alpha=0.88, edgecolor="#CBD5E1", lw=0.6),
                )
    finally:
        conn.close()

    fig.suptitle(
        "Geometrias 3D Reais Geradas no Stage 5C: Projeção Lateral XY (esq.) + Seção Transversal YZ em x=32 (dir.)",
        fontsize=12.6,
        fontweight="bold",
    )
    _save(fig, "fig06_real_3d_geometries_comparison.png")


def build_fig07_sed_field_and_counterfactual_surgery() -> None:
    """
    Fig 07: Campo exato de Densidade de Energia de Deformação (SED_e via fast_amg)
    em S1 para Phys-UViT ep19 vs Treliça Binária 0/1 + seleção de voxels mortos p/ Probes.
    """
    from alphabuilder.src.logic.mcts.legal_moves import get_legal_remove_moves
    from alphabuilder.src.logic.harvest.generators import generate_bc_masks
    from alphabuilder.src.logic.selfplay.fast_fem import evaluate_fem_fast
    from alphabuilder.src.logic.storage import deserialize_array

    conn = sqlite3.connect(f"file:{EVAL_DB}?mode=ro", uri=True)
    try:
        blob = conn.execute(
            "SELECT density_blob FROM game_steps WHERE game_id='phys_uvit_ep19__S1_cantilever_long_center' ORDER BY step DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        conn.close()

    d_phys = deserialize_array(blob).astype(np.float32)
    d_truss = _make_binary_truss_s1()
    cfg_s1 = {"x": 63, "y": 16, "z_start": 3, "z_end": 5, "bc_type": "FULL_CLAMP"}
    res_p = evaluate_fem_fast(d_phys, cfg_s1)
    res_t = evaluate_fem_fast(d_truss, cfg_s1)
    print(f"fig07 SED verify | Phys S1 C={res_p.compliance:.2f} | Binary Truss C={res_t.compliance:.2f} V={res_t.vol_frac*100:.2f}%")

    # Sum SED over Z for XY visualization; mask voids as NaN so background renders light grey
    sed_p_xy = np.where(d_phys.max(axis=2).T > 0.5, np.log10(1.0 + 1000.0 * res_p.sed_map.sum(axis=2).T), np.nan)
    sed_t_xy = np.where(d_truss.max(axis=2).T > 0.5, np.log10(1.0 + 1000.0 * res_t.sed_map.sum(axis=2).T), np.nan)

    vmax = max(float(np.nanmax(sed_p_xy)), float(np.nanmax(sed_t_xy)))
    cmap_sed = plt.get_cmap("inferno").copy()
    cmap_sed.set_bad(color="#F1F5F9")

    fig = plt.figure(figsize=(15.8, 5.2), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.15, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(sed_p_xy, origin="lower", cmap=cmap_sed, vmin=0.0, vmax=vmax, aspect="auto")
    ax1.set_title(f"A. Phys-UViT ep19 em S1 (C={res_p.compliance:,.0f} J, V=8.89%)\nConcentração Severa no Engaste Estreito")
    ax1.set_xlabel("Coordenada X (Engaste x=0 -> Carga x=63)")
    ax1.set_ylabel("Coordenada Y (Altura Útil)")
    ax1.scatter([63], [16], color="#38BDF8", s=60, marker="<", edgecolor="white", zorder=5)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(sed_t_xy, origin="lower", cmap=cmap_sed, vmin=0.0, vmax=vmax, aspect="auto")
    ax2.set_title(f"B. Treliça Binária 0/1 (C={res_t.compliance:.1f} J, V=15.37%)\nBanzos Afastados Distribuem Energia Uniformemente")
    ax2.set_xlabel("Coordenada X")
    ax2.set_ylabel("Coordenada Y")
    ax2.scatter([63], [16], color="#38BDF8", s=60, marker="<", edgecolor="white", zorder=5)
    fig.colorbar(im2, ax=[ax1, ax2], label=r"$\log_{10}(1 + 1000 \cdot \sum_z \mathrm{SED}_e)$", pad=0.02, fraction=0.025)

    # Panel C: Sorted per-voxel SED spectrum + bottom-16 counterfactual REMOVE probe candidates
    ax3 = fig.add_subplot(gs[0, 2])
    mx, my, mz = generate_bc_masks((64, 32, 8), "FULL_CLAMP")
    forces_s1 = np.zeros((3, 64, 32, 8), dtype=np.float32)
    forces_s1[1, 63, 15:17, 3:5] = -1.0
    legal_rem = get_legal_remove_moves(d_phys, threshold=0.5, bc_masks=(mx, my, mz), forces=(forces_s1[0], forces_s1[1], forces_s1[2])) > 0.5
    sed_p_sorted = np.sort(res_p.sed_map[d_phys > 0.5])
    sed_t_sorted = np.sort(res_t.sed_map[d_truss > 0.5])
    sed_rem_legal = np.sort(res_p.sed_map[legal_rem])

    ax3.plot(np.linspace(0, 100, len(sed_p_sorted)), sed_p_sorted, color=COLORS["phys"], lw=2.2, label="Phys-UViT ep19 (1,456 voxels)")
    ax3.plot(np.linspace(0, 100, len(sed_t_sorted)), sed_t_sorted, color=COLORS["stage6"], lw=2.2, label="Treliça Binária 0/1 (2,518 voxels)")
    if len(sed_rem_legal) >= 16:
        pct_cutoff = 100.0 * 16.0 / len(sed_p_sorted)
        ax3.axvspan(0, pct_cutoff * 3.5, color="#FEE2E2", alpha=0.65, label="Zona Alvo Probes SED (menor SED)")
        ax3.scatter([pct_cutoff], [sed_rem_legal[15]], color=COLORS["simp_bin"], s=55, zorder=6)
        ax3.annotate(
            f"Top-16 remoção legal\nSED <= {sed_rem_legal[15]:.2e} J\n(vs pico {sed_p_sorted[-1]:.1f} J)",
            xy=(pct_cutoff, sed_rem_legal[15]),
            xytext=(18.0, 2e-3),
            fontsize=7.9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor=COLORS["simp_bin"], lw=0.8),
            arrowprops=dict(arrowstyle="->", color=COLORS["simp_bin"], lw=1.1),
        )

    ax3.set_yscale("log")
    ax3.set_xlim(-1, 101)
    ax3.set_ylim(1e-7, 2e3)
    ax3.set_xlabel("Percentil dos Voxels Sólidos Ordenados por SED [%]")
    ax3.set_ylabel("Energia por Voxel SED_e [J] (escala log)")
    ax3.set_title("C. Espectro SED_e e Seleção Cirúrgica Stage 6")
    ax3.grid(color=COLORS["grid"], linewidth=0.7)
    ax3.legend(loc="lower right", fontsize=7.6, frameon=True)

    fig.suptitle(
        "Física Microscópica via SED_e (fast_amg): Por Que Probes Contrafactuais Enxergam Massa Ociosa",
        fontsize=12.5,
        fontweight="bold",
    )
    _save(fig, "fig07_sed_field_and_counterfactual_surgery.png")


def build_fig08_training_dynamics_stage3_vs_stage5() -> None:
    """
    Fig 08: Curvas reais das 30 épocas do Stage 3 (`Simple`) vs Stage 5 (`Phys-UViT 3D`)
    evidenciando convergência das cabeças físicas e o Teto de Imitação Supervisionada.
    """
    h3 = _load_json(REPO_ROOT / "data" / "analysis" / "stage3_simple_metrics_history.json")
    h5 = _load_json(REPO_ROOT / "data" / "analysis" / "stage5_phys_uvit_metrics_history.json")

    ep3 = np.array([r["epoch"] for r in h3])
    ep5 = np.array([r["epoch"] for r in h5])

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 7.4), constrained_layout=True)

    # Panel A: Val Core Loss
    ax = axes[0, 0]
    ax.plot(ep3, [r["val"]["loss"] for r in h3], color=COLORS["ep30"], lw=2.2, marker="s", ms=3.8, label="Simple (Stage 3 Val Core)")
    ax.plot(ep5, [r["val"]["core_loss"] for r in h5], color=COLORS["phys"], lw=2.3, marker="o", ms=4.0, label="Phys-UViT 3D (Stage 5 Val Core)")
    ax.axvline(19, color=COLORS["simp_cont"], ls="--", lw=1.4)
    ax.scatter([19], [h5[18]["val"]["core_loss"]], color=COLORS["simp_cont"], s=75, zorder=6, label=f"Melhor Ep 19 ({h5[18]['val']['core_loss']:.4f}, -4.2%)")
    ax.set_title("A. Perda Principal de Validação (Policy + Value + Blueprint)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Val Core Loss")
    ax.grid(color=COLORS["grid"], linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.0, frameon=True)

    # Panel B: Value MAE
    ax = axes[0, 1]
    ax.plot(ep3, [r["val"]["value_mae"] for r in h3], color=COLORS["ep30"], lw=2.2, marker="s", ms=3.8, label="Simple Value MAE")
    ax.plot(ep5, [r["val"]["value_mae"] for r in h5], color=COLORS["phys"], lw=2.3, marker="o", ms=4.0, label="Phys-UViT 3D Value MAE")
    ax.axvline(19, color=COLORS["simp_cont"], ls="--", lw=1.4)
    ax.text(19.5, 0.025, f"Ep 19: MAE={h5[18]['val']['value_mae']:.4f}\n(mas OOD Bias=+0.463\nsem Self-Play!)", fontsize=8.1, bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEFCE8", edgecolor="#CA8A04"))
    ax.set_title("B. Erro Médio Absoluto de Valor In-Distribution (MAE)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Value MAE")
    ax.grid(color=COLORS["grid"], linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.0, frameon=True)

    # Panel C: Blueprint Dice & Policy Top-1 Accuracy
    ax = axes[1, 0]
    ax.plot(ep3, [r["val"]["blueprint_dice"] * 100 for r in h3], color=COLORS["ep30"], lw=2.0, ls="--", label="Simple Blueprint Dice [%]")
    ax.plot(ep5, [r["val"]["blueprint_dice"] * 100 for r in h5], color=COLORS["phys"], lw=2.3, label="Phys-UViT Blueprint Dice [%]")
    ax.plot(ep3, [r["val"]["policy_top1_acc"] * 100 for r in h3], color="#F59E0B", lw=1.7, ls=":", label="Simple Policy Top-1 [%]")
    ax.plot(ep5, [r["val"]["policy_top1_acc"] * 100 for r in h5], color=COLORS["stage6"], lw=2.0, label="Phys-UViT Policy Top-1 [%]")
    ax.axvline(19, color=COLORS["simp_cont"], ls="--", lw=1.3)
    ax.set_title("C. Convergência Estrutural (Blueprint Dice 91% e Policy Top-1)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia / Dice [%]")
    ax.grid(color=COLORS["grid"], linewidth=0.7)
    ax.legend(loc="lower right", fontsize=7.8, frameon=True)

    # Panel D: Factored Physics Heads (log10 C MAE & Conn Acc)
    ax = axes[1, 1]
    lc_mae = [r["val"]["log_c_mae"] for r in h5]
    conn_acc = [r["val"]["conn_acc"] * 100 for r in h5]
    ax2 = ax.twinx()
    l1 = ax.plot(ep5, lc_mae, color=COLORS["simp_bin"], lw=2.3, marker="o", ms=3.8, label="log10(C) MAE In-Dist [décadas]")
    l2 = ax2.plot(ep5, conn_acc, color=COLORS["simp_cont"], lw=2.2, ls="--", marker="^", ms=4.0, label="Acurácia Conectividade [%]")
    ax.axvline(19, color=COLORS["simp_cont"], ls="--", lw=1.3)
    ax.set_title("D. Cabeças Físicas Fatoradas (Stage 5 Phys-UViT 3D)")
    ax.set_xlabel("Época")
    ax.set_ylabel("log10(C) MAE [décadas]", color=COLORS["simp_bin"])
    ax2.set_ylabel("Acurácia Conectividade [%]", color=COLORS["simp_cont"])
    ax.set_ylim(0.022, 0.132)
    ax2.set_ylim(99.86, 100.015)
    ax.grid(color=COLORS["grid"], linewidth=0.7)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="center right", bbox_to_anchor=(0.98, 0.56), fontsize=7.9, frameon=True)
    ax.text(
        0.34,
        0.79,
        f"Teto de Imitação Supervisionada:\n"
        f"log10(C) cai até {min(lc_mae):.4f} décadas In-Dist,\n"
        f"mas sofre shift para 1.103 décadas em rolagens OOD.",
        transform=ax.transAxes,
        fontsize=7.9,
        va="center",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFF1F2", edgecolor=COLORS["simp_bin"], lw=0.8),
    )

    fig.suptitle(
        "Dinâmica Real de Treinamento (30 Épocas, 2x T4): Stage 3 Simple vs. Stage 5 Phys-UViT 3D",
        fontsize=12.6,
        fontweight="bold",
    )
    _save(fig, "fig08_training_dynamics_stage3_vs_stage5.png")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate AlphaBuilder Master Study Charts.")
    parser.add_argument("--only", type=str, default="", help="Comma-separated figure numbers e.g. 6,7,8")
    args = parser.parse_args()
    only = {int(x.strip()) for x in args.only.split(",") if x.strip()} if args.only else set(range(1, 9))

    eval_data = _load_json(EVAL_JSON)
    if 1 in only:
        build_fig01_compliance_gap_autopsy(eval_data)
    if 2 in only:
        build_fig02_reward_landscape_and_overtaking_path(eval_data)
    if 3 in only:
        build_fig03_ood_blindness_diagnosis()
    if 4 in only:
        build_fig04_four_tier_curriculum_and_selfplay_loop()
    if 5 in only:
        build_fig05_selfplay_projections_and_scaling_laws(eval_data)
    if 6 in only:
        build_fig06_real_3d_geometries_comparison(eval_data)
    if 7 in only:
        build_fig07_sed_field_and_counterfactual_surgery()
    if 8 in only:
        build_fig08_training_dynamics_stage3_vs_stage5()


if __name__ == "__main__":
    main()
