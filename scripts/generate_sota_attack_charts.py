#!/usr/bin/env python3
"""Generate publication-grade matplotlib figures for the BESO 3D SOTA Attack technical study.

Reads all empirical baselines directly from:
  - data/analysis/beso3d_benchmark_results.json
  - data/analysis/beso3d_geometries.npz
  - data/analysis/eval_stage5_phys_uvit_results.json
and uses exact reward/PUCT equations from AlphaBuilder v4.1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphabuilder.src.logic.selfplay.reward import calculate_reward

BESO_JSON = REPO_ROOT / "data" / "analysis" / "beso3d_benchmark_results.json"
BESO_NPZ = REPO_ROOT / "data" / "analysis" / "beso3d_geometries.npz"
STAGE5_JSON = REPO_ROOT / "data" / "analysis" / "eval_stage5_phys_uvit_results.json"
FIG_DIR = REPO_ROOT / "docs" / "figures"

# Global publication typography & clean framing
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13.0,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "savefig.dpi": 280,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

# Colorblind-safe restrained academic palette
C_GEN0 = "#94A3B8"      # Slate gray (Gen 0 baseline)
C_SIMP = "#D97706"      # Amber/ochre (SIMP continuous)
C_BESO_ISO = "#1D4ED8"  # Deep royal blue (BESO iso-V)
C_BESO_PAR = "#0F766E"  # Deep teal (BESO Pareto-Best)
C_V41 = "#B91C1C"       # Crimson (Stage 6 v4.1 upgrades / SOTA target)
C_OOD = "#7C3AED"       # Purple (OOD G1-G4)


def _short_id(sid: str) -> str:
    return sid.split("_", 1)[0]


def make_fig_sota01(beso: Dict, stage5: Dict, out_path: Path) -> None:
    """Fig 1: Score v3.3, log-Compliance across S1-S6 + G1-G4, and BESO Pareto trajectories."""
    beso_by_id = {s["scenario_id"]: s for s in beso["scenarios"]}
    canon_ids = [s["scenario_id"] for s in beso["scenarios"] if s["suite"] == "canonical"]
    ood_ids = [s["scenario_id"] for s in beso["scenarios"] if s["suite"] == "ood"]
    all_ids = canon_ids + ood_ids

    simp = stage5["simp_baselines"]
    gen0 = {ep["scenario_id"]: ep for ep in stage5["episodes"]}

    fig = plt.figure(figsize=(15.5, 9.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.05, 1.0], hspace=0.36, wspace=0.22)

    # --- Panel A: Score v3.3 on S1..S6 + Mean ---
    ax_a = fig.add_subplot(gs[0, 0])
    labels_a = [_short_id(s) for s in canon_ids] + ["Média S1–S6"]
    x_a = np.arange(len(labels_a))
    w = 0.20

    s_gen0 = [gen0[k]["final_score"] for k in canon_ids]
    s_simp = [simp[k]["reward_cont"] for k in canon_ids]
    s_iso = [beso_by_id[k]["iso_v"]["total_score"] for k in canon_ids]
    s_par = [beso_by_id[k]["pareto_best"]["total_score"] for k in canon_ids]

    vals_g = s_gen0 + [float(np.mean(s_gen0))]
    vals_s = s_simp + [float(np.mean(s_simp))]
    vals_i = s_iso + [float(np.mean(s_iso))]
    vals_p = s_par + [float(np.mean(s_par))]

    b1 = ax_a.bar(x_a - 1.5 * w, vals_g, width=w, color=C_GEN0, label="Gen 0 (Phys-UViT ep19)")
    b2 = ax_a.bar(x_a - 0.5 * w, vals_s, width=w, color=C_SIMP, label="SIMP Contínuo (FEniTop)")
    b3 = ax_a.bar(x_a + 0.5 * w, vals_i, width=w, color=C_BESO_ISO, label="BESO 3D (iso-V ≈ SIMP)")
    b4 = ax_a.bar(x_a + 1.5 * w, vals_p, width=w, color=C_BESO_PAR, label="BESO 3D (Pareto-Best R)")

    ax_a.axhline(0.0, color="#334155", linewidth=0.9)
    ax_a.axvline(5.5, color="#64748B", linestyle=":", linewidth=1.0)
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(labels_a, fontweight="bold")
    ax_a.set_ylim(-0.22, 0.96)
    ax_a.set_ylabel("Score Composto v3.3 (R)")
    ax_a.set_title("A. Score Composto v3.3 nos Cenários Canônicos (S1–S6)")

    # Direct labels above mean bars (staggered vertically to prevent BESO iso-V / Pareto overlap)
    mean_offsets = [0.015, 0.015, 0.015, 0.055]
    for bar, val, dy in zip([b1[-1], b2[-1], b3[-1], b4[-1]], [vals_g[-1], vals_s[-1], vals_i[-1], vals_p[-1]], mean_offsets):
        if dy > 0.03:
            bx = bar.get_x() + bar.get_width() / 2.0
            ax_a.plot([bx, bx], [val + 0.006, val + dy - 0.004], color="#0F766E", lw=0.8)
        ax_a.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + dy,
            f"{val:+.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#0F172A",
        )

    ax_a.text(
        0.02,
        0.04,
        "Nota: SIMP binarizado (>0,5) desconectou em 5/6 cenários (R médio = −0,825).",
        transform=ax_a.transAxes,
        fontsize=7.9,
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=0.7),
    )
    ax_a.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.95)

    # --- Panel B: Compliance (log10) across all 10 scenarios ---
    ax_b = fig.add_subplot(gs[0, 1])
    x_b = np.arange(len(all_ids))
    labels_b = [_short_id(k) for k in all_ids]

    c_gen0 = [gen0[k]["final_compliance"] if k in gen0 else np.nan for k in all_ids]
    c_simp = [simp[k]["comp_cont"] if k in simp else np.nan for k in all_ids]
    c_iso = [beso_by_id[k]["iso_v"]["compliance"] for k in all_ids]
    c_par = [beso_by_id[k]["pareto_best"]["compliance"] for k in all_ids]

    ax_b.plot(x_b[:6], c_gen0[:6], "o--", color=C_GEN0, lw=1.7, ms=6.0, label="Gen 0 (S1–S6)")
    ax_b.plot(x_b[:6], c_simp[:6], "s-", color=C_SIMP, lw=1.8, ms=6.0, label="SIMP Contínuo (S1–S6)")
    ax_b.plot(x_b, c_iso, "^-", color=C_BESO_ISO, lw=2.0, ms=6.5, label="BESO 3D iso-V (S1–G4)")
    ax_b.plot(x_b, c_par, "D-", color=C_BESO_PAR, lw=1.8, ms=5.5, label="BESO 3D Pareto-Best (S1–G4)")

    ax_b.set_yscale("log")
    ax_b.axvline(5.5, color="#64748B", linestyle="--", linewidth=1.2)
    ax_b.axvspan(5.5, 9.6, color="#F5F3FF", alpha=0.55, zorder=0)
    ax_b.text(2.5, 1.15e5, "Suíte Canônica (64×32×8)", ha="center", fontsize=8.3, color="#334155", fontweight="bold")
    ax_b.text(7.5, 1.15e5, "Suíte OOD (G1–G4)", ha="center", fontsize=8.3, color="#5B21B6", fontweight="bold")

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(labels_b, fontweight="bold")
    ax_b.set_xlim(-0.4, 9.5)
    ax_b.set_ylim(28.0, 2.6e5)
    ax_b.set_ylabel("Compliance C [J] — escala log (menor = mais rígido)")
    ax_b.set_title("B. Rigidez Estrutural Empírica em Todos os 10 Cenários")
    ax_b.legend(loc="upper right", bbox_to_anchor=(0.99, 0.90), ncol=1, frameon=True, facecolor="white", framealpha=0.96, fontsize=8.0)

    # Annotate BESO vs SIMP stiffness gain in open mid-gap away from legend
    ax_b.annotate(
        "BESO iso-V: C = 241,8 J vs.\nSIMP_cont: C = 459,7 J (1,9× mais rígido)",
        xy=(2.0, c_iso[2]),
        xytext=(0.05, 1950.0),
        fontsize=7.8,
        color="#1E3A8A",
        arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#93C5FD", lw=0.7),
    )

    # --- Panel C: Real BESO evolutionary trajectories (V% -> Score) ---
    ax_c = fig.add_subplot(gs[1, :])
    traj_specs = [
        ("S1_cantilever_long_center", "S1: Cantilever longa (FULL_CLAMP)", "#1D4ED8", "-"),
        ("S3_cantilever_midspan_top", "S3: Cantilever meio-vão (Pareto 10,9%)", "#0F766E", "-"),
        ("S6_short_deep_clamp", "S6: Cantilever curta (Pareto 8,0%)", "#047857", "-."),
        ("G1_dual_pad_angled", "G1 OOD: Dupla aba 30° (colapso guloso <17%)", "#DC2626", "-"),
        ("G2_bridge_mbb", "G2 OOD: Ponte MBB (pino + rolete)", "#7C3AED", "--"),
    ]

    for sid, label, col, ls in traj_specs:
        rec = beso_by_id[sid]
        hist = [h for h in rec["history"] if not h.get("in_iso_hold", False) and h["vol_frac"] <= 0.62]
        vols = [h["vol_frac"] * 100.0 for h in hist]
        scores = [h["score"] for h in hist]
        ax_c.plot(vols, scores, linestyle=ls, color=col, linewidth=2.1, label=label)

        iv = rec["iso_v"]
        pb = rec["pareto_best"]
        ax_c.scatter([iv["volume_fraction"] * 100.0], [iv["total_score"]], s=48, color=col, edgecolor="white", zorder=5)
        ax_c.scatter(
            [pb["volume_fraction"] * 100.0],
            [pb["total_score"]],
            s=115,
            marker="*",
            color=col,
            edgecolor="black",
            linewidth=0.6,
            zorder=6,
        )

    # Custom legend entries for markers
    ax_c.scatter([], [], s=48, color="#334155", edgecolor="white", label="● Corte iso-V (~13,5%)")
    ax_c.scatter([], [], s=110, marker="*", color="#334155", edgecolor="black", lw=0.6, label="★ Máximo Pareto-Best (R)")

    ax_c.annotate(
        "Penhasco guloso em G1:\nPareto R=+0,518 (V=17,0%)\ncai para +0,415 em 13,5%",
        xy=(13.48, 0.4152),
        xytext=(24.5, 0.34),
        fontsize=8.2,
        color="#991B1B",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#FCA5A5", lw=0.8),
    )

    ax_c.set_xlim(60.0, 6.5)
    ax_c.set_ylim(0.28, 0.90)
    ax_c.set_xlabel("Fração Volumétrica Sólida (%) — erosão evolutiva da esquerda para a direita")
    ax_c.set_ylabel("Score Composto v3.3 (R)")
    ax_c.set_title("C. Trajetórias Evolutivas Reais do BESO 3D: Compromisso Rigidez–Leveza e Falha da Heurística Gulosa")
    ax_c.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.96)

    fig.suptitle(
        "Dados Empíricos Medidos — BESO 3D (Soft-Kill) vs. SIMP Contínuo vs. Phys-UViT Gen 0 (v4 Stage 5)",
        y=0.985,
    )
    fig.savefig(out_path)
    plt.close(fig)


def make_fig_sota02(beso: Dict, npz_path: Path, out_path: Path) -> None:
    """Fig 2: 6 real 3D BESO Pareto-Best topologies (XY Z-thickness + YZ cross section)."""
    z = np.load(npz_path)
    beso_by_id = {s["scenario_id"]: s for s in beso["scenarios"]}

    panels = [
        ("S1_cantilever_long_center", "S1: Cantilever longa (FULL_CLAMP)", (63, 16, 0.0, -1.0), "full", None),
        ("S3_cantilever_midspan_top", "S3: Meio-vão superior (FULL_CLAMP)", (44, 24, 0.0, -1.0), "full", None),
        ("S5_rail_xy_3d_eccentric", "S5: RAIL_XY excêntrico Z (3D assimétrico)", (52, 11, 0.0, -1.0), "rail", None),
        ("G1_dual_pad_angled", "G1 OOD: Dupla aba x=0 + carga 30°", (58, 16, 0.5, -0.866), "dual", None),
        ("G2_bridge_mbb", "G2 OOD: Ponte MBB (pino + rolete)", (32, 30, 0.0, -1.0), "mbb", None),
        ("G3_keepout_window", "G3 OOD: Janela proibida central", (60, 16, 0.0, -1.0), "full", (18, 38, 10, 22)),
    ]

    fig = plt.figure(figsize=(16.2, 9.0))
    outer = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.20)

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="#F8FAFC")

    for idx, (sid, label, load_vec, bc_kind, keepout) in enumerate(panels):
        rec = beso_by_id[sid]
        pb = rec["pareto_best"]
        dens = z[f"{sid}__pareto"].astype(np.float32)  # (nx, ny, nz)
        nx, ny, nz = dens.shape

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[idx], width_ratios=[4.3, 1.05], wspace=0.07
        )
        ax_xy = fig.add_subplot(inner[0])
        ax_yz = fig.add_subplot(inner[1])

        # XY view: thickness along Z
        xy_mean = dens.mean(axis=2).T  # (ny, nx)
        xy_mask = np.ma.masked_where(dens.sum(axis=2).T <= 0.0, xy_mean)
        ax_xy.imshow(xy_mask, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

        # YZ view: thickness along X
        yz_mean = dens.mean(axis=0)  # (ny, nz)
        yz_mask = np.ma.masked_where(dens.sum(axis=0) <= 0.0, yz_mean)
        ax_yz.imshow(yz_mask, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

        # Supports overlay
        if bc_kind in ("full", "rail"):
            col_sup = "#1E293B" if bc_kind == "full" else "#64748B"
            ax_xy.add_patch(patches.Rectangle((-0.4, -0.4), 1.3, ny - 0.2, facecolor=col_sup, edgecolor="none"))
        elif bc_kind == "dual":
            ax_xy.add_patch(patches.Rectangle((-0.4, 0.0), 1.4, 8.0, facecolor="#1E293B", edgecolor="none"))
            ax_xy.add_patch(patches.Rectangle((-0.4, 24.0), 1.4, 8.0, facecolor="#1E293B", edgecolor="none"))
        elif bc_kind == "mbb":
            ax_xy.add_patch(patches.Rectangle((-0.4, 0.0), 1.5, 6.0, facecolor="#1E293B", edgecolor="none"))
            ax_xy.add_patch(patches.Rectangle((nx - 1.2, 0.0), 1.5, 6.0, facecolor="#0284C7", edgecolor="none"))

        # Keep-out window in G3
        if keepout is not None:
            x0, x1, y0, y1 = keepout
            ax_xy.add_patch(
                patches.Rectangle(
                    (x0 - 0.5, y0 - 0.5),
                    x1 - x0,
                    y1 - y0,
                    facecolor="#FEE2E2",
                    edgecolor="#DC2626",
                    hatch="////",
                    linewidth=1.1,
                    alpha=0.85,
                )
            )
            ax_xy.text((x0 + x1) / 2.0, (y0 + y1) / 2.0, "Proibida", ha="center", va="center", fontsize=7.5, color="#991B1B", fontweight="bold")

        # Load arrow
        lx, ly, fx, fy = load_vec
        scale = 6.2
        dx, dy = fx * scale, fy * scale
        ax_xy.annotate(
            "",
            xy=(lx + dx, ly + dy),
            xytext=(lx, ly),
            arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color="#DC2626", lw=2.0),
        )
        ax_xy.scatter([lx], [ly], color="#DC2626", s=24, zorder=6)

        ax_xy.set_xlim(-1.0, nx - 0.5)
        ax_xy.set_ylim(-0.5, ny - 0.5)
        ax_xy.set_xlabel("X (vão)", fontsize=8.3)
        ax_xy.set_ylabel("Y (altura)", fontsize=8.3)
        ax_xy.grid(False)

        ax_yz.set_xticks([0, nz - 1])
        ax_yz.set_xticklabels(["0", str(nz - 1)], fontsize=7.5)
        ax_yz.set_yticks([])
        ax_yz.set_xlabel("Z (YZ)", fontsize=8.1)
        ax_yz.grid(False)

        # Highlight S5 Z centroid asymmetry
        if sid.startswith("S5"):
            zc = pb["geometry"]["z_centroid"]
            ax_yz.axvline(zc, color="#DC2626", linestyle="--", linewidth=1.2)
            ax_yz.text(zc + 0.3, ny * 0.86, f"z_c={zc:.2f}", fontsize=7.3, color="#B91C1C", fontweight="bold")

        g = pb["geometry"]
        stats_line = (
            f"V={pb['volume_fraction']*100:.2f}% | C={pb['compliance']:.1f} J | "
            f"R={pb['total_score']:+.3f} | hol={g['hollow_ratio_xy']*100:.0f}%"
        )
        ax_xy.set_title(f"{label}\n{stats_line}", fontsize=8.8, fontweight="bold", loc="left", pad=5)

    fig.suptitle(
        "Geometrias Reais 3D do Benchmark BESO 3D (Pareto-Best): Projeção XY e Seção Transversal YZ",
        y=0.985,
    )
    fig.savefig(out_path)
    plt.close(fig)


def make_fig_sota03(out_path: Path) -> None:
    """Fig 3: Exact mechanistic comparison of MCTS-DAG v4.1 & Episode Lifecycle fixes."""
    fig = plt.figure(figsize=(16.5, 5.3))
    gs = gridspec.GridSpec(1, 3, wspace=0.31)

    # --- Panel A: Exact PUCT vs N_c comparing FPU=0 vs relative FPU ---
    ax_a = fig.add_subplot(gs[0, 0])
    n_c = np.arange(0, 26)
    n_parent = 40
    q_parent = 0.45
    c_puct = 1.0
    p_sibling = 0.06  # Realistic REMOVE/SWAP edge prior

    # Legacy FPU=0: at n_c=0, q_value = 0.0; for n_c >= 1 assume discovered empirical Q = 0.48
    def _puct_curve(fpu_val: float, q_discovered: float = 0.48) -> np.ndarray:
        vals = []
        for n in n_c:
            q = fpu_val if n == 0 else q_discovered
            expl = c_puct * p_sibling * math.sqrt(n_parent) / (1.0 + n)
            vals.append(q + expl)
        return np.array(vals)

    y_fpu0 = _puct_curve(0.0)
    y_rel = _puct_curve(max(-1.0, q_parent - 0.25))  # q_parent - 0.25 = +0.20

    # Visited dominant ADD sibling already holding Q = +0.44 after 10 visits (P = 0.14)
    puct_visited_add = 0.44 + c_puct * 0.14 * math.sqrt(n_parent) / (1.0 + 10.0)

    ax_a.plot(n_c, y_fpu0, "o--", color="#64748B", lw=1.8, ms=4.2, label="Legado: FPU = 0,0 fixo")
    ax_a.plot(n_c, y_rel, "s-", color=C_V41, lw=2.1, ms=4.5, label="v4.1: FPU relativo (Q_pai − 0,25 = +0,20)")
    ax_a.axhline(
        puct_visited_add,
        color=C_BESO_ISO,
        linestyle=":",
        lw=1.7,
        label=f"Irmão ADD visitado (N=10, Q=+0,44): PUCT={puct_visited_add:.2f}",
    )

    ax_a.scatter([0], [y_fpu0[0]], color="#64748B", s=65, zorder=5)
    ax_a.scatter([0], [y_rel[0]], color=C_V41, s=75, zorder=5)
    ax_a.annotate(
        f"N=0 travado\nPUCT={y_fpu0[0]:.2f} < {puct_visited_add:.2f}",
        xy=(0.0, y_fpu0[0]),
        xytext=(5.2, 0.36),
        fontsize=8.0,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CBD5E1", lw=0.6),
        arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.0),
    )
    ax_a.annotate(
        f"Desbloqueado em N=0!\nPUCT={y_rel[0]:.2f} > {puct_visited_add:.2f}",
        xy=(0.0, y_rel[0]),
        xytext=(4.0, 0.615),
        fontsize=8.0,
        color="#991B1B",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_V41, lw=1.1),
    )

    ax_a.set_xlim(-0.8, 25.0)
    ax_a.set_ylim(0.18, 0.72)
    ax_a.set_xlabel("Visitas ao filho candidato ($N_c$)")
    ax_a.set_ylabel("Pontuação PUCT (c_puct = 1,0, P = 0,06)")
    ax_a.set_title("A. FPU Relativo: Desbloqueio em N=0", fontsize=10.3)
    ax_a.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.96, fontsize=7.5)

    # --- Panel B: Combinatorial path redundancy vs DAG + macro speedup ---
    ax_b = fig.add_subplot(gs[0, 1])
    depths = np.arange(1, 9)
    raw_tree_paths = np.array([math.factorial(int(d)) for d in depths], dtype=float)
    dag_merged_paths = np.ones_like(depths, dtype=float)

    ax_b.plot(depths, raw_tree_paths, "o-", color="#64748B", lw=2.0, ms=5, label="Árvore plana: $d!$ permutações comutativas")
    ax_b.plot(depths, dag_merged_paths, "s-", color=C_BESO_ISO, lw=2.1, ms=5, label="MCTS-DAG: estado único após fusão A+B = B+A")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Profundidade de busca ($d$ passos comutativos)")
    ax_b.set_ylabel("Caminhos redundantes por estado (log)")
    ax_b.set_title("B. Colapso DAG e Alcance de Macro-Ações", fontsize=10.3)
    ax_b.set_ylim(0.5, 1.5e5)
    ax_b.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.96, fontsize=7.5)

    ax_b_in = ax_b.inset_axes([0.52, 0.14, 0.44, 0.35])
    horiz_x = np.arange(1, 5)
    ax_b_in.bar(horiz_x - 0.18, horiz_x * 1.0, width=0.35, color="#94A3B8", label="1-voxel")
    ax_b_in.bar(horiz_x + 0.18, horiz_x * 8.0, width=0.35, color=C_V41, label="Macro (k=8)")
    ax_b_in.set_xticks(horiz_x)
    ax_b_in.set_title("Alcance (voxels)", fontsize=7.6, fontweight="bold")
    ax_b_in.tick_params(labelsize=7.0)
    ax_b_in.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax_b_in.grid(True, alpha=0.2)

    # --- Panel C: Phase-2 Termination & Best-of-Episode vs Poisoned Last-Step ---
    ax_c = fig.add_subplot(gs[0, 2])
    steps = np.arange(0, 29)
    # Construct physically realistic compliance & volume trajectory and evaluate exact v3.3 reward!
    # Step 0..4: thin fragile bridge just connected (C drops rapidly as chord thickens)
    # Step 5..14: cavity carving improves V while maintaining low C -> rises to peak at t*=14
    # Step 15..28: over-erosion degrades stiffness faster than V saves
    vols = np.array(
        [
            0.055, 0.068, 0.082, 0.095, 0.110, 0.125, 0.132, 0.128, 0.122, 0.118,
            0.114, 0.110, 0.106, 0.103, 0.100, 0.096, 0.092, 0.088, 0.084, 0.080,
            0.076, 0.072, 0.069, 0.066, 0.063, 0.060, 0.058, 0.055, 0.053,
        ]
    )
    comps = np.array(
        [
            18000, 7500, 3200, 1650, 950, 620, 440, 355, 295, 255,
            230, 212, 198, 190, 186, 235, 315, 430, 610, 880,
            1250, 1820, 2650, 3800, 5400, 7600, 10800, 14500, 19200,
        ],
        dtype=float,
    )
    scores = np.array([calculate_reward(c, v, is_valid=True) for c, v in zip(comps, vols)])

    peak_idx = int(np.argmax(scores))
    stop_v41_idx = min(len(steps) - 1, peak_idx + 12)  # patience=12 / MCTS_STOP
    gamma = 0.99

    z_legacy = np.array([scores[-1] * (gamma ** (len(steps) - 1 - t)) for t in steps])
    z_best = np.zeros_like(scores)
    # Match exact runner/storage best-of-episode: max R over future steps >= t discounted from argmax(R[t:])
    for t in range(len(steps)):
        end_slice = stop_v41_idx + 1 if t <= stop_v41_idx else len(steps)
        rel_idx = int(np.argmax(scores[t:end_slice]))
        z_best[t] = float(scores[t + rel_idx] * (gamma ** rel_idx))

    ax_c.axvspan(0, 4.5, color="#ECFDF5", alpha=0.8, label="Carência Pós-Conexão (5 passos)")
    ax_c.plot(steps, scores, "o-", color="#0F172A", lw=2.0, ms=3.8, label="Score imediato FEM $R_{\\mathrm{total}}(t)$")
    ax_c.plot(steps, z_legacy, "--", color="#64748B", lw=1.7, label="Alvo legado (último passo $t=T$)")
    ax_c.plot(steps[: stop_v41_idx + 1], z_best[: stop_v41_idx + 1], "-", color=C_V41, lw=2.1, label="Alvo v4.1 Best-of-Episode ($\\max_{\\tau\\geq t}$)")

    ax_c.scatter([peak_idx], [scores[peak_idx]], marker="*", s=140, color=C_BESO_PAR, edgecolor="black", zorder=6)
    ax_c.axvline(peak_idx + 6, color="#EA580C", linestyle="--", lw=1.2)
    ax_c.axvline(stop_v41_idx, color=C_V41, linestyle=":", lw=1.5)
    ax_c.text(peak_idx - 0.8, scores[peak_idx] + 0.035, f"Pico t*={peak_idx} (R*={scores[peak_idx]:+.2f})", fontsize=7.8, color="#0F766E", fontweight="bold", ha="center")
    ax_c.annotate(
        "MCTS_STOP\n(após ≥6)",
        xy=(peak_idx + 6, 0.48),
        xytext=(peak_idx + 8.5, 0.62),
        fontsize=7.3,
        color="#C2410C",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#EA580C", lw=0.9),
    )
    ax_c.annotate(
        "Paciência=12",
        xy=(stop_v41_idx, 0.21),
        xytext=(18.2, 0.23),
        fontsize=7.3,
        color="#991B1B",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#FCA5A5", lw=0.6),
        arrowprops=dict(arrowstyle="->", color=C_V41, lw=0.9),
    )

    ax_c.set_xlim(-0.5, 28.5)
    ax_c.set_ylim(-0.15, 0.85)
    ax_c.set_xlabel("Passo na Fase 2 (após conexão da ponte)")
    ax_c.set_ylabel("Recompensa / Alvo de Valor ($z_t$)")
    ax_c.set_title("C. Carência, Parada e Alvo Best-of-Episode", fontsize=10.3)
    ax_c.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=7.3)

    fig.suptitle(
        "Análise Mecanística das Atualizações MCTS-DAG v4.1 e Ciclo de Vida (Equações Exatas do Repositório)",
        y=1.02,
    )
    fig.savefig(out_path)
    plt.close(fig)


def make_fig_sota04(beso: Dict, stage5: Dict, out_path: Path) -> None:
    """Fig 4: Modeled Stage 6 attack projections and Pareto Score vs FEM budget frontier."""
    s1_6_iso = beso["summary"]["canonical_iso_v"]["mean_score"]
    s1_6_par = beso["summary"]["canonical_pareto_best"]["mean_score"]
    g1_4_par = beso["summary"]["ood_pareto_best"]["mean_score"]
    simp_cont = float(np.mean([v["reward_cont"] for v in stage5["simp_baselines"].values()]))
    gen0_mean = float(np.mean([ep["final_score"] for ep in stage5["episodes"]]))

    fig = plt.figure(figsize=(15.8, 5.8))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)

    # --- Panel A: Trajectory Gen 0 -> Gen 8 ---
    ax_a = fig.add_subplot(gs[0, 0])
    # X labels mapped cleanly to integer positions 0..9: Gen0 (emp), Gen0+v4.1 (sem retreino), Gen1..Gen8
    xs = np.arange(0, 10)
    x_labels = ["Gen 0\n(emp.)", "Gen 0\n+v4.1", "Gen 1", "Gen 2", "Gen 3", "Gen 4", "Gen 5", "Gen 6", "Gen 7", "Gen 8"]

    # Clearly modeled projection curves anchored at empirical Gen 0 = +0.0764
    s16_nom = np.array([gen0_mean, 0.28, 0.42, 0.52, 0.565, 0.595, 0.615, 0.628, 0.638, 0.645])
    s16_lo = np.array([gen0_mean, 0.22, 0.36, 0.47, 0.515, 0.545, 0.568, 0.585, 0.598, 0.606])
    s16_hi = np.array([gen0_mean, 0.33, 0.47, 0.56, 0.600, 0.625, 0.642, 0.654, 0.662, 0.668])

    ood_nom = np.array([np.nan, 0.18, 0.31, 0.44, 0.505, 0.550, 0.582, 0.605, 0.620, 0.630])
    ood_lo = np.array([np.nan, 0.12, 0.24, 0.37, 0.445, 0.495, 0.530, 0.556, 0.574, 0.586])
    ood_hi = np.array([np.nan, 0.23, 0.37, 0.49, 0.550, 0.590, 0.618, 0.638, 0.650, 0.658])

    # Empirical reference horizontal lines
    ax_a.axhline(s1_6_par, color=C_BESO_PAR, linestyle="--", lw=1.8, label=f"Empírico BESO 3D Pareto S1–S6 ({s1_6_par:+.3f})")
    ax_a.axhline(s1_6_iso, color=C_BESO_ISO, linestyle="-.", lw=1.6, label=f"Empírico BESO 3D iso-V S1–S6 ({s1_6_iso:+.3f})")
    ax_a.axhline(g1_4_par, color="#0D9488", linestyle=":", lw=1.5, label=f"Empírico BESO 3D Pareto G1–G4 ({g1_4_par:+.3f})")
    ax_a.axhline(simp_cont, color=C_SIMP, linestyle="--", lw=1.5, label=f"Empírico SIMP Contínuo S1–S6 ({simp_cont:+.3f})")

    # Modeled bands & curves
    ax_a.fill_between(xs, s16_lo, s16_hi, color=C_V41, alpha=0.14)
    ax_a.plot(xs, s16_nom, "o-", color=C_V41, lw=2.3, ms=5.5, label="Projeção Etapa 6 v4.1 — S1–S6 (nominal ± banda)")
    ax_a.scatter([0], [gen0_mean], color=C_GEN0, edgecolor="black", s=85, zorder=7)

    ax_a.fill_between(xs[1:], ood_lo[1:], ood_hi[1:], color=C_OOD, alpha=0.11)
    ax_a.plot(xs[1:], ood_nom[1:], "^--", color=C_OOD, lw=1.9, ms=5.2, label="Projeção Etapa 6 v4.1 — OOD G1–G4")

    # Gates anchored cleanly near top margin above data curves
    for gx, gtxt in [(3, "Gate 1 (Gen 2)"), (6, "Gate 2 (Gen 5)"), (9, "Gate 3 (Gen 8)")]:
        ax_a.axvline(gx, color="#94A3B8", linestyle=":", lw=1.0)
        ax_a.text(gx - 0.08, 0.705, gtxt, ha="right", va="bottom", fontsize=7.6, color="#334155", fontweight="bold")

    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(x_labels, fontsize=8.0)
    ax_a.set_xlim(-0.35, 9.35)
    ax_a.set_ylim(0.0, 0.75)
    ax_a.set_ylabel("Score Composto v3.3 (R)")
    ax_a.set_title("A. Trajetória Projetada de Ataque (vs. Baselines Empíricos)")
    ax_a.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=7.6)

    # --- Panel B: Pareto Frontier (Score vs FEM Evaluations) ---
    ax_b = fig.add_subplot(gs[0, 1])

    mean_beso_iters = float(np.mean([s["iterations_run"] for s in beso["scenarios"]]))
    mean_gen0_p2 = float(np.mean([ep["phase2_steps_run"] for ep in stage5["episodes"]]))

    # Empirical points
    emp_pts = [
        (mean_gen0_p2, gen0_mean, C_GEN0, "o", f"Gen 0 empírico\n(~{mean_gen0_p2:.0f} FEMs, +0,076)"),
        (120.0, simp_cont, C_SIMP, "s", f"SIMP Contínuo FEniTop\n(~120 FEMs, {simp_cont:+.3f})"),
        (mean_beso_iters, s1_6_iso, C_BESO_ISO, "^", f"BESO 3D iso-V\n({mean_beso_iters:.0f} FEMs, {s1_6_iso:+.3f})"),
        (mean_beso_iters, s1_6_par, C_BESO_PAR, "D", f"BESO 3D Pareto-Best\n({mean_beso_iters:.0f} FEMs, {s1_6_par:+.3f})"),
    ]
    for fx, fy, col, mk, txt in emp_pts:
        ax_b.scatter([fx], [fy], color=col, marker=mk, s=95, edgecolor="black", linewidth=0.7, zorder=6)

    ax_b.annotate("Gen 0 empírico\n(30 FEMs, +0,076)", xy=(mean_gen0_p2, gen0_mean), xytext=(22, 0.14), fontsize=8.0, color="#334155", arrowprops=dict(arrowstyle="->", color="#64748B"))
    ax_b.annotate(f"SIMP Contínuo\n(~120 FEMs, {simp_cont:+.3f})", xy=(120, simp_cont), xytext=(65, 0.44), fontsize=8.0, color="#92400E", arrowprops=dict(arrowstyle="->", color=C_SIMP))
    ax_b.annotate(f"BESO 3D iso-V ({mean_beso_iters:.0f} FEMs, {s1_6_iso:+.3f})", xy=(mean_beso_iters, s1_6_iso), xytext=(56, 0.555), fontsize=7.9, color="#1E3A8A", arrowprops=dict(arrowstyle="->", color=C_BESO_ISO))
    ax_b.annotate(f"BESO 3D Pareto\n({mean_beso_iters:.0f} FEMs, {s1_6_par:+.3f})", xy=(mean_beso_iters, s1_6_par), xytext=(116, 0.678), ha="center", fontsize=8.1, color="#065F46", fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_BESO_PAR))

    # Modeled AlphaBuilder v4.1 amortized Pareto curve (Gen 6..8 regimes)
    ab_fems = np.array([7.5, 24.0, 50.0])
    ab_scores = np.array([0.555, 0.625, 0.652])
    ax_b.plot(ab_fems, ab_scores, "*-", color=C_V41, lw=2.3, ms=12, label="Projeção AlphaBuilder v4.1 (Gen 6–8 amortizado)")
    ax_b.fill_between(ab_fems, ab_scores - 0.025, ab_scores + 0.020, color=C_V41, alpha=0.14)

    for fx, fy, lbl, tx, ty in [
        (7.5, 0.555, "Modo Rápido\n(6–10 FEMs, >SIMP)", 6.8, 0.615),
        (24.0, 0.625, "Modo Padrão DAG\n(20–30 FEMs, ≈BESO)", 19.0, 0.675),
        (50.0, 0.652, "Busca Profunda\n(45–55 FEMs)", 48.0, 0.702),
    ]:
        ax_b.annotate(
            lbl,
            xy=(fx, fy),
            xytext=(tx, ty),
            ha="center",
            fontsize=7.9,
            color="#991B1B",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_V41, lw=0.9),
        )

    # Shade 3x-10x speedup region
    ax_b.annotate(
        "Vantagem de amortização\n3× a 10× menos chamadas FEM\npor geometria inédita",
        xy=(16.0, 0.605),
        xytext=(5.2, 0.34),
        fontsize=8.2,
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FEF2F2", edgecolor="#FCA5A5", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_V41, lw=1.1),
    )

    ax_b.set_xscale("log")
    ax_b.set_xlim(4.0, 175.0)
    ax_b.set_ylim(0.03, 0.74)
    ax_b.set_xticks([5, 10, 20, 30, 50, 100, 150])
    ax_b.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_b.set_xlabel("Avaliações FEM Reais por Geometria Inédita (escala log)")
    ax_b.set_ylabel("Score Composto v3.3 (R)")
    ax_b.set_title("B. Fronteira de Pareto: Qualidade Discreta vs. Orçamento FEM")
    ax_b.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.0)

    fig.suptitle(
        "Projeções Modeladas da Etapa 6 v4.1 vs. Referências Empíricas (BESO 3D, SIMP e Gen 0)",
        y=1.01,
    )
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    beso = json.loads(BESO_JSON.read_text(encoding="utf-8"))
    stage5 = json.loads(STAGE5_JSON.read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    f1 = FIG_DIR / "fig_sota01_beso3d_vs_simp_vs_gen0.png"
    f2 = FIG_DIR / "fig_sota02_beso3d_real_geometries.png"
    f3 = FIG_DIR / "fig_sota03_mcts_and_termination_ablation.png"
    f4 = FIG_DIR / "fig_sota04_stage6_attack_projections.png"

    make_fig_sota01(beso, stage5, f1)
    make_fig_sota02(beso, BESO_NPZ, f2)
    make_fig_sota03(f3)
    make_fig_sota04(beso, stage5, f4)
    print("Generated charts:", [str(p) for p in (f1, f2, f3, f4)])


if __name__ == "__main__":
    main()
