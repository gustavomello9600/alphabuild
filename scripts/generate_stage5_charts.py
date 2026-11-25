#!/usr/bin/env python3
"""Generate publication-grade comparison charts for AlphaBuilder v4 Stage 5 (Phys-UViT 3D).

Modes:
  --mode training : Reads stage5_phys_uvit_metrics_history.json and
                    stage3_simple_metrics_history.json, generating:
                    - data/analysis/figures/stage5_losses_comparison.png
                    - data/analysis/figures/stage5_validation_heads_comparison.png
  --mode eval     : Reads eval_ep2_results.json, eval_ep30_results.json, and
                    eval_stage5_phys_uvit_results.json, generating:
                    - data/analysis/figures/stage5_eval_compliance_score.png
                    - data/analysis/figures/stage5_eval_geometry_and_calibration.png
  --mode all      : Runs both modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
FIG_DIR = ANALYSIS_DIR / "figures"

# Colorblind-friendly palette consistent with Stage 3 charts
C_PHYS_VAL = "#2563eb"     # Royal Blue
C_PHYS_TR = "#0f766e"      # Teal
C_SIMPLE_VAL = "#64748b"   # Slate Gray
C_SIMPLE_TR = "#94a3b8"    # Light Slate
C_EP2 = "#cbd5e1"          # Pale Slate
C_SIMP_BIN = "#f59e0b"     # Amber
C_SIMP_CONT = "#059669"    # Emerald
C_ALERT = "#dc2626"        # Coral Red
C_PURPLE = "#7c3aed"       # Violet
C_DARK = "#1e293b"         # Slate 800
C_GRID = "#e2e8f0"         # Slate 200


def _apply_clean_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": C_GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.2,
            "figure.titlesize": 13.5,
            "figure.titleweight": "bold",
            "text.color": C_DARK,
            "axes.labelcolor": C_DARK,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / filename
    fig.savefig(out, dpi=180, facecolor="#ffffff", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"Saved: {out.relative_to(REPO_ROOT)} ({out.stat().st_size / 1024:.1f} KB)")
    return out


def _fmt_pt(val: float, fmt: str = ".4f") -> str:
    return format(val, fmt).replace(".", ",")


def plot_stage5_losses_comparison(
    s5_hist: List[Dict[str, Any]],
    s3_hist: List[Dict[str, Any]],
) -> None:
    ep5 = np.array([h["epoch"] for h in s5_hist])
    ep3 = np.array([h["epoch"] for h in s3_hist])

    # Core v3.3 loss (Phys-UViT logs core_loss explicitly; Simple's total loss == core_loss)
    s5_tr_core = np.array([h["train"].get("core_loss", h["train"]["loss"]) for h in s5_hist])
    s5_va_core = np.array([h["val"].get("core_loss", h["val"]["loss"]) for h in s5_hist])
    s3_tr_core = np.array([h["train"]["loss"] for h in s3_hist])
    s3_va_core = np.array([h["val"]["loss"] for h in s3_hist])

    s5_tr_pol = np.array([h["train"]["policy_loss"] for h in s5_hist])
    s5_va_pol = np.array([h["val"]["policy_loss"] for h in s5_hist])
    s3_tr_pol = np.array([h["train"]["policy_loss"] for h in s3_hist])
    s3_va_pol = np.array([h["val"]["policy_loss"] for h in s3_hist])

    s5_tr_val = np.array([h["train"]["value_loss"] for h in s5_hist])
    s5_va_val = np.array([h["val"]["value_loss"] for h in s5_hist])
    s3_tr_val = np.array([h["train"]["value_loss"] for h in s3_hist])
    s3_va_val = np.array([h["val"]["value_loss"] for h in s3_hist])

    s5_tr_bp = np.array([h["train"]["blueprint_loss"] for h in s5_hist])
    s5_va_bp = np.array([h["val"]["blueprint_loss"] for h in s5_hist])
    s3_tr_bp = np.array([h["train"]["blueprint_loss"] for h in s3_hist])
    s3_va_bp = np.array([h["val"]["blueprint_loss"] for h in s3_hist])

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), sharex=True)

    def _plot_pair(ax, y5_tr, y5_va, y3_tr, y3_va, title, ylabel, log_scale=False):
        ax.axvspan(1, 3, color="#fef3c7", alpha=0.45, zorder=0, label="Warmup LR (Ep 1–3)")
        ax.plot(ep3, y3_tr, color=C_SIMPLE_TR, lw=1.4, ls="--", label="Simple Train")
        ax.plot(ep3, y3_va, color=C_SIMPLE_VAL, lw=1.9, ls="-", marker="o", ms=2.8, label="Simple Val")
        ax.plot(ep5, y5_tr, color=C_PHYS_TR, lw=1.7, ls="--", label="Phys-UViT Train")
        ax.plot(ep5, y5_va, color=C_PHYS_VAL, lw=2.3, ls="-", marker="s", ms=3.2, label="Phys-UViT Val")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.legend(loc="upper right", ncol=2, frameon=True, facecolor="#ffffff", framealpha=0.92, edgecolor="#e2e8f0", fontsize=7.6)

    # A. Core Loss v3.3
    ax = axes[0, 0]
    _plot_pair(ax, s5_tr_core, s5_va_core, s3_tr_core, s3_va_core,
               "A. Core Loss v3.3 (Policy + 2·Value + 0,5·Blueprint)", "Core Loss")
    i5_best = int(np.argmin(s5_va_core))
    i3_best = int(np.argmin(s3_va_core))
    ax.scatter([ep5[i5_best], ep3[i3_best]], [s5_va_core[i5_best], s3_va_core[i3_best]],
               color=[C_ALERT, C_SIMPLE_VAL], s=[55, 45], zorder=6)
    cross_idx = int(np.where(s5_va_core < s3_va_core[-1])[0][0])
    ax.scatter([ep5[cross_idx]], [s5_va_core[cross_idx]], color=C_SIMP_CONT, s=48, zorder=6)
    ax.annotate(
        f"Phys-UViT já bate\nEp 30 Simple na Ep {ep5[cross_idx]}\n({_fmt_pt(s5_va_core[cross_idx])})",
        xy=(ep5[cross_idx], s5_va_core[cross_idx]),
        xytext=(5.5, 0.72),
        fontsize=8.1, fontweight="bold", color=C_SIMP_CONT,
        bbox=dict(boxstyle="round,pad=0.25", fc="#ecfdf5", ec=C_SIMP_CONT, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_SIMP_CONT, lw=1.1),
    )
    ax.annotate(
        f"Melhor Phys-UViT (Ep {ep5[i5_best]}): {_fmt_pt(s5_va_core[i5_best])}\nvs Simple melhor (Ep {ep3[i3_best]}): {_fmt_pt(s3_va_core[i3_best])}",
        xy=(ep5[i5_best], s5_va_core[i5_best]),
        xytext=(14.0, 0.56),
        fontsize=8.2, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.26", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.2),
    )

    # B. Policy Loss (KL)
    ax = axes[0, 1]
    _plot_pair(ax, s5_tr_pol, s5_va_pol, s3_tr_pol, s3_va_pol,
               "B. Policy Loss (KL) — Overfitting Seletivo pós-Ep 20", "Policy Loss (KL)")
    ax.axvspan(20, 30, color="#fee2e2", alpha=0.35, zorder=0)
    ip5 = int(np.argmin(s5_va_pol))
    ip3 = int(np.argmin(s3_va_pol))
    ax.scatter([ep5[ip5], ep3[ip3]], [s5_va_pol[ip5], s3_va_pol[ip3]],
               color=[C_ALERT, C_SIMPLE_VAL], s=[55, 45], zorder=6)
    ax.annotate(
        f"Mín Phys-UViT (Ep {ep5[ip5]}): {_fmt_pt(s5_va_pol[ip5])}\nvs Simple (Ep {ep3[ip3]}): {_fmt_pt(s3_va_pol[ip3])}",
        xy=(ep5[ip5], s5_va_pol[ip5]),
        xytext=(6.2, 0.52),
        fontsize=8.2, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.26", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.2),
    )
    ax.text(25.0, 0.44, f"Pós-Ep 20: Train {_fmt_pt(s5_tr_pol[19])}→{_fmt_pt(s5_tr_pol[-1])}\nVal {_fmt_pt(s5_va_pol[19])}→{_fmt_pt(s5_va_pol[-1])}\n(saturação offline)",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.22", fc="#fff1f2", ec="#fca5a5", lw=0.8))

    # C. Value Loss (Huber, Log)
    ax = axes[1, 0]
    _plot_pair(ax, s5_tr_val, s5_va_val, s3_tr_val, s3_va_val,
               "C. Value Loss (Huber, Escala Log) — Zero Overfitting", "Value Loss (log)", log_scale=True)
    iv5 = int(np.argmin(s5_va_val))
    iv3 = int(np.argmin(s3_va_val))
    ax.scatter([ep5[iv5], ep3[iv3]], [s5_va_val[iv5], s3_va_val[iv3]],
               color=[C_ALERT, C_SIMPLE_VAL], s=[55, 45], zorder=6)
    red_val = (1.0 - s5_va_val[iv5] / s3_va_val[iv3]) * 100.0
    ax.annotate(
        f"Mín Phys-UViT (Ep {ep5[iv5]}): {_fmt_pt(s5_va_val[iv5], '.5f')}\nvs Simple mín (Ep {ep3[iv3]}): {_fmt_pt(s3_va_val[iv3], '.5f')}\n(-{_fmt_pt(red_val, '.1f')}% loss)",
        xy=(ep5[iv5], s5_va_val[iv5]),
        xytext=(12.2, 0.011),
        fontsize=8.1, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.26", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.2),
    )

    # D. Blueprint Loss
    ax = axes[1, 1]
    _plot_pair(ax, s5_tr_bp, s5_va_bp, s3_tr_bp, s3_va_bp,
               "D. Blueprint Loss (BCE + Dice) — Ganho Contínuo", "Blueprint Loss")
    ib5 = int(np.argmin(s5_va_bp))
    ib3 = int(np.argmin(s3_va_bp))
    ax.scatter([ep5[ib5], ep3[ib3]], [s5_va_bp[ib5], s3_va_bp[ib3]],
               color=[C_ALERT, C_SIMPLE_VAL], s=[55, 45], zorder=6)
    red_bp = (1.0 - s5_va_bp[ib5] / s3_va_bp[ib3]) * 100.0
    ax.annotate(
        f"Mín Phys-UViT (Ep {ep5[ib5]}): {_fmt_pt(s5_va_bp[ib5])}\nvs Simple mín (Ep {ep3[ib3]}): {_fmt_pt(s3_va_bp[ib3])}\n(-{_fmt_pt(red_bp, '.1f')}%)",
        xy=(ep5[ib5], s5_va_bp[ib5]),
        xytext=(12.5, 0.175),
        fontsize=8.2, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.26", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.2),
    )

    axes[1, 0].set_xlabel("Época")
    axes[1, 1].set_xlabel("Época")
    fig.suptitle("Stage 5 (Phys-UViT 3D, 4,91M) vs Stage 3 (SimpleBackbone, 4,43M): Evolução das Losses (Kaggle 2 × T4)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "stage5_losses_comparison.png")


def plot_stage5_validation_heads(
    s5_hist: List[Dict[str, Any]],
    s3_hist: List[Dict[str, Any]],
) -> None:
    ep5 = np.array([h["epoch"] for h in s5_hist])
    ep3 = np.array([h["epoch"] for h in s3_hist])

    mae5 = np.array([h["val"]["value_mae"] for h in s5_hist])
    mae3 = np.array([h["val"]["value_mae"] for h in s3_hist])

    top5 = np.array([h["val"]["policy_top1_acc"] * 100.0 for h in s5_hist])
    top3 = np.array([h["val"]["policy_top1_acc"] * 100.0 for h in s3_hist])
    top5_ma = np.convolve(np.pad(top5, (2, 0), mode="edge"), np.ones(3) / 3.0, mode="valid")
    top3_ma = np.convolve(np.pad(top3, (2, 0), mode="edge"), np.ones(3) / 3.0, mode="valid")

    dice5 = np.array([h["val"]["blueprint_dice"] * 100.0 for h in s5_hist])
    dice3 = np.array([h["val"]["blueprint_dice"] * 100.0 for h in s3_hist])

    logc5 = np.array([h["val"]["log_c_mae"] for h in s5_hist])
    conn5 = np.array([h["val"]["conn_acc"] * 100.0 for h in s5_hist])

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.8))

    # A. Value MAE
    ax = axes[0, 0]
    ax.plot(ep3, mae3, color=C_SIMPLE_VAL, lw=1.9, marker="o", ms=3.0, label="SimpleBackbone Val MAE")
    ax.plot(ep5, mae5, color=C_PHYS_VAL, lw=2.3, marker="s", ms=3.2, label="Phys-UViT 3D Val MAE")
    im5 = int(np.argmin(mae5))
    im3 = int(np.argmin(mae3))
    ax.scatter([ep5[im5], ep3[im3]], [mae5[im5], mae3[im3]], color=[C_ALERT, C_SIMPLE_VAL], s=[58, 45], zorder=6)
    red_mae = (1.0 - mae5[im5] / mae3[im3]) * 100.0
    ax.annotate(
        f"Phys-UViT mín Ep {ep5[im5]}: {_fmt_pt(mae5[im5])}\n(Ep 19 best: {_fmt_pt(mae5[18])})\nvs Simple mín: {_fmt_pt(mae3[im3])} (-{_fmt_pt(red_mae, '.1f')}%)",
        xy=(ep5[im5], mae5[im5]),
        xytext=(10.5, 0.036),
        fontsize=8.2, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.26", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.2),
    )
    ax.set_title("A. Value Head — MAE na Validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Value MAE (|V_pred − V_target|)")
    ax.set_ylim(0.008, 0.078)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    # B. Policy Top-1 Accuracy
    ax = axes[0, 1]
    ax.plot(ep3, top3, color="#cbd5e1", lw=1.0, marker="o", ms=2.5, label="Simple Top-1 (bruto)")
    ax.plot(ep3, top3_ma, color=C_SIMPLE_VAL, lw=2.0, ls="--", label="Simple Média Móvel (3 ép.)")
    ax.plot(ep5, top5, color="#93c5fd", lw=1.2, marker="s", ms=2.8, label="Phys-UViT Top-1 (bruto)")
    ax.plot(ep5, top5_ma, color=C_PHYS_VAL, lw=2.4, label="Phys-UViT Média Móvel (3 ép.)")
    ipk5 = int(np.argmax(top5))
    ipk3 = int(np.argmax(top3))
    ax.scatter([ep5[ipk5], ep3[ipk3]], [top5[ipk5], top3[ipk3]], color=[C_ALERT, C_SIMPLE_VAL], s=[58, 45], zorder=6)
    ax.annotate(
        f"Pico Phys-UViT Ep {ep5[ipk5]}: {_fmt_pt(top5[ipk5], '.2f')}%\nEp 19: {_fmt_pt(top5[18], '.2f')}% | Ep 30: {_fmt_pt(top5[-1], '.2f')}%\nvs Simple Ep 30: {_fmt_pt(top3[-1], '.2f')}%",
        xy=(ep5[ipk5], top5[ipk5]),
        xytext=(12.0, 6.2),
        fontsize=8.1, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.26", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.2),
    )
    ax.set_title("B. Policy Head — Acurácia Top-1 (%)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Policy Top-1 Accuracy (%)")
    ax.set_ylim(1.5, 19.0)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(loc="lower right", ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=7.4)

    # C. Blueprint Dice
    ax = axes[1, 0]
    ax.axhline(90.0, color=C_SIMP_CONT, ls=":", lw=1.4, label="Patamar 90,0%")
    ax.plot(ep3, dice3, color=C_SIMPLE_VAL, lw=1.9, marker="o", ms=3.0, label="SimpleBackbone Dice")
    ax.plot(ep5, dice5, color=C_PHYS_TR, lw=2.3, marker="s", ms=3.2, label="Phys-UViT 3D Dice")
    c90_5 = int(np.where(dice5 >= 90.0)[0][0])
    c90_3 = int(np.where(dice3 >= 90.0)[0][0])
    id5 = int(np.argmax(dice5))
    ax.scatter([ep5[c90_5], ep3[c90_3], ep5[id5]], [dice5[c90_5], dice3[c90_3], dice5[id5]],
               color=[C_SIMP_CONT, C_SIMPLE_VAL, C_ALERT], s=[55, 45, 58], zorder=6)
    ax.annotate(
        f"Cruza 90% na Ep {ep5[c90_5]} ({_fmt_pt(dice5[c90_5], '.2f')}%)\n14 épocas antes do Simple (Ep {ep3[c90_3]})!",
        xy=(ep5[c90_5], dice5[c90_5]),
        xytext=(5.5, 79.0),
        fontsize=8.1, fontweight="bold", color=C_SIMP_CONT,
        bbox=dict(boxstyle="round,pad=0.24", fc="#ecfdf5", ec=C_SIMP_CONT, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_SIMP_CONT, lw=1.1),
    )
    ax.annotate(
        f"Pico Ep {ep5[id5]}: {_fmt_pt(dice5[id5], '.2f')}%\n(Ep 19: {_fmt_pt(dice5[18], '.2f')}%) vs Simple {_fmt_pt(float(np.max(dice3)), '.2f')}%",
        xy=(ep5[id5], dice5[id5]),
        xytext=(13.5, 84.5),
        fontsize=8.1, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.1),
    )
    ax.set_title("C. Blueprint Head — Dice Score (%)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Blueprint Dice (%)")
    ax.set_ylim(70.0, 92.4)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    # D. Factored Physics Heads (Phys-UViT 3D Exclusive)
    ax = axes[1, 1]
    l1 = ax.plot(ep5, logc5, color=C_PURPLE, lw=2.3, marker="s", ms=3.4, label="log10(Compliance) MAE (esq.)")
    ilc = int(np.argmin(logc5))
    ax.scatter([ep5[ilc]], [logc5[ilc]], color=C_ALERT, s=55, zorder=6)
    ax.set_ylim(0.020, 0.130)
    ax.set_ylabel("log10(Compliance) MAE", color=C_PURPLE)
    ax.tick_params(axis="y", labelcolor=C_PURPLE)
    ax.set_xlabel("Época")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    l2 = ax2.plot(ep5, conn5, color=C_SIMP_CONT, lw=2.1, ls="--", marker="^", ms=3.3, label="Connectivity Acc % (dir.)")
    ax2.set_ylim(99.55, 100.06)
    ax2.set_ylabel("Topological Connectivity Acc (%)", color=C_SIMP_CONT)
    ax2.tick_params(axis="y", labelcolor=C_SIMP_CONT)

    mult_err = (10.0 ** logc5[ilc] - 1.0) * 100.0
    ax.annotate(
        f"Mín logC MAE Ep {ep5[ilc]}: {_fmt_pt(logc5[ilc])}\n(±{_fmt_pt(mult_err, '.1f')}% fator de C!) | Ep 30: {_fmt_pt(logc5[-1])}",
        xy=(ep5[ilc], logc5[ilc]),
        xytext=(5.2, 0.088),
        fontsize=8.1, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f5f3ff", ec=C_PURPLE, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.1),
    )
    ax2.text(
        0.97, 0.56,
        f"Conn Acc >= {_fmt_pt(min(conn5[3:]), '.2f')}% (Ep 4+)\n100,000% exatos nas Ep 25–30",
        transform=ax2.transAxes, ha="right", va="center", fontsize=8.0, fontweight="bold", color=C_SIMP_CONT,
        bbox=dict(boxstyle="round,pad=0.24", fc="#ecfdf5", ec=C_SIMP_CONT, lw=0.8),
    )
    lines = l1 + l2
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right", bbox_to_anchor=(0.98, 0.36),
              frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=7.9)
    ax.set_title("D. Cabeças Fatoradas de Física (Exclusivo Phys-UViT 3D)")

    fig.suptitle("Métricas de Validação por Cabeça: Phys-UViT 3D vs SimpleBackbone (30 Épocas)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "stage5_validation_heads_comparison.png")


def plot_stage5_eval_compliance_score(
    ep2: Dict[str, Any],
    ep30: Dict[str, Any],
    s5: Dict[str, Any],
) -> None:
    short_names = [
        "S1\nClamp\n(x=63)",
        "S2\nTip Bot.\n(x=63,y=6)",
        "S3\nMidspan\n(x=44)",
        "S4\nRail XY\n(x=58)",
        "S5\nRail 3D\n(x=52)",
        "S6\nShort\n(x=36)",
    ]
    e2_list = ep2["episodes"]
    e30_list = ep30["episodes"]
    s5_list = s5["episodes"]
    smap = s5.get("simp_baselines") or ep30["simp_baselines"]

    c_ep2 = np.array([e["final_compliance"] if e["final_compliance"] is not None else np.nan for e in e2_list])
    c_ep30 = np.array([e["final_compliance"] if e["final_compliance"] is not None else np.nan for e in e30_list])
    c_s5 = np.array([e["final_compliance"] if e["final_compliance"] is not None else np.nan for e in s5_list])
    c_sbin = np.array([smap[e["scenario_id"]]["comp_bin"] for e in s5_list])
    c_scont = np.array([smap[e["scenario_id"]]["comp_cont"] for e in s5_list])

    sc_ep2 = np.array([e["final_score"] for e in e2_list])
    sc_ep30 = np.array([e["final_score"] for e in e30_list])
    sc_s5 = np.array([e["final_score"] for e in s5_list])
    sc_scont = np.array([smap[e["scenario_id"]]["reward_cont"] for e in s5_list])
    sc_sbin_s6 = smap["S6_short_deep_clamp"]["reward_bin"]

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.6))

    # A: Final Compliance (Log)
    ax = axes[0]
    x = np.arange(len(short_names))
    w = 0.165
    ax.bar(x - 2 * w, c_ep2, width=w, color=C_EP2, edgecolor="#64748b", lw=0.6, label="Simple ep2")
    ax.bar(x - 1 * w, c_ep30, width=w, color=C_SIMPLE_VAL, edgecolor="#334155", lw=0.6, label="Simple ep30")
    ax.bar(x + 0 * w, c_s5, width=w, color=C_PHYS_VAL, edgecolor="#1d4ed8", lw=0.7, label="Phys-UViT ep19")
    ax.bar(x + 1 * w, c_sbin, width=w, color=C_SIMP_BIN, edgecolor="#d97706", lw=0.6, label="SIMP Bin (ρ>0,5)")
    ax.bar(x + 2 * w, c_scont, width=w, color=C_SIMP_CONT, edgecolor="#047857", lw=0.6, label="SIMP Cont (β=4,0)")

    for i in range(6):
        if np.isfinite(c_s5[i]) and np.isfinite(c_ep30[i]) and c_s5[i] > 0:
            delta_pct = (c_s5[i] - c_ep30[i]) / c_ep30[i] * 100.0
            col = C_SIMP_CONT if delta_pct <= 0 else C_ALERT
            top_y = max(c_s5[i], c_ep30[i]) * 1.24
            ax.text(x[i], top_y, f"{delta_pct:+.0f}%".replace(".", ","), ha="center", va="bottom",
                    fontsize=8.0, fontweight="bold", color=col)
        elif not np.isfinite(c_s5[i]):
            ax.text(x[i], 120.0, "N/C", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C_ALERT)

    ax.set_yscale("log")
    ax.set_ylim(50, 1.4e6)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Compliance Final (menor = mais rígido, log)")
    ax.set_title("A. Compliance Final: Phys-UViT ep19 vs Simple ep30 vs SIMP (Δ% vs ep30)")
    ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=7.8)

    # B: Score v3.3
    ax = axes[1]
    labels_b = short_names + ["Média\n(6 cen.)"]
    xb = np.arange(len(labels_b))
    wb = 0.25
    y2 = np.append(sc_ep2, float(np.mean(sc_ep2)))
    y30 = np.append(sc_ep30, float(np.mean(sc_ep30)))
    y5 = np.append(sc_s5, float(np.mean(sc_s5)))

    ax.axhline(0.0, color="#64748b", lw=1.0, zorder=1)
    ax.bar(xb - wb, y2, width=wb, color=C_EP2, edgecolor="#64748b", lw=0.6, label="Simple ep2")
    ax.bar(xb, y30, width=wb, color=C_SIMPLE_VAL, edgecolor="#334155", lw=0.6, label="Simple ep30")
    ax.bar(xb + wb, y5, width=wb, color=C_PHYS_VAL, edgecolor="#1d4ed8", lw=0.7, label="Phys-UViT ep19")
    ax.scatter(np.arange(6), sc_scont, marker="D", s=42, color=C_SIMP_CONT, zorder=5, label="SIMP Cont Reward")
    if sc_sbin_s6 is not None:
        ax.scatter([5 - wb], [sc_sbin_s6], marker="*", s=105, color=C_SIMP_BIN, edgecolor="#92400e", zorder=6,
                   label=f"SIMP Bin S6 ({_fmt_pt(sc_sbin_s6, '+.3f')})")

    for i in range(len(xb)):
        v30, v5 = y30[i], y5[i]
        off5 = 0.015 if v5 >= 0 else -0.018
        ax.text(xb[i] + wb, v5 + off5, _fmt_pt(v5, "+.2f"),
                ha="center", va="bottom" if v5 >= 0 else "top", fontsize=7.6, fontweight="bold",
                color=C_PHYS_VAL if v5 >= v30 else C_ALERT)
        off30 = 0.015 if v30 >= 0 else -0.018
        ax.text(xb[i], v30 + off30, _fmt_pt(v30, "+.2f"),
                ha="center", va="bottom" if v30 >= 0 else "top", fontsize=7.3, color="#475569")

    pos_30 = sum(1 for v in sc_ep30 if v > 0)
    pos_5 = sum(1 for v in sc_s5 if v > 0)
    conv_30 = sum(1 for e in e30_list if e["end_reason"] == "CONVERGENCE")
    conv_5 = sum(1 for e in s5_list if e["end_reason"] == "CONVERGENCE")
    c_mean_30 = float(np.nanmean(c_ep30))
    c_mean_5 = float(np.nanmean(c_s5))
    ax.text(
        0.025, 0.96,
        f"Score médio: ep30={_fmt_pt(float(np.mean(sc_ep30)), '+.3f')} → Phys-UViT={_fmt_pt(float(np.mean(sc_s5)), '+.3f')}\n"
        f"Score > 0: {pos_30}/6 → {pos_5}/6 | CONVERGENCE: {conv_30}/6 → {conv_5}/6\n"
        f"Vol médio: 9,53% → 7,67% | C méd: 15.179 → 23.646",
        transform=ax.transAxes, va="top", fontsize=7.8, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.26", fc="#f8fafc", ec="#cbd5e1"),
    )
    ax.set_ylim(-0.32, 0.80)
    ax.set_xticks(xb)
    ax.set_xticklabels(labels_b)
    ax.set_ylabel("Score / Recompensa v3.3")
    ax.set_title("B. Score Final v3.3 e Volume de Convergência")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=7.6)

    fig.suptitle("Stage 5C — Inferência Real + MCTS (160 sims) + FEniCSx: Phys-UViT 3D (ep19) vs SimpleBackbone (ep30)", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "stage5_eval_compliance_score.png")


def plot_stage5_eval_geometry_calibration(
    ep30: Dict[str, Any],
    s5: Dict[str, Any],
) -> None:
    short_6 = ["S1", "S2", "S3", "S4", "S5", "S6"]
    e30_list = ep30["episodes"]
    s5_list = s5["episodes"]
    smap = s5.get("simp_baselines") or ep30["simp_baselines"]

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.2))
    x = np.arange(6)

    # A: inertia_y_norm + hollow_ratio_xy annotation
    ax = axes[0, 0]
    w = 0.26
    iy_30 = np.array([e["geometry"]["inertia_y_norm"] for e in e30_list])
    iy_5 = np.array([e["geometry"]["inertia_y_norm"] for e in s5_list])
    iy_simp = np.array([smap[e["scenario_id"]]["geometry"]["inertia_y_norm"] for e in s5_list])
    hol_30 = [e["geometry"]["hollow_ratio_xy"] for e in e30_list]
    hol_5 = [e["geometry"]["hollow_ratio_xy"] for e in s5_list]

    ax.bar(x - w, iy_30, width=w, color=C_SIMPLE_VAL, edgecolor="#334155", lw=0.6, label="Simple ep30")
    ax.bar(x, iy_5, width=w, color=C_PHYS_VAL, edgecolor="#1d4ed8", lw=0.6, label="Phys-UViT ep19")
    ax.bar(x + w, iy_simp, width=w, color=C_SIMP_CONT, edgecolor="#047857", lw=0.6, label="SIMP (β=4,0)")
    for i in range(6):
        if hol_5[i] >= 0.01 or hol_30[i] >= 0.01:
            ax.text(x[i] - w, iy_30[i] + 0.012, f"h={_fmt_pt(hol_30[i], '.2f')}", ha="center", va="bottom",
                    rotation=75, fontsize=6.9, color="#475569")
            ax.text(x[i] + 0.03, iy_5[i] + 0.012, f"h={_fmt_pt(hol_5[i], '.2f')}", ha="center", va="bottom",
                    rotation=75, fontsize=7.1, fontweight="bold", color=C_PHYS_VAL)
    ax.text(
        0.025, 0.96,
        "Em S2, Simple ep30 teve 2 ilhas\n(I_y=0,112 inflado; ilha pr.=0,058)\nPhys-UViT fechou 1 ilha limpa (0,082)",
        transform=ax.transAxes, va="top", fontsize=7.5, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.22", fc="#eff6ff", ec=C_PHYS_VAL, lw=0.8),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_6)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("Inércia à Flexão Normalizada (I_y)")
    ax.set_title("A. Uso de Altura Estrutural (I_y) e Vazamento hollow_xy")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=7.8)

    # B: z_centroid (upper band) + overshoot_frac (lower band)
    ax = axes[0, 1]
    ov_30 = np.array([e["geometry"]["overshoot_frac"] * 100.0 for e in e30_list])
    ov_5 = np.array([e["geometry"]["overshoot_frac"] * 100.0 for e in s5_list])
    zc_30 = np.array([e["geometry"]["z_centroid"] if e["geometry"]["z_centroid"] is not None else np.nan for e in e30_list])
    zc_5 = np.array([e["geometry"]["z_centroid"] if e["geometry"]["z_centroid"] is not None else np.nan for e in s5_list])
    zc_simp = np.array([smap[e["scenario_id"]]["geometry"]["z_centroid"] for e in s5_list])

    wb = 0.28
    b1 = ax.bar(x - wb / 2, ov_30, width=wb, color=C_SIMPLE_VAL, alpha=0.82, label="Overshoot X ep30 (% esq.)")
    b2 = ax.bar(x + wb / 2, ov_5, width=wb, color=C_PHYS_VAL, alpha=0.85, label="Overshoot X Phys-UViT (% esq.)")
    for i in range(6):
        if ov_30[i] > 0.1 or ov_5[i] > 0.1:
            ax.text(x[i] - wb / 2, ov_30[i] + 0.35, f"{_fmt_pt(ov_30[i], '.1f')}%", ha="center", va="bottom", fontsize=7.0, color="#475569")
            ax.text(x[i] + wb / 2, ov_5[i] + 0.35, f"{_fmt_pt(ov_5[i], '.1f')}%", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color=C_PHYS_VAL)
        elif i == 2:
            pass
    ax.text(
        0.025, 0.43,
        "Phys-UViT 3D:\n0,0% overshoot\nem 6/6 cenários!",
        transform=ax.transAxes, va="top", fontsize=7.5, fontweight="bold", color=C_PHYS_VAL,
        bbox=dict(boxstyle="round,pad=0.22", fc="#eff6ff", ec=C_PHYS_VAL, lw=0.8),
    )
    ax.set_ylim(0.0, 29.0)
    ax.set_ylabel("Overshoot Pós-Carga em X (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_6)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    l1 = ax2.plot(x, zc_30, color="#475569", marker="o", ms=4.8, lw=1.4, ls=":", label="z_centroid ep30 (dir.)")
    l2 = ax2.plot(x, zc_5, color=C_PURPLE, marker="s", ms=5.3, lw=2.0, ls="-", label="z_centroid Phys-UViT (dir.)")
    l3 = ax2.plot(x, zc_simp, color=C_SIMP_CONT, marker="D", ms=4.8, lw=1.5, ls="--", label="z_centroid SIMP (dir.)")
    ax2.set_ylim(-2.6, 8.2)
    ax2.set_ylabel("Centroide em Z (voxels, Z∈[0..7])", color=C_PURPLE)
    ax2.tick_params(axis="y", labelcolor=C_PURPLE)
    ax2.annotate(
        f"S5 (Z=[1,3]): SIMP={_fmt_pt(zc_simp[4], '.2f')}\nvs ep30={_fmt_pt(zc_30[4], '.2f')}, Phys={_fmt_pt(zc_5[4], '.2f')}",
        xy=(4, zc_simp[4]), xytext=(1.9, 1.4),
        fontsize=7.4, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.22", fc="#f5f3ff", ec=C_PURPLE, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=C_SIMP_CONT, lw=1.1),
    )
    handles = [b1, b2] + l1 + l2 + l3
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left", ncol=2,
              frameon=True, facecolor="#ffffff", framealpha=0.94, edgecolor="#e2e8f0", fontsize=7.0)
    ax.set_title("B. Overshoot Pós-Carga X (S3–S6) e Centroide em Z")

    # C: Value Head OOD Calibration (Bias & MAE in Phase 2)
    ax = axes[1, 0]
    bias_30 = np.array([e["value_calibration_p2"]["bias_net_minus_fem"] if e["value_calibration_p2"]["bias_net_minus_fem"] is not None else np.nan for e in e30_list])
    bias_5 = np.array([e["value_calibration_p2"]["bias_net_minus_fem"] if e["value_calibration_p2"]["bias_net_minus_fem"] is not None else np.nan for e in s5_list])
    mae_30 = np.array([e["value_calibration_p2"]["mae"] if e["value_calibration_p2"]["mae"] is not None else np.nan for e in e30_list])
    mae_5 = np.array([e["value_calibration_p2"]["mae"] if e["value_calibration_p2"]["mae"] is not None else np.nan for e in s5_list])

    lbl_c = short_6 + ["Média\nP2"]
    xc = np.arange(len(lbl_c))
    wc = 0.34
    y30_c = np.append(mae_30, float(np.nanmean(mae_30)))
    y5_c = np.append(mae_5, float(np.nanmean(mae_5)))
    b30_c = np.append(bias_30, float(np.nanmean(bias_30)))
    b5_c = np.append(bias_5, float(np.nanmean(bias_5)))

    ax.axhline(0.0, color="#64748b", lw=0.9)
    ax.bar(xc - wc / 2, y30_c, width=wc, color=C_SIMPLE_VAL, edgecolor="#334155", lw=0.6, label="MAE P2 Simple ep30")
    ax.bar(xc + wc / 2, y5_c, width=wc, color=C_PHYS_VAL, edgecolor="#1d4ed8", lw=0.6, label="MAE P2 Phys-UViT ep19")
    ax.scatter(xc - wc / 2, b30_c, marker="o", s=36, color=C_SIMP_BIN, zorder=5, label="Viés (V−R) ep30")
    ax.scatter(xc + wc / 2, b5_c, marker="^", s=40, color=C_ALERT, zorder=5, label="Viés (V−R) Phys-UViT")
    for i in range(len(xc)):
        if np.isfinite(y30_c[i]):
            ax.text(xc[i] - wc / 2 - 0.03, y30_c[i] + 0.016, _fmt_pt(y30_c[i], ".2f"), ha="center", va="bottom", fontsize=7.0, color="#475569")
        if np.isfinite(y5_c[i]):
            ax.text(xc[i] + wc / 2 + 0.03, y5_c[i] + 0.016, _fmt_pt(y5_c[i], ".2f"), ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=C_PHYS_VAL)
    ax.set_xticks(xc)
    ax.set_xticklabels(lbl_c)
    ax.set_ylim(-0.08, 0.80)
    ax.set_ylabel("MAE / Viés (V_net − R_FEM) na Fase 2")
    ax.set_title("C. Calibração OOD da Value Head na Fase 2 (Rollouts Reais)")
    ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=7.3)

    # D: Phys-UViT 3D Factored log10(C) OOD Calibration
    ax = axes[1, 1]
    aux_list = [e.get("physics_aux_diagnostics") or {} for e in s5_list]
    lc_pred = np.array([a.get("final_log_c_pred") if a.get("final_log_c_pred") is not None else np.nan for a in aux_list])
    lc_true = np.array([a.get("final_log_c_true") if a.get("final_log_c_true") is not None else np.nan for a in aux_list])
    lc_mae = np.array([a.get("log_c_mae") if a.get("log_c_mae") is not None else np.nan for a in aux_list])
    lc_r = [a.get("log_c_pearson_r") for a in aux_list]

    wd = 0.34
    ax.bar(x - wd / 2, lc_true, width=wd, color=C_SIMP_CONT, edgecolor="#047857", lw=0.6, label="log10(C_FEM) Real (FEniCSx)")
    ax.bar(x + wd / 2, lc_pred, width=wd, color=C_PURPLE, edgecolor="#5b21b6", lw=0.6, label="log10(C_pred) Phys-UViT")
    for i in range(6):
        if np.isfinite(lc_true[i]) and np.isfinite(lc_pred[i]):
            top = max(lc_true[i], lc_pred[i])
            r_str = f"r={_fmt_pt(lc_r[i], '.2f')}" if lc_r[i] is not None else "r=---"
            mae_str = f"MAE={_fmt_pt(lc_mae[i], '.2f')}" if np.isfinite(lc_mae[i]) else ""
            ax.text(x[i], top + 0.12, f"{mae_str}\n{r_str}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=C_DARK)

    mean_lc_mae = float(np.nanmean(lc_mae)) if np.any(np.isfinite(lc_mae)) else float("nan")
    ax.text(
        0.025, 0.96,
        f"log10(C) OOD MAE médio: {_fmt_pt(mean_lc_mae, '.3f')}\n(Subestima C OOD → explica viés V_net)\nConn Acc OOD: P1=0,01% | P2=100%",
        transform=ax.transAxes, va="top", fontsize=7.4, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="#f5f3ff", ec=C_PURPLE, lw=0.8),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_6)
    ax.set_ylim(0.0, 6.4)
    ax.set_ylabel("log10(Compliance)")
    ax.set_title("D. Cabeça Física Fatorada: Predição de log10(C) nos Finais P2")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=7.5)

    fig.suptitle("Stage 5C — Geometria 3D, Shift Covariável da Value Head e Calibração Fatorada de log10(C)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, "stage5_eval_geometry_and_calibration.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 5 comparison charts.")
    parser.add_argument("--mode", choices=("training", "eval", "all"), default="all")
    args = parser.parse_args()

    _apply_clean_style()

    if args.mode in ("training", "all"):
        s5_hist = json.loads((ANALYSIS_DIR / "stage5_phys_uvit_metrics_history.json").read_text(encoding="utf-8"))
        s3_hist = json.loads((ANALYSIS_DIR / "stage3_simple_metrics_history.json").read_text(encoding="utf-8"))
        plot_stage5_losses_comparison(s5_hist, s3_hist)
        plot_stage5_validation_heads(s5_hist, s3_hist)

    if args.mode in ("eval", "all"):
        ep2 = json.loads((ANALYSIS_DIR / "eval_ep2_results.json").read_text(encoding="utf-8"))
        ep30 = json.loads((ANALYSIS_DIR / "eval_ep30_results.json").read_text(encoding="utf-8"))
        s5 = json.loads((ANALYSIS_DIR / "eval_stage5_phys_uvit_results.json").read_text(encoding="utf-8"))
        plot_stage5_eval_compliance_score(ep2, ep30, s5)
        plot_stage5_eval_geometry_calibration(ep30, s5)


if __name__ == "__main__":
    main()
