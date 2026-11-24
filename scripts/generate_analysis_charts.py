#!/usr/bin/env python3
"""Generate publication-grade analysis charts for AlphaBuilder v3.3 reports.

Reads verified persisted data from:
- data/analysis/stage3_simple_metrics_history.json
- data/analysis/eval_ep2_results.json
- data/analysis/eval_ep30_results.json
- data/eval_inference_v3_3.db (read-only)
- checkpoints/simple_v3_3_stage3_best.pt (read-only weight & activation audit)

Outputs 8 high-DPI white-background PNG figures under data/analysis/figures/.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphabuilder.src.logic.mcts.engine import build_state_tensor
from alphabuilder.src.logic.storage import deserialize_array
from alphabuilder.src.neural.model import AlphaBuilderV31
from scripts.evaluate_inference_scenarios import RESOLUTION, SCENARIOS, _build_masks_and_forces

FIG_DIR = REPO_ROOT / "data" / "analysis" / "figures"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
DB_PATH = REPO_ROOT / "data" / "eval_inference_v3_3.db"
CKPT_EP30_PATH = REPO_ROOT / "checkpoints" / "simple_v3_3_stage3_best.pt"

# Colorblind-friendly palette
C_TRAIN = "#0f766e"      # Teal
C_VAL = "#2563eb"        # Royal Blue
C_EP2 = "#94a3b8"        # Slate Gray
C_EP30 = "#2563eb"       # Royal Blue
C_SIMP_BIN = "#f59e0b"   # Amber
C_SIMP_CONT = "#059669"  # Emerald
C_ALERT = "#dc2626"      # Coral Red
C_DARK = "#1e293b"       # Slate 800
C_GRID = "#e2e8f0"       # Slate 200


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
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
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


def plot_stage3_losses(hist: List[Dict[str, Any]]) -> None:
    epochs = np.array([h["epoch"] for h in hist])
    tr_loss = np.array([h["train"]["loss"] for h in hist])
    va_loss = np.array([h["val"]["loss"] for h in hist])
    tr_pol = np.array([h["train"]["policy_loss"] for h in hist])
    va_pol = np.array([h["val"]["policy_loss"] for h in hist])
    tr_val = np.array([h["train"]["value_loss"] for h in hist])
    va_val = np.array([h["val"]["value_loss"] for h in hist])
    tr_bp = np.array([h["train"]["blueprint_loss"] for h in hist])
    va_bp = np.array([h["val"]["blueprint_loss"] for h in hist])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    panels = [
        (axes[0, 0], tr_loss, va_loss, "A. Total Loss (Combo Ponderada)", "Loss", False,
         f"Melhor Val: Ep 30 = {va_loss[-1]:.4f}\n(gap gen. = +{va_loss[-1]-tr_loss[-1]:.4f})", (30, va_loss[-1]), (18.5, 0.78)),
        (axes[0, 1], tr_pol, va_pol, "B. Policy Loss (Cross-Entropy)", "Policy Loss", False,
         f"Val -60,3%\n({va_pol[0]:.4f} → {va_pol[-1]:.4f})", (30, va_pol[-1]), (18.5, 0.56)),
        (axes[1, 0], tr_val, va_val, "C. Value Loss (MSE, Escala Log)", "Value Loss (log)", True,
         f"Val ~21× menor\n({va_val[0]:.4f} → {va_val[-1]:.4f})", (30, va_val[-1]), (17.5, 0.012)),
        (axes[1, 1], tr_bp, va_bp, "D. Blueprint Loss (BCE + Dice)", "Blueprint Loss", False,
         f"Val -67,8%\n({va_bp[0]:.4f} → {va_bp[-1]:.4f})", (30, va_bp[-1]), (18.5, 0.23)),
    ]

    for ax, tr, va, title, ylabel, use_log, ann_text, xy_pt, xy_txt in panels:
        ax.axvspan(1, 3, color="#fef3c7", alpha=0.45, zorder=0, label="Warmup (Ep 1–3)")
        ax.plot(epochs, tr, color=C_TRAIN, linewidth=2.1, marker="o", markersize=3.2, label="Train")
        ax.plot(epochs, va, color=C_VAL, linewidth=2.3, marker="s", markersize=3.4, label="Validation")
        ax.scatter([30], [va[-1]], color=C_ALERT, s=48, zorder=5)
        ax.annotate(
            ann_text,
            xy=xy_pt,
            xytext=xy_txt,
            fontsize=8.8,
            fontweight="bold",
            color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
            arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.2, shrinkA=2, shrinkB=4),
        )
        if use_log:
            ax.set_yscale("log")
        ax.set_title(title, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.92, edgecolor="#e2e8f0")

    axes[1, 0].set_xlabel("Época")
    axes[1, 1].set_xlabel("Época")
    fig.suptitle("Stage 3 (SimpleBackbone, Kaggle 2 × Tesla T4): Evolução das Losses em 30 Épocas", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    _save_fig(fig, "stage3_losses_evolution.png")


def plot_stage3_val_metrics(hist: List[Dict[str, Any]]) -> None:
    epochs = np.array([h["epoch"] for h in hist])
    mae = np.array([h["val"]["value_mae"] for h in hist])
    top1 = np.array([h["val"]["policy_top1_acc"] * 100.0 for h in hist])
    dice = np.array([h["val"]["blueprint_dice"] * 100.0 for h in hist])
    # 3-epoch trailing moving average for noisy Top-1
    top1_ma = np.convolve(np.pad(top1, (2, 0), mode="edge"), np.ones(3) / 3.0, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    # A: Value MAE
    ax = axes[0]
    ax.plot(epochs, mae, color=C_VAL, linewidth=2.2, marker="o", markersize=3.5, label="Val Value MAE")
    ax.scatter([2, 30], [mae[1], mae[29]], color=[C_EP2, C_ALERT], s=[55, 60], zorder=5)
    ax.annotate(f"ep2: {mae[1]:.4f}", xy=(2, mae[1]), xytext=(5.5, mae[1] + 0.011),
                fontsize=8.8, fontweight="bold", color="#475569",
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.0))
    ax.annotate(f"ep30: {mae[29]:.4f}\n(-79,5% vs Ep 1)", xy=(30, mae[29]), xytext=(16.5, 0.032),
                fontsize=8.8, fontweight="bold", color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1"),
                arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.2))
    ax.set_title("A. Value Head — MAE na Validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("MAE (|V_pred − V_target|)")
    ax.set_ylim(0.008, 0.073)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])

    # B: Policy Top-1 Accuracy
    ax = axes[1]
    ax.plot(epochs, top1, color="#93c5fd", linewidth=1.3, marker="o", markersize=3.0, label="Top-1 por época")
    ax.plot(epochs, top1_ma, color=C_VAL, linewidth=2.4, label="Média móvel (3 ép.)")
    peak_idx = int(np.argmax(top1))
    ax.scatter([2, epochs[peak_idx], 30], [top1[1], top1[peak_idx], top1[29]],
               color=[C_EP2, C_SIMP_CONT, C_ALERT], s=[55, 65, 60], zorder=5)
    ax.annotate(f"ep2: {top1[1]:.2f}%", xy=(2, top1[1]), xytext=(5.0, 5.2),
                fontsize=8.6, fontweight="bold", color="#475569",
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.0))
    ax.annotate(f"Pico ep{epochs[peak_idx]}: {top1[peak_idx]:.2f}%", xy=(epochs[peak_idx], top1[peak_idx]),
                xytext=(10.5, 17.5), fontsize=8.8, fontweight="bold", color=C_SIMP_CONT,
                arrowprops=dict(arrowstyle="->", color=C_SIMP_CONT, lw=1.1))
    ax.annotate(f"ep30: {top1[29]:.2f}%", xy=(30, top1[29]), xytext=(22.2, 11.2),
                fontsize=8.8, fontweight="bold", color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1"),
                arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.1))
    ax.set_title("B. Policy Head — Acurácia Top-1 (%)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_ylim(1.0, 19.2)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    # C: Blueprint Dice
    ax = axes[2]
    ax.plot(epochs, dice, color=C_TRAIN, linewidth=2.3, marker="o", markersize=3.4, label="Val Blueprint Dice")
    ax.axhline(90.0, color=C_SIMP_CONT, linestyle="--", linewidth=1.2, alpha=0.85, label="Referência 90%")
    ax.scatter([2, 29, 30], [dice[1], dice[28], dice[29]],
               color=[C_EP2, C_SIMP_CONT, C_ALERT], s=[55, 58, 60], zorder=5)
    ax.annotate(f"ep1: {dice[0]:.2f}%", xy=(1, dice[0]), xytext=(3.8, 72.0),
                fontsize=8.5, color="#475569",
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.0))
    ax.annotate(f"ep2: {dice[1]:.2f}%", xy=(2, dice[1]), xytext=(5.8, 81.2),
                fontsize=8.6, fontweight="bold", color="#475569",
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.0))
    ax.annotate(f"Pico ep29: {dice[28]:.2f}%\nep30: {dice[29]:.2f}%", xy=(30, dice[29]), xytext=(16.5, 83.2),
                fontsize=8.6, fontweight="bold", color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1"),
                arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.2))
    ax.set_title("C. Blueprint Head — Dice Score (%)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Dice (%)")
    ax.set_ylim(69.5, 92.2)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    fig.suptitle("Stage 3: Evolução das Métricas de Validação (ep2 Smoke → ep30 Melhor Checkpoint)", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "stage3_validation_metrics.png")


def plot_stage3_dynamics(hist: List[Dict[str, Any]]) -> None:
    epochs = np.array([h["epoch"] for h in hist])
    lrs = np.array([h["lr"] * 1e4 for h in hist])  # x1e-4
    gr = np.array([h["train"]["bp_pol_grad_ratio"] for h in hist])
    sps = np.array([h["samples_per_sec"] for h in hist])
    ep_min = np.array([h["elapsed_sec"] / 60.0 for h in hist])
    total_h = sum(h["elapsed_sec"] for h in hist) / 3600.0

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    # A: LR Schedule
    ax = axes[0]
    ax.axvspan(1, 3, color="#fef3c7", alpha=0.55, label="Warmup (Ep 1–3)")
    ax.plot(epochs, lrs, color=C_VAL, linewidth=2.3, marker="o", markersize=3.2, label="Learning Rate")
    ax.scatter([3, 30], [lrs[2], lrs[29]], color=[C_SIMP_BIN, C_ALERT], s=55, zorder=5)
    ax.annotate("Pico: 3,00e-4\n(Ep 3)", xy=(3, lrs[2]), xytext=(8.8, 2.72),
                fontsize=8.6, fontweight="bold", color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_SIMP_BIN, lw=1.1))
    ax.annotate("Final: 3,00e-6\n(Cosine Decay)", xy=(30, lrs[29]), xytext=(18.5, 0.72),
                fontsize=8.6, fontweight="bold", color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1"),
                arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.1))
    ax.set_title("A. Agenda de Learning Rate")
    ax.set_xlabel("Época")
    ax.set_ylabel("Learning Rate (× 10⁻⁴)")
    ax.set_ylim(-0.1, 3.35)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    # B: Gradient ratio bp/pol
    ax = axes[1]
    ax.bar(epochs, gr, color="#99f6e4", edgecolor=C_TRAIN, linewidth=0.7, width=0.68, label="Ratio bp/pol")
    gr_mean = float(np.mean(gr))
    ax.axhline(gr_mean, color=C_DARK, linestyle="--", linewidth=1.4, label=f"Média = {gr_mean:.3f}")
    ax.axhline(0.40, color=C_SIMP_BIN, linestyle=":", linewidth=1.3, label="Teto saudável (0,40)")
    pk = int(np.argmax(gr))
    ax.annotate(f"Pico Ep {epochs[pk]}: {gr[pk]:.3f}", xy=(epochs[pk], gr[pk]), xytext=(3.8, 0.36),
                fontsize=8.5, fontweight="bold", color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.22", fc="#f8fafc", ec="#cbd5e1"),
                arrowprops=dict(arrowstyle="->", color=C_TRAIN, lw=1.1))
    ax.set_title("B. Razão de Gradientes (Blueprint / Policy)")
    ax.set_xlabel("Época")
    ax.set_ylabel("||∇ Blueprint|| / ||∇ Policy||")
    ax.set_ylim(0, 0.47)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.3)

    # C: Throughput stability (separated bands on twin axes)
    ax = axes[2]
    l1 = ax.plot(epochs, sps, color=C_TRAIN, linewidth=2.2, marker="o", markersize=3.2, label="Vazão (samples/s)")
    ax.set_ylim(316.0, 349.0)  # keeps samples/s (~340) in upper band
    ax.set_ylabel("Vazão (samples/s)", color=C_TRAIN)
    ax.tick_params(axis="y", labelcolor=C_TRAIN)
    ax.set_xlabel("Época")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    l2 = ax2.plot(epochs, ep_min, color=C_VAL, linewidth=2.0, linestyle="--", marker="s", markersize=3.0, label="Duração (min/época)")
    ax2.set_ylim(20.1, 23.8)  # keeps min/epoch (~21.04) cleanly in lower band
    ax2.set_ylabel("Duração (min / época)", color=C_VAL)
    ax2.tick_params(axis="y", labelcolor=C_VAL)

    ax.text(
        0.5, 0.52,
        f"Média: {np.mean(sps):.1f} samples/s | {np.mean(ep_min):.2f} min/época\nTotal 30 ép.: {total_h:.2f} h wall (~21,04 GPU-h)",
        transform=ax.transAxes, ha="center", va="center", fontsize=8.4, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", fc="#f8fafc", ec="#cbd5e1"),
    )
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    ax.set_title("C. Estabilidade de Vazão no Treino DDP")

    fig.suptitle("Stage 3: Dinâmica de Otimização e Estabilidade de Hardware (2 × Tesla T4)", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "stage3_dynamics_schedule.png")


def plot_eval_compliance_and_score(ep2: Dict[str, Any], ep30: Dict[str, Any]) -> None:
    short_names = ["S1\nClamp\n(x=63)", "S2\nTip Bot.\n(x=63,y=6)", "S3\nMidspan\n(x=44)", "S4\nRail XY\n(x=58)", "S5\nRail 3D\n(x=52)", "S6\nShort\n(x=36)"]
    e2_list = ep2["episodes"]
    e30_list = ep30["episodes"]
    simp_map = ep30["simp_baselines"]

    c_ep2 = np.array([e["final_compliance"] for e in e2_list])
    c_ep30 = np.array([e["final_compliance"] for e in e30_list])
    c_sbin = np.array([simp_map[e["scenario_id"]]["comp_bin"] for e in e30_list])
    c_scont = np.array([simp_map[e["scenario_id"]]["comp_cont"] for e in e30_list])

    s_ep2 = np.array([e["final_score"] for e in e2_list])
    s_ep30 = np.array([e["final_score"] for e in e30_list])
    s_scont = np.array([simp_map[e["scenario_id"]]["reward_cont"] for e in e30_list])
    s_sbin_s6 = simp_map["S6_short_deep_clamp"]["reward_bin"]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.4))

    # A: Final Compliance (Log)
    ax = axes[0]
    x = np.arange(len(short_names))
    w = 0.20
    b1 = ax.bar(x - 1.5 * w, c_ep2, width=w, color=C_EP2, edgecolor="#64748b", lw=0.6, label="ep2 (2 épocas)")
    b2 = ax.bar(x - 0.5 * w, c_ep30, width=w, color=C_EP30, edgecolor="#1d4ed8", lw=0.6, label="ep30 (30 épocas)")
    b3 = ax.bar(x + 0.5 * w, c_sbin, width=w, color=C_SIMP_BIN, edgecolor="#d97706", lw=0.6, label="SIMP Bin (ρ>0,5)")
    b4 = ax.bar(x + 1.5 * w, c_scont, width=w, color=C_SIMP_CONT, edgecolor="#047857", lw=0.6, label="SIMP Cont (β=4,0)")

    for i in range(len(x)):
        ratio = c_ep2[i] / c_ep30[i]
        tag = f"{ratio:.1f}×".replace(".", ",") if i != 1 else "0,8×*"
        col = C_EP30 if i != 1 else C_ALERT
        top_y = max(c_ep2[i], c_ep30[i]) * 1.20
        ax.text(x[i] - 0.5 * w, top_y, tag, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=col)

    ax.annotate("S6: ep30 (5.732) < SIMP_bin (6.801)\ncom metade da massa", xy=(5 - 0.5 * w, c_ep30[5]),
                xytext=(3.2, 2.2e5), fontsize=8.2, fontweight="bold", color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec=C_EP30, lw=0.9),
                arrowprops=dict(arrowstyle="->", color=C_EP30, lw=1.2, connectionstyle="arc3,rad=-0.12"))
    ax.set_yscale("log")
    ax.set_ylim(50, 1.2e6)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Compliance Final (menor = mais rígido, log)")
    ax.set_title("A. Rigidez Física Final (ep2 × ep30 × SIMP)")
    ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.3)

    # B: Score v3.3
    ax = axes[1]
    labels_b = short_names + ["Média\n(6 cen.)"]
    xb = np.arange(len(labels_b))
    wb = 0.34
    y2 = np.append(s_ep2, float(np.mean(s_ep2)))
    y30 = np.append(s_ep30, float(np.mean(s_ep30)))

    ax.axhline(0.0, color="#64748b", linewidth=1.1, zorder=1)
    ax.bar(xb - wb / 2, y2, width=wb, color=C_EP2, edgecolor="#64748b", lw=0.6, label="ep2 Score Final")
    ax.bar(xb + wb / 2, y30, width=wb, color=C_EP30, edgecolor="#1d4ed8", lw=0.6, label="ep30 Score Final")
    ax.scatter(np.arange(6), s_scont, marker="D", s=46, color=C_SIMP_CONT, zorder=5, label="SIMP Cont Reward")
    ax.scatter([5 - wb / 2], [s_sbin_s6], marker="*", s=115, color=C_SIMP_BIN, edgecolor="#92400e", zorder=6, label="SIMP Bin S6 (+0,048)")

    for i in range(len(xb)):
        v2, v30 = y2[i], y30[i]
        y2_off = 0.032 if (i == 5) else (0.015 if v2 >= 0 else -0.022)
        ax.text(xb[i] - wb / 2, v2 + y2_off, f"{v2:+.2f}".replace(".", ","),
                ha="center", va="bottom" if v2 >= 0 else "top", fontsize=7.6, color="#475569")
        y30_off = -0.026 if i == 1 else (0.015 if v30 >= 0 else -0.020)
        ax.text(xb[i] + wb / 2, v30 + y30_off, f"{v30:+.2f}".replace(".", ","),
                ha="center", va="bottom" if v30 >= 0 else "top", fontsize=7.8, fontweight="bold", color=C_EP30 if i != 1 else C_ALERT)

    ax.text(0.03, 0.94, "Score > 0: ep2 = 1/6 → ep30 = 4/6\nCONVERGENCE: ep2 = 1/6 → ep30 = 4/6",
            transform=ax.transAxes, va="top", fontsize=8.3, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc="#f8fafc", ec="#cbd5e1"))
    ax.set_ylim(-0.27, 0.78)
    ax.set_xticks(xb)
    ax.set_xticklabels(labels_b)
    ax.set_ylabel("Score / Recompensa v3.3")
    ax.set_title("B. Score Final v3.3: Salto Médio -0,092 → +0,057")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.0)

    fig.suptitle("Inferência Real + MCTS (160 sims): Comparativo de Compliance e Score v3.3", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "eval_compliance_score_comparison.png")


def plot_eval_geometry(ep2: Dict[str, Any], ep30: Dict[str, Any]) -> None:
    short_6 = ["S1", "S2", "S3", "S4", "S5", "S6"]
    e2 = ep2["episodes"]
    e30 = ep30["episodes"]
    smap = ep30["simp_baselines"]

    iy_2 = np.array([e["geometry"]["inertia_y_norm"] for e in e2])
    iy_30 = np.array([e["geometry"]["inertia_y_norm"] for e in e30])
    iy_simp = np.array([smap[e["scenario_id"]]["geometry"]["inertia_y_norm"] for e in e30])
    s2_main_iy = 0.05833586055867759

    tap_2 = np.array([e["geometry"]["y_span_root"] - e["geometry"]["y_span_tip"] for e in e2])
    tap_30 = np.array([e["geometry"]["y_span_root"] - e["geometry"]["y_span_tip"] for e in e30])

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0))

    # A: inertia_y_norm
    ax = axes[0]
    x = np.arange(6)
    w = 0.26
    ax.bar(x - w, iy_2, width=w, color=C_EP2, edgecolor="#64748b", lw=0.6, label="ep2")
    iy_30_plot = iy_30.copy()
    iy_30_plot[1] = s2_main_iy
    ax.bar(x, iy_30_plot, width=w, color=C_EP30, edgecolor="#1d4ed8", lw=0.6, label="ep30 (ilha principal)")
    # Hatched cap on S2 showing 2-island artifact inflation
    ax.bar([1], [iy_30[1] - s2_main_iy], bottom=[s2_main_iy], width=w,
           color="#dbeafe", edgecolor=C_ALERT, hatch="///", lw=0.9, label="S2 bruto (c/ 2ª ilha)")
    ax.bar(x + w, iy_simp, width=w, color=C_SIMP_CONT, edgecolor="#047857", lw=0.6, label="SIMP (β=4,0)")
    ax.annotate(f"S2: 0,058 ilha pr.\n(0,112 c/ ilha solta)", xy=(1, iy_30[1]), xytext=(1.35, 0.23),
                fontsize=8.0, fontweight="bold", color=C_ALERT,
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff1f2", ec=C_ALERT, lw=0.8),
                arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(short_6)
    ax.set_ylim(0, 0.59)
    ax.set_ylabel("inertia_y_norm")
    ax.set_title("A. Uso de Altura Estrutural (I_y)")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.0)

    # B: Root - Tip Taper
    ax = axes[1]
    wb = 0.34
    ax.axhline(0.0, color="#64748b", lw=1.1)
    ax.bar(x - wb / 2, tap_2, width=wb, color=C_EP2, edgecolor="#64748b", lw=0.6, label="ep2 (5/6 invertido)")
    ax.bar(x + wb / 2, tap_30, width=wb, color=C_EP30, edgecolor="#1d4ed8", lw=0.6, label="ep30 (6/6 raiz ≥ ponta)")
    for i in range(6):
        lbl2 = "0" if int(tap_2[i]) == 0 else f"{int(tap_2[i]):+d}"
        ax.text(x[i] - wb / 2, tap_2[i] + (0.35 if tap_2[i] >= 0 else -0.45), lbl2,
                ha="center", va="bottom" if tap_2[i] >= 0 else "top", fontsize=8.0, color="#475569")
        lbl = f"{int(tap_30[i]):+d}" if i != 1 else "+8*"
        ax.text(x[i] + wb / 2, tap_30[i] + 0.35, lbl,
                ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=C_EP30)
    ax.text(0.03, 0.03, "*S2 ilha principal isolada: span 9/10/9 (Δ=0)", transform=ax.transAxes,
            fontsize=7.9, color="#475569", bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#e2e8f0", alpha=0.9))
    ax.set_xticks(x)
    ax.set_xticklabels(short_6)
    ax.set_ylim(-12.6, 10.8)
    ax.set_ylabel("y_span_root − y_span_tip (voxels)")
    ax.set_title("B. Afilamento Mecânico Raiz → Ponta")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.2)

    # C: Post-load X overshoot (S3..S6)
    ax = axes[2]
    idx_ov = [2, 3, 4, 5]
    lbl_ov = ["S3\n(x=44)", "S4\n(x=58)", "S5\n(x=52)", "S6\n(x=36)"]
    ov_2 = np.array([e2[i]["geometry"]["overshoot_frac"] * 100.0 for i in idx_ov])
    ov_30 = np.array([e30[i]["geometry"]["overshoot_frac"] * 100.0 for i in idx_ov])
    xo = np.arange(len(idx_ov))
    ax.bar(xo - wb / 2, ov_2, width=wb, color=C_EP2, edgecolor="#64748b", lw=0.6, label="ep2")
    ax.bar(xo + wb / 2, ov_30, width=wb, color=C_EP30, edgecolor="#1d4ed8", lw=0.6, label="ep30")
    for i in range(len(xo)):
        ax.text(xo[i] - wb / 2, ov_2[i] + 0.4, f"{ov_2[i]:.1f}%".replace(".", ","), ha="center", va="bottom", fontsize=8.0, color="#475569")
        ax.text(xo[i] + wb / 2, ov_30[i] + 0.4, f"{ov_30[i]:.1f}%".replace(".", ","), ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=C_EP30)
    ax.axhline(0.0, color=C_SIMP_CONT, lw=2.0, label="SIMP alvo (0,0%)")
    ax.set_xticks(xo)
    ax.set_xticklabels(lbl_ov)
    ax.set_ylim(0, 19.5)
    ax.set_ylabel("Fração de massa além da carga (%)")
    ax.set_title("C. Overshoot Pós-Carga em X (lx < 63)")
    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.2)

    fig.suptitle("Padrões Geométricos Aprendidos: Inércia, Afilamento e Disciplina Espacial", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "eval_geometry_patterns.png")


def _fetch_p2_curve(con: sqlite3.Connection, game_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = con.execute(
        "SELECT step, value, fem_reward, reward_components_json "
        "FROM game_steps WHERE game_id = ? AND phase = 'REFINEMENT' ORDER BY step ASC",
        (game_id,),
    ).fetchall()
    steps = np.arange(len(rows))
    vals = np.array([float(r[1]) for r in rows])
    fems = []
    for _, _, fr, rc_json in rows:
        if fr is not None:
            fems.append(float(fr))
        else:
            fems.append(float(json.loads(rc_json)["fem_reward"]))
    return steps, vals, np.array(fems)


def plot_value_calibration_shift(hist: List[Dict[str, Any]], ep2: Dict[str, Any], ep30: Dict[str, Any]) -> None:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        st2, v2, f2 = _fetch_p2_curve(con, "ep2__S6_short_deep_clamp")
        st30, v30, f30 = _fetch_p2_curve(con, "ep30__S6_short_deep_clamp")
    finally:
        con.close()

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2))

    # A: Step-by-step S6 Phase 2 calibration
    ax = axes[0]
    ax.plot(st2, v2, color=C_EP2, linestyle="--", linewidth=1.9, label="ep2 V_net (predito)")
    ax.plot(st2, f2, color="#475569", linestyle="-", linewidth=1.9, marker="o", markersize=3.0, label="ep2 R_FEM (real)")
    ax.plot(st30, v30, color=C_EP30, linestyle="--", linewidth=2.2, label="ep30 V_net (predito)")
    ax.plot(st30, f30, color=C_SIMP_CONT, linestyle="-", linewidth=2.3, marker="^", markersize=4.0, label="ep30 R_FEM (real)")
    ax.fill_between(st30, f30, v30, color="#bfdbfe", alpha=0.38, label="Viés otimista ep30 (+0,44)")
    ax.axhline(0.0, color="#cbd5e1", lw=1.0)
    ax.text(0.03, 0.96, "Correlação r(V_net, R_FEM) em S6:\nep2 = −0,101  →  ep30 = +0,884",
            transform=ax.transAxes, va="top", fontsize=8.3, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc="#f8fafc", ec="#cbd5e1"))
    ax.set_xlabel("Passo relativo na Fase 2 (REFINEMENT)")
    ax.set_ylabel("Valor / Reward v3.3")
    ax.set_ylim(-0.08, 0.98)
    ax.set_title("A. Trajetória Real em S6: V_net vs R_FEM na Fase 2")
    ax.legend(loc="upper right", ncol=1, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=7.9)

    # B: Supervised Val MAE vs Live Phase-2 MCTS MAE
    ax = axes[1]
    labels = ["Val\nSup.", "S1", "S2", "S3", "S4", "S5", "S6", "Média\nP2"]
    mae2_p2 = [e["value_calibration_p2"]["mae"] for e in ep2["episodes"]]
    mae30_p2 = [e["value_calibration_p2"]["mae"] for e in ep30["episodes"]]
    y2 = [hist[1]["val"]["value_mae"]] + mae2_p2 + [float(np.mean(mae2_p2))]
    y30 = [hist[-1]["val"]["value_mae"]] + mae30_p2 + [float(np.mean(mae30_p2))]

    xb = np.arange(len(labels))
    w = 0.36
    ax.bar(xb - w / 2, y2, width=w, color=C_EP2, edgecolor="#64748b", lw=0.6, label="ep2 (2 épocas)")
    ax.bar(xb + w / 2, y30, width=w, color=C_EP30, edgecolor="#1d4ed8", lw=0.6, label="ep30 (30 épocas)")
    ax.axvline(0.5, color="#94a3b8", linestyle=":", lw=1.3)
    for i in range(len(xb)):
        ax.text(xb[i] - w / 2, y2[i] + 0.014, f"{y2[i]:.3f}".replace(".", ",") if i == 0 else f"{y2[i]:.2f}".replace(".", ","),
                ha="center", va="bottom", fontsize=7.5, color="#475569")
        ax.text(xb[i] + w / 2, y30[i] + 0.014, f"{y30[i]:.4f}".replace(".", ",") if i == 0 else f"{y30[i]:.2f}".replace(".", ","),
                ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=C_EP30 if i != 0 else C_ALERT)

    ax.annotate("Shift covariável (~30×):\n0,0134 em validação\nvs 0,401 médio na Fase 2",
                xy=(0 + w / 2, y30[0] + 0.055), xytext=(-0.15, 0.77),
                ha="left", va="top",
                fontsize=8.1, fontweight="bold", color=C_ALERT,
                bbox=dict(boxstyle="round,pad=0.26", fc="#fff1f2", ec=C_ALERT, lw=0.9),
                arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.2, connectionstyle="arc3,rad=0.10"))
    ax.set_xticks(xb)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Value Head MAE")
    ax.set_title("B. Deslocamento de Distribuição (Validação vs Rollout Real)")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.4)

    fig.suptitle("Diagnóstico da Fase 2: Calibração da Value Head e Shift Covariável", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "eval_value_calibration_shift.png")


def plot_hardware_cost_viability() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))

    # A: 30-Epoch Pre-training Cost (GPU-h)
    ax = axes[0]
    configs = ["Swin · 1×A100\n(est.)", "Swin · 1×4090\n(est.)", "Simple · 2×T4\n(medido)", "Swin · 2×T4\n(est.)"]
    mids = [12.5, 15.75, 21.04, 116.0]
    lows = [2.5, 3.25, 0.0, 20.0]
    highs = [2.5, 3.25, 0.0, 20.0]
    cols = [C_SIMP_CONT, C_SIMP_CONT, C_EP30, C_ALERT]
    y = np.arange(len(configs))
    ax.barh(y, mids, color=cols, height=0.52, edgecolor="#334155", lw=0.6)
    ax.errorbar(mids, y, xerr=[lows, highs], fmt="none", ecolor=C_DARK, elinewidth=1.6, capsize=4)
    ax.axvline(30.0, color=C_SIMP_BIN, linestyle="--", lw=1.6, label="Cota semanal Kaggle (30h)")
    lbls = ["10–15 GPU-h\n(~US$14–21)", "12,5–19 GPU-h\n(~US$5–8)", "21,04 GPU-h\n(10,52 h wall)", "96–136 GPU-h\n(5–7 resumes)"]
    for i, (m, hi, txt) in enumerate(zip(mids, highs, lbls)):
        xpos = m + hi + 3.0 if i < 3 else m - hi - 3.0
        ha = "left" if i < 3 else "right"
        tcol = C_DARK if i < 3 else C_ALERT
        ax.text(xpos, y[i], txt, va="center", ha=ha, fontsize=8.0, fontweight="bold", color=tcol,
                bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff", ec="none", alpha=0.85) if i == 3 else None)
    ax.set_yticks(y)
    ax.set_yticklabels(configs)
    ax.set_xlim(0, 148)
    ax.set_xlabel("Custo Total (GPU-horas para 30 épocas)")
    ax.set_title("A. Custo de Pré-Treino (477k amostras)")
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.0)

    # B: Local Raio-X BS=1 (ms)
    ax = axes[1]
    names_b = ["Simple\n(4,43M)\nFP32 XPU", "Swin-UNETR\n(6,83M)\nFP32 XPU"]
    vals_b = [70.8, 169.3]
    xb = np.arange(2)
    ax.bar(xb, vals_b, width=0.45, color=[C_SIMP_CONT, C_SIMP_BIN], edgecolor="#334155", lw=0.7)
    ax.axhline(120.0, color=C_VAL, linestyle="--", lw=1.5, label="Debounce Studio (120 ms)")
    ax.text(0, 70.8 + 5, "70,8 ms\n(~14,1 FPS)\nTempo real", ha="center", va="bottom", fontsize=8.4, fontweight="bold", color=C_SIMP_CONT)
    ax.text(1, 169.3 + 5, "169,3 ms\n(~5,9 FPS)\nAjustar ~200 ms", ha="center", va="bottom", fontsize=8.4, fontweight="bold", color="#b45309")
    ax.set_xticks(xb)
    ax.set_xticklabels(names_b)
    ax.set_ylim(0, 225)
    ax.set_ylabel("Latência BS=1 (ms) em Intel Graphics [0x7dd1]")
    ax.set_title("B. Raio-X Neural Interativo (BS=1)")
    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.2)

    # C: Local MCTS Latency (s/step)
    ax = axes[2]
    names_c = ["Simple\n160 sims\n(medido)", "Swin\n48–64 sims\n(est.)", "Swin\n160 sims\n(est.)"]
    m_c = [4.25, 6.0, 18.5]
    lo_c = [0.75, 1.5, 1.5]
    hi_c = [0.75, 1.5, 1.5]
    cols_c = [C_EP30, C_SIMP_BIN, C_ALERT]
    xc = np.arange(3)
    ax.axhspan(0, 8.0, color="#dcfce7", alpha=0.45, label="Zona interativa (< 8 s/passo)")
    ax.bar(xc, m_c, width=0.48, color=cols_c, edgecolor="#334155", lw=0.7)
    ax.errorbar(xc, m_c, yerr=[lo_c, hi_c], fmt="none", ecolor=C_DARK, elinewidth=1.6, capsize=4)
    lbls_c = ["3,5–5,0 s\n(F1 / F2+FEM)", "4,5–7,5 s\n(viável c/ corte)", "17–20 s\n(pesado ao vivo)"]
    for i, (m, h, t) in enumerate(zip(m_c, hi_c, lbls_c)):
        ax.text(xc[i], m + h + 0.6, t, ha="center", va="bottom", fontsize=8.2, fontweight="bold",
                color=C_DARK if i < 2 else C_ALERT)
    ax.set_xticks(xc)
    ax.set_xticklabels(names_c)
    ax.set_ylim(0, 24.5)
    ax.set_ylabel("Tempo por Passo MCTS (s)")
    ax.set_title("C. MCTS ao Vivo no Hardware Local")
    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.2)

    fig.suptitle("Swin-UNETR vs SimpleBackbone: Custo de Treino em Nuvem/Kaggle e Latência Local", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "hardware_cost_latency_viability.png")


def plot_ep30_weight_inspection(ep30_json: Dict[str, Any]) -> None:
    ckpt30 = torch.load(CKPT_EP30_PATH, map_location="cpu", weights_only=False)
    sd30 = {k.replace("module.", ""): v.float() for k, v in ckpt30["model_state_dict"].items()}

    # Panel A: Input channel weight energy (%) in stage1.conv1 vs raw_phys_proj (7 physical channels)
    w_conv1 = sd30["encoder.stage1.conv1.weight"][:, :7]  # (24, 7, 3, 3, 3)
    w_raw = sd30["raw_phys_proj.weight"].squeeze()        # (24, 7)
    # Use full 11ch denom for stage1.conv1 as measured across all channels
    w_conv1_all = sd30["encoder.stage1.conv1.weight"]
    e_conv1 = (w_conv1 ** 2).sum(dim=(0, 2, 3, 4)).numpy()
    pct_conv1 = e_conv1 / float((w_conv1_all ** 2).sum()) * 100.0
    e_raw = (w_raw ** 2).sum(dim=0).numpy()
    pct_raw = e_raw / float(e_raw.sum()) * 100.0
    ch_labels = ["density", "mask_x", "mask_y", "mask_z", "fx", "fy", "fz"]

    # Panel B & C: Live Phase-2 Policy logits & in-memory Value bias surgery sweep on ep30 final & SIMP states
    model = AlphaBuilderV31(in_channels=7, out_channels=2, feature_size=24, use_swin=False)
    model.load_state_dict(sd30)
    model.eval()

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    add_logits, rem_logits = [], []
    v_ep30_raw, r_ep30_true, v_simp_raw, r_simp_true = [], [], [], []
    try:
        for sc in SCENARIOS:
            sid = sc["id"]
            bc_m, f_m = _build_masks_and_forces(RESOLUTION, sc["load_config"], sc["bc_type"])
            bc_t, f_t = (bc_m[0], bc_m[1], bc_m[2]), (f_m[0], f_m[1], f_m[2])

            row30 = con.execute(
                "SELECT density_blob, fem_reward FROM game_steps WHERE game_id = ? ORDER BY step DESC LIMIT 1",
                (f"ep30__{sid}",),
            ).fetchone()
            dens30 = deserialize_array(row30[0]).astype(np.float32)
            x30 = torch.from_numpy(build_state_tensor(dens30, bc_t, f_t)).float().unsqueeze(0)

            rowsimp = con.execute(
                "SELECT density_blob FROM game_steps WHERE game_id = ? ORDER BY step DESC LIMIT 1",
                (f"simp__{sid}",),
            ).fetchone()
            dens_s = deserialize_array(rowsimp[0]).astype(np.float32)
            xs = torch.from_numpy(build_state_tensor(dens_s, bc_t, f_t)).float().unsqueeze(0)

            with torch.no_grad():
                pol30, val30 = model(x30)
                _, vals = model(xs)

            pa = pol30[0, 0].numpy()
            pr = pol30[0, 1].numpy()
            add_logits.append(float(pa[dens30 <= 0.5].max()))
            rem_logits.append(float(pr[dens30 > 0.5].max()))

            v_ep30_raw.append(float(val30.item()))
            r_ep30_true.append(float(row30[1]))
            v_simp_raw.append(float(vals.item()))
            r_simp_true.append(float(ep30_json["simp_baselines"][sid]["reward_cont"]))
    finally:
        con.close()

    deltas = [0.0, -0.10, -0.20, -0.30, -0.35]
    mae_ep30_curve, mae_simp_curve = [], []
    for d in deltas:
        e30_preds = [float(np.tanh(np.arctanh(np.clip(v, -0.9999, 0.9999)) + d)) for v in v_ep30_raw]
        simp_preds = [float(np.tanh(np.arctanh(np.clip(v, -0.9999, 0.9999)) + d)) for v in v_simp_raw]
        mae_ep30_curve.append(float(np.mean(np.abs(np.array(e30_preds) - np.array(r_ep30_true)))))
        mae_simp_curve.append(float(np.mean(np.abs(np.array(simp_preds) - np.array(r_simp_true)))))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    # A: Channel energies
    ax = axes[0]
    xa = np.arange(len(ch_labels))
    wa = 0.36
    ax.bar(xa - wa / 2, pct_conv1, width=wa, color=C_EP30, edgecolor="#1d4ed8", lw=0.6, label="encoder.stage1.conv1 (% total 11ch)")
    ax.bar(xa + wa / 2, pct_raw, width=wa, color=C_SIMP_BIN, edgecolor="#d97706", lw=0.6, label="raw_phys_proj (atalho 1×1)")
    ax.annotate("fz nunca visto no warmup:\n27,3% energia residual OOD!", xy=(6 + wa / 2, pct_raw[6]), xytext=(3.1, 31.0),
                fontsize=8.1, fontweight="bold", color=C_ALERT,
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff1f2", ec=C_ALERT, lw=0.8),
                arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.1))
    ax.text(0.03, 0.95, "Simetria D4 aprendida:\nfx ≈ fy (26,8% vs 26,4%)\nmask_x ≈ mask_y (3,3% vs 3,2%)",
            transform=ax.transAxes, va="top", fontsize=7.9,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1"))
    ax.set_xticks(xa)
    ax.set_xticklabels(ch_labels, rotation=20)
    ax.set_ylim(0, 39.5)
    ax.set_ylabel("Fração da Energia L2² dos Pesos (%)")
    ax.set_title("A. Sensibilidade por Canal na Entrada")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=7.8)

    # B: Action Logit Gap on ep30 final states
    ax = axes[1]
    lbl_b = ["S1", "S2", "S3", "S4", "S5", "S6", "Média"]
    xb = np.arange(len(lbl_b))
    add_plot = add_logits + [float(np.mean(add_logits))]
    rem_plot = rem_logits + [float(np.mean(rem_logits))]
    ax.axhline(0.0, color="#64748b", lw=1.0)
    ax.bar(xb - wa / 2, add_plot, width=wa, color=C_TRAIN, edgecolor="#0f766e", lw=0.6, label="max_logit ADD (vazios)")
    ax.bar(xb + wa / 2, rem_plot, width=wa, color=C_ALERT, edgecolor="#991b1b", lw=0.6, label="max_logit REMOVE (sólidos)")
    for i in range(len(xb)):
        ax.text(xb[i] - wa / 2, add_plot[i] + 0.22, f"{add_plot[i]:+.1f}".replace(".", ","),
                ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=C_TRAIN)
        ax.text(xb[i] + wa / 2, rem_plot[i] + (0.22 if rem_plot[i] >= 0 else -0.32), f"{rem_plot[i]:+.1f}".replace(".", ","),
                ha="center", va="bottom" if rem_plot[i] >= 0 else "top", fontsize=7.6, color=C_ALERT)
    ax.text(0.025, 0.96, "Δ = +9,2 (finais) | +8,9 (18 P2)\n0/18 estados com REM > ADD",
            transform=ax.transAxes, va="top", fontsize=7.8, fontweight="bold", color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.24", fc="#f8fafc", ec="#cbd5e1"))
    ax.set_xticks(xb)
    ax.set_xticklabels(lbl_b)
    ax.set_ylim(-2.6, 14.0)
    ax.set_ylabel("Logit Máximo Válido na Policy Head")
    ax.set_title("B. Competição ADD vs REMOVE nos Estados Finais ep30")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=7.8)

    # C: Value bias shift trade-off
    ax = axes[2]
    ax.plot(deltas, mae_ep30_curve, color=C_EP30, lw=2.3, marker="o", markersize=5.5, label="MAE Finais ep30 (OOD MCTS)")
    ax.plot(deltas, mae_simp_curve, color=C_SIMP_CONT, lw=2.3, marker="D", markersize=5.2, label="MAE Finais SIMP Contínuo")
    for d, ye, ys in zip(deltas, mae_ep30_curve, mae_simp_curve):
        if d in (0.0, -0.20, -0.35):
            ax.text(d, ye + (0.022 if ye > ys else -0.040), f"{ye:.3f}".replace(".", ","), ha="center", fontsize=8.0, fontweight="bold", color=C_EP30)
            ax.text(d, ys + (0.020 if ys > ye else -0.040), f"{ys:.3f}".replace(".", ","), ha="center", fontsize=8.0, fontweight="bold", color=C_SIMP_CONT)
    ax.annotate("Troca destrutiva:\ncorrigir ep30 (-0,35)\ndegrada SIMP 2,2×", xy=(-0.35, mae_simp_curve[-1]),
                xytext=(-0.03, 0.52), fontsize=8.0, fontweight="bold", color=C_ALERT,
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff1f2", ec=C_ALERT, lw=0.8),
                arrowprops=dict(arrowstyle="->", color=C_ALERT, lw=1.1, connectionstyle="arc3,rad=-0.12"))
    ax.set_xlabel("Deslocamento manual Δ em value_head.5.bias (atual = +0,1059)")
    ax.set_ylabel("Value MAE nos 6 Cenários")
    ax.set_ylim(0.05, 0.64)
    ax.invert_xaxis()
    ax.set_title("C. Efeito de Cirurgia no Bias da Value Head")
    ax.legend(loc="lower left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.0)

    fig.suptitle("Inspeção Direta do Checkpoint ep30: Canais, Logits ADD/REMOVE e Cirurgia de Pesos", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "ep30_weight_inspection.png")


def main() -> None:
    _apply_clean_style()
    hist = json.loads((ANALYSIS_DIR / "stage3_simple_metrics_history.json").read_text())
    ep2 = json.loads((ANALYSIS_DIR / "eval_ep2_results.json").read_text())
    ep30 = json.loads((ANALYSIS_DIR / "eval_ep30_results.json").read_text())

    plot_stage3_losses(hist)
    plot_stage3_val_metrics(hist)
    plot_stage3_dynamics(hist)
    plot_eval_compliance_and_score(ep2, ep30)
    plot_eval_geometry(ep2, ep30)
    plot_value_calibration_shift(hist, ep2, ep30)
    plot_hardware_cost_viability()
    plot_ep30_weight_inspection(ep30)


if __name__ == "__main__":
    main()
