#!/usr/bin/env python3
"""Generate 5 publication-grade first-principles architecture diagrams for AlphaBuilder.

Outputs crisp white-background schematics under data/analysis/figures/:
1. arch_01_simple_backbone_anatomy.png
2. arch_02_first_principles_vs_swin.png
3. arch_03_macro_architecture.png
4. arch_04_internal_mechanisms.png
5. arch_05_mcts_selfplay_roadmap.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "data" / "analysis" / "figures"

# Cohesive palette
C_CONV = "#0f766e"       # Teal (3D Conv / Spatial)
C_CONV_BG = "#f0fdfa"
C_TRANS = "#2563eb"      # Royal Blue (Global Transformer / RoPE)
C_TRANS_BG = "#eff6ff"
C_PHYS = "#7c3aed"       # Purple (Physics FiLM / AdaLN / SED)
C_PHYS_BG = "#f5f3ff"
C_MCTS = "#d97706"       # Amber (Policy / MCTS)
C_MCTS_BG = "#fffbeb"
C_VAL = "#059669"        # Emerald (Value / FEniCSx Verifier)
C_VAL_BG = "#ecfdf5"
C_WARN = "#dc2626"       # Coral Red (Limitations / Zero-Padding Waste)
C_WARN_BG = "#fef2f2"
C_SLATE = "#1e293b"      # Slate 800
C_MUTED = "#475569"      # Slate 600
C_BORDER = "#cbd5e1"     # Slate 300
C_LIGHT = "#f8fafc"      # Slate 50


def _setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "text.color": C_SLATE,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=180, facecolor="#ffffff", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"Saved: {out.relative_to(REPO_ROOT)} ({out.stat().st_size / 1024:.1f} KB)")


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    fc: str = C_LIGHT,
    ec: str = C_SLATE,
    lw: float = 1.5,
    title_fs: float = 9.8,
    body_fs: float = 8.2,
    pad: float = 0.35,
    title_color: str | None = None,
    zorder: int = 3,
    ls: str = "-",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size=1.0",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    tcol = title_color or ec
    if body:
        # Place title cleanly in upper strip and body below
        lines_b = body.count("\n") + 1
        title_y = y + h * (0.80 if lines_b >= 3 else 0.72)
        body_y = y + h * (0.37 if lines_b >= 3 else 0.32)
        ax.text(x + w / 2, title_y, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=tcol, zorder=zorder + 1)
        ax.text(x + w / 2, body_y, body, ha="center", va="center", fontsize=body_fs, color=C_SLATE, linespacing=1.28, zorder=zorder + 1)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=tcol, zorder=zorder + 1)


def _arrow(
    ax: plt.Axes,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    color: str = C_SLATE,
    lw: float = 1.6,
    rad: float = 0.0,
    ls: str = "-",
    style: str = "-|>",
    label: str = "",
    lpos: Tuple[float, float] | None = None,
    lfs: float = 7.8,
    zorder: int = 2,
) -> None:
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arr)
    if label:
        mx = lpos[0] if lpos else 0.5 * (p0[0] + p1[0])
        my = lpos[1] if lpos else 0.5 * (p0[1] + p1[1])
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=lfs,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff", ec="none", alpha=0.92),
            zorder=zorder + 3,
        )


def _callout_badge(
    ax: plt.Axes,
    bx: float,
    by: float,
    bw: float,
    bh: float,
    target: Tuple[float, float],
    tag: str,
    text: str,
    rad: float = 0.0,
) -> None:
    _box(
        ax, bx, by, bw, bh,
        tag, text,
        fc=C_WARN_BG, ec=C_WARN, lw=1.3,
        title_fs=8.5, body_fs=7.6, pad=0.25, title_color=C_WARN, zorder=6,
    )
    sx = bx + bw / 2
    sy = by if by > target[1] else by + bh
    if bx > target[0] + 5 and abs(by + bh / 2 - target[1]) < 4:
        sx, sy = bx, by + bh / 2
    _arrow(ax, (sx, sy), target, color=C_WARN, lw=1.3, ls="--", rad=rad, zorder=5)


def plot_arch_01_simple_backbone() -> None:
    fig, ax = plt.subplots(figsize=(17.0, 8.8))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 66)
    ax.axis("off")

    ax.text(55, 63.8, "SimpleBackbone (v3.3 · 4.433.132 parâmetros): Anatomia Atual e 5 Limitações Diagnosticadas",
            ha="center", va="center", fontsize=14, fontweight="bold", color=C_SLATE)
    ax.text(55, 61.2, "Código-fonte: alphabuilder/src/neural/model.py | Auditoria empírica no checkpoint ep30 (Stage 3, 30 épocas)",
            ha="center", va="center", fontsize=9.5, color=C_MUTED)

    # Top skip rail cleanly positioned below top callouts
    ax.text(52.5, 44.8, "Skip Connections espaciais (enc1 1×, enc2 1/2×, enc3 1/4×)",
            ha="center", va="center", fontsize=8.1, fontweight="bold", color=C_CONV,
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=C_CONV, lw=0.8), zorder=4)

    # Main backbone flow y=21..38
    _box(ax, 2.0, 21.5, 15.5, 16.5,
         "Entrada Aumentada",
         "(B, 11, 64, 32, 8)\n7 fís: ρ, mask_xyz, fx,fy,fz\n+3 CoordConv (-1..1)\n+1 domain_mask (1s)",
         fc=C_LIGHT, ec=C_SLATE)

    _box(ax, 21.0, 21.5, 17.5, 16.5,
         "SimpleEncoder (68,6%)",
         "3.040.992 params\nStage 1 (1×): 24ch\nStage 2 (1/2×): 48ch\nStage 3 (1/4×): 96ch\nResBlock + GN + SiLU",
         fc=C_CONV_BG, ec=C_CONV)

    _box(ax, 42.0, 20.0, 20.5, 19.5,
         "Bottleneck Dilatado (1/8×)",
         "Grid 8 × 4 × 1 · C = 192\nb1: Conv 3³ (1→2×)\nb2: Dilated (2,2,1)\nb3: Dilated (4,2,1) [RF=148]\n+ global_ctx (AvgPool→MLP)\n+ phys_film (FiLM)",
         fc=C_TRANS_BG, ec=C_TRANS)

    _box(ax, 66.0, 21.5, 17.0, 16.5,
         "SimpleDecoder (27,1%)",
         "1.202.976 params\nConvT 3D (1/4→1/2→1×)\nConcat skips (96, 48, 24)\n+ bottleneck_to_full\n+ raw_phys_proj (1×1)",
         fc=C_CONV_BG, ec=C_CONV)

    # 3 Output Heads cleanly spaced on far right (x=87.5..108)
    _box(ax, 87.5, 45.0, 20.5, 11.5,
         "Value Head (2,97%)",
         "[AvgPool, MaxPool, phys_emb]\n448 → 256 → 64 → 1 → Tanh\nV(s) ∈ [-1, 1]",
         fc=C_VAL_BG, ec=C_VAL)

    _box(ax, 87.5, 25.0, 20.5, 11.5,
         "Policy Head (0,35%)",
         "Conv 3³ + GN + SiLU + 1³\n(B, 2, 64, 32, 8)\nLogits: ADD / REMOVE",
         fc=C_MCTS_BG, ec=C_MCTS)

    _box(ax, 87.5, 7.5, 20.5, 11.5,
         "Blueprint Head (0,35%)",
         "Conv 3³ + GN + SiLU + 1³\n(B, 1, 64, 32, 8)\nAlvo: V_constructed (SIMP)",
         fc=C_PHYS_BG, ec=C_PHYS)

    # Bottom global physics rail
    _box(ax, 15.0, 4.5, 24.5, 9.5,
         "9 Descritores Físicos Globais",
         "vol_mean, bc_xy, bc_z, f_max, ...\n→ phys_film & phys_value_proj",
         fc=C_PHYS_BG, ec=C_PHYS, title_fs=8.8, body_fs=7.6)

    # Connecting arrows
    _arrow(ax, (17.5, 29.8), (21.0, 29.8), C_SLATE, lw=1.7)
    _arrow(ax, (38.5, 29.8), (42.0, 29.8), C_CONV, lw=1.7)
    _arrow(ax, (62.5, 29.8), (66.0, 29.8), C_TRANS, lw=1.7)
    _arrow(ax, (29.8, 38.0), (74.5, 38.0), C_CONV, lw=1.3, ls="--", rad=-0.20)

    # Bottleneck -> Value Head
    _arrow(ax, (52.5, 39.5), (87.5, 50.5), C_VAL, lw=1.7, rad=-0.10, label="AvgPool + MaxPool", lpos=(72.0, 43.0))

    # Decoder -> Policy & Blueprint
    _arrow(ax, (83.0, 30.2), (87.5, 30.7), C_MCTS, lw=1.7)
    _arrow(ax, (83.0, 26.0), (87.5, 13.5), C_PHYS, lw=1.7, rad=0.14)

    # Global physics rail arrows
    _arrow(ax, (9.5, 21.5), (15.0, 9.2), C_PHYS, lw=1.3, rad=0.14)
    _arrow(ax, (39.5, 9.2), (52.2, 20.0), C_PHYS, lw=1.4, rad=-0.12, label="γ, β FiLM")
    _arrow(ax, (39.5, 11.8), (87.5, 47.0), C_VAL, lw=1.2, ls=":", rad=0.16, label="phys_emb (bc_z 50,4%)", lpos=(65.0, 17.5))

    # 5 Diagnostic Limitation Badges ([L1]..[L5]) — generous widths and clean gaps
    _callout_badge(
        ax, 1.5, 47.5, 26.5, 10.5, (9.5, 38.0),
        "[L1] Entrada rígida e carga única",
        "CoordConv fixo; sem keep-in/out ou multi-carga.\nstem[fy] 82% plano em Z; fz OOD = 27,3% em atalho",
    )
    _callout_badge(
        ax, 31.5, 48.0, 27.5, 10.2, (50.0, 39.5),
        "[L2] Dilatação rígida + global_ctx escalar",
        "Kernel fixo independente do conteúdo;\nsem atenção par-a-par nem 3D RoPE relativa",
    )
    _callout_badge(
        ax, 62.0, 48.0, 23.0, 10.2, (87.5, 52.5),
        "[L3] Value 1-escalar em Tanh",
        "value_head.0 em posto 16,7%;\nviés otimista +0,40 fora do treino",
        rad=-0.06,
    )
    _callout_badge(
        ax, 42.0, 1.5, 20.0, 9.5, (87.5, 28.0),
        "[L5] Policy cega à fase/volume",
        "Abismo +8,9 logits pró-ADD Fase 2\n(0/18 estados c/ REM > ADD)",
        rad=0.15,
    )
    _callout_badge(
        ax, 64.5, 1.5, 20.5, 9.5, (87.5, 11.5),
        "[L4] Blueprint exige oráculo SIMP",
        "Ótima no warmup (Dice 90,1%),\nmas fica sem alvo no Self-Play",
    )

    _save(fig, "arch_01_simple_backbone_anatomy.png")


def plot_arch_02_first_principles_vs_swin() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.0))

    # Panel A: Global Elliptic PDE + SED Physics
    ax = axes[0]
    ax.set_xlim(-4, 70)
    ax.set_ylim(-4, 46)
    ax.axis("off")
    ax.set_title("A. Princípio 1: Elasticidade Elíptica Global\n(∇·σ + f = 0 exige alcance imediato)", fontsize=11.3, fontweight="bold", pad=8)

    ax.add_patch(Rectangle((2, 3), 60, 25, facecolor="#f8fafc", edgecolor="#94a3b8", lw=1.2, ls="--"))
    ax.plot([2, 2], [3, 28], color=C_SLATE, lw=3.0)
    for y_tri in np.linspace(5, 26, 6):
        tri = Polygon([[2, y_tri], [-1.5, y_tri - 1.4], [-1.5, y_tri + 1.4]], facecolor="#64748b", edgecolor=C_SLATE, lw=0.8)
        ax.add_patch(tri)
    ax.text(-0.5, 31.2, "Engaste\n(x = 0)", ha="center", va="center", fontsize=8.4, fontweight="bold", color=C_SLATE)

    ax.plot([2, 62], [26.5, 16.0], color=C_WARN, lw=4.3, alpha=0.85)
    ax.plot([2, 62], [4.5, 15.0], color=C_WARN, lw=4.3, alpha=0.85)
    for x0, x1 in [(12, 26), (26, 40), (40, 54)]:
        ax.plot([x0, x1], [24.6 - 0.16 * x0, 6.4 + 0.16 * x1], color=C_MCTS, lw=2.1, alpha=0.8)
        ax.plot([x0, x1], [6.4 + 0.16 * x0, 24.6 - 0.16 * x1], color=C_MCTS, lw=2.1, alpha=0.8)
    ax.add_patch(FancyBboxPatch((8, 12.0), 22, 7.0, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="#dbeafe", edgecolor=C_TRANS, lw=1.1, ls=":"))
    ax.text(19, 15.5, "Núcleo neutro\nSED ≈ 0 → REMOVE", ha="center", va="center", fontsize=7.9, fontweight="bold", color=C_TRANS)

    _arrow(ax, (62, 24), (62, 16.5), color=C_WARN, lw=2.8)
    ax.text(62, 25.5, "Carga F\n(x = 63)", ha="center", va="bottom", fontsize=8.6, fontweight="bold", color=C_WARN)

    _arrow(ax, (60, 17.5), (3.5, 26.8), color=C_PHYS, lw=2.0, rad=0.30, style="<|-|>")
    ax.text(36, 40.5, "Função de Green G(x, x'):\ncarga na ponta define momento no engaste!",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_PHYS,
            bbox=dict(boxstyle="round,pad=0.25", fc=C_PHYS_BG, ec=C_PHYS, lw=0.8))

    # Panel B: Padding Waste & Window Locality comparison
    ax = axes[1]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    ax.set_title("B. Swin-UNETR vs Phys-UViT 3D\n(Zero-Padding em Z + Localidade vs Tokens Globais)", fontsize=11.3, fontweight="bold", pad=8)

    # Row 1: SimpleBackbone
    _box(ax, 1.5, 44.5, 97.0, 16.5, "", "", fc=C_LIGHT, ec=C_CONV, lw=1.2)
    ax.text(4, 57.5, "1. SimpleBackbone (4,43M · divisor = 8)", fontsize=8.9, fontweight="bold", color=C_CONV)
    ax.add_patch(Rectangle((4, 46.8), 29, 7.8, facecolor="#99f6e4", edgecolor=C_CONV, lw=1.2))
    ax.text(18.5, 50.7, "64×32×8\n(16.384 vox · 0% pad)", ha="center", va="center", fontsize=7.2, fontweight="bold")
    ax.text(35.5, 50.7, "Conv3D + Dilatação rígida (4,2,1)\n+ global_ctx escalar\n✓ 70,8 ms BS=1  |  ✗ sem atenção par-a-par", ha="left", va="center", fontsize=7.4, color=C_SLATE)

    # Row 2: Swin-UNETR
    _box(ax, 1.5, 23.0, 97.0, 18.0, "", "", fc=C_WARN_BG, ec=C_WARN, lw=1.3)
    ax.text(4, 37.8, "2. Swin-UNETR (6,83M · divisor = 32)", fontsize=8.9, fontweight="bold", color=C_WARN)
    ax.add_patch(Rectangle((4, 25.2), 7.2, 10.0, facecolor="#fde68a", edgecolor=C_MCTS, lw=1.1))
    ax.add_patch(Rectangle((11.2, 25.2), 21.8, 10.0, facecolor="#fee2e2", edgecolor=C_WARN, hatch="///", lw=1.1))
    ax.text(18.5, 30.2, "75% Zero-Pad\n(Z: 8 → 32)", ha="center", va="center", fontsize=7.3, fontweight="bold", color=C_WARN,
            bbox=dict(boxstyle="round,pad=0.12", fc="#ffffff", ec="none", alpha=0.90))
    ax.text(35.5, 30.2, "65.536 vox (4×) + janela local 7×7×7\nAcoplamento global só no estágio 4\n✗ 169,3 ms BS=1 | ~4,5–6,5× treino | ~17–20s", ha="left", va="center", fontsize=7.4, color=C_SLATE)

    # Row 3: Phys-UViT 3D
    _box(ax, 1.5, 2.0, 97.0, 17.8, "", "", fc=C_TRANS_BG, ec=C_TRANS, lw=1.6)
    ax.text(4, 16.7, "3. Phys-UViT 3D Proposta (Conv + Global)", fontsize=8.9, fontweight="bold", color=C_TRANS)
    gx, gy, gw, gh = 4, 4.2, 29, 9.6
    ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor="#dbeafe", edgecolor=C_TRANS, lw=1.2))
    pts = [(gx + 3, gy + 2), (gx + 14.5, gy + 7.4), (gx + 26, gy + 2), (gx + 26, gy + 7.6), (gx + 3, gy + 7.6)]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=C_TRANS, lw=0.8, alpha=0.55)
        ax.scatter([pts[i][0]], [pts[i][1]], color=C_TRANS, s=15, zorder=4)
    ax.text(gx + gw / 2, gy + gh / 2, "16×8×2 = 256 tokens\n(0% zero-padding)", ha="center", va="center", fontsize=7.2, fontweight="bold", color=C_SLATE,
            bbox=dict(boxstyle="round,pad=0.14", fc="#ffffff", ec="none", alpha=0.88))
    ax.text(35.5, 9.0, "Atenção global plena + 3D RoPE na 1ª camada\n+ Multi-Load Cross-Attn + Cabeça SED\n✓ 0% pad | ~3–4× mais leve que Swin p/ MCTS", ha="left", va="center", fontsize=7.4, color=C_SLATE)

    # Panel C: MCTS Pareto Frontier
    ax = axes[2]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.grid(True, color="#e2e8f0", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("C. Fronteira de Pareto no MCTS\n(Força = Qualidade da Rede × Sims/s)", fontsize=11.3, fontweight="bold", pad=8)
    ax.set_xlabel("Vazão de Busca Local (maior = mais simulações / passo)")
    ax.set_ylabel("Expressividade Física Global & BCs Arbitrárias")
    ax.set_xticks([18, 50, 82])
    ax.set_xticklabels(["Lento\n(17–20s/passo)", "Rápido\n(4–6s/passo)", "Muito Rápido\n(3–4s/passo)"])
    ax.set_yticks([25, 55, 85])
    ax.set_yticklabels(["Local / Caixa Fixa", "Dilation / Semi-Global", "Global + BCs Arbitrárias"])

    ax.fill([56, 98, 98, 66], [96, 96, 62, 74], color="#dcfce7", alpha=0.55, zorder=1)
    ax.plot([42, 68, 86], [94, 86, 56], color=C_VAL, linestyle="--", lw=1.6, zorder=2)

    ax.scatter([82], [52], s=140, color=C_CONV, edgecolor=C_SLATE, lw=1.2, zorder=5)
    ax.text(78, 36, "SimpleBackbone (v3.3)\n70,8 ms BS=1 | ~3,5–5 s MCTS\n(bom p/ TCC c/ Etapa 6)", ha="center", fontsize=7.8, fontweight="bold", color=C_CONV,
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=C_CONV, lw=0.8))

    ax.scatter([22], [68], s=140, color=C_WARN, edgecolor=C_SLATE, lw=1.2, zorder=5)
    ax.text(28, 53, "Swin-UNETR (v3.3)\n169,3 ms BS=1 | ~17–20 s\n(75% pad + janela 7³)", ha="center", fontsize=7.8, fontweight="bold", color=C_WARN,
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=C_WARN, lw=0.8))

    ax.scatter([68], [86], s=190, marker="*", color=C_TRANS, edgecolor=C_SLATE, lw=1.2, zorder=6)
    ax.annotate("Phys-UViT 3D (Alvo v4)\n0% pad | 256 tokens RoPE | SED",
                xy=(68, 86), xytext=(36, 86), ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_TRANS,
                bbox=dict(boxstyle="round,pad=0.22", fc=C_TRANS_BG, ec=C_TRANS, lw=1.0),
                arrowprops=dict(arrowstyle="->", color=C_TRANS, lw=1.2))

    fig.tight_layout(w_pad=1.8)
    _save(fig, "arch_02_first_principles_vs_swin.png")


def plot_arch_03_macro_architecture() -> None:
    fig, ax = plt.subplots(figsize=(17.6, 9.8))
    ax.set_xlim(0, 108)
    ax.set_ylim(0, 64)
    ax.axis("off")

    ax.text(54, 61.8, "Arquitetura Alvo: Phys-UViT 3D (ConvNeXt/Res3D + Global 3D-RoPE Transformer + Cabeça SED)",
            ha="center", va="center", fontsize=14, fontweight="bold", color=C_SLATE)
    ax.text(54, 59.3, "Derivada dos 5 primeiros princípios: 0% zero-padding, acoplamento global na camada 1, domínios/cargas arbitrários e sinal físico denso no Self-Play",
            ha="center", va="center", fontsize=9.4, color=C_MUTED)

    # Inputs column (x=1.5..18)
    _box(ax, 1.5, 35.5, 16.5, 17.0,
         "Entrada Espacial",
         "(B, 7, D, H, W)\n• density ρ\n• domain_mask ternária:\n  [free / keep-in / keep-out]\n• suportes mask_x, mask_y, mask_z",
         fc=C_LIGHT, ec=C_SLATE)

    _box(ax, 1.5, 18.0, 16.5, 12.0,
         "Conjunto Multi-Carga",
         "(B, K, 3, D, H, W) → tokens\n{(fx_k, fy_k, fz_k)}_{k=1..K}\nK casos independentes\n(set-equivariante)",
         fc=C_MCTS_BG, ec=C_MCTS)

    _box(ax, 1.5, 4.0, 16.5, 10.5,
         "Contexto Físico Global",
         "phys_stats (||F||, log||F||,\nbc_regime, vol_frac)\n+ (phase, target_vol)",
         fc=C_PHYS_BG, ec=C_PHYS)

    # Encoder 3D (x=21.5..37.5)
    _box(ax, 21.5, 35.5, 16.0, 17.0,
         "Encoder 3D Raso",
         "ConvNeXt / ResBlock 3D\nGroupNorm + SiLU (0% pad)\n• 1× (64×32×8, C=32)\n• 1/2× (32×16×4, C=64)\n• 1/4× (16×8×2, C=192)",
         fc=C_CONV_BG, ec=C_CONV)
    ax.text(29.5, 33.4, "✓ Apenas divisor = 4 (vs 32 da Swin)", ha="center", fontsize=7.9, fontweight="bold", color=C_CONV,
            bbox=dict(boxstyle="round,pad=0.14", fc="#ffffff", ec="none", alpha=0.9))

    # Global Physics Transformer Bottleneck (x=41..64.5)
    _box(ax, 41.0, 29.0, 23.5, 22.5,
         "Bottleneck Global Transformer",
         "4–6× Blocos | N = 256 tokens (16×8×2) | d = 192\n\n1. 3D RoPE (Δx, Δy, Δz) + Masked MHSA\n    → acopla apoio ↔ carga em 1 camada!\n2. Multi-Load Cross-Attention (K cargas)\n3. AdaLN(phys) + SwiGLU FFN",
         fc=C_TRANS_BG, ec=C_TRANS, lw=2.0)

    # Factored Value Head (upper-right x=68..101.5)
    _box(ax, 68.0, 46.5, 33.5, 10.5,
         "Value Head Fatorada (Cura Baixo Posto & Viés OOD)",
         "[AvgPool, MaxPool (veto desconexão), AttnPool] ⊕ phys_emb → MLP\n3 Saídas Desacopladas:  log(C)  |  P(conectado)  |  V(s) ∈ [-1, 1]",
         fc=C_VAL_BG, ec=C_VAL, lw=1.7)

    # Decoder 3D (x=68..82.5)
    _box(ax, 68.0, 22.5, 14.5, 18.0,
         "Decoder 3D",
         "1/4× → 1/2× → 1×\nFunde skips 1/2× e 1×\n+ Global Broadcast\n+ skip limpo ρ/BCs\n(B, 32, 64, 32, 8)",
         fc=C_CONV_BG, ec=C_CONV)

    # 3 Spatial Full-Res Heads (x=85.5..101.5)
    _box(ax, 85.5, 31.0, 16.0, 11.5,
         "Policy Head",
         "(B, 2, 64, 32, 8)\nADD / REMOVE\nAdaLN(phase, vol)\n+ máscara keep-in/out",
         fc=C_MCTS_BG, ec=C_MCTS, lw=1.7)

    _box(ax, 85.5, 17.5, 16.0, 11.0,
         "Physics SED + u Head",
         "(B, 4, 64, 32, 8)\nEnergia deformação SED\ne deslocamentos (ux,uy,uz)\n(grátis no Self-Play!)",
         fc=C_PHYS_BG, ec=C_PHYS, lw=1.7)

    _box(ax, 85.5, 4.5, 16.0, 10.0,
         "Blueprint Head (opc.)",
         "(B, 1, 64, 32, 8)\nAuxiliar opcional\napenas no warmup SIMP",
         fc=C_LIGHT, ec="#94a3b8", ls="--")

    # Skip connections overhead
    _arrow(ax, (29.5, 52.5), (75.2, 40.5), color=C_CONV, lw=1.4, ls="--", rad=-0.24,
           label="Skips sub-voxel (1×, 1/2×)", lpos=(52.5, 56.2))

    # Dataflow arrows
    _arrow(ax, (18.0, 44.0), (21.5, 44.0), C_SLATE, lw=1.7)
    _arrow(ax, (37.5, 44.0), (41.0, 44.0), C_CONV, lw=1.8, label="(16×8×2)", lpos=(39.2, 46.8))
    _arrow(ax, (64.5, 46.5), (68.0, 51.5), C_VAL, lw=1.8)
    _arrow(ax, (64.5, 34.0), (68.0, 34.0), C_TRANS, lw=1.8)

    # Decoder -> Spatial Heads
    _arrow(ax, (82.5, 35.5), (85.5, 36.8), C_MCTS, lw=1.7)
    _arrow(ax, (82.5, 29.0), (85.5, 23.0), C_PHYS, lw=1.7)
    _arrow(ax, (82.5, 24.0), (85.5, 9.5), "#64748b", lw=1.3, ls="--", rad=0.12)

    # Multi-load -> Cross-Attn inside bottleneck (clean lower route avoiding green label)
    _arrow(ax, (18.0, 24.0), (41.0, 32.2), C_MCTS, lw=1.6, rad=-0.06, label="Tokens de Carga K_L", lpos=(29.5, 25.0))

    # Bottom Conditioning Rail routed cleanly around right margin (x=104.5)
    rail_y = 9.2
    ax.plot([18.0, 84.0], [rail_y, rail_y], color=C_PHYS, lw=1.6, ls="-.", zorder=2)
    ax.plot([84.0, 104.5], [2.2, 2.2], color=C_PHYS, lw=1.6, ls="-.", zorder=2)
    ax.plot([84.0, 84.0], [2.2, rail_y], color=C_PHYS, lw=1.6, ls="-.", zorder=2)
    ax.plot([104.5, 104.5], [2.2, 36.8], color=C_PHYS, lw=1.6, ls="-.", zorder=2)
    _arrow(ax, (104.5, 36.8), (101.5, 36.8), C_PHYS, lw=1.6, ls="-.")
    ax.text(104.8, 22.0, "AdaLN\nfase/vol", ha="left", va="center", fontsize=7.8, fontweight="bold", color=C_PHYS)

    _arrow(ax, (52.7, rail_y), (52.7, 29.0), C_PHYS, lw=1.5, ls="-.", label="AdaLN(phys)", lpos=(52.7, 19.0))
    _arrow(ax, (66.5, rail_y), (68.0, 48.0), C_PHYS, lw=1.3, ls="-.", rad=-0.18)
    ax.text(36.0, 10.8, "Barramento global: phys_stats + (phase, vol_frac)", ha="center", fontsize=8.0, fontweight="bold", color=C_PHYS,
            bbox=dict(boxstyle="round,pad=0.16", fc="#ffffff", ec="none"))

    _save(fig, "arch_03_macro_architecture.png")


def plot_arch_04_internal_mechanisms() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.4))

    # Panel A: 1 Transformer Block Anatomy
    ax = axes[0]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("A. Bloco Transformer Físico 3D\n(Modulado por Física + RoPE + Multi-Load)", fontsize=11.3, fontweight="bold", pad=8)

    ax.add_patch(FancyBboxPatch((8, 6), 84, 88, boxstyle="round,pad=0.4,rounding_size=2.0",
                                facecolor="#f8fafc", edgecolor=C_TRANS, lw=1.5, ls="--"))
    _box(ax, 22, 85, 56, 7.0, "Tokens x_tok (B, 256, 192)", "", fc=C_LIGHT, ec=C_SLATE, title_fs=8.8)
    _box(ax, 18, 70, 64, 9.5, "1. AdaLN(phys) + Masked Global MHSA", "3D RoPE (Δx, Δy, Δz); ignora keep-out", fc=C_TRANS_BG, ec=C_TRANS, title_fs=8.8, body_fs=7.6)
    _box(ax, 18, 52, 64, 9.5, "2. Multi-Load Cross-Attention", "Q = tokens | K,V = K casos de carga [fx,fy,fz]", fc=C_MCTS_BG, ec=C_MCTS, title_fs=8.8, body_fs=7.6)
    _box(ax, 18, 34, 64, 9.5, "3. AdaLN(phys) + SwiGLU FFN", "Mistura não-linear por voxel latente", fc=C_PHYS_BG, ec=C_PHYS, title_fs=8.8, body_fs=7.6)
    _box(ax, 22, 12, 56, 8.5, "Saída p/ Decoder & Value", "Acoplamento global desde a 1ª camada!", fc=C_CONV_BG, ec=C_CONV, title_fs=8.8, body_fs=7.6)

    for y0, y1 in [(85, 79.8), (70, 61.8), (52, 43.8), (34, 20.8)]:
        _arrow(ax, (50, y0), (50, y1), C_SLATE, lw=1.4)
    for ya, yb in [(82.5, 65.5), (65.5, 47.5), (47.5, 27.0)]:
        _arrow(ax, (18, ya), (18, yb), "#64748b", lw=1.2, ls=":", rad=0.55)
        ax.text(11.5, 0.5 * (ya + yb), "+", ha="center", va="center", fontsize=11, fontweight="bold", color="#475569")

    # Panel B: Curing ep30 empirical bottlenecks
    ax = axes[1]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("B. Cura Direta das Falhas Medidas no ep30\n(De Imitação Travada → Escultura Ativa)", fontsize=11.3, fontweight="bold", pad=8)

    _box(ax, 4, 54, 92, 40,
         "Gargalos Empíricos em v3.3 (ep30)",
         "• Value 1-escalar em tanh → posto 16,7%, viés OOD +0,40\n"
         "• Policy cega à fase → +8,9 logits pró-ADD (99,93% ADD)\n"
         "• Blueprint Head só existe com SIMP (sem sinal no Self-Play)\n"
         "• Canal fz morto no warmup → 27,3% ruído em raw_phys_proj",
         fc=C_WARN_BG, ec=C_WARN, lw=1.4, title_fs=9.5, body_fs=8.0)

    _arrow(ax, (50, 53.5), (50, 44.5), C_VAL, lw=2.4, label="Upgrade + Etapa 6", lpos=(67, 49.0))

    _box(ax, 4, 4, 92, 40,
         "Soluções Estruturais na v4 (Phys-UViT 3D)",
         "• Value Fatorada: prevê log(C), P(conn) e V(s) separados\n"
         "• Cabeça Física SED + u: FEniCSx dá SED real no Self-Play;\n"
         "  voxels com SED ≈ 0 sinalizam diretamente onde REMOVER\n"
         "• Policy Head c/ AdaLN(fase, vol) + treino c/ cargas 3D fz ≠ 0",
         fc=C_VAL_BG, ec=C_VAL, lw=1.6, title_fs=9.5, body_fs=8.0)

    # Panel C: Non-rectangular arbitrary domain + multi-load + 3D RoPE
    ax = axes[2]
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("C. Domínios & Condições Arbitrárias\n(Máscara Ternária + 3D RoPE + Multi-Carga)", fontsize=11.3, fontweight="bold", pad=8)

    ax.add_patch(Rectangle((8, 16), 74, 72, facecolor="#f1f5f9", edgecolor="#94a3b8", lw=1.1, ls="--"))
    ax.add_patch(Rectangle((42, 52), 40, 36, facecolor="#fee2e2", edgecolor=C_WARN, hatch="///", lw=1.2))
    ax.text(62, 70, "Fora do Domínio\n(mask = 0 na MHSA)", ha="center", va="center", fontsize=8.0, fontweight="bold", color=C_WARN,
            bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff", ec=C_WARN, lw=0.7))

    l_poly = Polygon([[8, 16], [82, 16], [82, 52], [42, 52], [42, 88], [8, 88]],
                     facecolor="#ecfeff", edgecolor=C_CONV, lw=2.0, alpha=0.75, zorder=2)
    ax.add_patch(l_poly)

    hole = Circle((44, 34), 9.2, facecolor="#fecaca", edgecolor=C_WARN, lw=1.6, hatch="xxx", zorder=3)
    ax.add_patch(hole)
    ax.text(44, 34, "keep-out\n(furo)", ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_WARN,
            bbox=dict(boxstyle="round,pad=0.12", fc="#ffffff", ec="none", alpha=0.9), zorder=4)

    ax.add_patch(Rectangle((8, 74), 10, 14, facecolor="#bbf7d0", edgecolor=C_VAL, lw=1.6, zorder=3))
    ax.text(13, 81, "keep-in\n+ apoio", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_VAL, zorder=4)

    _arrow(ax, (84, 42), (72, 42), C_MCTS, lw=2.5, zorder=5)
    ax.text(85.5, 42, "Carga F₁", ha="left", va="center", fontsize=8.4, fontweight="bold", color=C_MCTS)
    _arrow(ax, (64, 15), (64, 25), C_PHYS, lw=2.5, zorder=5)
    ax.text(64, 10.5, "Carga F₂ (caso 2)", ha="center", fontsize=8.4, fontweight="bold", color=C_PHYS)

    ax.scatter([18, 68], [62, 34], color=C_TRANS, s=45, zorder=5)
    _arrow(ax, (18, 62), (68, 34), C_TRANS, lw=1.8, style="<|-|>", rad=-0.18,
           label="3D RoPE(Δx, Δy, Δz)", lpos=(47, 56), lfs=8.0, zorder=5)

    ax.text(48, 3.8, "Sem tamanho fixo nem engaste único em x=0", ha="center", fontsize=8.4, fontweight="bold", color=C_SLATE)

    fig.tight_layout(w_pad=1.6)
    _save(fig, "arch_04_internal_mechanisms.png")


def plot_arch_05_selfplay_roadmap() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 6.0))

    # Panel A: Closed-Loop AlphaZero-Structural
    ax = axes[0]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("A. O Motor que Supera o Professor: Loop Fechado Self-Play", fontsize=11.5, fontweight="bold", pad=8)

    _box(ax, 34, 78, 32, 13, "1. Estado Estrutural s_t", "Geometria ρ, máscaras keep-in/out,\nsuportes e conjunto K cargas", fc=C_LIGHT, ec=C_SLATE)
    _box(ax, 66, 48, 31, 15, "2. Rede Phys-UViT 3D", "Prior Policy (ADD/REM)\nValue fatorada + campo SED", fc=C_TRANS_BG, ec=C_TRANS)
    _box(ax, 54, 10, 40, 16, "3. Busca MCTS (PUCT)", "Estágio 1 (PBRS) → Estágio 2 (Top-K\nbalanceado p/ destravar REMOVE)", fc=C_MCTS_BG, ec=C_MCTS)
    _box(ax, 6, 10, 40, 16, "4. Verificador FEniCSx (Verdade)", "Resolve PDE elástica real (u, C, SED)\n+ análise topológica de ilhas", fc=C_VAL_BG, ec=C_VAL)
    _box(ax, 3, 48, 31, 15, "5. Replay + Treino RL", "Alvos reais: (π_MCTS, V_FEM, SED)\nElimina viés OOD + teto do SIMP", fc=C_PHYS_BG, ec=C_PHYS)

    _arrow(ax, (66, 84), (82, 63), C_SLATE, lw=1.8, rad=-0.18)
    _arrow(ax, (82, 48), (76, 26), C_TRANS, lw=1.8, rad=-0.12)
    _arrow(ax, (54, 18), (46, 18), C_MCTS, lw=2.0)
    _arrow(ax, (18, 26), (18, 48), C_VAL, lw=1.8)
    _arrow(ax, (18, 63), (34, 84), C_PHYS, lw=1.8, rad=-0.18)

    ax.text(50, 46, "Ciclo Autônomo:\nRede propõe →\nFEniCSx verifica →\nRede aprende do erro real",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=C_SLATE,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f8fafc", ec=C_BORDER, lw=1.0))

    # Panel B: Evolutionary Roadmap (TCC -> v4 -> Industrial CAD)
    ax = axes[1]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("B. Curva de Progresso: Do TCC Hoje à Escala Industrial", fontsize=11.5, fontweight="bold", pad=8)

    _box(ax, 4, 6, 54, 23,
         "Degrau 1 · TCC Agora (Prova da Hipótese)",
         "SimpleBackbone (4,43M) + Etapa 6 Self-Play\n"
         "• Grade (64×32×8) · 3,5–5 s/passo local · 0% zero-padding\n"
         "• Meta: quebrar o teto do ep30, destravar REMOVE,\n"
         "  abrir vazios (hollow) e provar aprendizado via FEM",
         fc=C_CONV_BG, ec=C_CONV, lw=1.8, title_fs=9.3, body_fs=7.9)

    _box(ax, 22, 37, 56, 24,
         "Degrau 2 · AlphaBuilder v4 (BCs & Física Gerais)",
         "Phys-UViT 3D (Conv3D + Global 3D-RoPE + SED)\n"
         "• Keep-in / keep-out ternário + Multi-Load Cross-Attn\n"
         "• Cabeça SED guiando remoções sem oráculo SIMP\n"
         "• Física onde SIMP sofre: flambagem, tensão VM, manufatura",
         fc=C_TRANS_BG, ec=C_TRANS, lw=1.8, title_fs=9.3, body_fs=7.9)

    _box(ax, 40, 69, 56, 24,
         "Degrau 3 · Escala Industrial CAD (>10⁶ elementos)",
         "Octree Esparso + Transolver / MeshGraphNets\n"
         "• Operadores em nuvem/malha fora da grade densa\n"
         "• MCTS Hierárquico (Coarse-to-Fine / primitivas SDF)\n"
         "• Amortização instantânea + refinamento fino guiado",
         fc=C_PHYS_BG, ec=C_PHYS, lw=1.8, title_fs=9.3, body_fs=7.9)

    _arrow(ax, (32, 29.2), (42, 36.8), C_CONV, lw=2.2)
    _arrow(ax, (50, 61.2), (60, 68.8), C_TRANS, lw=2.2)

    fig.tight_layout(w_pad=1.8)
    _save(fig, "arch_05_mcts_selfplay_roadmap.png")


def main() -> None:
    _setup()
    plot_arch_01_simple_backbone()
    plot_arch_02_first_principles_vs_swin()
    plot_arch_03_macro_architecture()
    plot_arch_04_internal_mechanisms()
    plot_arch_05_selfplay_roadmap()


if __name__ == "__main__":
    main()
