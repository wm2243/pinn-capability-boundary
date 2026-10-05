# -*- coding: utf-8 -*-
"""Regenerate the English framework figure (fig 1) with properly sized text.

The shipped Chinese figure (figures/zh/fig_framework.*) is kept untouched.
Outputs:
  figures/en/fig_framework.pdf / .png        (canonical archive copy)
  paper/figures/fig_framework_en.pdf / .png  (copy used by paper1_v1_20_en.tex)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]

BOX_EDGE = "#4a4a4a"
BROWN = "#8a6d3b"

def box(ax, x, y, w, h, fc, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=BOX_EDGE, lw=lw, mutation_aspect=0.5))

def main():
    fig, ax = plt.subplots(figsize=(13.2, 7.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- top box ----
    box(ax, 0.17, 0.875, 0.66, 0.105, "#eef2f7")
    ax.text(0.50, 0.927, r"Spatial diagonal weight   $W=\mathrm{diag}(w_1,\dots,w_N)$,   $0<w_0\leq w_i\leq w_1$",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color="#1a1a2e")

    # ---- three channel boxes ----
    ytop, ybot = 0.30, 0.70
    xs = [(0.025, 0.315), (0.355, 0.645), (0.685, 0.975)]
    fcs = ["#e3edf7", "#f6efdd", "#e7f0e3"]
    titles = ["Channel $\\mathcal{S}$ | zeroth-order: solution / fit",
              "Level $R$ | frozen linearization\n(not an independent channel)",
              "Channel $\\theta$ | first-order: algorithm"]
    bodies = ["zero-residual set, global minimizer,\nresidual\u2013error bridge;\nindependent of $W$ for $w>0$",
              "$H_{\\mathrm{lin}}=J^{\\top}WJ$ and its spectrum;\nacts only indirectly via\nmulti-step gradient evolution",
              "$g_w=J^{\\top}Wr$, iteration trajectory,\nstationary points and\nbasins of attraction"]
    for (x0, x1), fc, tt, bd in zip(xs, fcs, titles, bodies):
        box(ax, x0, ytop, x1 - x0, ybot - ytop, fc)
        ax.text((x0 + x1) / 2, ybot - 0.075, tt, ha="center", va="top", fontsize=12.5,
                fontweight="bold", color="#1a1a2e")
        ax.text((x0 + x1) / 2, ybot - 0.24, bd, ha="center", va="top", fontsize=11.5, color="#333")

    # ---- arrows top -> boxes ----
    arrow_x = [0.17, 0.50, 0.83]
    arrow_lab = ["solution-preserving;\nstability constant unchanged",
                 "frozen spectrum: directions\nunreachable, scale bounded",
                 "the only route into\nthe one-step update"]
    for xx, lab in zip(arrow_x, arrow_lab):
        ax.annotate("", xy=(xx, ybot + 0.012), xytext=(xx, 0.868),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.6))
        ax.text(xx, 0.845, lab, ha="center", va="top", fontsize=10.5, color="#333")

    # ---- dashed brown arrows between R and theta boxes ----
    ax.add_patch(FancyArrowPatch((0.645, 0.660), (0.685, 0.700), connectionstyle="arc3,rad=-0.25",
                                 arrowstyle="-|>", color=BROWN, lw=1.8, ls="--"))
    ax.add_patch(FancyArrowPatch((0.685, 0.620), (0.645, 0.600), connectionstyle="arc3,rad=-0.25",
                                 arrowstyle="-|>", color=BROWN, lw=1.8, ls="--"))

    # ---- brown caption ----
    ax.text(0.50, 0.245, "Level $R$ acts on channel $\\theta$ only indirectly: within frozen training windows, via multi-step $g_w$ evolution (dashed arrows)",
            ha="center", va="center", fontsize=10.5, color=BROWN)

    # ---- bottom box ----
    box(ax, 0.025, 0.02, 0.95, 0.185, "#f2f2f2")
    ax.text(0.50, 0.152, "Controlled remainder $E_{\\mathrm{rem}}$ (not a fourth channel)",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#1a1a2e")
    ax.text(0.50, 0.095, "residual curvature $H_{nl}=\\sum_i w_i r_i\\nabla^2 r_i$,  3rd+-order Taylor terms,  in-window curvature drift,\npreconditioner variation,  exogenous jumps",
            ha="center", va="center", fontsize=11, color="#333")
    ax.text(0.50, 0.043, "\u2014\u2014 only explicit, controlled upper bounds",
            ha="center", va="center", fontsize=11, color="#555")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    outs = [ROOT / "figures" / "en" / "fig_framework",
            ROOT / "paper" / "figures" / "fig_framework_en"]
    for o in outs:
        fig.savefig(str(o) + ".pdf")
        fig.savefig(str(o) + ".png", dpi=200)
        print("saved", o)
    plt.close(fig)

if __name__ == "__main__":
    main()
