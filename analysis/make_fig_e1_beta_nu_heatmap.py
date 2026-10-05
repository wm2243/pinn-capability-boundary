# -*- coding: utf-8 -*-
"""重画 E1 图：左 panel 为格内中位 L2 热力图，右 panel 为逐种子最优 beta* 分布。

输出中文/英文双版，并同步复制到 paper/figures/ 供 LaTeX 使用。
"""
from pathlib import Path
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "aggregated" / "v7_v1_exp_E1_beta_nu"
FIG_ROOT = ROOT / "figures"
PAPER_FIG = ROOT / "paper" / "figures"

for f in ["Microsoft YaHei", "SimHei"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt_beta(b):
    return f"{b:.0f}" if float(b).is_integer() else f"{b:g}"


def plot(lang):
    agg = load_csv(DATA / "e1_agg.csv")
    runs = load_csv(DATA / "e1_runs.csv")

    nus = sorted({float(r["nu"]) for r in agg})
    # 左图只画四个粘度公共的 6 档 beta；nu=.005 独有的加密档 {2,3,7,20,40,100} 见附录全量表
    betas = [1.0, 5.0, 10.0, 15.0, 30.0, 60.0]
    data = np.full((len(nus), len(betas)), np.nan)
    for r in agg:
        nu = float(r["nu"])
        beta = float(r["beta"])
        if beta not in betas:
            continue
        i = nus.index(nu)
        j = betas.index(beta)
        # nu=.001、beta>=5 的塌缩失败态按实值显示（中位≈0.573，色标上限内呈深红）
        data[i, j] = min(float(r["L2_med"]), 0.6)  # 上限截断便于显示

    # 逐种子最优 beta*
    best = {nu: [] for nu in nus}
    by_key = {}
    for r in runs:
        nu = float(r["nu"])
        seed = int(float(r["seed"]))
        beta = float(r["beta"])
        l2 = float(r["L2"])
        key = (nu, seed)
        if key not in by_key or l2 < by_key[key][1]:
            by_key[key] = (beta, l2)
    for (nu, seed), (beta, l2) in sorted(by_key.items()):
        if nu in best:
            best[nu].append((seed, beta))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.14, wspace=0.30)

    if lang == "en":
        suptitle = "E1: median heatmap and per-seed optimal strength"
        t1 = "(a) Cell-median terminal $L_2$ (central tendency only)"
        t2 = "(b) Per-seed optimal $\\beta^*$: dispersed and non-monotone"
        xlabel = "weight strength $\\beta$"
        ylabel = "viscosity $\\nu$"
        cbar_label = "$L_2$ (median, capped at 0.6)"
        y2 = "per-seed optimal $\\beta^*$ (log scale)"
        median_label = "median"
    else:
        suptitle = "E1：中位热力图与逐种子最优强度分布"
        t1 = "(a) 格内中位终态 $L_2$（仅为中心趋势）"
        t2 = "(b) 逐种子最优 $\\beta^*$：散布大且非单调"
        xlabel = "加权强度 $\\beta$"
        ylabel = "粘性 $\\nu$"
        cbar_label = "$L_2$（中位，上限 0.6）"
        y2 = "逐种子最优 $\\beta^*$（对数刻度）"
        median_label = "中位数"

    fig.suptitle(suptitle, fontsize=11.5, fontweight="bold", y=0.97)

    im = ax1.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.6, origin="lower")
    ax1.set_xticks(range(len(betas)))
    ax1.set_xticklabels([fmt_beta(b) for b in betas], fontsize=8)
    ax1.set_yticks(range(len(nus)))
    ax1.set_yticklabels([f"{n:.3f}" for n in nus])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(t1, fontsize=10)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.025)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    for i in range(len(nus)):
        for j in range(len(betas)):
            if not np.isnan(data[i, j]):
                v = data[i, j]
                color = "white" if v > 0.35 else "black"
                ax1.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.3, color=color)

    rng = np.random.default_rng(20261004)
    medians = []
    for i, nu in enumerate(nus):
        vals = sorted(best.get(nu, []), key=lambda t: t[0])
        bs = np.array([b for _, b in vals], dtype=float)
        medians.append(float(np.median(bs)))
        jit = np.linspace(-0.20, 0.20, len(bs)) if len(bs) > 1 else np.array([0.0])
        # 相同 beta 的点再按种子稳定错开，避免完全重叠
        ax2.scatter(np.full(len(bs), i) + jit, bs, s=28, color="#4C72B0", alpha=0.78,
                    edgecolor="white", linewidth=0.45, zorder=3)
        ax2.hlines(np.median(bs), i - 0.28, i + 0.28, color="#C44E52", lw=2.0, zorder=4)
        ax2.text(i + 0.31, np.median(bs), f"{fmt_beta(np.median(bs))}", va="center", ha="left",
                 fontsize=8, color="#C44E52")

    ax2.set_xticks(range(len(nus)))
    ax2.set_xticklabels([f"{n:.3f}" for n in nus])
    ax2.set_yscale("log")
    yticks = [1, 2, 5, 10, 20, 50, 100]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([str(y) for y in yticks])
    ax2.set_ylim(0.75, 125)
    ax2.set_xlim(-0.55, len(nus) - 0.25)
    ax2.set_xlabel(ylabel)
    ax2.set_ylabel(y2)
    ax2.set_title(t2, fontsize=10)
    ax2.grid(True, axis="y", alpha=0.28, lw=0.5, which="major")
    ax2.plot([], [], color="#C44E52", lw=2.0, label=median_label)
    ax2.legend(loc="upper left", fontsize=8, frameon=False)

    suffix = "_en" if lang == "en" else "_zh"
    name = f"fig_E1_beta_nu_heatmap{suffix}"
    outdir = FIG_ROOT / lang
    outdir.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        out = outdir / f"{name}.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        shutil.copy2(out, PAPER_FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {name} -> {outdir} and {PAPER_FIG}; medians={medians}")


if __name__ == "__main__":
    plot("zh")
    plot("en")
