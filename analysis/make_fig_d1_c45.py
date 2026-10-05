# -*- coding: utf-8 -*-
"""第七章新图：
  fig_d1_caseb_{zh,en}   —— §7.3 双面板：(a) D1 rational 族 β 扫描（新数据，n=11/档）；
                                       (b) caseB 人为坏盆地各干预（对照/减权/增权/裁剪）。
  fig_c45_advdiff_{zh,en} —— §7.4 单面板：C4/C5 二维线性对流—扩散 33 run 逐种子散点。

输出中文/英文双版（png+pdf），同步复制到 figures/{zh,en}/ 与 paper/figures/。
"""
from pathlib import Path
import csv
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "data" / "aggregated"
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

TAU = 0.06
C_PT = "#4C72B0"
C_MED = "#C44E52"


def jitter(n, width=0.22, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, n)


def strip(ax, xs, ys, color=C_PT):
    ax.scatter(xs, ys, s=26, color=color, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
    ax.hlines(np.median(ys), xs.mean() - 0.28, xs.mean() + 0.28, color=C_MED, lw=2.0, zorder=4)


def save(fig, name, lang):
    suffix = "_en" if lang == "en" else "_zh"
    outdir = FIG_ROOT / lang
    outdir.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        out = outdir / f"{name}{suffix}.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        shutil.copy2(out, PAPER_FIG / f"{name}{suffix}.{ext}")
    plt.close(fig)
    print(f"saved {name}{suffix} -> {outdir} and {PAPER_FIG}")


# ---------------- 数据 ----------------
def load_d1r():
    with open(AGG / "v8_time_varying_d1r" / "d1r_runs.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out.setdefault(float(r["beta"]), []).append(float(r["L2_total"]))
    return {b: np.array(v) for b, v in sorted(out.items())}


def load_caseb():
    """caseB/s8 人为坏盆地（空间线性族，canonical 代 summary_11..18）"""
    base = AGG / "v8_time_varying"
    def rd(name):
        with open(base / name, encoding="utf-8-sig") as f:
            return np.array([float(r["L2_relative"]) for r in csv.DictReader(f)])
    return [
        ("ctrl",  rd("summary_11.csv")),
        ("down3", rd("summary_12.csv")),
        ("down8", rd("summary_13.csv")),
        ("up3",   rd("summary_14.csv")),
        ("up8",   rd("summary_15.csv")),
        ("clip01", rd("summary_16.csv")),
        ("clip1",  rd("summary_17.csv")),
    ]


def load_c45():
    d = json.load(open(AGG / "v8_burgers_2d" / "summary_2.json"))
    out = {}
    for rec in d:
        out[rec["beta"]] = np.array([r["L2_total"] for r in rec["results"]])
    return dict(sorted(out.items()))


# ---------------- 图 1：D1 + caseB ----------------
def plot_d1(lang):
    d1 = load_d1r()
    cb = load_caseb()
    zh = lang == "zh"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.0, 1.35]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.87, bottom=0.13, wspace=0.30)

    # (a) D1 rational β 扫描
    betas = list(d1.keys())
    for i, b in enumerate(betas):
        strip(ax1, i + jitter(len(d1[b]), seed=i), d1[b])
    ax1.axhline(TAU, color="gray", ls="--", lw=1.2, zorder=1)
    ax1.text(3.42, TAU * 0.72, f"τ={TAU}", fontsize=8, color="gray", ha="right")
    ax1.set_yscale("log")
    ax1.set_ylim(2e-3, 0.6)
    ax1.set_xticks(range(len(betas)))
    ax1.set_xticklabels([f"{b:g}" for b in betas])
    ax1.set_xlabel(r"$\beta$" if not zh else r"增权强度 $\beta$")
    ax1.set_ylabel(r"terminal relative $L_2$" if not zh else r"时空全域相对 $L_2$")
    ax1.set_title(("(a) D1 time-dependent Burgers, rational family: near-neutral"
                  if not zh else "(a) D1 时变 Burgers（有理族）：各档无显著差异"), fontsize=10)
    ax1.grid(True, axis="y", alpha=0.28, lw=0.5, which="major")

    # (b) caseB 人为坏盆地
    labels_zh = {"ctrl": "对照", "down3": r"减权 $1/3$", "down8": r"减权 $1/8$",
                 "up3": r"增权 $3$", "up8": r"增权 $8$",
                 "clip01": "裁剪 .01", "clip1": "裁剪 .1"}
    labels_en = {"ctrl": "control", "down3": r"down $1/3$", "down8": r"down $1/8$",
                 "up3": r"up $3$", "up8": r"up $8$",
                 "clip01": "clip .01", "clip1": "clip .1"}
    labels = labels_zh if zh else labels_en
    for i, (k, v) in enumerate(cb):
        strip(ax2, i + jitter(len(v), seed=10 + i), v)
    ax2.set_xticks(range(len(cb)))
    ax2.set_xticklabels([labels[k] for k, _ in cb], fontsize=8.5)
    ax2.set_ylim(2.2, 3.05)
    note = ("all interventions stay in the bad basin (τ=0.06 far below)"
            if not zh else "所有干预均留在坏盆地（τ=0.06 远低于此范围）")
    ax2.text(0.02, 0.035, note, transform=ax2.transAxes, fontsize=8.5, color="gray")
    ax2.set_ylabel(r"terminal relative $L_2$" if not zh else r"终态全域相对 $L_2$")
    ax2.set_title(("(b) caseB artificial bad basin: no rescue by down-weighting or clipping"
                   if not zh else "(b) caseB 人为坏盆地：减权与裁剪均救不出"), fontsize=10)
    ax2.grid(True, axis="y", alpha=0.28, lw=0.5)

    save(fig, "fig_d1_caseb", lang)


# ---------------- 图 2：C4/C5 ----------------
def plot_c45(lang):
    d = load_c45()
    zh = lang == "zh"
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.13)

    betas = list(d.keys())
    for i, b in enumerate(betas):
        strip(ax, i + jitter(len(d[b]), seed=20 + i), d[b])
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_ylim(0.038, 0.068)
    ax.set_xlabel(r"$\beta$" if not zh else r"增权强度 $\beta$")
    ax.set_ylabel(r"terminal relative $L_2$" if not zh else r"时空全域相对 $L_2$")
    ax.set_title(("C4/C5 2-D linear advection–diffusion: all 33 runs in good basins"
                  if not zh else "C4/C5 二维线性对流—扩散：33 个 run 全部进入好盆地"),
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.28, lw=0.5)
    med_note = (f"median {np.median(d[1.0]):.4f} → {np.median(d[8.0]):.4f}"
                if not zh else f"中位 {np.median(d[1.0]):.4f} → {np.median(d[8.0]):.4f}")
    ax.text(0.03, 0.94, med_note, transform=ax.transAxes, fontsize=8.5, color=C_MED)

    save(fig, "fig_c45_advdiff", lang)


if __name__ == "__main__":
    for lang in ["zh", "en"]:
        plot_d1(lang)
        plot_c45(lang)
