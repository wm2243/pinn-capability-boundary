# -*- coding: utf-8 -*-
"""fig 10: internal-singularity closure panel (S2c/S2d/S2e), zh + en.

Adapted from the original dev script to read the release aggregated CSVs
(data/aggregated/...). Changes vs the original: x tick labels of panels (a)/(b)
are rotated to fix the overlap; jitter is seeded for reproducibility.

Outputs (both languages):
  figures/{zh,en}/fig_s2_singular_panel*.png
  paper/figures/fig_s2_singular_panel*.png   (copies used by the tex files)
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "data" / "aggregated"
FONT_ZH = r"C:\Windows\Fonts\msyh.ttc"

OK, BAD, NEU, UP, DOWN = "#3A8A18", "#C0504D", "#9A9A9A", "#E8A24B", "#4B86B4"

def read_csv(rel):
    with open(AGG / rel, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def f(x):
    try: return float(x)
    except: return np.nan

# ---------- load ----------
s2d = read_csv("v7_v2_exp_V2_S2d_internal_layer/s2d_runs.csv")
s2e_base = read_csv("v7_v2_exp_V2_S2e_spike/s2e_runs.csv")
s2e_seed = read_csv("v7_v2_exp_V2_S2e_spike/diag_seed.csv")
basin = read_csv("v7_v2_exp_V2_S2c_aligned_twostage/diag_basin_hit.csv")
stretch = read_csv("v7_v2_exp_V2_S2c_aligned_twostage/diag_stretch.csv")
lift = read_csv("v7_v2_exp_V2_S2c_aligned_twostage/diag_lift.csv")

def group(rows, key):
    g = {}
    for r in rows: g.setdefault(r[key], []).append(r)
    return g
g_st, g_lf = group(stretch, "config"), group(lift, "config")
def vals(rows, col="best_L2"):
    return np.array([f(r[col]) for r in rows], float)
bmap = {r["config"]: r for r in basin}

A_series = [
    ("bur",   None, "bur nu=.005 sig15 K8"),
    ("pos",   None, "pos eps=.01 sig15 K8"),
    ("B",     vals(g_st["B_identity_eps01_sig3"]), None),
    ("A",     vals(g_st["A_stretch_eps01_sig3"]), None),
    ("C",     vals(g_st["C_stretch_eps010_sig3"]), None),
    ("F",     vals(g_lf["F_logistic_eps01_sig15"]), None),
    ("E",     vals(g_lf["E_exp_eps01_sig15"]), None),
]
seed_order = [("sech","sech"),("gauss_low","gauss_low"),("gauss_std","gauss_std"),
              ("gauss_high","gauss_high"),("gauss_wide","gauss_wide"),("gauss_shift","gauss_shift")]
smap = {r["name"]: r for r in s2e_seed}
noseed = next(r for r in s2e_base if abs(f(r["eps"])-0.0044) < 1e-9)
d_eps = [f(r["eps"]) for r in s2d]
d_glob = [f(r["L2"]) for r in s2d]
d_layer = [f(r["L2_layer"]) for r in s2d]

T = {
 "zh": dict(
   ta="(a) 贴边对流—扩散层（S1 边界编码，层在 x=1）",
   tb="(b) 内部 Allen–Cahn 尖峰（S3 拓扑种子，ε=.0044）",
   tc="(c) 内部转向点层（S2 坐标/测度，erf 内层）",
   ya="最佳全域 $L_2$ 相对误差", yb="全域 $L_2$ 相对误差", yc="选中解 $L_2$ 相对误差",
   xc="层厚参数 ε",
   thresh="命中阈值 $L_2=.06$",
   lab=dict(bur="Burgers\n可精修(K8)", pos="贴边\n无干预(K8)",
            B="恒等\n(薄ε=.01)", A="坐标拉伸\n(薄ε=.01)", C="坐标拉伸\n(厚ε=.1)",
            F="粗logistic\n提升", E="精确指数\n提升"),
   seed=dict(sech="精确sech\n种子", gauss_low="粗高斯\n幅.8", gauss_std="粗高斯\n标准",
             gauss_high="粗高斯\n幅2", gauss_wide="粗高斯\n加宽3ε", gauss_shift="错位\nx0=.15"),
   noseed="无种子\n(K4)", glob="全域", layer="层内",
   note_a="n=4/组（K8 对照组 n=8，横杠为中位、左端为最优）",
   note_b="柱高为 4 起点聚合误差；绿=4/4 成峰，红=4/4 失败",
   note_c="多起点均命中，报告选中解；β=1、K4 两阶段",
 ),
 "en": dict(
   ta="(a) Wall advection-diffusion layer (S1 edge encoding, layer at x=1)",
   tb="(b) Interior Allen-Cahn spike (S3 topological seed, eps=.0044)",
   tc="(c) Interior turning-point layer (S2 coordinate/measure, erf inner layer)",
   ya="best global relative $L_2$", yb="global relative $L_2$", yc="selected-run relative $L_2$",
   xc="layer parameter eps",
   thresh="hit threshold $L_2=.06$",
   lab=dict(bur="Burgers\nrefinable(K8)", pos="wall\nno fix (K8)",
            B="identity\n(thin .01)", A="stretch\n(thin .01)", C="stretch\n(thick .1)",
            F="coarse logistic\nlift", E="exact exp\nlift"),
   seed=dict(sech="exact sech\nseed", gauss_low="coarse G\namp .8", gauss_std="coarse G\nstd",
             gauss_high="coarse G\namp 2", gauss_wide="coarse G\nwide 3eps", gauss_shift="shifted\nx0=.15"),
   noseed="no seed\n(K4)", glob="global", layer="in-layer",
   note_a="n=4/group (K8 controls n=8; bar=median, left tip=best)",
   note_b="bar=aggregate over 4 starts; green=4/4 peaked, red=4/4 failed",
   note_c="all multistarts hit; selected run shown; beta=1, K4 two-stage",
 ),
}

def make(lang):
    rng = np.random.default_rng(0)
    if lang == "zh":
        fp = fm.FontProperties(fname=FONT_ZH); plt.rcParams["axes.unicode_minus"] = False
    else:
        fp = fm.FontProperties(family="DejaVu Sans")
    t = T[lang]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2))
    # ---- (a) ----
    a = ax[0]; a.set_yscale("log"); a.axhline(.06, color=NEU, ls="--", lw=1.2)
    a.text(0.02, .075, t["thresh"], fontproperties=fp, fontsize=8.5, color="#555")
    for i,(key,vals4,bcfg) in enumerate(A_series):
        if vals4 is not None:
            col = OK if np.nanmedian(vals4) < .06 else BAD
            a.scatter(np.full(len(vals4), i)+rng.uniform(-.07,.07,len(vals4)),
                      vals4, color=col, s=26, alpha=.8, zorder=3)
            a.hlines(np.nanmedian(vals4), i-.28, i+.28, color=col, lw=2.4, zorder=2)
        else:
            r = bmap[bcfg]; mn, md = f(r["min"]), f(r["median"])
            if md < .06: col = OK
            elif mn < .06: col = UP
            else: col = BAD
            a.hlines(md, i-.28, i+.28, color=col, lw=2.4)
            if mn < .06:
                a.plot(i, mn, marker="D", ms=6, color=OK, zorder=4)
            else:
                a.plot([i-.28],[mn], marker="|", ms=10, color=col)
            a.scatter([i],[md], color=col, s=30, marker="s", zorder=3)
    a.set_xticks(range(len(A_series)))
    a.set_xticklabels([t["lab"][k] for k,_,_ in A_series], fontproperties=fp, fontsize=8,
                      rotation=28, ha="right", rotation_mode="anchor")
    a.set_ylabel(t["ya"], fontproperties=fp, fontsize=10); a.set_title(t["ta"], fontproperties=fp, fontsize=10)
    a.text(.01,-.40,t["note_a"],transform=a.transAxes,fontproperties=fp,fontsize=8,color="#555")
    a.grid(alpha=.25, which="both")
    # ---- (b) ----
    b = ax[1]; b.set_yscale("log"); b.axhline(.06, color=NEU, ls="--", lw=1.2)
    bx = list(range(7)); bvals=[]; bcols=[]; blab=[t["noseed"]]+[t["seed"][k] for k,_ in seed_order]
    bvals.append(f(noseed["L2"])); bcols.append(BAD)
    for _,k in seed_order:
        r = smap[k]; bvals.append(f(r["L2"])); bcols.append(OK if int(float(r["n_peak"]))==4 else BAD)
    b.bar(bx, bvals, color=bcols, alpha=.82, width=.62)
    for i,v in enumerate(bvals): b.text(i, v*1.25, ("%.1e"%v), ha="center", fontsize=7.6,
                                        fontproperties=fp, rotation=0)
    b.set_xticks(bx)
    b.set_xticklabels(blab, fontproperties=fp, fontsize=7.6, rotation=28, ha="right",
                      rotation_mode="anchor")
    b.set_ylabel(t["yb"], fontproperties=fp, fontsize=10); b.set_title(t["tb"], fontproperties=fp, fontsize=10)
    b.text(.01,-.40,t["note_b"],transform=b.transAxes,fontproperties=fp,fontsize=8,color="#555")
    b.grid(alpha=.25, axis="y", which="both")
    # ---- (c) ----
    c = ax[2]
    x = np.arange(len(d_eps)); w=.36
    c.bar(x-w/2, d_glob, w, color=DOWN, label=t["glob"], alpha=.9)
    c.bar(x+w/2, d_layer, w, color=UP, label=t["layer"], alpha=.9)
    ymax = max(max(d_glob), max(d_layer))
    for i,(g,l) in enumerate(zip(d_glob,d_layer)):
        c.text(i-w/2, g+ymax*.02, "%.1e"%g, ha="center", fontsize=7.4)
        c.text(i+w/2, l+ymax*.02, "%.1e"%l, ha="center", fontsize=7.4)
    c.set_ylim(0, ymax*1.18)
    c.set_xticks(x); c.set_xticklabels([("%g"%e) for e in d_eps])
    c.set_xlabel(t["xc"], fontproperties=fp, fontsize=10)
    c.set_ylabel(t["yc"], fontproperties=fp, fontsize=10); c.set_title(t["tc"], fontproperties=fp, fontsize=10)
    c.legend(prop=fp, fontsize=9, loc="upper right")
    c.text(.01,-.22,t["note_c"],transform=c.transAxes,fontproperties=fp,fontsize=8,color="#555")
    c.grid(alpha=.25, axis="y", which="both")
    for aa in ax: aa.tick_params(labelsize=8.5)
    fig.tight_layout()
    name = "fig_s2_singular_panel.png" if lang=="zh" else "fig_s2_singular_panel_en.png"
    outs = [ROOT / "figures" / lang / name, ROOT / "paper" / "figures" / name]
    for o in outs:
        fig.savefig(o, dpi=200, bbox_inches="tight")
        print("saved", o)
    plt.close(fig)

if __name__ == "__main__":
    make("zh"); make("en")
