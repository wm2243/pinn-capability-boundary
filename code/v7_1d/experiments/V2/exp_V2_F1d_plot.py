# -*- coding: utf-8 -*-
"""F1d 作图：真实 Burgers 冻结 Jacobian 的 r--N / kappa(R)--N / 有效秩 PR--N
读 f1d_real_jacobian.csv，自动识别 kappa_diag 触 1e12 保护地板的数值秩亏点（空心、不计入趋势）。
出中英文两版图，归档 theory/final/figs/{zh,en}，数据归档 theory/final/data。
"""
import os, csv, shutil
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_HERE = Path(__file__).resolve()
_SRC = _HERE.parent.parent.parent.parent          # 源代码/（v7/experiments/V2 上四级）
DATA = _SRC / "results" / "v7" / "V2" / "exp_V2_F1d_real_jacobian" / "f1d_real_jacobian.csv"
OUT = _SRC / "results" / "v7" / "V2" / "exp_V2_F1d_real_jacobian"
FINAL_DATA = _SRC / "theory" / "final" / "data"
FINAL_ZH = _SRC / "theory" / "final" / "figs" / "zh"
FINAL_EN = _SRC / "theory" / "final" / "figs" / "en"
for d in (FINAL_DATA, FINAL_ZH, FINAL_EN): d.mkdir(parents=True, exist_ok=True)

FLOOR = 1e12 - 1e9       # kappa_diag 保护地板阈值


def load():
    rows = []
    with open(DATA, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    rows.sort(key=lambda z: z["N"])
    for r in rows:
        r["floor"] = bool(r["kappa_diag"] >= FLOOR)
    return rows


def setup_font(zh):
    if zh:
        for c in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
            if os.path.exists(c):
                fm.fontManager.addfont(c); plt.rcParams["font.family"] = fm.FontProperties(fname=c).get_name(); break
        plt.rcParams["axes.unicode_minus"] = False
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"; plt.rcParams["axes.unicode_minus"] = True


def make(rows, zh=True):
    setup_font(zh)
    good = [r for r in rows if not r["floor"]]
    bad = [r for r in rows if r["floor"]]
    N = np.array([r["N"] for r in rows], float)
    Ng = np.array([r["N"] for r in good], float)
    rg = np.array([r["ratio_r"] for r in good], float)
    Nb = np.array([r["N"] for r in bad], float)
    rb = np.array([r["ratio_r"] for r in bad], float)
    kR = np.array([r["kappa_R"] for r in rows], float)
    PR = np.array([r["eff_rank_PR"] for r in rows], float)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # ---- 左：r--N ----
    a = ax[0]
    NN = np.geomspace(Ng.min(), N.max(), 120)
    a.plot(NN, 1.0 / NN, "--", color="#999999", lw=1.2,
           label=("van der Sluis 下界 $1/N$" if zh else "van der Sluis lower bound $1/N$"))
    a.plot(Ng, rg, "o-", color="#4B86B4", lw=2, ms=7,
           label=("可信区 $r=\\kappa^*_{\\rm diag}/\\kappa(R)$" if zh
                  else "Trustworthy $r=\\kappa^*_{\\rm diag}/\\kappa(R)$"))
    if len(Nb):
        a.scatter(Nb, rb, marker="x", s=90, color="#C0504D", zorder=5,
                  label=("数值秩亏点（double 地板，剔除）" if zh
                         else "Numerical rank-deficient (double floor, excluded)"))
    # r 企稳带
    a.axhline(rg[-1], ls=":", color="#4B86B4", alpha=.5)
    a.annotate(("r 企稳 ≈ %.2f" % rg[-1] if zh else "r plateaus ≈ %.2f" % rg[-1]),
               xy=(Ng[-1], rg[-1]), xytext=(Ng[-1] * .18, rg[-1] + .06),
               fontsize=10, color="#2f5f87")
    a.set_xscale("log"); a.set_ylim(-.02, 1.02)
    a.set_xlabel(("配点数 $N$" if zh else "Collocation points $N$"))
    a.set_ylabel(("最优对角缩放比 $r$" if zh else "Best diagonal ratio $r$"))
    a.set_title(("真实冻结 Jacobian：$N{\\leq}512$ 时 $r$ 企稳于约 0.77" if zh
                 else "Real frozen Jacobian: $r$ plateaus $\\approx$0.77 for $N{\\leq}512$"))
    a.grid(alpha=.3, which="both"); a.legend(fontsize=8, loc="lower left")

    # ---- 右：kappa(R) 与 PR 双轴 ----
    b = ax[1]
    b.plot(N, kR, "s-", color="#C0504D", lw=1.8, ms=6,
           label=("$\\kappa(R)$（僵硬程度）" if zh else "$\\kappa(R)$ (stiffness)"))
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel(("配点数 $N$" if zh else "Collocation points $N$"))
    b.set_ylabel(("$\\kappa(R)$" if zh else "$\\kappa(R)$"), color="#C0504D")
    b.tick_params(axis="y", labelcolor="#C0504D")
    if len(Nb):
        b.axvspan(Nb.min() / 1.3, Nb.max() * 1.3, color="#C0504D", alpha=.07)
        b.annotate(("数值秩亏区\n$\\kappa(R)>10^{13}$" if zh else "numerical\nrank-deficient"),
                   xy=(Nb.min(), kR[list(N).index(Nb.min())]), xytext=(Nb.min()*.5, 1e9),
                   fontsize=8.5, color="#C0504D")
    b2 = b.twinx()
    b2.plot(N, PR, "^-", color="#3f8f5f", lw=1.8, ms=6,
            label=("有效秩 PR" if zh else "effective rank PR"))
    b2.axhline(PR[-1], ls=":", color="#3f8f5f", alpha=.6)
    b2.annotate(("PR 饱和 ≈ %.0f" % PR[-1] if zh else "PR saturates ≈ %.0f" % PR[-1]),
                xy=(N[-1], PR[-1]), xytext=(N[-1]*.12, PR[-1] + 25), fontsize=10, color="#2f6b47")
    b2.set_ylabel(("有效秩参与率 $PR=(\\sum\\lambda)^2/\\sum\\lambda^2$" if zh
                   else "Effective rank PR"), color="#3f8f5f")
    b2.tick_params(axis="y", labelcolor="#3f8f5f"); b2.set_ylim(0, max(PR)*1.5)
    b.set_title(("配点冗余：$\\kappa(R)$ 暴涨而有效秩饱和" if zh
                 else "Collocation redundancy: $\\kappa(R)$ blows up while effective rank saturates"))
    b.grid(alpha=.3, which="both")
    l1, la1 = b.get_legend_handles_labels(); l2, la2 = b2.get_legend_handles_labels()
    b.legend(l1 + l2, la1 + la2, fontsize=8, loc="upper left")

    fig.tight_layout()
    zz = "zh" if zh else "en"
    fp = OUT / ("fig_F1d_real_jacobian_%s.png" % zz)
    fig.savefig(fp, dpi=160); plt.close(fig)
    return fp


def main():
    rows = load()
    print("N  r  kappa_R  CV  PR  floor:")
    for r in rows:
        print("  N=%5d r=%.4f kR=%.3e CV=%.3f PR=%.1f %s"
              % (r["N"], r["ratio_r"], r["kappa_R"], r["cv_g"], r["eff_rank_PR"],
                 "<-- FLOOR/剔除" if r["floor"] else ""))
    good = [r for r in rows if not r["floor"]]
    print("\n可信区 r:", [round(r["ratio_r"], 3) for r in good])
    print("有效秩 PR:", [round(r["eff_rank_PR"], 1) for r in rows])
    fz = make(rows, True); fe = make(rows, False)
    shutil.copy2(fz, FINAL_ZH / fz.name); shutil.copy2(fe, FINAL_EN / fe.name)
    shutil.copy2(DATA, FINAL_DATA / "f1d_real_jacobian.csv")
    print("已归档:", fz.name, fe.name, "-> theory/final；数据 -> final/data")


if __name__ == "__main__":
    main()
