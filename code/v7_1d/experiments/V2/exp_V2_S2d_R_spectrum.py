# -*- coding: utf-8 -*-
"""
V2 实验 S2d-R：内部转向点层上的 R 层（冻结截面）谱不可能性跨方程诊断（零训练，秒级）
================================================================================
目的：第一篇 thm:local-no-go（空间局部对角权改不了冻结 Gram 的特征方向、够不到谱白化、
条件数改善被 van der Sluis 夹住）在【第二个方程类型】上复核。证明本身方程无关（只用对称
谱定理/相关性恒等式），本实验给转向点线性算子 -eps u'' - x u'=0（系数 x 在 x=0 变号，
即“转向点”）提供与 Burgers F1c 同口径的数值佐证：

  (A) 三族局部权（层带 |x|<3√eps 内增/减权）κ 随对比度 β：有界、双向不定，不趋于 1；
  (B) N=64/128 局部权 + 随机对角权 κ 云：全部落在 van der Sluis 带 [κ(R)/N, κ(R)]，
      Jacobi 归一 w_i=1/K_ii 达上界 κ(R)，稠密 W*=K^{-1} 才给 κ=1；
  (C) 相关矩阵不变量 R(W^{1/2} K W^{1/2})=R(K)：任意正对角权下到机器精度（~1e-14）；
  (D) eps 扫描：eps↓、算子近奇异时 κ(R) 与数值最优 κ*_diag 同步发散，改善比
      κ*_diag/κ(R) 有界于 [1/N,1]、不趋于 0——局部权压不回有界条件数。

约定（与 exp_V2_F1c_corr_vandersluis 完全一致）：合成周期有限差分算子仅用于提取冻结线性化
算子谱结构，不是真实 BVP 离散；配点 Gram K=J J^T（J 过参数化、P>N 正定）。
产出 results/v7/V2/exp_V2_S2d_R_spectrum/ 与 theory/final/{data,figs/{zh,en}} 冻结副本。
================================================================================
"""
import os, sys, csv
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_HERE = Path(__file__).resolve(); _V7 = _HERE.parents[2]
for _p in (str(_V7), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import exp_V2_F1c_corr_vandersluis as F1c   # 复用全部线代工具与作图规范

_ROOT = _V7.parent
OUT = _ROOT / "results" / "v7" / "V2" / "exp_V2_S2d_R_spectrum"
FINAL = _ROOT / "theory" / "final"
for d in [OUT, FINAL / "data", FINAL / "figs" / "zh", FINAL / "figs" / "en"]:
    d.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260918)
ZH_FONT = r"C:\Windows\Fonts\msyh.ttc"
COL_UP, COL_DL, COL_DI, COL_UNI, COL_WS = "#E8A24B", "#7FB3D5", "#4B86B4", "#9A9A9A", "#C0504D"
EPS_MAIN = 1e-4
BETAS = [1.5, 2, 3, 5, 8, 12, 20, 40, 80]
# 注：N=8 数值优化面板只扫到 ε=1e-4。更薄 ε=1e-6 在 N=8（dx=.25≫层宽 .001）上转向点
# x=0 行的 -εD2 正则项近乎消失、矩阵数值秩退化，κ 定义失稳（非定理违反）；极薄层另用加密 N。
EPS_SCAN = [1e-2, 1e-3, 1e-4]


def _font(lang):
    if lang == "zh" and os.path.exists(ZH_FONT):
        fm.fontManager.addfont(ZH_FONT)
        plt.rcParams["font.family"] = fm.FontProperties(fname=ZH_FONT).get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def tp_operator(n, eps):
    """转向点线性算子 L=-eps D2 - diag(x) D1（周期有限差分，与 F1c 同约定）。"""
    D1, D2 = F1c.periodic_D(n)
    x = np.linspace(-1, 1, n, endpoint=False)
    return -float(eps) * D2 - np.diag(x) @ D1, x


def local_weights_tp(x, eps, beta, band_c=3.0):
    delta = band_c * np.sqrt(float(eps))          # 层带半宽 3√eps
    mask = (np.abs(x) < delta).astype(float)
    return {
        "up": 1.0 + (beta - 1) * mask,
        "down_lin": 1.0 - (1.0 - 1.0 / beta) * mask,
        "down_inv": 1.0 / (1.0 + (beta - 1) * mask),
    }


def gram(n, eps, seed=0):
    Dop, x = tp_operator(n, eps)
    J = F1c.build_J(Dop, seed=seed)
    return J @ J.T, x


# (A) 三族 κ-β 曲线
def panel_curves(n=128, eps=EPS_MAIN):
    K, x = gram(n, eps)
    rows = []
    for b in BETAS:
        for wn, w in local_weights_tp(x, eps, float(b)).items():
            rows.append(dict(beta=b, family=wn, kappa=F1c.kappa_spd(F1c.weight_Ktilde(K, w))))
    return rows, dict(kappa_R=F1c.kappa_spd(F1c.corr_mat(K)),
                      kappa_Wstar=F1c.kappa_dense(K, np.linalg.inv(K)))


# (B) 高维云 + van der Sluis
def panel_cloud(n=128, eps=EPS_MAIN, n_rand=400):
    K, x = gram(n, eps)
    R = F1c.corr_mat(K); kR = F1c.kappa_spd(R)
    w_jac = 1.0 / np.diag(K); k_jac = F1c.kappa_spd(F1c.weight_Ktilde(K, w_jac))
    k_ws = F1c.kappa_dense(K, np.linalg.inv(K))
    cloud = []
    for b in BETAS:
        for wn, w in local_weights_tp(x, eps, float(b)).items():
            cloud.append(dict(kind=wn, beta=b, contrast=b, kappa=F1c.kappa_spd(F1c.weight_Ktilde(K, w))))
    for _ in range(n_rand):
        z = rng.uniform(-1, 1, n); L = rng.uniform(0, np.log(80)); ptp = np.ptp(z)
        w = np.exp((z - z.min()) / ptp * L) if ptp > 0 else np.ones(n)
        cloud.append(dict(kind="rand", beta=w.max() / w.min(), contrast=w.max() / w.min(),
                          kappa=F1c.kappa_spd(F1c.weight_Ktilde(K, w))))
    meta = dict(n=n, eps=eps, kappa_R=kR, kappa_lower=kR / n, kappa_jacobi=k_jac, kappa_Wstar=k_ws)
    return cloud, meta


# (C) 相关矩阵不变量
def panel_invariant():
    rows = []
    for n in [32, 64, 128, 256]:
        K, x = gram(n, EPS_MAIN, seed=n)
        R0 = F1c.corr_mat(K)
        for b in [3.0, 10.0, 40.0]:
            for wn, w in local_weights_tp(x, EPS_MAIN, b).items():
                Rw = F1c.corr_mat(F1c.weight_Ktilde(K, w))
                rows.append(dict(n=n, weight=f"{wn}_b{b:g}",
                                 rel_err=np.linalg.norm(Rw - R0, "fro") / np.linalg.norm(R0, "fro")))
        for k in range(6):
            w = np.exp(rng.uniform(-2, 2, n))
            Rw = F1c.corr_mat(F1c.weight_Ktilde(K, w))
            rows.append(dict(n=n, weight=f"rand{k}",
                             rel_err=np.linalg.norm(Rw - R0, "fro") / np.linalg.norm(R0, "fro")))
    return rows


# (D) eps 扫描（N=8 数值优化 κ*_diag）
def panel_eps(n=8):
    rows = []
    for eps in EPS_SCAN:
        K, _ = gram(n, eps, seed=n)
        kK = F1c.kappa_spd(K); kR = F1c.kappa_spd(F1c.corr_mat(K))
        kstar = F1c.kappa_diag_opt(K, n_restart=32, seed=1)
        rows.append(dict(eps=eps, kappa_K=kK, kappa_R=kR, kappa_star=kstar, lower=kR / n,
                         ratio_star_over_R=kstar / kR))
    return rows


def _write(name, rows):
    if not rows: return
    with open(OUT / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(FINAL / "data" / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def make_figure(lang, curves, cmeta, cloud, meta, inv_rows, eps_rows):
    _font(lang); L = lang == "zh"
    t = dict(
        A=("转向点层：三族局部权 κ 随对比度（有界、不达 1）" if L else
           "Turning-point layer: local-weight κ vs contrast (bounded, never 1)"),
        B=("局部权 κ 云与 van der Sluis 夹逼" if L else "Local-weight κ cloud & van der Sluis bounds"),
        C=("相关矩阵不变量（机器精度）" if L else "Correlation invariant (machine precision)"),
        D=("ε 扫描：病态时局部权同步失效" if L else "ε-scan: local weights fail as layer thins"),
        xA=("权重对比度 β" if L else "weight contrast β"), yA=("加权条件数 κ" if L else "weighted κ"),
        xB=("w_max/w_min" if L else "w_max/w_min"), yB=("κ(K~)" if L else "κ(K~)"),
        xC=("配点数 N" if L else "collocation N"),
        yC=("‖R(K~)-R(K)‖_F/‖R(K)‖_F" if L else "‖R(K~)-R(K)‖_F/‖R(K)‖_F"),
        xD=("ε（层厚 √ε）" if L else "ε (layer width √ε)"), yD=("条件数" if L else "condition number"),
        up=("局部增权" if L else "local up"), dl=("线性减权" if L else "linear down"),
        di=("逆减权" if L else "inverse down"), rnd=("随机对角权" if L else "random diagonal"),
        ws=(r"全局 $W_*=K^{-1}$，κ=1" if L else r"global $W_*=K^{-1}$, κ=1"))
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 9.0))
    a = ax[0, 0]
    for fam, col, mk in [("up", COL_UP, "s"), ("down_lin", COL_DL, "^"), ("down_inv", COL_DI, "v")]:
        z = sorted((q for q in curves if q["family"] == fam), key=lambda q: q["beta"])
        a.plot([q["beta"] for q in z], [q["kappa"] for q in z], mk + "-", color=col, ms=5,
               label=t[{"up": "up", "down_lin": "dl", "down_inv": "di"}[fam]])
    a.axhline(cmeta["kappa_R"], color=COL_UNI, ls="--", lw=1.4, label=f"κ(R)={cmeta['kappa_R']:.2e}")
    a.axhline(cmeta["kappa_Wstar"], color=COL_WS, ls="-", lw=1.6, label=t["ws"])
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(t["xA"]); a.set_ylabel(t["yA"])
    a.set_title("(A) " + t["A"], fontsize=11); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    a = ax[0, 1]
    styles = {"up": (COL_UP, "s", t["up"]), "down_lin": (COL_DL, "^", t["dl"]),
              "down_inv": (COL_DI, "v", t["di"]), "rand": (COL_UNI, ".", t["rnd"])}
    for kind, (col, mk, lab) in styles.items():
        z = [q for q in cloud if q["kind"] == kind]
        a.scatter([q["contrast"] for q in z], [q["kappa"] for q in z],
                  s=(22 if kind != "rand" else 8), c=col, marker=mk, label=lab,
                  alpha=(.85 if kind != "rand" else .35))
    a.axhline(meta["kappa_R"], color=COL_UP, ls="--", lw=1.6, label=f"κ(R)={meta['kappa_R']:.2e}")
    a.axhline(meta["kappa_lower"], color=COL_UP, ls=":", lw=1.6, label=f"κ(R)/N={meta['kappa_lower']:.2e}")
    a.axhline(meta["kappa_Wstar"], color=COL_WS, ls="-", lw=1.6, label=t["ws"])
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(t["xB"]); a.set_ylabel(t["yB"])
    a.set_title(f"(B) N={meta['n']}, ε={meta['eps']:g}", fontsize=11)
    a.grid(alpha=.3, which="both"); a.legend(fontsize=7.5, loc="lower right", ncol=2)
    a = ax[1, 0]
    agg = {}
    for q in inv_rows: agg.setdefault(q["n"], []).append(q["rel_err"])
    ns = sorted(agg)
    a.plot(ns, [np.median(agg[n]) for n in ns], "s-", color=COL_DI, label=("中位" if L else "median"))
    a.plot(ns, [np.max(agg[n]) for n in ns], "^--", color=COL_DI, alpha=.5, label=("最大" if L else "max"))
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(t["xC"]); a.set_ylabel(t["yC"])
    a.set_title("(C) " + t["C"], fontsize=11); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    a = ax[1, 1]
    es = [q["eps"] for q in eps_rows]
    a.plot(es, [q["kappa_R"] for q in eps_rows], "s-", color=COL_UP, label="κ(R)")
    a.plot(es, [q["kappa_star"] for q in eps_rows], "o-", color="k", label=("数值 κ*_diag (N=8)" if L else "num. κ*_diag (N=8)"))
    a.plot(es, [q["lower"] for q in eps_rows], ":", color=COL_UP, label="κ(R)/N")
    a.set_xscale("log"); a.set_yscale("log"); a.invert_xaxis()
    a.set_xlabel(t["xD"]); a.set_ylabel(t["yD"]); a.set_title("(D) " + t["D"], fontsize=11)
    a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    a2 = a.twinx()
    a2.plot(es, [q["ratio_star_over_R"] for q in eps_rows], "^--", color=COL_DI, ms=6, alpha=.7)
    a2.set_ylabel(("改善比 κ*_diag/κ(R)" if L else "ratio κ*_diag/κ(R)"), color=COL_DI)
    a2.tick_params(axis="y", labelcolor=COL_DI); a2.set_yscale("linear"); a2.set_ylim(0, 1.05)
    fig.suptitle(("转向点层上 R 层谱不可能性跨方程复核：局部对角权改不了特征方向" if L else
                  "Cross-equation R-layer no-go on turning-point layer"), fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for l2 in (["zh", "en"] if lang == "zh" else ["en"]):
        p = FINAL / "figs" / l2 / "fig_s2d_R_panel.png"; fig.savefig(p, dpi=150); print("saved", p)
    fig.savefig(OUT / f"fig_s2d_R_panel_{lang}.png", dpi=150); plt.close(fig)


def main():
    print("[S2d-R] (A) family κ-β ...")
    curves, cmeta = panel_curves()
    print(f"   κ(R)={cmeta['kappa_R']:.3e}  κ(W*)={cmeta['kappa_Wstar']:.3e}")
    print("[S2d-R] (B) cloud + van der Sluis ...")
    cloud, meta = panel_cloud()
    print(f"   κ(R)={meta['kappa_R']:.3e} lower={meta['kappa_lower']:.3e} "
          f"Jacobi={meta['kappa_jacobi']:.3e} W*={meta['kappa_Wstar']:.3e}")
    assert abs(meta["kappa_jacobi"] - meta["kappa_R"]) / meta["kappa_R"] < 1e-6
    assert meta["kappa_Wstar"] < 1 + 1e-4
    assert all(q["kappa"] >= meta["kappa_lower"] * (1 - 1e-6) for q in cloud), "van der Sluis 下界被违反"
    print("[S2d-R] (C) correlation invariant ...")
    inv_rows = panel_invariant(); worst = max(q["rel_err"] for q in inv_rows)
    print(f"   worst R-discrepancy = {worst:.3e}"); assert worst < 1e-10, worst
    print("[S2d-R] (D) eps scan ...")
    eps_rows = panel_eps()
    for q in eps_rows:
        print(f"   eps={q['eps']:<8g} κR={q['kappa_R']:.3e} κ*={q['kappa_star']:.3e} ratio={q['ratio_star_over_R']:.3f}")
        assert 1.0 / 8 * 0.95 <= q["ratio_star_over_R"] <= 1 + 1e-6
    _write("s2d_R_curves.csv", curves); _write("s2d_R_cloud.csv", cloud)
    _write("s2d_R_invariant.csv", inv_rows); _write("s2d_R_eps_scan.csv", eps_rows)
    with open(OUT / "s2d_R_meta.txt", "w", encoding="utf-8") as f: f.write(str(meta))
    make_figure("zh", curves, cmeta, cloud, meta, inv_rows, eps_rows)
    make_figure("en", curves, cmeta, cloud, meta, inv_rows, eps_rows)
    print("[S2d-R] done ->", OUT)


if __name__ == "__main__":
    main()
