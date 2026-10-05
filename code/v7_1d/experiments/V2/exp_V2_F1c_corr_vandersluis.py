# -*- coding: utf-8 -*-
"""
F1c（定理 4.2 / thm:local-no-go 的纯线代诊断，不训练、秒级~分钟级）
======================================================================
配套论文定理 4.2「空间局部乘性加权的谱能力边界」。全部为静态矩阵数值实验，
不实例化网络、不训练、不读 checkpoint；合成可控微分算子（与 F1 同一构造），
在配点 NTK Gram 约定 K_r = A A^T（N×N，过参数化 P>N 时正定）下验证四条可证伪预测：

  (A) N=2 闭式范例：K=U diag(1,100) U^T（θ=30°），局部对角权 W=diag(t,1) 的
      κ(t) 扫描；闭式最优点 t*=K22/K11（AM-GM 对角均衡），有限改善
      κ(K)=100 -> κ(R)=(1+ρ)/(1-ρ)=75.4，但够不到 1；全局 W*=K^{-1} 给 κ=1。
  (B) 高维（N=64 激波线性化算子）局部权 κ 云：随对比度 β 扫描，任何局部权都落在
      van der Sluis 带 [κ(R)/N, κ(R)] 之上/之内，Jacobi 归一 w_i=1/K_ii 达到
      上界 κ(R)，数值最优 κ*_diag（N=8 多次重启优化）落在带内，W*=K^{-1} 给 κ=1。
  (C) 相关矩阵不变量（定理 4.2(iii-a)）：对任意正对角 W，R(W^{1/2} K W^{1/2})=R(K)，
      残差应到机器精度（~1e-14），与 N、算子、权重无关。
  (D) ν 扫描：ν↓ 对流占优、K 近奇异时 κ(R) 与 κ*_diag 同步发散、改善比
      κ*_diag/κ(R) 有界（∈[1/N,1]）、不趋于 0——任何局部权压不回有界条件数。

记号（与论文第 4 章一致）：
  相关矩阵  R(A) = Δ_A^{-1/2} A Δ_A^{-1/2},  Δ_A=diag(A_11,...,A_NN)；
  加权配点 Gram  K̃ = W^{1/2} K W^{1/2}；条件数一律指（对称正定）非零谱条件数。

产出：
  results/v7/V2/exp_V2_F1c_corr_vandersluis/  （csv + 中英双版 2×2 图）
  theory/final/data/f1c_*.csv 与 theory/final/figs/{zh,en}/fig_f1c_corr_panel.png（冻结副本）
"""
import os, csv
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------- 路径 ----------
_HERE = Path(__file__).resolve()
_PKG = _HERE.parent.parent                       # v7/
_ROOT = _PKG.parent.parent                       # 源代码/
OUT = _ROOT / "results" / "v7" / "V2" / "exp_V2_F1c_corr_vandersluis"
FINAL = _ROOT / "theory" / "final"
for d in [OUT, FINAL / "data", FINAL / "figs" / "zh", FINAL / "figs" / "en"]:
    d.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260916)
ZH_FONT = r"C:\Windows\Fonts\msyh.ttc"
COL_UP, COL_DL, COL_DI, COL_UNI, COL_WS = "#E8A24B", "#7FB3D5", "#4B86B4", "#9A9A9A", "#C0504D"


def _font(lang):
    if lang == "zh" and os.path.exists(ZH_FONT):
        fm.fontManager.addfont(ZH_FONT)
        plt.rcParams["font.family"] = fm.FontProperties(fname=ZH_FONT).get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


# ---------- 合成微分算子（与 F1 一致，周期有限差分） ----------
def periodic_D(n, L=2.0):
    h = L / n
    D1 = np.zeros((n, n)); D2 = np.zeros((n, n))
    for i in range(n):
        D1[i, (i + 1) % n] = 1 / (2 * h); D1[i, (i - 1) % n] = -1 / (2 * h)
        D2[i, (i + 1) % n] = 1 / h ** 2; D2[i, i] = -2 / h ** 2; D2[i, (i - 1) % n] = 1 / h ** 2
    return D1, D2


def shock_operator(n, nu=0.05):
    D1, D2 = periodic_D(n)
    x = np.linspace(-1, 1, n, endpoint=False)
    us = -np.tanh(x / (2 * nu))          # 稳态 Burgers 行波剖面 u*=-tanh(x/2ν)
    return -nu * D2 + np.diag(us) @ D1, x


def lin_operator(n):
    _, D2 = periodic_D(n)
    return -D2, np.linspace(-1, 1, n, endpoint=False)


def build_J(Dop, seed=0, lift=1e-3):
    """过参数化雅可比 J=[D, εE]∈R^{N×2N}（P=2N>N），使配点 Gram K=JJ^T≻0。
    周期微分算子 D 含常数零模；额外随机列 εE 以远小于最小非零奇异值的尺度 lift 零空间
    （模拟 P>N 时 NTK Gram 满秩），不改变主导谱结构与相关矩阵的非对角形态。"""
    n = Dop.shape[0]
    s = np.linalg.svd(Dop, compute_uv=False)
    nz = s[s > 1e-8 * s.max()]
    s1 = nz.min() if len(nz) else 1.0
    r = np.random.default_rng(seed)
    eps = np.sqrt(lift * s1 ** 2 / n)
    return np.hstack([Dop, eps * r.standard_normal((n, n))])


# ---------- 线代工具 ----------
def corr_mat(A):
    """R(A)=Δ^{-1/2} A Δ^{-1/2}，Δ=diag(A_ii)。"""
    d = np.diag(A).copy()
    d = np.where(d > 0, d, np.nan)
    s = 1.0 / np.sqrt(d)
    return (A * s[:, None]) * s[None, :]


def kappa_spd(A, eps=1e-13):
    v = np.linalg.eigvalsh((A + A.T) / 2)
    v = v[v > eps * v.max()]
    return v.max() / v.min()


def weight_Ktilde(K, w):
    d = np.sqrt(w)
    return (K * d[:, None]) * d[None, :]      # 对角 W：W^{1/2} K W^{1/2}


def sym_sqrtm(A):
    w, V = np.linalg.eigh((A + A.T) / 2)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def kappa_dense(K, W):
    """稠密（一般非对角）权 W 下 κ(W^{1/2} K W^{1/2})；谱白化 W*=K^{-1} 给 I、κ=1。"""
    Wh = sym_sqrtm(W)
    return kappa_spd(Wh @ K @ Wh)


def local_weights(x, beta, delta=0.05):
    mask = (np.abs(x) < delta).astype(float)
    return {
        "up": 1.0 + (beta - 1) * mask,
        "down_lin": 1.0 - (1 - 1.0 / beta) * mask,
        "down_inv": 1.0 / (1.0 + (beta - 1) * mask),
    }


def kappa_diag_opt(K, n_restart=24, seed=0):
    """数值求 κ*_diag = min_{正对角 W} κ(W^{1/2} K W^{1/2})，log w 空间多起点 Nelder-Mead。"""
    n = K.shape[0]
    r = np.random.default_rng(seed)
    d0 = np.diag(K)

    def obj(z):
        w = np.exp(z)
        return kappa_spd(weight_Ktilde(K, w))

    z_jac = -np.log(d0)                    # Jacobi 归一 w_i=1/K_ii -> K̃=R(K)
    starts = [z_jac, np.zeros(n), z_jac + 0.2 * r.standard_normal((n_restart - 2, n))]
    best = np.inf
    for z0 in starts:
        try:
            res = minimize(obj, z0, method="Nelder-Mead",
                           options=dict(maxiter=4000, xatol=1e-8, fatol=1e-8, disp=False))
            best = min(best, res.fun)
        except Exception:
            pass
    return best


# ---------- (A) N=2 闭式 ----------
def panel_n2():
    theta = np.deg2rad(30.0)
    c, s = np.cos(theta), np.sin(theta)
    U = np.array([[c, -s], [s, c]])
    K = U @ np.diag([1.0, 100.0]) @ U.T
    a, b, cxy = K[0, 0], K[1, 1], K[1, 0]
    rho = abs(cxy) / np.sqrt(a * b)
    kR = (1 + rho) / (1 - rho)
    t_star = b / a
    ts = np.exp(np.linspace(np.log(0.05), np.log(20), 4001))
    ks = np.array([kappa_spd(weight_Ktilde(K, np.array([t, 1.0]))) for t in ts])
    Ws = np.linalg.inv(K)                    # 稠密、非局部谱白化 W*=K^{-1}
    k_ws = kappa_dense(K, Ws)
    row = dict(a=a, b=b, K12=cxy, rho=rho, kappa_R=kR, t_star=t_star,
               kappa_at_tstar=kappa_spd(weight_Ktilde(K, np.array([t_star, 1.0]))),
               kappa_K=100.0, kappa_Wstar=k_ws)
    return K, ts, ks, row


# ---------- (C) 相关矩阵不变性 ----------
def panel_invariant():
    rows = []
    for op_name, op in [("lin", lin_operator), ("shock", lambda n: shock_operator(n, 0.05))]:
        for n in [32, 64, 128, 256]:
            Dop, x = op(n)
            J = build_J(Dop, seed=n)
            K = J @ J.T
            R0 = corr_mat(K)
            for beta in [3.0, 10.0, 40.0]:
                for wname, w in local_weights(x, beta).items():
                    Rw = corr_mat(weight_Ktilde(K, w))
                    rel = np.linalg.norm(Rw - R0, "fro") / np.linalg.norm(R0, "fro")
                    rows.append(dict(op=op_name, n=n, weight=f"{wname}_b{beta:g}", rel_err=rel))
            for k in range(6):                       # 随机正对角权
                w = np.exp(rng.uniform(-2, 2, n))
                Rw = corr_mat(weight_Ktilde(K, w))
                rel = np.linalg.norm(Rw - R0, "fro") / np.linalg.norm(R0, "fro")
                rows.append(dict(op=op_name, n=n, weight=f"rand{k}", rel_err=rel))
    return rows


# ---------- (B) 高维云 + van der Sluis ----------
def panel_cloud(n=64, nu=0.05, n_rand=400):
    Dop, x = shock_operator(n, nu)
    J = build_J(Dop, seed=n)
    K = J @ J.T
    R = corr_mat(K)
    kR = kappa_spd(R)
    w_jac = 1.0 / np.diag(K)
    k_jac = kappa_spd(weight_Ktilde(K, w_jac))
    Ws = np.linalg.inv(K)
    k_ws = kappa_dense(K, Ws)
    cloud = []
    betas = [1.5, 2, 3, 5, 8, 12, 20, 40, 80]
    for beta in betas:
        for wname, w in local_weights(x, float(beta)).items():
            k = kappa_spd(weight_Ktilde(K, w))
            cloud.append(dict(kind=wname, beta=beta, contrast=beta, kappa=k))
    for _ in range(n_rand):                        # 随机对数权，对比度=max/min
        z = rng.uniform(-1, 1, n)
        L = rng.uniform(0, np.log(80))
        rng_ = np.ptp(z)
        w = np.exp((z - z.min()) / rng_ * L) if rng_ > 0 else np.ones(n)
        k = kappa_spd(weight_Ktilde(K, w))
        cloud.append(dict(kind="rand", beta=w.max() / w.min(), contrast=w.max() / w.min(), kappa=k))
    meta = dict(n=n, nu=nu, kappa_R=kR, kappa_lower=kR / n,
                kappa_jacobi=k_jac, kappa_Wstar=k_ws)
    return cloud, meta


# ---------- (D) ν 扫描（N=8 数值优化 κ*_diag） ----------
def panel_nu(n=8):
    nus = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005]
    rows = []
    for nu in nus:
        Dop, x = shock_operator(n, nu)
        J = build_J(Dop, seed=n)
        K = J @ J.T
        kK = kappa_spd(K)
        kR = kappa_spd(corr_mat(K))
        kstar = kappa_diag_opt(K, n_restart=32, seed=1)
        rows.append(dict(nu=nu, kappa_K=kK, kappa_R=kR, kappa_star=kstar,
                         lower=kR / n, ratio_star_over_R=kstar / kR))
    return rows


def _write_csv(name, rows):
    if not rows:
        return
    p = OUT / name
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(FINAL / "data" / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def make_figure(lang, n2, inv_rows, cloud, meta, nu_rows):
    _font(lang)
    K, ts, ks, r2 = n2
    L = lang == "zh"
    t = dict(
        A=("N=2：局部权的有限改善与全局天花板" if L else "N=2: limited local gain, global ceiling"),
        B=("高维局部权 κ 云与 van der Sluis 夹逼" if L else "High-D local-weight κ cloud & van der Sluis bounds"),
        C=("相关矩阵不变量 $R(\\widetilde K)=R(K)$（机器精度）" if L else "Correlation invariant $R(\\widetilde K)=R(K)$ (machine precision)"),
        D=("ν 扫描：病态时局部权同步失效" if L else "ν-scan: local weights fail as problem stiffens"),
        xA=("局部权参数 t （W=diag(t,1)）" if L else "local weight t  (W=diag(t,1))"),
        yA=("条件数 κ" if L else "condition number κ"),
        xB=("权重对比度 $w_{\\max}/w_{\\min}$" if L else "weight contrast $w_{\\max}/w_{\\min}$"),
        yB=("加权条件数 $\\kappa(\\widetilde K)$" if L else "weighted $\\kappa(\\widetilde K)$"),
        xC=("配点数 N" if L else "collocation N"),
        yC=("$\\|R(\\widetilde K)-R(K)\\|_F\\,/\\,\\|R(K)\\|_F$" if L
            else "$\\|R(\\widetilde K)-R(K)\\|_F\\,/\\,\\|R(K)\\|_F$"),
        xD=("粘度 ν" if L else "viscosity ν"),
        yD=("条件数" if L else "condition number"),
        loc=("局部增权" if L else "local up"), locdl=("线性减权" if L else "linear down"),
        locdi=("逆减权" if L else "inverse down"), rnd=("随机对角权" if L else "random diagonal"),
        kR=("Jacobi 上界 κ(R)" if L else "Jacobi upper κ(R)"),
        kRl=("van der Sluis 下界 κ(R)/N" if L else "lower κ(R)/N"),
        ws=("全局 $W_*=K^{-1}$，κ=1" if L else "global $W_*=K^{-1}$, κ=1"),
        kstar=("数值 $\\kappa^*_{\\mathrm{diag}}$" if L else "numerical $\\kappa^*_{\\mathrm{diag}}$"),
    )
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 9.0))

    # A
    a = ax[0, 0]
    a.plot(ts, ks, color=COL_DI, lw=2.2)
    a.axhline(100, color=COL_UNI, ls=":", lw=1.4)
    a.axhline(r2["kappa_R"], color=COL_UP, ls="--", lw=1.6, label=t["kR"] + f"={r2['kappa_R']:.1f}")
    a.axhline(1.0, color=COL_WS, ls="-", lw=1.6, label=t["ws"])
    a.plot(r2["t_star"], r2["kappa_at_tstar"], "o", color="k", ms=6)
    a.annotate(f"t*={r2['t_star']:.2f}, κ={r2['kappa_at_tstar']:.1f}",
               (r2["t_star"], r2["kappa_at_tstar"]), textcoords="offset points", xytext=(8, 10), fontsize=9)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(t["xA"]); a.set_ylabel(t["yA"])
    a.set_title("(A) " + t["A"], fontsize=11); a.grid(alpha=.3, which="both"); a.legend(fontsize=8, loc="upper center")

    # B
    a = ax[0, 1]
    styles = {"up": (COL_UP, "s", t["loc"]), "down_lin": (COL_DL, "^", t["locdl"]),
              "down_inv": (COL_DI, "v", t["locdi"]), "rand": (COL_UNI, ".", t["rnd"])}
    for kind, (col, mk, lab) in styles.items():
        z = [q for q in cloud if q["kind"] == kind]
        a.scatter([q["contrast"] for q in z], [q["kappa"] for q in z], s=(22 if kind != "rand" else 8),
                  c=col, marker=mk, label=lab, alpha=(.85 if kind != "rand" else .35))
    a.axhline(meta["kappa_R"], color=COL_UP, ls="--", lw=1.6, label=t["kR"] + f"={meta['kappa_R']:.2e}")
    a.axhline(meta["kappa_lower"], color=COL_UP, ls=":", lw=1.6,
              label=t["kRl"] + f"={meta['kappa_lower']:.2e}")
    a.axhline(meta["kappa_Wstar"], color=COL_WS, ls="-", lw=1.6, label=t["ws"])
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(t["xB"]); a.set_ylabel(t["yB"])
    a.set_title(f"(B) {t['B']}  (N={meta['n']}, ν={meta['nu']})", fontsize=11)
    a.grid(alpha=.3, which="both"); a.legend(fontsize=7.5, loc="lower right", ncol=2)

    # C
    a = ax[1, 0]
    for op, col, mk in [("lin", COL_DL, "^"), ("shock", COL_DI, "s")]:
        z = [q for q in inv_rows if q["op"] == op]
        agg = {}
        for q in z:
            agg.setdefault(q["n"], []).append(q["rel_err"])
        ns = sorted(agg)
        med = [np.median(agg[n]) for n in ns]; mx = [np.max(agg[n]) for n in ns]
        a.plot(ns, med, mk + "-", color=col, ms=6,
               label=("线性算子（中位）" if L else "linear (median)") if op == "lin"
               else ("激波算子（中位）" if L else "shock (median)"))
        a.plot(ns, mx, mk + "--", color=col, ms=5, alpha=.5,
               label=("最大" if L else "max") if op == "shock" else None)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(t["xC"]); a.set_ylabel(t["yC"])
    a.set_title("(C) " + t["C"], fontsize=11); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)

    # D
    a = ax[1, 1]
    nus = [q["nu"] for q in nu_rows]
    a.plot(nus, [q["kappa_R"] for q in nu_rows], "s-", color=COL_UP, label=t["kR"])
    a.plot(nus, [q["kappa_star"] for q in nu_rows], "o-", color="k", label=t["kstar"] + "(N=8)")
    a.plot(nus, [q["lower"] for q in nu_rows], ":", color=COL_UP, label=t["kRl"])
    a.set_xscale("log"); a.set_yscale("log"); a.invert_xaxis()
    a.set_xlabel(t["xD"]); a.set_ylabel(t["yD"]); a.set_title("(D) " + t["D"], fontsize=11)
    a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    a2 = a.twinx()
    a2.plot(nus, [q["ratio_star_over_R"] for q in nu_rows], "^--", color=COL_DI, ms=6, alpha=.7)
    a2.set_ylabel(("$\\kappa^*_{\\mathrm{diag}}/\\kappa(R)$" if not L else "改善比 $\\kappa^*_{\\mathrm{diag}}/\\kappa(R)$"), color=COL_DI)
    a2.tick_params(axis="y", labelcolor=COL_DI); a2.set_yscale("linear"); a2.set_ylim(0, 1.05)

    fig.suptitle(("定理 4.2 数值诊断：局部对角权改不了相关矩阵、够不到谱白化" if L
                  else "Thm 4.2 diagnostic: local diagonal weights cannot alter R(K) nor reach whitening"),
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for lang2 in (["zh", "en"] if lang == "zh" else ["en"]):
        p = FINAL / "figs" / lang2 / "fig_f1c_corr_panel.png"
        fig.savefig(p, dpi=150); print("saved", p)
    fig.savefig(OUT / f"fig_f1c_corr_panel_{lang}.png", dpi=150)
    plt.close(fig)


def main():
    print("[F1c] (A) N=2 closed form ...")
    n2 = panel_n2()
    K, ts, ks, r2 = n2
    print(f"   rho={r2['rho']:.5f}  kappa(R)={r2['kappa_R']:.3f}  t*={r2['t_star']:.4f}  "
          f"kappa(t*)={r2['kappa_at_tstar']:.3f}  kappa(W*)={r2['kappa_Wstar']:.3e}")
    assert abs(r2["kappa_R"] - 75.4) < 1.0 and abs(r2["t_star"] - 2.922) < 0.05
    assert r2["kappa_Wstar"] < 1 + 1e-6

    print("[F1c] (C) correlation invariant ...")
    inv_rows = panel_invariant()
    worst = max(q["rel_err"] for q in inv_rows)
    print(f"   worst relative R-discrepancy = {worst:.3e}")
    assert worst < 1e-10, worst

    print("[F1c] (B) high-D cloud + van der Sluis ...")
    cloud, meta = panel_cloud(n=64, nu=0.05, n_rand=400)
    print(f"   kappa(R)={meta['kappa_R']:.3e}  lower={meta['kappa_lower']:.3e}  "
          f"Jacobi={meta['kappa_jacobi']:.3e}  W*={meta['kappa_Wstar']:.3e}")
    assert abs(meta["kappa_jacobi"] - meta["kappa_R"]) / meta["kappa_R"] < 1e-6
    assert meta["kappa_Wstar"] < 1 + 1e-4, (meta["kappa_Wstar"], meta["kappa_R"])
    # 没有任何局部权越过 κ*_diag 下界（van der Sluis）：云点 κ ≥ κ(R)/N
    assert all(q["kappa"] >= meta["kappa_lower"] * (1 - 1e-6) for q in cloud)

    print("[F1c] (D) nu scan (N=8 optimization) ...")
    nu_rows = panel_nu(8)
    for q in nu_rows:
        print(f"   nu={q['nu']:<6} kR={q['kappa_R']:.3e} k*={q['kappa_star']:.3e} "
              f"ratio={q['ratio_star_over_R']:.3f}")
        assert 1.0 / 8 * 0.95 <= q["ratio_star_over_R"] <= 1.0 + 1e-6

    _write_csv("f1c_n2_closedform.csv", [r2])
    _write_csv("f1c_corr_invariant.csv", inv_rows)
    _write_csv("f1c_cloud.csv", cloud)
    _write_csv("f1c_nu_scan.csv", nu_rows)
    with open(OUT / "f1c_meta.txt", "w", encoding="utf-8") as f:
        f.write(str(meta))

    make_figure("zh", n2, inv_rows, cloud, meta, nu_rows)
    make_figure("en", n2, inv_rows, cloud, meta, nu_rows)
    print("[F1c] done ->", OUT)


if __name__ == "__main__":
    main()
