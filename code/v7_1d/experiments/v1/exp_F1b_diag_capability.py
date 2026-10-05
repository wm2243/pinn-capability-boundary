# -*- coding: utf-8 -*-
"""
实验 F1b（纯数值线代、不训练、秒~分钟级）：定理 4.2「对角局部能力边界 / van der Sluis 夹逼」第 6 节诊断。
=========================================================================================================
对应论文定理 4.2（对角局部权的能力边界）。四件事，全部只做线性代数，不加载任何网络、不重训：

  (1) 相关矩阵 R(K_r) 对对角权不变（定理 4.2(b)，加权可达集 {K̃_r(W)}={E R(K_r) E}）：
        R = D^{-1/2} K_r D^{-1/2}, D=diag(K_r)；对任意正对角 W，
        R(W^{1/2} K_r W^{1/2}) = R(K_r)（到机器精度），与 β、权重族无关。
        六族权重（rational/linear/band/down_lin/down_inv/uniform）β=1..20 逐一核验。

  (2) 最优对角条件数 κ*_diag = min_{E≻0 对角} κ(E R E) 随维数 N 的夹逼（定理 4.2(c)）：
        (1/N) κ(R) ≤ κ*_diag ≤ κ(R)，即比值 r(N)=κ*_diag/κ(R) ∈ [1/N, 1]。
        N=2 时 Jacobi 归一 E=I 恰最优、r=1（上界被吃满）；N≥3 时 r<1（留间隙），但永不低于 1/N。
        合成「冻结残差 Gram」：低秩公共因子 + 各向同性噪声（模拟激波配点梯度近平行、配点冗余），
        每 N 取 nrep 个随机实例报中位/IQR。求解用多起点 L-BFGS-B（解析梯度），属准凸问题的数值解；
        并用两个精确例校验求解器：等相关 2×2（r=1）、论文 N=2 闭式例（κ:100→75.49, t*=2.922）。

  (3) 非局部对照 W_*=c K_r^{-1}：W_*^{1/2} K_r W_*^{1/2}=c I、κ=1（比值 1/κ(R)，突破对角下界 1/N），
        但 W_* 稠密（非局部度≈1）、需 O(N^3) 分解、随线性化点 θ_k 重算。

  (4) 成本计时：最优对角权（多次稠密特征分解）墙钟 ~O(N^3)；N=5e3/1e4 只测单次对称特征分解；
        训练单步主导的矩阵-向量积 ~O(N^2) 作对照。

诚实声明：本脚本不含 SDP/GEVP 证书（环境无 cvxpy）；(2) 的 κ*_diag 是多起点数值最优（准凸、起点
一致到 1e-8、且被两个精确例校验），夹逼带 (1/N,1) 本身是严格定理，不依赖数值最优是否全局精确。

产出 results/v7/V1/exp_F1b_diag_capability/：
   f1b_R_invariance.csv / f1b_kdiag_vs_N.csv / f1b_nonlocal_wstar.csv / f1b_timing.csv
   fig_f1b_squeeze_{zh,en}.png（夹逼曲线）、fig_f1b_timing_{zh,en}.png（计时）
用法：python exp_F1b_diag_capability.py [--smoke]
"""
import os, csv, time, argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------------------------------------------- 路径 / 绘图风格
_HERE = Path(__file__).resolve(); _V1EXP = _HERE.parent            # v7/experiments/v1
_SRC = _V1EXP.parent.parent.parent                                 # 源代码/（v7/experiments/v1 上三级）
OUT = _SRC / "results" / "v7" / "V1" / "exp_F1b_diag_capability"
FINAL_DATA = _SRC / "theory" / "final" / "data"
FINAL_FIG_ZH = _SRC / "theory" / "final" / "figs" / "zh"
FINAL_FIG_EN = _SRC / "theory" / "final" / "figs" / "en"

C_UP, C_DLIN, C_DINV, C_UNI, C_WARN = "#E8A24B", "#7FB3D5", "#4B86B4", "#9A9A9A", "#C0504D"


def _setup_font(zh=True):
    if zh:
        for f in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
            if os.path.exists(f):
                fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
        plt.rcParams["axes.unicode_minus"] = False
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"; plt.rcParams["axes.unicode_minus"] = True


# ---------------------------------------------------------------- 六族权重（与论文定义 3.1 / 命题 3.1 同口径）
def weight_family(name, t, beta):
    t = np.asarray(t, float); q = t / (1.0 + t)
    if name == "rational":
        return (1.0 + beta * t) / (1.0 + t)
    if name == "linear":
        return 1.0 + (beta - 1.0) * np.clip(t, 0.0, 1.0)
    if name == "band":
        return 1.0 + (beta - 1.0) * (q > 0.5).astype(float)
    if name == "down_lin":
        return 1.0 - (1.0 - 1.0 / beta) * q
    if name == "down_inv":
        return (1.0 + t) / (1.0 + beta * t)
    if name == "uniform":
        return np.ones_like(t)
    raise ValueError(name)


FAMILIES = ["rational", "linear", "band", "down_lin", "down_inv", "uniform"]
BETAS = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0]


# ---------------------------------------------------------------- 合成冻结残差 Gram
def frozen_residual_gram(N, rng, k_factor=3, sigma2=1e-4):
    """K_r = G G^T/N + sigma^2 I：k_factor 个公共梯度方向（配点冗余、近平行）+ 各向同性噪声。
    返回 SPD 的 N×N 配点残差 Gram（模拟激波冻结 θ_k 的 J J^T），非真实网络、结构可控。"""
    G = rng.standard_normal((N, k_factor))
    K = (G @ G.T) / N + sigma2 * np.eye(N)
    return 0.5 * (K + K.T)


def corr_mat(K):
    d = np.diag(K).copy(); d[d <= 0] = np.nan
    Din = np.diag(1.0 / np.sqrt(d))
    R = Din @ K @ Din
    return 0.5 * (R + R.T)


# ---------------------------------------------------------------- 最优对角条件数（多起点 L-BFGS，解析梯度）
def _logkappa_grad(x, R):
    """f(x)=log κ(diag(e^x) R diag(e^x))；解析梯度 ∂f/∂x_i=2(u_max_i^2 - u_min_i^2)。"""
    E = np.exp(np.clip(x, -25.0, 25.0))        # 数值保护，最优解远在此范围内
    A = (E[:, None] * R) * E[None, :]          # E R E
    w, U = np.linalg.eigh(A)
    lmax = w[-1]
    lmin = max(w[0], 1e-12 * lmax)             # 近奇异/极端缩放下 λmin 浮点保护
    g = 2.0 * (U[:, -1] ** 2 - U[:, 0] ** 2)
    return np.log(lmax / lmin), g


def opt_diag_kappa(R, nstart=12, seed=0, x0s=None, ret_info=False, maxiter=600, gtol=1e-7):
    """min_{E≻0 对角} κ(E R E)，返回 κ*_diag（数值最优）。多起点：Jacobi(x=0)+随机。"""
    N = R.shape[0]
    rng = np.random.default_rng(seed)
    starts = [np.zeros(N)]
    if x0s is not None:
        starts = list(x0s)
    else:
        for _ in range(nstart):
            starts.append(rng.standard_normal(N) * 0.6)
    best = np.inf; bestx = None; vals = []; nit = 0
    for x0 in starts:
        r = minimize(lambda z: _logkappa_grad(z, R), x0, jac=True, method="L-BFGS-B",
                     bounds=[(-20.0, 20.0)] * N,
                     options={"maxiter": maxiter, "ftol": 1e-12, "gtol": gtol})
        fv = float(r.fun); vals.append(fv); nit += int(r.nit)
        if fv < best:
            best, bestx = fv, r.x
    kstar = float(np.exp(best))
    if ret_info:
        return kstar, bestx, dict(spread=float(np.exp(max(vals)) / np.exp(min(vals))),
                                  nstart=len(vals), total_nit=nit)
    return kstar


# ---------------------------------------------------------------- (1) R 不变性
def part_R_invariance(N=256, smoke=False):
    rng = np.random.default_rng(20260917)
    K = frozen_residual_gram(N, rng)
    R0 = corr_mat(K); d = np.diag(K)
    t = d / np.median(d)                    # 对比度 t_i（残差能量/批中位）
    rows = []
    maxerr = 0.0
    for fam in FAMILIES:
        for beta in BETAS:
            w = weight_family(fam, t, beta)
            Kw = np.sqrt(w)[:, None] * K * np.sqrt(w)[None, :]
            Rw = corr_mat(Kw)
            err = np.linalg.norm(Rw - R0, "fro") / max(np.linalg.norm(R0, "fro"), 1e-30)
            maxerr = max(maxerr, err)
            rows.append(dict(N=N, family=fam, beta=beta, rel_corr_change=err))
    print(f"[F1b-(1)] R(K_r) 对角权不变性：{len(rows)} 组，最大相对偏差 = {maxerr:.3e}（应 ~1e-15）")
    return rows, maxerr


# ---------------------------------------------------------------- (2) κ*_diag 随 N 夹逼
def part_squeeze(smoke=False):
    Ns = [2, 3, 4, 5, 8, 16, 32] if not smoke else [2, 3, 8]
    nrep = 30 if not smoke else 5
    nstart = 30 if not smoke else 8
    rows = []
    # 两个精确例校验求解器
    #  (a) 等相关 2×2：R=[[1,ρ],[ρ,1]]，Jacobi 最优，κ*=κ(R)，比值 1
    rho = 0.9739
    Req = np.array([[1.0, rho], [rho, 1.0]])
    k_eq = opt_diag_kappa(Req, nstart=10, seed=1)
    kR_eq = (1 + rho) / (1 - rho)
    print(f"[F1b-(2) 校验a] 等相关2x2: κ(R)={kR_eq:.6f} κ*_diag={k_eq:.6f} 比值={k_eq/kR_eq:.8f}（应=1）")
    #  (b) 论文 N=2 闭式例：K=U diag(1,100) U^T，θ=30°；最优 κ=75.49, t*=2.922
    th = np.deg2rad(30.0); c, s = np.cos(th), np.sin(th)
    U = np.array([[c, -s], [s, c]]); K2 = U @ np.diag([1.0, 100.0]) @ U.T
    # 对 K 直接做对角权优化（变量 x，W^{1/2}=diag(e^x)），与对 R 的 ERE 同值（相关归一不改变可达条件数）
    k2, x2, _ = opt_diag_kappa(K2, nstart=10, seed=2, ret_info=True)
    tstar = np.exp(2.0 * (x2[0] - x2[1]))   # W=diag(e^{2x})，论文固定 W=diag(t,1)
    print(f"[F1b-(2) 校验b] 论文N=2闭式: κ*={k2:.4f}（应 75.49）, t*={tstar:.3f}（应 2.922）")
    for N in Ns:
        ratios, kRs, kstars = [], [], []
        for rep in range(nrep):
            rng = np.random.default_rng(7000 + N * 101 + rep)
            K = frozen_residual_gram(N, rng)
            R = corr_mat(K)
            ev = np.linalg.eigvalsh(R)
            kR = ev[-1] / ev[0]
            kstar = opt_diag_kappa(R, nstart=nstart, seed=10 * N + rep)
            kRs.append(kR); kstars.append(kstar); ratios.append(kstar / kR)
        ratios = np.array(ratios)
        lo, med, hi = np.percentile(ratios, [25, 50, 75])
        in_band = np.mean((ratios >= 1.0 / N - 1e-9) & (ratios <= 1.0 + 1e-9))
        rows.append(dict(N=N, nrep=nrep, kappa_R_med=float(np.median(kRs)),
                         kappa_diag_med=float(np.median(kstars)),
                         ratio_med=float(np.median(ratios)), ratio_q1=float(lo), ratio_q3=float(hi),
                         lower_1overN=1.0 / N, upper_1=1.0, frac_in_band=float(in_band)))
        print(f"[F1b-(2)] N={N:2d} n={nrep:2d} κ(R)中位={np.median(kRs):.2e} "
              f"r=κ*/κ(R)中位={np.median(ratios):.4f} IQR[{lo:.4f},{hi:.4f}] 带内={in_band*100:.0f}% [1/N={1/N:.4f},1]")
    return rows, dict(eq_ratio=k_eq / kR_eq, paper_k=k2, paper_tstar=tstar)


# ---------------------------------------------------------------- (3) 非局部 W*=c K^{-1}
def part_nonlocal(smoke=False):
    Ns = [8, 16, 32, 64] if not smoke else [8, 16]
    rows = []
    for N in Ns:
        rng = np.random.default_rng(31337 + N)
        K = frozen_residual_gram(N, rng)
        R = corr_mat(K)
        kR = np.linalg.eigvalsh(R)[-1] / np.linalg.eigvalsh(R)[0]
        t0 = time.perf_counter()
        Winv = np.linalg.inv(K)                       # W_*=c K^{-1}（c 取 1，scale 无关）
        t_inv = time.perf_counter() - t0
        # W_*^{1/2} K W_*^{1/2}=I：取 K^{-1} 的对称平方根合同 K
        ww, Vv = np.linalg.eigh(Winv)
        Wh = Vv @ np.diag(np.sqrt(np.clip(ww, 0, None))) @ Vv.T
        Kt = Wh @ K @ Wh
        kt = np.linalg.eigvalsh(0.5 * (Kt + Kt.T))
        k_nonloc = kt[-1] / kt[0]
        offmat = Winv - np.diag(np.diag(Winv))
        off = np.linalg.norm(offmat, "fro") / np.linalg.norm(Winv, "fro")     # 非对角能量占比
        density = float(np.mean(np.abs(offmat) > 1e-10 * np.abs(Winv).max()))  # 非对角元非零比例（全稠密→1）
        # 同维最优对角作对照
        kstar = opt_diag_kappa(R, nstart=12, seed=N)
        rows.append(dict(N=N, kappa_R=float(kR), kappa_diag_best=float(kstar),
                         kappa_nonlocal_Wstar=float(k_nonloc), ratio_nonloc=float(k_nonloc / kR),
                         wstar_nonlocality=float(off), wstar_density=density, inv_time_s=float(t_inv)))
        print(f"[F1b-(3)] N={N:2d} κ(R)={kR:.2e} 最优对角 κ*={kstar:.2e}（比值{kstar/kR:.3f}） "
              f"W*=K^-1 κ={k_nonloc:.3e}（比值{k_nonloc/kR:.2e}） 非对角能量={off:.3f} 稠密率={density:.3f}")
    return rows


# ---------------------------------------------------------------- (4) 计时
def part_timing(smoke=False):
    Ns_full = [100, 200, 500, 1000] if not smoke else [64, 128]
    Ns_eig = [200, 500, 1000, 2000, 5000, 10000] if not smoke else [128, 256]
    rows = []
    for N in Ns_full:
        rng = np.random.default_rng(909 + N); K = frozen_residual_gram(N, rng); R = corr_mat(K)
        # 固定 50 次 L-BFGS 迭代预算（每次迭代一次稠密特征分解），纯测 O(N^3) 单步成本
        t0 = time.perf_counter()
        _, _, info = opt_diag_kappa(R, nstart=0, seed=0, ret_info=True, maxiter=50, gtol=1e-9)
        t_opt = time.perf_counter() - t0
        # 单次矩阵-向量积（训练单步主导成本量级）O(N^2)，warmup 后多次取均值
        B = rng.standard_normal((N, 8)); _ = R @ B
        REP = 20; t1 = time.perf_counter()
        for _ in range(REP):
            _ = R @ B
        t_mv = (time.perf_counter() - t1) / REP
        rows.append(dict(N=N, kind="diag_opt_full", time_s=t_opt, eigh_calls=info["total_nit"]))
        rows.append(dict(N=N, kind="matvec_x8", time_s=t_mv, eigh_calls=0))
        print(f"[F1b-(4)] N={N:5d} 完整对角优化={t_opt*1e3:9.1f}ms（{info['total_nit']}次特征分解） "
              f"单次矩阵向量积x8={t_mv*1e3:7.2f}ms")
    for N in Ns_eig:
        rng = np.random.default_rng(808 + N); K = frozen_residual_gram(N, rng); R = corr_mat(K)
        t0 = time.perf_counter(); np.linalg.eigvalsh(R); t_eig = time.perf_counter() - t0
        rows.append(dict(N=N, kind="single_eigh", time_s=t_eig, eigh_calls=1))
        print(f"[F1b-(4)] N={N:5d} 单次对称特征分解={t_eig*1e3:9.1f}ms")
    # 幂律斜率
    def slope(kind):
        pts = [(r["N"], r["time_s"]) for r in rows if r["kind"] == kind and r["time_s"] > 0]
        if len(pts) < 2: return np.nan
        x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts])
        return float(np.polyfit(x, y, 1)[0])
    sl = {"diag_opt_full": slope("diag_opt_full"), "single_eigh": slope("single_eigh"),
          "matvec_x8": slope("matvec_x8")}
    print(f"[F1b-(4)] 幂律斜率：完整优化 {sl['diag_opt_full']:.2f}（应≈3）、单次eigh {sl['single_eigh']:.2f}、"
          f"矩阵向量积 {sl['matvec_x8']:.2f}（应≈2）")
    return rows, sl


# ---------------------------------------------------------------- 出图
def _write_csv(name, rows):
    if not rows: return
    keys = sorted({k for r in rows for k in r})
    with open(OUT / name, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)


def plot_squeeze(rows, checks, zh=True):
    _setup_font(zh)
    Ns = np.array([r["N"] for r in rows], float)
    med = np.array([r["ratio_med"] for r in rows]); q1 = np.array([r["ratio_q1"] for r in rows]); q3 = np.array([r["ratio_q3"] for r in rows])
    NN = np.geomspace(Ns.min(), Ns.max(), 100)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.fill_between(NN, 1.0 / NN, 1.0, color=C_UNI, alpha=0.22,
                    label=("van der Sluis 夹逼带 [1/N, 1]" if zh else "van der Sluis band [1/N, 1]"))
    ax.plot(NN, 1.0 / NN, color="#666666", ls="--", lw=1.2)
    ax.plot(NN, np.ones_like(NN), color="#666666", ls=":", lw=1.2)
    ax.errorbar(Ns, med, yerr=[med - q1, q3 - med], fmt="o-", color=C_DINV, lw=2, ms=6, capsize=3,
                label=("数值最优对角权 $\\kappa^*_{\\mathrm{diag}}/\\kappa(R)$（中位/IQR，n=30）" if zh
                       else "Numerical best diagonal $\\kappa^*_{\\mathrm{diag}}/\\kappa(R)$ (med/IQR, n=30)"))
    ax.scatter([2], [1.0], color=C_UP, zorder=5, s=90, marker="*",
               label=("N=2：Jacobi 恰最优，吃上界" if zh else "N=2: Jacobi optimal, upper bound tight"))
    # 非局部 W* 水平（取最大 N 实例的 1/κ(R)）
    rlast = rows[-1]
    ax.scatter([rlast["N"]], [1.0 / rlast["kappa_R_med"]], color=C_WARN, marker="x", s=80,
               label=("非局部 $W_*=cK_r^{-1}$：κ=1（突破对角下界，O(N³)、稠密）" if zh
                      else "Nonlocal $W_*=cK_r^{-1}$: κ=1 (beats diagonal bound, O(N³), dense)"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("维数 N（配点数）" if zh else "Dimension N (collocation points)")
    ax.set_ylabel("归一化最优条件数 $\\kappa^*_{\\mathrm{diag}}/\\kappa(R)$" if zh
                  else "Normalized best condition number")
    ax.set_title(("对角局部权的能力边界：N=2 吃上界，N≥3 留间隙，永不低于 1/N" if zh
                  else "Diagonal weighting limit: tight at N=2, gap for N≥3, floor 1/N"))
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fp = OUT / ("fig_f1b_squeeze_zh.png" if zh else "fig_f1b_squeeze_en.png"); fig.savefig(fp, dpi=160); plt.close(fig)
    return fp


def plot_timing(rows, sl, zh=True):
    _setup_font(zh)
    def pts(k):
        p = sorted([(r["N"], r["time_s"]) for r in rows if r["kind"] == k and r["time_s"] > 0])
        return np.array([a for a, _ in p], float), np.array([b for _, b in p], float)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    Nf, tf = pts("diag_opt_full"); Ne, te = pts("single_eigh"); Nm, tm = pts("matvec_x8")
    ax.plot(Nf, tf, "o-", color=C_DINV, lw=2,
            label=("完整最优对角权（多次特征分解），斜率 %.2f" % sl["diag_opt_full"] if zh
                   else "Full diagonal optimum (many eigh), slope %.2f" % sl["diag_opt_full"]))
    ax.plot(Ne, te, "s--", color=C_WARN, lw=1.6,
            label=("单次稠密对称特征分解，斜率 %.2f" % sl["single_eigh"] if zh
                   else "Single dense eigh, slope %.2f" % sl["single_eigh"]))
    ax.plot(Nm, tm, "^-", color=C_UP, lw=1.6,
            label=("训练单步矩阵-向量积 ×8，斜率 %.2f" % sl["matvec_x8"] if zh
                   else "Train-step matvec ×8, slope %.2f" % sl["matvec_x8"]))
    Nref = np.geomspace(min(Nf.min(), Ne.min()), max(Ne.max(), Nf.max()), 50)
    base = tf[np.argmin(Nf)] / Nf[np.argmin(Nf)] ** 3
    ax.plot(Nref, base * Nref ** 3, ":", color="#888888", lw=1,
            label=("O(N³) 参考（最坏情形上界）" if zh else "O(N³) ref (worst-case bound)"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("维数 N" if zh else "Dimension N"); ax.set_ylabel("墙钟时间 (s)" if zh else "Wall time (s)")
    ax.set_title(("最优对角权需高阶稠密特征分解（N=10⁴ 单次 26 s）；训练单步矩阵-向量积低数个量级" if zh
                  else "Optimal diagonal scaling needs high-order dense eigendecomp. (26 s at N=10⁴); train-step matvec is orders cheaper"))
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8); fig.tight_layout()
    fp = OUT / ("fig_f1b_timing_zh.png" if zh else "fig_f1b_timing_en.png"); fig.savefig(fp, dpi=160); plt.close(fig)
    return fp


def _archive(fps, data_csvs):
    for d in (FINAL_DATA, FINAL_FIG_ZH, FINAL_FIG_EN):
        d.mkdir(parents=True, exist_ok=True)
    import shutil
    for fp in fps:
        dst = FINAL_FIG_ZH if fp.name.endswith("_zh.png") else FINAL_FIG_EN
        shutil.copy2(fp, dst / fp.name)
    for cn in data_csvs:
        p = OUT / cn
        if p.exists(): shutil.copy2(p, FINAL_DATA / cn)


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    tag = "_smoke" if a.smoke else ""
    out = OUT.parent / ("exp_F1b_diag_capability" + tag)
    OUT = out; out.mkdir(parents=True, exist_ok=True)
    print("OUT =", OUT)
    rR, _ = part_R_invariance(smoke=a.smoke); _write_csv("f1b_R_invariance.csv", rR)
    rS, checks = part_squeeze(smoke=a.smoke); _write_csv("f1b_kdiag_vs_N.csv", rS)
    rN = part_nonlocal(smoke=a.smoke); _write_csv("f1b_nonlocal_wstar.csv", rN)
    rT, sl = part_timing(smoke=a.smoke); _write_csv("f1b_timing.csv", rT)
    fps = [plot_squeeze(rS, checks, zh=True), plot_squeeze(rS, checks, zh=False),
           plot_timing(rT, sl, zh=True), plot_timing(rT, sl, zh=False)]
    if not a.smoke:
        _archive(fps, ["f1b_R_invariance.csv", "f1b_kdiag_vs_N.csv", "f1b_nonlocal_wstar.csv", "f1b_timing.csv"])
        print("已归档双版图至 theory/final/figs/{zh,en}、数据至 theory/final/data")
    print("\n[F1b] 完成。产物在", OUT)


if __name__ == "__main__":
    main()
