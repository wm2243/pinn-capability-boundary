# -*- coding: utf-8 -*-
"""
残差空间谱对齐分析（V8 新增；对应理论框架第 6 章假设 [SA] 与定理 6.1）

与 utils/spectral_estimator.py 的分工（审稿人会追问，务必分清）：
  * spectral_estimator 估【参数空间】Hessian H = ∇²L 的极端特征值（维度 = 参数数 p，巨大，
    只能 Lanczos 迭代），用于判断训练点局部是否凸（λmin 符号 / 负曲率维数）；
  * 本模块分析【残差空间】算子：残差 Jacobian J_r ∈ R^{n×p}、残差 Gram K_r = J_r J_r^T ∈ R^{n×n}、
    冻结权重 W = diag(w_i) ∈ R^{n×n}。V8 的谱对齐 [SA]（非对角能量 E_off = Õ(n^{-1/2})）、
    条件数改善 κ_new、慢模态权重 β_sat 都在这个 n×n 空间；n = 配点数（实验取 64~512），可显式分解。

冻结前提（重要）：调用前 criterion 必须已处于冻结态（load_frozen_field / freeze_residual_field），
此时 W 只随 β 变、不依赖当前残差，才对应论文“静态冻结算子”；未冻结时得到的是 state-dependent 量，
只能作旁证、不能用于检验定理 6.1。

所有量均为【离散求积】口径（均匀配点 ×1/n 已吸收进 J_r 的列尺度，不影响条件数与 E_off 的比值）。
"""

import numpy as np
import torch


def residual_vector(model, pde, x):
    """在配点 x(n×1) 上计算残差向量 r ∈ R^n（建图，供逐行求 Jacobian）。"""
    xx = x.detach().clone().requires_grad_(True)
    return pde.compute_residual(model, xx).reshape(-1)


def residual_jacobian(model, pde, x, show=False):
    """J_r[i, j] = ∂r_i/∂θ_j，尺寸 n×p。逐行反向（n≤512 可接受；仅在冻结点计算一次）。"""
    params = [pp for pp in model.parameters() if pp.requires_grad]
    n = x.shape[ 0 ]
    r = residual_vector(model, pde, x)
    rows = []
    for i in range(n):
        gs = torch.autograd.grad(r[ i ], params, retain_graph=(i < n - 1),
                                 create_graph=False, allow_unused=True)
        vec = torch.cat([(g if g is not None else torch.zeros_like(pp)).reshape(-1)
                         for g, pp in zip(gs, params)])
        rows.append(vec.detach())
    J = torch.stack(rows, dim=0)
    if show:
        print(f"  [ResSpec] J_r={tuple(J.shape)} (n={n}, p={J.shape[ 1 ]})")
    return J, r.detach()


def _cond_of_pos(vals, cond_tol):
    """对严格正谱算条件数（剔除 ≤tol·λmax 的数值零空间），返回 (κ, λmin_pos)。"""
    pos = vals[vals > cond_tol * float(vals.max())]
    if pos.numel() == 0:
        return float("nan"), float("nan")
    return float(pos.max()) / float(pos.min()), float(pos.min())


def spectral_alignment(model, pde, crit, x, beta=None, cond_tol=1e-10, w_override=None):
    """
    单点冻结谱对齐诊断。
    w_override: 可选，注入自定义冻结权重（与 x 同长的 1D tensor），用于【受控反例】——
                固定 K_r（同一 x、同一网络），只把权重在配点间置换/错位，单一变量破坏 [SA]。
    返回 (scalars:dict, arrays:dict)。
      scalars: E_off（非对角能量，假设 SA-d 期望 Õ(n^-1/2)）、κ_orig（未加权 K_r 条件数）、
               κ_new（严格加权 W^{1/2} K_r W^{1/2} 条件数，定理 6.1 主量）、κ_diag（对角近似 μ 条件数）、
               β_sat（最慢模态上的 ŵ_k，理论应趋近 β）、ε*（命题 6.2 失配阈值）、SA_holds、improve_ratio。
      arrays : lam/what/mu/w/x/r（供 ŵ-λ log-log 估 a、μ 全谱图）。
    """
    if beta is not None and hasattr(crit, "set_beta"):
        crit.set_beta(beta)
    n = x.shape[ 0 ]
    J, r = residual_jacobian(model, pde, x)
    with torch.no_grad():
        if w_override is None:
            w = crit.get_current_weights(r, x_pde=x.detach()).reshape(-1)   # 冻结权重 w_i
        else:
            w = w_override.detach().to(J.device).reshape(-1)                # 受控注入（错位权重）
            if w.numel() != n:
                raise ValueError(f"w_override 长度 {w.numel()} 与配点数 {n} 不一致")
        K = J @ J.t()
        K = 0.5 * (K + K.t())                                            # 对称化去数值非对称
        lam, U = torch.linalg.eigh(K)                                    # 升序特征值/向量
        lam = lam.clamp_min(0.0)
        What = U.t() @ torch.diag(w) @ U                                 # 模态空间权重 Ŵ = U^T W U
        wh = torch.diagonal(What)
        off = What - torch.diag(wh)
        fro = lambda M: float(torch.linalg.norm(M))
        e_off = fro(off) / max(fro(What), 1e-30)

        sw = torch.sqrt(w.clamp_min(1e-12))
        A = (sw[:, None]) * K * (sw[ None, : ])                          # W^{1/2} K W^{1/2}
        A = 0.5 * (A + A.t())
        lw = torch.linalg.eigvalsh(A).clamp_min(0.0)

        kappa_orig, lmin_o = _cond_of_pos(lam, cond_tol)
        kappa_new, lmin_n = _cond_of_pos(lw, cond_tol)
        mu = (wh.clamp_min(0.0) * lam).cpu().numpy()                     # 对角近似 μ_k = ŵ_k λ_k
        kappa_diag = _safe_cond_np(mu, cond_tol)

        order = torch.argsort(lam)
        beta_sat = float(wh[ order[ 0 ] ])                               # 最慢模态上的权重
        beta_use = float(getattr(crit, "beta", float("nan")))
        eps_star = (beta_use - 1.0) / (1.0 + kappa_orig) if np.isfinite(kappa_orig) else float("nan")
        sa_holds = bool(e_off < eps_star) if np.isfinite(eps_star) else None
        scalars = dict(
            n=int(n), beta=beta_use, E_off=float(e_off),
            kappa_orig=kappa_orig, kappa_new=kappa_new, kappa_diag=kappa_diag,
            lam_min_orig=lmin_o, lam_min_new=lmin_n, beta_sat=beta_sat,
            eps_star=eps_star, SA_holds=sa_holds,
            improve_ratio=kappa_new / max(kappa_orig, 1e-30) if np.isfinite(kappa_orig) else float("nan"),
            w_min=float(w.min()), w_max=float(w.mean()),
        )
        arrays = dict(
            lam=lam.cpu().numpy(), what=wh.cpu().numpy(), mu=mu,
            w=w.detach().cpu().numpy(),
            x=x.detach().cpu().numpy().reshape(-1),
            r=r.detach().cpu().numpy(),
        )
    return scalars, arrays


def _safe_cond_np(mu, tol):
    mu = np.asarray(mu, float)
    if mu.size == 0:
        return float("nan")
    pos = mu[mu > tol * mu.max()]
    return float(pos.max() / pos.min()) if pos.size else float("nan")


# ---------------- log-log 幂律拟合（ŵ-λ 谱形估 a；E_off-n 缩放） ----------------
def fit_loglog_slope(x, y):
    """log y = slope·log x + intercept 的 OLS，返回 slope/intercept/r2/n。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    msk = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lx, ly = np.log(x[msk]), np.log(y[msk])
    if lx.size < 2:
        return dict(slope=float("nan"), intercept=float("nan"), r2=float("nan"), n=int(lx.size))
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    slope, icept = float(coef[ 0 ]), float(coef[ 1 ])
    pred = A @ coef
    r2 = 1.0 - float(((ly - pred) ** 2).sum()) / max(float(((ly - ly.mean()) ** 2).sum()), 1e-30)
    return dict(slope=slope, intercept=icept, r2=r2, n=int(lx.size))


def bootstrap_slope_ci(x, y, n_boot=400, seed=0, alpha=0.05):
    """对 log-log 斜率做 bootstrap，返回 (lo, hi)（1-alpha 置信区间）。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    msk = (x > 0) & (y > 0)
    lx, ly = np.log(x[msk]), np.log(y[msk])
    N = lx.size
    if N < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx_all = np.arange(N)
    slopes = []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=N, replace=True)
        if np.unique(lx[idx]).size < 2:
            continue
        A = np.vstack([lx[idx], np.ones(N)]).T
        c, *_ = np.linalg.lstsq(A, ly[idx], rcond=None)
        slopes.append(float(c[ 0 ]))
    if not slopes:
        return float("nan"), float("nan")
    slopes = np.sort(slopes)
    lo = slopes[max(int(alpha / 2 * len(slopes)), 0)]
    hi = slopes[min(int((1 - alpha / 2) * len(slopes)) - 1, len(slopes) - 1)]
    return float(lo), float(hi)


def estimate_spectral_a(lam, what, n_boot=400, seed=0):
    """
    谱形指数：ŵ_k ∝ λ_k^{-a}（log ŵ = const - a·log λ）。
    返回 a=-slope、95% CI（对 a）、r2、有效点数。a→0 即线性改善段消失（V8 要求独立估 a 并报敏感性）。
    """
    fit = fit_loglog_slope(lam, what)
    a = -fit["slope"]
    lo, hi = bootstrap_slope_ci(lam, what, n_boot, seed)
    return dict(a=float(a), a_ci=(-hi, -lo), r2=fit["r2"], n=fit["n"])
