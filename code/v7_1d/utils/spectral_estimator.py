# -*- coding: utf-8 -*-
"""
Hessian 谱估计器（修订版）

旧版问题：用 100 个随机向量的 Rayleigh 商取最小来估 λ_min —— 高维参数空间中随机向量
几乎正交于最小特征方向，结果严重偏大且随机；再取 abs() 会把负曲率（非凸信号）抹成正的。

新版（无需矩阵求逆，只需 HVP）：
  λ_max : Lanczos LM 求 H 最大特征值
  λ_min : 谱平移 —— 对 B = λ_max I − H（半正定）做 Lanczos LM 得 μ_max，则 λ_min = λ_max − μ_max
          Lanczos 会收敛到极端特征方向，比随机 RQ 可靠；λ_min<0 即真实负曲率，不再 abs 掩盖
  κ     = λ_max / max(|λ_min|, eps)，并返回负曲率标志 n_neg
"""

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh


class HessianSpectralEstimator:
    @staticmethod
    def estimate_condition_number(model, loss_fn, pde_engine, x_spectral,
                                  num_lanczos=40, k_min=2):
        params = [p for p in model.parameters() if p.requires_grad]
        shapes = [p.shape for p in params]
        total_dim = sum(p.numel() for p in params)
        if total_dim < 2:
            raise ValueError(f"Parameter dimension {total_dim} too small")
        call_count = [0]

        def hvp(v_np):
            call_count[0] += 1
            v_t = torch.from_numpy(v_np).to(x_spectral.device).float()
            v_list = _unflatten(v_t, shapes)
            with torch.enable_grad():
                res = pde_engine.compute_residual(model, x_spectral)
                loss, _ = loss_fn(res, x_pde=x_spectral)
                grads = torch.autograd.grad(loss, params, create_graph=True)
                dot = sum((g * v).sum() for g, v in zip(grads, v_list))
                hv = torch.autograd.grad(dot, params)
            out = torch.cat([h.reshape(-1) for h in hv]).detach().cpu().numpy()
            return np.ascontiguousarray(out.ravel(), dtype=np.float64)

        op = LinearOperator((total_dim, total_dim), matvec=hvp, rmatvec=hvp, dtype=np.float64)
        k = min(num_lanczos, total_dim - 2)
        ncv = min(2 * k + 1, total_dim)

        # ---- λ_max（保留符号不取 abs：训练 Hessian 最大特征值应≥0，若为负本身就是退化信号）----
        try:
            emax = eigsh(op, k=1, which='LA', tol=1e-4, maxiter=total_dim, ncv=ncv)
            lam_max = float(emax[0][0])
        except Exception:
            # LA 不收敛时退回 LM（模最大），仅在确为正时采用
            emax = eigsh(op, k=1, which='LM', tol=1e-4, maxiter=total_dim, ncv=ncv)
            lam_max = float(emax[0][0])
        if lam_max <= 0:
            print(f"  [Spectral] 警告：λ_max={lam_max:.3e}≤0，Hessian 半负定，κ/λmin 不可信（Phase0 非凸信号）")

        # ---- λ_min via shifted B = lam_max I − H ----
        def shifted(v):
            return lam_max * v - hvp(v)
        op_shift = LinearOperator((total_dim, total_dim), matvec=shifted, rmatvec=shifted, dtype=np.float64)
        ks = min(k_min, k)
        n_neg = 0
        try:
            bvals = eigsh(op_shift, k=ks, which='LM', tol=1e-4, maxiter=total_dim,
                          ncv=min(2 * ks + 20, total_dim), return_eigenvectors=False)
            mu_max = float(np.max(bvals))
            lam_min = lam_max - mu_max
            # 负曲率维数：B 特征值 > lam_max（等价 H 特征值<0）的个数
            n_neg = int(np.sum(bvals > lam_max + 1e-8 * (1 + lam_max)))
        except Exception as e:
            print(f"  [Spectral] shifted-Lanczos for λ_min failed ({e}); λ_min 置为不可用(NaN)")
            lam_min = float('nan')

        kappa = float(lam_max / max(abs(lam_min), 1e-12)) if np.isfinite(lam_min) else float('nan')
        spectrum = [lam_max, lam_min]
        print(f"  [Spectral] HVP calls={call_count[0]}  λmax={lam_max:.4e} λmin={lam_min:.4e} "
              f"κ={kappa:.3e} neg_curv={n_neg}", flush=True)
        return kappa, lam_max, lam_min, n_neg, spectrum


def _unflatten(flat_tensor, shapes):
    out, off = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(flat_tensor[off:off + n].reshape(s)); off += n
    return out
