# -*- coding: utf-8 -*-
"""F1d λmin 严格验证：反幂法 + 残差检验 + 多方法交叉验证
================================================================
验证 N=1024, 2048 的最小特征值是否真实。
方法：
  1. np.linalg.eigvalsh (稠密分治)
  2. scipy.sparse.linalg.eigsh shift-invert
  3. 反幂法 (scipy.linalg.solve 直接解方程)
  4. 特征向量残差检验 ||Rv - λv|| / ||v||
  5. Rayleigh 商验证
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve as scipy_solve
from scipy.sparse.linalg import eigsh

_HERE = Path(__file__).resolve()
_V1EXP = _HERE.parent.parent / "v1"
for _p in (str(_V1EXP), str(_V1EXP.parent.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
from models.pinn_FourierFeatures_model import HardBCPINN
from physics.burgers_pde import SteadyBurgersPDE
from exp_F1b_diag_capability import corr_mat
from v8_matrix_common import enter_basin_cached, GATE_L2_MAX

GATE_KW = dict(sigma_hi=15.0, k=4, phase0_epochs=2000)
GATE_SEEDS = [0, 10, 20, 30, 40, 5, 15, 25, 35, 45, 50]
DEVICE = r2s.DEVICE


def find_gate(nu, xt, ut):
    fallback = None
    for s in GATE_SEEDS:
        b = enter_basin_cached(nu, seed=s, xt=xt, ut=ut, reuse=True, **GATE_KW)
        if b["l2"] < GATE_L2_MAX:
            print(f"[verify] nu={nu}: seed={s} L2={b['l2']:.3e}", flush=True)
            return b, s
        if fallback is None or b["l2"] < fallback["l2"]:
            fallback = b
    return None, fallback


def jacobian_rows(model, pde, x):
    model.zero_grad(set_to_none=True)
    r = pde.compute_residual(model, x).reshape(-1)
    N = r.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(N):
        gi = torch.autograd.grad(r[i], params, retain_graph=(i < N - 1), allow_unused=True)
        gi = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(gi, params)]).detach().cpu()
        rows.append(gi)
    model.zero_grad(set_to_none=True)
    return torch.stack(rows, 0).numpy().astype(np.float64)


def inverse_iteration(R, v0=None, max_iter=200, tol=1e-14):
    """反幂法：求最小特征值和特征向量。
    每步解 R x = v，归一化，Rayleigh商得λ。
    """
    N = R.shape[0]
    if v0 is None:
        rng = np.random.default_rng(42)
        v = rng.standard_normal(N)
    else:
        v = v0.copy()
    v = v / np.linalg.norm(v)
    
    lam_prev = 0
    for it in range(max_iter):
        # 解 R x = v
        try:
            x = scipy_solve(R, v, assume_a='pos')
        except Exception:
            # 如果正定假设失败，用通用解法
            x = np.linalg.solve(R, v)
        x = x / np.linalg.norm(x)
        # Rayleigh 商
        lam = float(x @ R @ x)
        # 残差
        res = np.linalg.norm(R @ x - lam * x) / (abs(lam) * np.linalg.norm(x) + 1e-30)
        if abs(lam - lam_prev) / (abs(lam) + 1e-30) < tol and res < 1e-10:
            print(f"    反幂法收敛: iter={it+1} λ={lam:.6e} res={res:.3e}", flush=True)
            return lam, x, res, it + 1
        v = x
        lam_prev = lam
    print(f"    反幂法未收敛: iter={max_iter} λ={lam:.6e} res={res:.3e}", flush=True)
    return lam, x, res, max_iter


def verify_N(N, nu, model, pde):
    print(f"\n{'='*70}", flush=True)
    print(f"  N={N}, ν={nu}", flush=True)
    print(f"{'='*70}", flush=True)
    
    x = torch.linspace(-1, 1, N, device=DEVICE).view(-1, 1)
    t0 = time.perf_counter()
    G = jacobian_rows(model, pde, x)
    K = 0.5 * (G @ G.T + (G @ G.T).T)
    R = corr_mat(K)
    print(f"  Jacobian+Gram 构建: {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  R shape: {R.shape}, 对角线范围: [{np.diag(R).min():.4f}, {np.diag(R).max():.4f}]", flush=True)
    
    # 方法1: np.linalg.eigvalsh
    t0 = time.perf_counter()
    ev = np.linalg.eigvalsh(R)
    t1 = time.perf_counter() - t0
    lmax_1 = ev[-1]
    lmin_1 = ev[0]
    print(f"\n  [方法1] np.linalg.eigvalsh ({t1:.2f}s)", flush=True)
    print(f"    λmax = {lmax_1:.6e}", flush=True)
    print(f"    λmin = {lmin_1:.6e}", flush=True)
    print(f"    κ    = {lmax_1/max(lmin_1,1e-30):.3e}", flush=True)
    
    # 方法2: shift-invert eigsh
    t0 = time.perf_counter()
    try:
        ev2, vec2 = eigsh(R, k=3, sigma=0.0, which='LM', 
                           return_eigenvectors=True, maxiter=20000, tol=1e-14)
        ev2 = np.sort(ev2)
        lmin_2 = ev2[0]
        # 残差检验
        res2 = np.linalg.norm(R @ vec2[:, 0] - lmin_2 * vec2[:, 0]) / (abs(lmin_2) * np.linalg.norm(vec2[:, 0]) + 1e-30)
        print(f"\n  [方法2] eigsh shift-invert ({time.perf_counter()-t0:.2f}s)", flush=True)
        print(f"    λmin = {lmin_2:.6e}", flush=True)
        print(f"    残差 ||Rv-λv||/(|λ|·||v||) = {res2:.3e}", flush=True)
        print(f"    最小3个特征值: {ev2[:3]}", flush=True)
    except Exception as e:
        print(f"\n  [方法2] eigsh 失败: {e}", flush=True)
        lmin_2 = float('nan')
        res2 = float('nan')
    
    # 方法3: 反幂法
    t0 = time.perf_counter()
    lmin_3, vec3, res3, niter = inverse_iteration(R, max_iter=300, tol=1e-15)
    print(f"\n  [方法3] 反幂法 ({time.perf_counter()-t0:.2f}s, {niter}次迭代)", flush=True)
    print(f"    λmin = {lmin_3:.6e}", flush=True)
    print(f"    残差 ||Rv-λv||/(|λ|·||v||) = {res3:.3e}", flush=True)
    
    # 方法4: 用方法1的特征向量做残差检验
    # 重新eigh拿特征向量
    ev_full, vec_full = np.linalg.eigh(R)
    lmin_4 = ev_full[0]
    vec_min = vec_full[:, 0]
    res4 = np.linalg.norm(R @ vec_min - lmin_4 * vec_min) / (abs(lmin_4) * np.linalg.norm(vec_min) + 1e-30)
    # Rayleigh 商验证
    rayleigh = float(vec_min @ R @ vec_min) / float(vec_min @ vec_min)
    print(f"\n  [方法4] eigh特征向量残差检验", flush=True)
    print(f"    λmin = {lmin_4:.6e}", flush=True)
    print(f"    Rayleigh商 = {rayleigh:.6e}", flush=True)
    print(f"    残差 ||Rv-λv||/(|λ|·||v||) = {res4:.3e}", flush=True)
    
    # 汇总对比
    print(f"\n  {'='*50}", flush=True)
    print(f"  汇总对比:", flush=True)
    print(f"    方法1 (eigvalsh):     λmin = {lmin_1:.6e}", flush=True)
    print(f"    方法2 (shift-invert): λmin = {lmin_2:.6e}  残差={res2:.3e}", flush=True)
    print(f"    方法3 (反幂法):       λmin = {lmin_3:.6e}  残差={res3:.3e}", flush=True)
    print(f"    方法4 (eigh+残差):    λmin = {lmin_4:.6e}  残差={res4:.3e}", flush=True)
    
    # 相对差异
    vals = [lmin_1, lmin_2, lmin_3, lmin_4]
    valid = [v for v in vals if not np.isnan(v) and v > 0]
    if len(valid) >= 2:
        vmin, vmax = min(valid), max(valid)
        rel_diff = (vmax - vmin) / vmin
        print(f"\n  最大相对差异: {rel_diff:.3e} ({rel_diff*100:.3f}%)", flush=True)
        if rel_diff < 0.01:
            print(f"  → 四种方法一致（差异<1%），λmin 可信", flush=True)
        elif rel_diff < 0.1:
            print(f"  → 四种方法基本一致（差异<10%），λmin 大致可信", flush=True)
        else:
            print(f"  → ⚠️ 方法间差异较大，需要进一步调查", flush=True)
    
    # 检查是否有聚类的小特征值
    print(f"\n  最小10个特征值: {ev[:10]}", flush=True)
    print(f"  最大10个特征值: {ev[-10:]}", flush=True)
    print(f"  有效秩(λ>1e-10): {np.sum(ev > 1e-10)}/{N}", flush=True)
    
    return dict(N=N, lmin_1=lmin_1, lmin_2=lmin_2, lmin_3=lmin_3, lmin_4=lmin_4,
                res2=res2, res3=res3, res4=res4)


def main():
    nu = 0.005
    xt, ut = r2s.build_test(nu)
    basin, gate_seed = find_gate(nu, xt, ut)
    if basin is None:
        print("无好盆地，退出", flush=True)
        return
    
    cfg = r2s.base_cfg(nu, "F1d_verify")
    model = HardBCPINN(cfg).to(DEVICE)
    pde = SteadyBurgersPDE(cfg)
    model.load_state_dict(basin["state"])
    model.set_sigma(basin["sigma_end"])
    model.eval()
    
    results = []
    for N in [1024, 2048]:
        r = verify_N(N, nu, model, pde)
        results.append(r)
    
    print(f"\n\n{'='*70}", flush=True)
    print(f"  最终结论", flush=True)
    print(f"{'='*70}", flush=True)
    for r in results:
        print(f"  N={r['N']}: 四种方法λmin = "
              f"{r['lmin_1']:.4e} / {r['lmin_2']:.4e} / {r['lmin_3']:.4e} / {r['lmin_4']:.4e}", flush=True)


if __name__ == "__main__":
    main()
