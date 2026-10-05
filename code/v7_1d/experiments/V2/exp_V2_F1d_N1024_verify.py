# -*- coding: utf-8 -*-
"""F1d N=1024 验证：多种子稳定性 + 混合精度Rayleigh商验证
================================================================
1. 多种子（3个）双精度下计算N=1024的r，看稳定性
2. 对每个种子，用mpmath高精度计算最小特征向量的Rayleigh商，
   对比双精度λmin，验证特征值准确性
3. N=2048不做（审稿人指出双精度极限）
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
import mpmath as mp

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
VERIFY_SEEDS = [0, 10, 20]  # 验证用的种子
DEVICE = r2s.DEVICE

_SRC = _V1EXP.parent.parent.parent
OUT = _SRC / "results" / "v7" / "V2" / "exp_V2_F1d_N1024_verify"
OUT.mkdir(parents=True, exist_ok=True)

model_holder, pde_holder = {}, {}


def find_gate(nu, xt, ut, seed):
    """按指定 seed 直接加载好盆地缓存（不遍历）。"""
    b = enter_basin_cached(nu, seed=seed, xt=xt, ut=ut, reuse=True, **GATE_KW)
    return b, seed


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


def opt_diag_kappa_fixed(R, nstart=6, seed=0, floor=1e-30):
    """修复版：地板1e-30。"""
    from scipy.optimize import minimize
    N = R.shape[0]
    rng = np.random.default_rng(seed)
    starts = [np.zeros(N)]
    for _ in range(nstart):
        starts.append(rng.standard_normal(N) * 0.6)

    def obj(x):
        E = np.exp(np.clip(x, -25.0, 25.0))
        A = (E[:, None] * R) * E[None, :]
        w, U = np.linalg.eigh(A)
        lmax = w[-1]
        lmin = max(w[0], floor * lmax)
        g = 2.0 * (U[:, -1] ** 2 - U[:, 0] ** 2)
        return np.log(lmax / lmin), g

    best = np.inf
    for x0 in starts:
        r = minimize(obj, x0, jac=True, method="L-BFGS-B",
                     bounds=[(-20.0, 20.0)] * N,
                     options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-7})
        if float(r.fun) < best:
            best = float(r.fun)
    return float(np.exp(best))


def rayleigh_quotient_mpmath(R_np, v_np):
    """用mpmath高精度计算Rayleigh商 v^T R v / v^T v。"""
    # 转mpmath矩阵
    R_mp = mp.matrix(R_np.tolist())
    v_mp = mp.matrix(v_np.tolist())
    # v^T R v
    vTR = (v_mp.T * R_mp * v_mp)[0, 0]
    vTv = (v_mp.T * v_mp)[0, 0]
    return float(vTR / vTv)


def measure(N, nu, basin, gate_seed, nstart=6):
    x = torch.linspace(-1, 1, N, device=DEVICE).view(-1, 1)
    t0 = time.perf_counter()
    G = jacobian_rows(model_holder["model"], pde_holder["pde"], x)
    K = 0.5 * (G @ G.T + (G @ G.T).T)
    R = corr_mat(K)
    t_jac = time.perf_counter() - t0

    # 双精度特征值
    t0 = time.perf_counter()
    ev_np, U_np = np.linalg.eigh(R)
    t_eig = time.perf_counter() - t0
    lmax_np = ev_np[-1]
    lmin_np = ev_np[0]
    v_min_np = U_np[:, 0]
    kR_np = lmax_np / max(lmin_np, 1e-30)

    # 双精度特征向量残差
    res_np = np.linalg.norm(R @ v_min_np - lmin_np * v_min_np) / (abs(lmin_np) * np.linalg.norm(v_min_np) + 1e-30)

    # mpmath Rayleigh商验证
    t0 = time.perf_counter()
    lmin_mp = rayleigh_quotient_mpmath(R, v_min_np)
    t_mp = time.perf_counter() - t0
    rel_diff = abs(lmin_mp - lmin_np) / abs(lmin_np)

    # 最优对角权（双精度，地板1e-30）
    t0 = time.perf_counter()
    kstar = opt_diag_kappa_fixed(R, nstart=nstart, seed=1000 + N, floor=1e-30)
    t_opt = time.perf_counter() - t0
    r_val = kstar / kR_np

    row = dict(
        gate_seed=gate_seed, N=N, nu=nu,
        lmax=float(lmax_np),
        lmin_np=float(lmin_np),
        lmin_mp_rayleigh=float(lmin_mp),
        lmin_rel_diff=float(rel_diff),
        eigvec_residual=float(res_np),
        kR=float(kR_np),
        kstar_diag=float(kstar),
        r=float(r_val),
        t_jac=float(t_jac), t_eig=float(t_eig), t_mp=float(t_mp), t_opt=float(t_opt),
        total=float(t_jac + t_eig + t_mp + t_opt),
    )
    return row


def setup(nu, basin):
    cfg = r2s.base_cfg(nu, "F1d_verify")
    model = HardBCPINN(cfg).to(DEVICE)
    pde = SteadyBurgersPDE(cfg)
    model.load_state_dict(basin["state"])
    model.set_sigma(basin["sigma_end"])
    model.eval()
    model_holder["model"] = model
    pde_holder["pde"] = pde


def main():
    nu = 0.005
    N = 1024
    xt, ut = r2s.build_test(nu)

    fn = OUT / f"f1d_N1024_verify.csv"
    fieldnames = ["gate_seed", "N", "nu", "lmax", "lmin_np", "lmin_mp_rayleigh",
                   "lmin_rel_diff", "eigvec_residual", "kR", "kstar_diag", "r",
                   "t_jac", "t_eig", "t_mp", "t_opt", "total"]
    done = set()
    if fn.exists():
        with open(fn, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add(int(r["gate_seed"]))
    new_file = not fn.exists()
    fh = open(fn, "a", newline="", encoding="utf-8-sig")
    wr = csv.DictWriter(fh, fieldnames=fieldnames)
    if new_file:
        wr.writeheader(); fh.flush()

    for gs in VERIFY_SEEDS:
        if gs in done:
            print(f"  gate_seed={gs}: 已存在，跳过", flush=True); continue
        print(f"\n{'='*60}", flush=True)
        print(f"  gate_seed={gs}, N={N}, nu={nu}", flush=True)
        print(f"{'='*60}", flush=True)

        # 按指定 seed 加载好盆地
        basin, actual_seed = find_gate(nu, xt, ut, gs)
        if basin is None or basin.get("l2", 9) >= GATE_L2_MAX:
            print(f"  seed={gs} 无好盆地，跳过", flush=True)
            continue
        setup(nu, basin)

        row = measure(N, nu, basin, gs, nstart=6)
        wr.writerow(row); fh.flush()
        print(f"\n  结果汇总 (gate_seed={gs}):", flush=True)
        print(f"    λmin (双精度)    = {row['lmin_np']:.6e}", flush=True)
        print(f"    λmin (mp Rayleigh)= {row['lmin_mp_rayleigh']:.6e}", flush=True)
        print(f"    相对差异          = {row['lmin_rel_diff']:.3e} ({row['lmin_rel_diff']*100:.3f}%)", flush=True)
        print(f"    特征向量残差      = {row['eigvec_residual']:.3e}", flush=True)
        print(f"    κ(R)              = {row['kR']:.3e}", flush=True)
        print(f"    κ*_diag           = {row['kstar_diag']:.3e}", flush=True)
        print(f"    r = κ*_diag/κ(R)  = {row['r']:.4f}", flush=True)
        print(f"    总时间            = {row['total']:.1f}s", flush=True)

    fh.close()

    # 汇总
    print(f"\n\n{'='*60}", flush=True)
    print(f"  多种子汇总 (N=1024, nu=0.005)", flush=True)
    print(f"{'='*60}", flush=True)
    if fn.exists():
        with open(fn, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        print(f"  {'seed':>6} {'λmin_np':>12} {'λmin_mp':>12} {'rel_diff':>10} {'residual':>10} {'r':>8}", flush=True)
        for r in rows:
            print(f"  {r['gate_seed']:>6} {float(r['lmin_np']):>12.4e} {float(r['lmin_mp_rayleigh']):>12.4e} "
                  f"{float(r['lmin_rel_diff']):>10.3e} {float(r['eigvec_residual']):>10.3e} {float(r['r']):>8.4f}", flush=True)
        if len(rows) >= 2:
            r_vals = [float(r['r']) for r in rows]
            print(f"\n  r的范围: [{min(r_vals):.4f}, {max(r_vals):.4f}], 均值={np.mean(r_vals):.4f}, 标准差={np.std(r_vals):.4f}", flush=True)
            rel_diffs = [float(r['lmin_rel_diff']) for r in rows]
            print(f"  λmin相对差异范围: [{min(rel_diffs):.3e}, {max(rel_diffs):.3e}]", flush=True)
            if max(rel_diffs) < 0.01:
                print(f"  → λmin相对差异<1%，双精度特征值可信", flush=True)
            else:
                print(f"  → ⚠️ λmin相对差异>1%，需进一步调查", flush=True)

    print(f"\n  结果写回: {fn}", flush=True)


if __name__ == "__main__":
    main()
