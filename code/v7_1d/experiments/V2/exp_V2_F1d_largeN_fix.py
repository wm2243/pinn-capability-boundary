# -*- coding: utf-8 -*-
"""F1d 大N补实验：修复 1e-12 地板问题 + shift-invert 验证 λmin
================================================================================
问题：opt_diag_kappa 的 _logkappa_grad 里 lmin = max(w[0], 1e-12*lmax)，
     N>=1024 时真实 λmin < 1e-12*λmax，κ_diag 被地板限制，r=κ_diag/κ_R 失真。
修复：地板改为 1e-30；同时用 scipy shift-invert 验证最小特征值。
只跑 N=1024, 2048，ν=0.005（数据正常的那个）。
产出 results/v7/V2/exp_V2_F1d_largeN_fix/
用法：python exp_V2_F1d_largeN_fix.py [--smoke]
"""
import os, sys, csv, argparse, time
from pathlib import Path
import numpy as np
import torch
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

_SRC = _V1EXP.parent.parent.parent
OUT = _SRC / "results" / "v7" / "V2" / "exp_V2_F1d_largeN_fix"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = r2s.DEVICE

model_holder, pde_holder = {}, {}


def find_gate(nu, xt, ut):
    fallback = None
    for s in GATE_SEEDS:
        b = enter_basin_cached(nu, seed=s, xt=xt, ut=ut, reuse=True, **GATE_KW)
        if b["l2"] < GATE_L2_MAX:
            print(f"[F1d-fix] nu={nu}: seed={s} L2={b['l2']:.3e}", flush=True)
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


def opt_diag_kappa_fixed(R, nstart=6, seed=0, floor=1e-30):
    """修复版：地板从 1e-12 改为 floor（默认 1e-30）。"""
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


def shift_invert_lmin(R, k=5):
    """用 shift-invert 模式算最小的 k 个特征值。"""
    try:
        # shift-invert: 找 R 的最小特征值 = 找 R^{-1} 的最大特征值
        # sigma=0 附近，which='LM' 找模最大的（即 R^{-1} 的大特征值）
        ev, _ = eigsh(R, k=k, sigma=0.0, which='LM', return_eigenvectors=True,
                       maxiter=10000, tol=1e-14)
        return np.sort(ev)
    except Exception as e:
        print(f"  shift-invert 失败: {e}", flush=True)
        return None


def measure(N, nu, basin, nstart=6):
    x = torch.linspace(-1, 1, N, device=DEVICE).view(-1, 1)
    G = jacobian_rows(model_holder["model"], pde_holder["pde"], x)
    K = 0.5 * (G @ G.T + (G @ G.T).T)
    R = corr_mat(K)

    # 普通 eigh
    t0 = time.perf_counter()
    ev = np.linalg.eigvalsh(R)
    t_eigh = time.perf_counter() - t0
    lmax_eigh = ev[-1]
    lmin_eigh = ev[0]
    kR_eigh = lmax_eigh / max(lmin_eigh, 1e-30)

    # shift-invert 验证
    t0 = time.perf_counter()
    ev_si = shift_invert_lmin(R, k=min(5, N-2))
    t_si = time.perf_counter() - t0
    if ev_si is not None:
        lmin_si = ev_si[0]
        lmin_si_k = ev_si  # 最小的k个
    else:
        lmin_si = float('nan')
        lmin_si_k = None

    # 修复版优化（地板 1e-30）
    t0 = time.perf_counter()
    kstar_fix = opt_diag_kappa_fixed(R, nstart=nstart, seed=1000 + N, floor=1e-30)
    t_opt_fix = time.perf_counter() - t0

    # 原版优化（地板 1e-12，作为对比）
    from exp_F1b_diag_capability import opt_diag_kappa
    t0 = time.perf_counter()
    kstar_orig = opt_diag_kappa(R, nstart=nstart, seed=1000 + N)
    t_opt_orig = time.perf_counter() - t0

    row = dict(
        N=N, nu=nu,
        lmax_eigh=float(lmax_eigh),
        lmin_eigh=float(lmin_eigh),
        kR_eigh=float(kR_eigh),
        lmin_si=float(lmin_si) if ev_si is not None else float('nan'),
        kstar_orig_floor1e12=float(kstar_orig),
        kstar_fix_floor1e30=float(kstar_fix),
        r_orig=float(kstar_orig / kR_eigh),
        r_fix=float(kstar_fix / kR_eigh),
        t_eigh=float(t_eigh), t_si=float(t_si),
        t_opt_orig=float(t_opt_orig), t_opt_fix=float(t_opt_fix),
    )
    if lmin_si_k is not None:
        for i, v in enumerate(lmin_si_k):
            row[f'lmin_si_{i}'] = float(v)
    return row


def setup(nu, basin):
    cfg = r2s.base_cfg(nu, "F1d_fix")
    model = HardBCPINN(cfg).to(DEVICE)
    pde = SteadyBurgersPDE(cfg)
    model.load_state_dict(basin["state"])
    model.set_sigma(basin["sigma_end"])
    model.eval()
    model_holder["model"] = model
    pde_holder["pde"] = pde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    # smoke 档用 N=128 走通 jacobian→eigh→shift-invert→两套对角优化全链路（分钟级）；
    # N=1024/2048 正式测量单次约 1–2 小时（dense shift-invert + 多起点 eigh 优化），仅正式重跑使用。
    Ns = [128] if a.smoke else [1024, 2048]
    nus = [0.005]

    fn = OUT / ("f1d_largeN_fix_smoke.csv" if a.smoke else "f1d_largeN_fix.csv")
    fieldnames = ["N", "nu", "lmax_eigh", "lmin_eigh", "kR_eigh",
                  "lmin_si", "kstar_orig_floor1e12", "kstar_fix_floor1e30",
                  "r_orig", "r_fix", "t_eigh", "t_si", "t_opt_orig", "t_opt_fix",
                  "lmin_si_0", "lmin_si_1", "lmin_si_2", "lmin_si_3", "lmin_si_4", "total_s"]
    done = set()
    if fn.exists():
        with open(fn, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((float(r["nu"]), int(r["N"])))
    new_file = not fn.exists()
    fh = open(fn, "a", newline="", encoding="utf-8-sig")
    wr = csv.DictWriter(fh, fieldnames=fieldnames)
    if new_file:
        wr.writeheader(); fh.flush()

    for nu in nus:
        xt, ut = r2s.build_test(nu)
        basin, gate_seed = find_gate(nu, xt, ut)
        if basin is None:
            print(f"[F1d-fix] nu={nu}: 无好盆地，跳过", flush=True)
            continue
        setup(nu, basin)
        print(f"[F1d-fix] nu={nu}: gate L2={basin['l2']:.3e} N={Ns}", flush=True)
        for N in Ns:
            if (nu, N) in done:
                print(f"  N={N}: 已存在，跳过", flush=True); continue
            t0 = time.perf_counter()
            nstart = 2 if a.smoke else 6
            try:
                row = measure(N, nu, basin, nstart=nstart)
            except Exception as e:
                print(f"  N={N}: 失败 {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()
                fh.close(); raise
            row["total_s"] = time.perf_counter() - t0
            wr.writerow(row); fh.flush()
            print(f"  N={N:5d}: lmin_eigh={row['lmin_eigh']:.3e} lmin_si={row['lmin_si']:.3e} "
                  f"kR={row['kR_eigh']:.3e} r_orig={row['r_orig']:.4f} r_fix={row['r_fix']:.4f} "
                  f"({row['total_s']:.1f}s)", flush=True)
    fh.close()
    print("[F1d-fix] 写回", fn, flush=True)


if __name__ == "__main__":
    main()
