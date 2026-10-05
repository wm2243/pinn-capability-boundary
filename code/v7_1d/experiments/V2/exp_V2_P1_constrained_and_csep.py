# -*- coding: utf-8 -*-
"""P1+P5: 约束家族内最优权 + 三元组口径 + C-Sep 直接测量"""
import os, sys, csv, argparse, time
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_V1EXP = _HERE.parent.parent / "v1"
for _p in (str(_V1EXP), str(_V1EXP.parent.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
from models.pinn_FourierFeatures_model import HardBCPINN
from physics.burgers_pde import SteadyBurgersPDE
from exp_F1b_diag_capability import corr_mat, opt_diag_kappa
from v8_matrix_common import enter_basin_cached, GATE_L2_MAX

GATE_KW = dict(sigma_hi=15.0, k=4, phase0_epochs=2000)
GATE_SEEDS = [0, 10, 20, 30, 40, 5, 15, 25, 35, 45, 50]
RATIONAL_BETAS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 100.0]

_SRC = _V1EXP.parent.parent.parent
OUT = _SRC / "results" / "v7" / "V2" / "exp_V2_P1_constrained"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = r2s.DEVICE
model_holder, pde_holder = {}, {}


def find_gate(nu, xt, ut):
    fallback = None
    for s in GATE_SEEDS:
        b = enter_basin_cached(nu, seed=s, xt=xt, ut=ut, reuse=True, **GATE_KW)
        if b["l2"] < GATE_L2_MAX:
            return b, s
        if fallback is None or b["l2"] < fallback["l2"]:
            fallback = b
    return None, fallback


def jacobian_and_residual(model, pde, x):
    model.zero_grad(set_to_none=True)
    r = pde.compute_residual(model, x).reshape(-1)
    N = r.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    r_np = r.detach().cpu().numpy().astype(np.float64)
    for i in range(N):
        gi = torch.autograd.grad(r[i], params, retain_graph=(i < N - 1), allow_unused=True)
        gi = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(gi, params)]).detach().cpu()
        rows.append(gi)
    model.zero_grad(set_to_none=True)
    return torch.stack(rows, 0).numpy().astype(np.float64), r_np


def kappa_of_weighted(K, w):
    Wsqrt = np.sqrt(w)
    H = Wsqrt[:, None] * K * Wsqrt[None, :]
    H = 0.5 * (H + H.T)
    ev = np.linalg.eigvalsh(H)
    return float(ev[-1] / max(ev[0], 1e-30))


def measure(N, nu, basin, nstart=10):
    x = torch.linspace(-1, 1, N, device=DEVICE).view(-1, 1)
    x_np = x.detach().cpu().numpy().ravel()
    G, r = jacobian_and_residual(model_holder["model"], pde_holder["pde"], x)
    K = G @ G.T
    K = 0.5 * (K + K.T)

    ev_K = np.linalg.eigvalsh(K)
    kappa_uniform = float(ev_K[-1] / max(ev_K[0], 1e-30))

    R = corr_mat(K)
    ev_R = np.linalg.eigvalsh(R)
    kappa_jacobi = float(ev_R[-1] / max(ev_R[0], 1e-30))

    t0 = time.perf_counter()
    kappa_unconstr = opt_diag_kappa(R, nstart=nstart, seed=2000 + N)
    t_opt = time.perf_counter() - t0

    t_contrast = np.abs(r) / max(np.median(np.abs(r)), 1e-30)
    best_kappa, best_beta = np.inf, None
    beta_kappas = {}
    for beta in RATIONAL_BETAS:
        w = (1.0 + beta * t_contrast) / (1.0 + t_contrast)
        w = np.clip(w, 1.0, beta)
        k = kappa_of_weighted(K, w)
        beta_kappas[beta] = k
        if k < best_kappa:
            best_kappa, best_beta = k, beta

    shock_mask = np.abs(x_np) < 3.0 * nu
    smooth_mask = ~shock_mask
    if shock_mask.sum() > 0 and smooth_mask.sum() > 0:
        t_S = t_contrast[shock_mask]
        t_O = t_contrast[smooth_mask]
        csep_strict = float(t_S.min() / max(t_O.max(), 1e-30))
        csep_median = float(np.median(t_S) / max(np.median(t_O), 1e-30))
        csep_mean = float(t_S.mean() / max(t_O.mean(), 1e-30))
    else:
        csep_strict = csep_median = csep_mean = float("nan")

    gnorm = np.linalg.norm(G, axis=1)
    cv = float(gnorm.std() / max(gnorm.mean(), 1e-30))

    row = dict(
        N=N, nu=nu,
        kappa_uniform=kappa_uniform,
        kappa_jacobi=kappa_jacobi,
        kappa_unconstr=float(kappa_unconstr),
        r_unconstr=float(kappa_unconstr / max(kappa_jacobi, 1e-30)),
        kappa_rational_best=float(best_kappa),
        rational_best_beta=float(best_beta),
        r_rational=float(best_kappa / max(kappa_jacobi, 1e-30)),
        rational_vs_unconstr=float(best_kappa / max(kappa_unconstr, 1e-30)),
        uniform_vs_jacobi=float(kappa_uniform / max(kappa_jacobi, 1e-30)),
        csep_strict=csep_strict,
        csep_median=csep_median,
        csep_mean=csep_mean,
        cv_g=cv,
        n_shock=int(shock_mask.sum()),
        n_smooth=int(smooth_mask.sum()),
        opt_s=float(t_opt),
    )
    for beta in RATIONAL_BETAS:
        row[f"kappa_beta_{beta:g}"] = beta_kappas[beta]
    return row


def setup(nu, basin):
    cfg = r2s.base_cfg(nu, "P1_probe")
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
    ap.add_argument("--Ns", default="")
    ap.add_argument("--nus", default="")
    a = ap.parse_args()
    Ns = [int(z) for z in a.Ns.split(",") if z.strip()] if a.Ns else (
        [64, 128] if a.smoke else [64, 128, 256, 512])
    nus = [float(z) for z in a.nus.split(",") if z.strip()] if a.nus else (
        [0.005] if a.smoke else [0.005, 0.01, 0.05])

    fn = OUT / ("p1_constrained_smoke.csv" if a.smoke else "p1_constrained.csv")
    fieldnames = ["N", "nu", "gate_seed", "gate_l2", "kappa_uniform", "kappa_jacobi",
                  "kappa_unconstr", "r_unconstr", "kappa_rational_best", "rational_best_beta",
                  "r_rational", "rational_vs_unconstr", "uniform_vs_jacobi",
                  "csep_strict", "csep_median", "csep_mean", "cv_g", "n_shock", "n_smooth",
                  "opt_s"] + [f"kappa_beta_{b:g}" for b in RATIONAL_BETAS]
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
            print(f"[P1] ν={nu}: 无好盆地，跳过", flush=True); continue
        setup(nu, basin)
        print(f"[P1] ν={nu}: gate L2={basin['l2']:.3e} σ={basin['sigma_end']:g} N={Ns}", flush=True)
        for N in Ns:
            if (nu, N) in done:
                print(f"  ν={nu} N={N:5d}: 已存在，跳过", flush=True); continue
            t0 = time.perf_counter()
            nstart = 10 if N <= 256 else 6
            try:
                row = measure(N, nu, basin, nstart=nstart)
            except Exception as e:
                print(f"  ν={nu} N={N:5d}: 失败 {type(e).__name__}: {e}", flush=True)
                fh.close(); raise
            row["gate_seed"] = gate_seed
            row["gate_l2"] = float(basin["l2"])
            wr.writerow(row); fh.flush()
            print(f"  ν={nu} N={N:5d}: κ_unif={row['kappa_uniform']:.3e} "
                  f"κ_jac={row['kappa_jacobi']:.3e} κ_unc={row['kappa_unconstr']:.3e} "
                  f"r_unc={row['r_unconstr']:.4f} κ_rat={row['kappa_rational_best']:.3e}(β={row['rational_best_beta']:g}) "
                  f"r_rat={row['r_rational']:.4f} C-Sep_med={row['csep_median']:.2f} "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
    fh.close()
    print("[P1] 写回", fn, flush=True)


if __name__ == "__main__":
    main()
