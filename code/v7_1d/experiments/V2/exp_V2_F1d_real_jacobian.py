# -*- coding: utf-8 -*-
"""实验 F1d（真实 Burgers 冻结 Jacobian，回应审稿人 N>=1e3 要求）
================================================================================
目标：在一个【真实 Gate 冻结点】（ν=.005 好盆地，新边界编码）上，跨配点数 N=64..2048
      现算残差对参数的 Jacobian J_i=∇θ r_i、构 K=J J^T，复用 F1b/B 实验的同一判据：
        R = corr_mat(K) = D^{-1/2} K D^{-1/2}
        κ(R)=λmax/λmin(R)；κ*_diag = opt_diag_kappa(R)（多起点 L-BFGS-B）
        r = κ*_diag/κ(R)
      并报告：
        幅度失配 CV(g)=std(||g_i||)/mean(||g_i||)（g_i=∇θ r_i）
        软模态参与率/有效秩  PR=(Σλ)^2/Σλ^2（R 的特征值）
      检验：真实激波冻结点的 r 是否随 N>=1e3 仍钉在 ~1（近等相关/方向共线类），
            从而证实 F1c(d) 的高 r 不是 N=8/64 外推、而是结构性质（命题 prop:equicorr）。
      纯诊断、不训练；从 _gate_cache 读冻结点（ν>=.1 的旧边界编码点已被 archive 移走，不用）。
产出 results/v7/V2/exp_V2_F1d_real_jacobian/
用法：python exp_V2_F1d_real_jacobian.py [--smoke]
"""
import os, sys, csv, argparse, time
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_V1EXP = _HERE.parent.parent / "v1"          # v7/experiments/v1
for _p in (str(_V1EXP), str(_V1EXP.parent.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s                 # 复用 build_test/base_cfg/build_all/DEVICE
from models.pinn_FourierFeatures_model import HardBCPINN
from physics.burgers_pde import SteadyBurgersPDE
from exp_F1b_diag_capability import corr_mat, opt_diag_kappa
from v8_matrix_common import enter_basin_cached, GATE_L2_MAX

# 默认 gate：σ=15 / K=4 / phase0=2000；seed 按规则等距、预固定顺序枚举，命中好盆地即取（不挑种子）
GATE_KW = dict(sigma_hi=15.0, k=4, phase0_epochs=2000)
GATE_SEEDS = [0, 10, 20, 30, 40, 5, 15, 25, 35, 45, 50]


def find_gate(nu, xt, ut):
    """按 GATE_SEEDS 顺序枚举 K=4 phase0 池，返回第一个 L2<GATE_L2_MAX 的好盆地；
    全部不达标则返回 (None, 最优盆地)。命中即取，不跨种子挑全局最优，避免挑种子。"""
    fallback = None
    for s in GATE_SEEDS:
        b = enter_basin_cached(nu, seed=s, xt=xt, ut=ut, reuse=True, **GATE_KW)
        if b["l2"] < GATE_L2_MAX:
            print(f"[F1d] ν={nu}: 选用规则种子 seed={s} 的好盆地 L2={b['l2']:.3e}", flush=True)
            return b, s
        if fallback is None or b["l2"] < fallback["l2"]:
            fallback = b
    return None, fallback
_SRC = _V1EXP.parent.parent.parent            # 源代码/
OUT = _SRC / "results" / "v7" / "V2" / "exp_V2_F1d_real_jacobian"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = r2s.DEVICE


def jacobian_rows(model, pde, x):
    """返回 G (N,p) = [∇θ r_i]，逐行 CPU；K=G G^T。x:(N,1)。"""
    model.zero_grad(set_to_none=True)
    r = pde.compute_residual(model, x).reshape(-1)
    N = r.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(N):
        gi = torch.autograd.grad(r[i], params, retain_graph=(i < N - 1),
                                 allow_unused=True)
        gi = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(gi, params)]).detach().cpu()
        rows.append(gi)
    model.zero_grad(set_to_none=True)
    return torch.stack(rows, 0).numpy().astype(np.float64)


def measure(N, nu, basin, nstart=12):
    x = torch.linspace(-1, 1, N, device=DEVICE).view(-1, 1)
    G = jacobian_rows(model_holder["model"], pde_holder["pde"], x)
    K = (G @ G.T)
    K = 0.5 * (K + K.T)
    R = corr_mat(K)
    ev = np.linalg.eigvalsh(R)
    kR = ev[-1] / max(ev[0], 1e-30)
    t0 = time.perf_counter()
    kstar = opt_diag_kappa(R, nstart=nstart, seed=1000 + N)
    t_opt = time.perf_counter() - t0
    gnorm = np.linalg.norm(G, axis=1)
    cv = float(gnorm.std() / max(gnorm.mean(), 1e-30))
    s1, s2 = ev.sum(), (ev ** 2).sum()
    pr = float((s1 ** 2) / max(s2, 1e-30))            # 有效秩参与率（近等相关 ~N）
    row = dict(N=N, nu=nu, kappa_R=float(kR), kappa_diag=float(kstar),
               ratio_r=float(kstar / kR), cv_g=cv, eff_rank_PR=pr,
               lambda_max_R=float(ev[-1]), lambda_min_R=float(ev[0]), opt_s=float(t_opt))
    return row


# 模块级 model/pde（basin 加载一次）
model_holder, pde_holder = {}, {}


def setup(nu, basin):
    cfg = r2s.base_cfg(nu, "F1d_probe")
    model = HardBCPINN(cfg).to(DEVICE)
    pde = SteadyBurgersPDE(cfg)
    model.load_state_dict(basin["state"])
    model.set_sigma(basin["sigma_end"])
    model.eval()
    model_holder["model"] = model
    pde_holder["pde"] = pde


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--Ns", default="", help="逗号分隔的 N 列表，覆盖默认")
    ap.add_argument("--nus", default="", help="逗号分隔的 ν 列表，覆盖默认")
    a = ap.parse_args()
    if a.Ns:
        Ns = [int(z) for z in a.Ns.split(",") if z.strip()]
    else:
        Ns = [64, 256] if a.smoke else [64, 128, 256, 512]
    if a.nus:
        nus = [float(z) for z in a.nus.split(",") if z.strip()]
    else:
        nus = [0.005] if a.smoke else [0.003, 0.005, 0.01, 0.05, 0.001]

    fn = OUT / ("f1d_real_jacobian_smoke.csv" if a.smoke else "f1d_real_jacobian.csv")
    fieldnames = ["N", "nu", "gate_seed", "gate_l2", "kappa_R", "kappa_diag", "ratio_r", "cv_g",
                  "eff_rank_PR", "lambda_max_R", "lambda_min_R", "opt_s", "total_s"]
    done = set()
    if fn.exists():                      # 断点续跑：读已完成的 (nu,N)
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
            print(f"[F1d] ν={nu}: 枚举 {len(GATE_SEEDS)} 个规则种子池均无 L2<{GATE_L2_MAX:g} 的好盆地"
                  f"（最优 L2={gate_seed['l2']:.3e}），跳过（极僵端好盆地罕见本身是信息）", flush=True)
            continue
        setup(nu, basin)
        print(f"[F1d] ν={nu}: gate L2={basin['l2']:.3e} σ={basin['sigma_end']:g} "
              f"device={DEVICE}；N={Ns}", flush=True)
        for N in Ns:
            if (nu, N) in done:
                print(f"  ν={nu} N={N:5d}: 已存在，跳过", flush=True); continue
            t0 = time.perf_counter()
            nstart = 12 if N <= 128 else (8 if N == 256 else 6)
            try:
                row = measure(N, nu, basin, nstart=nstart)
            except Exception as e:
                print(f"  ν={nu} N={N:5d}: 失败 {type(e).__name__}: {e}", flush=True)
                fh.close(); raise
            row["gate_seed"] = gate_seed; row["gate_l2"] = float(basin["l2"])
            row["total_s"] = time.perf_counter() - t0
            wr.writerow(row); fh.flush()
            print(f"  ν={nu} N={N:5d}: κR={row['kappa_R']:.3e} κdiag={row['kappa_diag']:.3e} "
                  f"r={row['ratio_r']:.4f} CV={row['cv_g']:.3f} PR={row['eff_rank_PR']:.1f} "
                  f"({row['total_s']:.1f}s)", flush=True)
    fh.close()
    print("[F1d] 写回", fn, flush=True)


if __name__ == "__main__":
    main()
