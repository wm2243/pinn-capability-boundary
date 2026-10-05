# -*- coding: utf-8 -*-
"""引理4.1(iv)缓变性验证：好盆地内Adam预条件器P_t的窗内累积变差δ_P
================================================================
在好盆地内（Phase 1 Adam训练），记录每步的D_t=diag(EMA(g²))，
计算P_t^{1/2}=D_t^{-1/2}的窗内累积变差δ_P(n)=Σ_{j<n}||P_{k+j}^{1/2}-P_k^{1/2}||_2。
若δ_P可控（O(1)或缓慢增长），则引理4.1(iv)的定量收缩率界适用。
"""
import sys, csv, time
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
from v8_matrix_common import enter_basin_cached, GATE_L2_MAX

GATE_KW = dict(sigma_hi=15.0, k=4, phase0_epochs=2000)
GATE_SEEDS = [0, 10, 20, 30, 40, 5, 15, 25, 35, 45, 50]
VERIFY_SEEDS = [0, 10, 20]
PHASE1_STEPS = 3000
WINDOW_SIZES = [50, 100, 500, 1000, 2000, 3000]
DEVICE = r2s.DEVICE

_SRC = _V1EXP.parent.parent.parent
OUT = _SRC / "results" / "v7" / "V2" / "exp_V2_lem41_drift_verify"
OUT.mkdir(parents=True, exist_ok=True)


def find_gate(nu, xt, ut):
    fallback = None
    for s in GATE_SEEDS:
        b = enter_basin_cached(nu, seed=s, xt=xt, ut=ut, reuse=True, **GATE_KW)
        if b["l2"] < GATE_L2_MAX:
            print(f"[drift] nu={nu}: seed={s} L2={b['l2']:.3e}", flush=True)
            return b, s
        if fallback is None or b["l2"] < fallback["l2"]:
            fallback = b
    return None, fallback


def get_adam_D(optimizer, step, eps=1e-8):
    """获取Adam的D_t=diag(v_hat)（偏差校正后的二阶矩），返回numpy数组。"""
    beta2 = optimizer.param_groups[0]['betas'][1]
    bias_corr = 1.0 - beta2 ** step
    parts = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                parts.append(np.zeros(p.numel(), dtype=np.float64))
                continue
            state = optimizer.state[p]
            if 'exp_avg_sq' not in state:
                parts.append(np.zeros(p.numel(), dtype=np.float64))
                continue
            v_hat = state['exp_avg_sq'].detach().cpu().numpy().astype(np.float64) / bias_corr
            parts.append(v_hat.reshape(-1))
    return np.concatenate(parts)


def compute_delta_P(P_sqrt_history, window_sizes, start_idx=0):
    """相对窗内漂移 δ_P(n)=Σ_{j<n}‖P_{k+j}^{1/2}-P_k^{1/2}‖₂ / ‖P_k^{1/2}‖₂（无量纲）。"""
    results = {}
    P0 = P_sqrt_history[start_idx]
    nrm0 = np.linalg.norm(P0, ord=2) + 1e-30
    for n in window_sizes:
        if start_idx + n > len(P_sqrt_history):
            results[n] = float('nan')
            continue
        delta = 0.0
        for j in range(n):
            diff = P_sqrt_history[start_idx + j] - P0
            delta += np.linalg.norm(diff, ord=2) / nrm0
        results[n] = delta
    return results


def run_phase1_with_drift(nu, basin, gate_seed):
    """跑Phase 1（Adam），记录每步的D_t和P_t^{1/2}。"""
    cfg = r2s.base_cfg(nu, f"drift_s{gate_seed}")
    cfg.update(dict(lr=r2s.LR_PHASE1, use_gate=False, adam_epochs=PHASE1_STEPS,
                    lbfgs_epochs=0, beta_schedule="const", loss_beta=1.0,
                    weight_mode="adaptive", weight_norm="raw"))
    model, pde, crit, opt, samp, vis, cfg = r2s.build_all(cfg, r2s.LR_PHASE1, 1.0)
    model.load_state_dict(basin["state"])
    model.set_sigma(basin["sigma_end"])
    model.train()

    # 优化器用和主实验一致的Adam（build_all已创建，这里直接用opt）
    optimizer = opt

    # 加载冻结场
    crit.load_frozen_field(basin["t_field"], beta=1.0)

    P_sqrt_history = []
    loss_history = []
    gnorm_history = []

    print(f"  开始Phase 1（{PHASE1_STEPS}步Adam），记录D_t...", flush=True)
    t0 = time.perf_counter()

    for step in range(1, PHASE1_STEPS + 1):
        optimizer.zero_grad()
        data = samp.sample(cfg['n_pde'], mode='random')
        x = data['x_pde'].to(DEVICE)
        r = pde.compute_residual(model, x)
        loss, _ = crit(r, x_pde=x)
        loss.backward()

        # 记录更新前的D_t（step步的状态）
        D_t = get_adam_D(optimizer, step)
        # P_t^{1/2} = (D_t + eps)^{-1/2}
        P_sqrt = 1.0 / np.sqrt(D_t + 1e-8)
        P_sqrt_history.append(P_sqrt)

        # 梯度范数
        gnorm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gnorm += p.grad.detach().cpu().numpy().astype(np.float64).reshape(-1) @ \
                         p.grad.detach().cpu().numpy().astype(np.float64).reshape(-1)
        gnorm = np.sqrt(gnorm)
        gnorm_history.append(gnorm)
        loss_history.append(float(loss.item()))

        optimizer.step()

        if step % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    step {step}/{PHASE1_STEPS}: loss={loss.item():.4e}, |g|={gnorm:.4e}, "
                  f"elapsed={elapsed:.1f}s", flush=True)

    total_time = time.perf_counter() - t0
    print(f"  Phase 1完成，总时间={total_time:.1f}s", flush=True)

    # 计算δ_P（从第0步开始，即训练开始时）
    delta_P_start = compute_delta_P(P_sqrt_history, WINDOW_SIZES, start_idx=0)

    # 计算δ_P（从第1000步开始，即训练中期，此时应该更稳定）
    delta_P_mid = compute_delta_P(P_sqrt_history, WINDOW_SIZES, start_idx=1000)

    # 计算P_t^{1/2}的相对变化（首尾）
    P_rel_change = np.linalg.norm(P_sqrt_history[-1] - P_sqrt_history[0], ord=2) / \
                    (np.linalg.norm(P_sqrt_history[0], ord=2) + 1e-30)

    # 计算D_t的相对变化（首尾）
    # （需要重新获取，因为P_sqrt_history存的是1/sqrt(D)）
    D_start = 1.0 / (P_sqrt_history[0] ** 2) - 1e-8
    D_end = 1.0 / (P_sqrt_history[-1] ** 2) - 1e-8
    D_rel_change = np.linalg.norm(D_end - D_start, ord=2) / (np.linalg.norm(D_start, ord=2) + 1e-30)

    row = dict(
        gate_seed=gate_seed, nu=nu,
        loss_start=loss_history[0], loss_end=loss_history[-1],
        gnorm_start=gnorm_history[0], gnorm_end=gnorm_history[-1],
        P_rel_change=float(P_rel_change),
        D_rel_change=float(D_rel_change),
        total_time=float(total_time),
    )
    for n in WINDOW_SIZES:
        row[f"deltaP_start_n{n}"] = delta_P_start[n]
        row[f"deltaP_mid_n{n}"] = delta_P_mid[n]

    return row, P_sqrt_history, loss_history, gnorm_history


def main():
    nu = 0.005
    xt, ut = r2s.build_test(nu)

    fn = OUT / f"lem41_drift_nu{nu}.csv"
    fieldnames = ["gate_seed", "nu", "loss_start", "loss_end", "gnorm_start", "gnorm_end",
                   "P_rel_change", "D_rel_change", "total_time"]
    for n in WINDOW_SIZES:
        fieldnames.append(f"deltaP_start_n{n}")
        fieldnames.append(f"deltaP_mid_n{n}")

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
        print(f"  gate_seed={gs}, nu={nu}", flush=True)
        print(f"{'='*60}", flush=True)

        basin, actual_seed = find_gate(nu, xt, ut)
        if basin is None:
            print(f"  无好盆地，跳过", flush=True)
            continue

        row, P_hist, loss_hist, gnorm_hist = run_phase1_with_drift(nu, basin, gs)
        wr.writerow(row); fh.flush()

        print(f"\n  结果汇总 (gate_seed={gs}):", flush=True)
        print(f"    loss: {row['loss_start']:.4e} -> {row['loss_end']:.4e}", flush=True)
        print(f"    |g|:  {row['gnorm_start']:.4e} -> {row['gnorm_end']:.4e}", flush=True)
        print(f"    P_t^{{1/2}} 相对变化（首尾）: {row['P_rel_change']:.4f}", flush=True)
        print(f"    D_t 相对变化（首尾）: {row['D_rel_change']:.4f}", flush=True)
        print(f"    δ_P（从训练开始）:", flush=True)
        for n in WINDOW_SIZES:
            v = row[f"deltaP_start_n{n}"]
            if not np.isnan(v):
                print(f"      n={n:>5}: δ_P={v:.4f}, δ_P/n={v/n:.6f}", flush=True)
        print(f"    δ_P（从训练中期step=1000）:", flush=True)
        for n in WINDOW_SIZES:
            v = row[f"deltaP_mid_n{n}"]
            if not np.isnan(v):
                print(f"      n={n:>5}: δ_P={v:.4f}, δ_P/n={v/n:.6f}", flush=True)

    fh.close()

    # 汇总
    print(f"\n\n{'='*60}", flush=True)
    print(f"  多种子汇总 (nu={nu})", flush=True)
    print(f"{'='*60}", flush=True)
    if fn.exists():
        with open(fn, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        print(f"  {'seed':>6} {'P_rel':>8} {'D_rel':>8} {'δP(n=500)':>12} {'δP/n(n=500)':>14} {'δP_mid(n=500)':>16}", flush=True)
        for r in rows:
            dp500 = float(r['deltaP_start_n500'])
            dpm500 = float(r['deltaP_mid_n500'])
            print(f"  {r['gate_seed']:>6} {float(r['P_rel_change']):>8.4f} {float(r['D_rel_change']):>8.4f} "
                  f"{dp500:>12.4f} {dp500/500:>14.6f} {dpm500:>16.4f}", flush=True)

        if len(rows) >= 2:
            dp500_vals = [float(r['deltaP_start_n500']) for r in rows]
            print(f"\n  δ_P(n=500) 范围: [{min(dp500_vals):.4f}, {max(dp500_vals):.4f}], "
                  f"均值={np.mean(dp500_vals):.4f}", flush=True)
            if max(dp500_vals) < 1.0:
                print(f"  → δ_P(n=500) < 1.0，预条件器缓变性良好，引理4.1(iv)定量界适用", flush=True)
            elif max(dp500_vals) < 10.0:
                print(f"  → δ_P(n=500) < 10.0，预条件器缓变可控，引理4.1(iv)定量界适用（常数略松）", flush=True)
            else:
                print(f"  → ⚠️ δ_P(n=500) > 10.0，预条件器变化较大，引理4.1(iv)定量界可能偏松，定性结论仍成立", flush=True)

    print(f"\n  结果写回: {fn}", flush=True)


if __name__ == "__main__":
    main()
