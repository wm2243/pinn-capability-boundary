# -*- coding: utf-8 -*-
"""S3补测：Adam预条件器绝对平台项 vs 白化梯度范数的比值
基于exp_V2_lem41_drift_verify.py的框架，额外记录白化梯度||g̃||=||P^{1/2}g||。
"""
import sys, csv, time, json
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_V1EXP = _HERE.parent.parent / "v1"
for _p in (str(_V1EXP), str(_V1EXP.parent.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
from v8_matrix_common import enter_basin_cached, GATE_L2_MAX

GATE_KW = dict(sigma_hi=15.0, k=4, phase0_epochs=2000)
GATE_SEEDS = [0, 10, 20, 30, 40, 5, 15, 25, 35, 45, 50]
VERIFY_SEEDS = [0, 10, 20]
PHASE1_STEPS = 1000
DEVICE = r2s.DEVICE

_SRC = _V1EXP.parent.parent.parent
OUT = _SRC / "results" / "v7" / "V2" / "exp_V2_s3_platform_ratio"
OUT.mkdir(parents=True, exist_ok=True)


def find_gate(nu, xt, ut):
    fallback = None
    for s in GATE_SEEDS:
        b = enter_basin_cached(nu, seed=s, xt=xt, ut=ut, reuse=True, **GATE_KW)
        if b["l2"] < GATE_L2_MAX:
            print(f"[s3] nu={nu}: seed={s} L2={b['l2']:.3e}", flush=True)
            return b, s
        if fallback is None or b["l2"] < fallback["l2"]:
            fallback = b
    return None, fallback


def get_adam_Psqrt_and_g(optimizer, step, eps=1e-8):
    """获取Adam的P^{1/2}=D^{-1/2}（偏差校正后）和当前梯度g。"""
    beta2 = optimizer.param_groups[0]['betas'][1]
    bias_corr = 1.0 - beta2 ** max(step, 1)
    psqrt_parts = []
    g_parts = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                psqrt_parts.append(np.ones(p.numel(), dtype=np.float64))
                g_parts.append(np.zeros(p.numel(), dtype=np.float64))
                continue
            g = p.grad.detach().cpu().numpy().astype(np.float64).reshape(-1)
            g_parts.append(g)
            state = optimizer.state[p]
            if 'exp_avg_sq' not in state:
                psqrt_parts.append(np.ones(p.numel(), dtype=np.float64))
                continue
            v_hat = state['exp_avg_sq'].detach().cpu().numpy().astype(np.float64).reshape(-1) / bias_corr
            psqrt = 1.0 / np.sqrt(v_hat + eps)
            psqrt_parts.append(psqrt)
    return np.concatenate(psqrt_parts), np.concatenate(g_parts)


def run_one(nu, basin, gate_seed):
    cfg = r2s.base_cfg(nu, f"s3_s{gate_seed}")
    cfg.update(dict(lr=r2s.LR_PHASE1, use_gate=False, adam_epochs=PHASE1_STEPS,
                    lbfgs_epochs=0, beta_schedule="const", loss_beta=1.0,
                    weight_mode="adaptive", weight_norm="raw"))
    model, pde, crit, opt, samp, vis, cfg = r2s.build_all(cfg, r2s.LR_PHASE1, 1.0)
    model.load_state_dict(basin["state"])
    model.set_sigma(basin["sigma_end"])
    model.train()
    crit.load_frozen_field(basin["t_field"], beta=1.0)
    optimizer = opt

    P_sqrt_history = []  # (step, P_sqrt)
    g_history = []       # (step, g)
    records = []

    for step in range(1, PHASE1_STEPS + 1):
        optimizer.zero_grad()
        data = samp.sample(cfg['n_pde'], mode='random')
        x = data['x_pde'].to(DEVICE)
        r = pde.compute_residual(model, x)
        loss, _ = crit(r, x_pde=x)
        loss.backward()

        # 记录更新前的P^{1/2}和g
        if step % 10 == 0 or step <= 50:
            P_sqrt, g = get_adam_Psqrt_and_g(optimizer, step)
            P_sqrt_history.append((step, P_sqrt))
            g_history.append((step, g))

            P_norm = float(np.linalg.norm(P_sqrt, ord=2))
            g_tilde = P_sqrt * g
            g_tilde_norm = float(np.linalg.norm(g_tilde, ord=2))
            g_norm = float(np.linalg.norm(g, ord=2))
            records.append(dict(step=step, loss=float(loss.item()),
                                P_sqrt_norm=P_norm, g_norm=g_norm, g_tilde_norm=g_tilde_norm))

        optimizer.step()

    # 计算稳态窗（step>=500）的平台/梯度比值
    steady_idx = [i for i, r in enumerate(records) if r['step'] >= 500]
    ratios = []
    for idx in steady_idx[1:]:  # 从第二个开始，需要前一个
        prev = records[idx-1]; cur = records[idx]
        p0 = dict(P_sqrt_history)[prev['step']]
        p1 = dict(P_sqrt_history)[cur['step']]
        n_steps = cur['step'] - prev['step']
        delta_rel = np.linalg.norm(p1 - p0, ord=2) / (np.linalg.norm(p0, ord=2) + 1e-30)
        per_step = delta_rel / n_steps
        platform_per_step = prev['P_sqrt_norm'] * per_step
        ratio = platform_per_step / (prev['g_tilde_norm'] + 1e-30)
        ratios.append(dict(step=prev['step'], platform_per_step=platform_per_step,
                           g_tilde_norm=prev['g_tilde_norm'], ratio=ratio))

    return records, ratios


def main():
    nu = 0.005
    xt, ut = r2s.build_test(nu)

    all_ratios = []
    for gs in VERIFY_SEEDS:
        basin, seed = find_gate(nu, xt, ut)
        if basin is None:
            print(f"[s3] seed={gs}: no basin, skip"); continue
        records, ratios = run_one(nu, basin, seed)
        all_ratios.extend(ratios)
        if ratios:
            print(f"[s3] seed={seed}: {len(ratios)} samples, "
                  f"ratio median={np.median([r['ratio'] for r in ratios]):.3e}, "
                  f"platform/step median={np.median([r['platform_per_step'] for r in ratios]):.3e}, "
                  f"||g̃|| median={np.median([r['g_tilde_norm'] for r in ratios]):.3e}", flush=True)

    if all_ratios:
        ra = np.array([r['ratio'] for r in all_ratios])
        pa = np.array([r['platform_per_step'] for r in all_ratios])
        ga = np.array([r['g_tilde_norm'] for r in all_ratios])
        summary = dict(nu=nu, n_seeds=len(VERIFY_SEEDS), n_samples=len(all_ratios),
                       ratio_median=float(np.median(ra)), ratio_q25=float(np.percentile(ra,25)),
                       ratio_q75=float(np.percentile(ra,75)),
                       platform_per_step_median=float(np.median(pa)),
                       g_tilde_norm_median=float(np.median(ga)))
        with open(OUT / 's3_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(OUT / 's3_ratios.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=['step','platform_per_step','g_tilde_norm','ratio'])
            w.writeheader(); w.writerows(all_ratios)
        print("\n=== S3 SUMMARY ===")
        for k,v in summary.items(): print(f"  {k}: {v}")
        print(f"\nSaved to {OUT}")


if __name__ == '__main__':
    main()
