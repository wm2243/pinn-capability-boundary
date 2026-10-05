# -*- coding: utf-8 -*-
"""实验 Gstab-B：减权"梯度尺度稳定器 vs 预条件器"配对检验（严谨版）
================================================================
目的：严谨区分两个对立假设——
  假设A（本文主张，尺度稳定器）：减权降低早期梯度峰值/裁剪触发率，
      但【不】扩大稳定学习率、【不】改善条件数；二阶法（自带尺度归一化）下无差异。
  假设B（预条件器）：减权改善条件数、扩大稳定学习率。

现有 Gstab（results/v7/V1/exp_Gstab_grad）的 Stage-E（n=11）已支持"峰值/裁剪率降低"，
但 Stage-B1（lr 裕度）仅 n=3、lr 网格仅 4 个粗值，Stage-B2（L-BFGS）仅 n=3。
本实验补齐 B1/B2 到 n=11 配对，并加密 lr 网格，另记录逐步梯度曲线。

【优化器与理论准入类（定义 def:admit / 引理 lem:linrec）】
  Part 1 用固定步长 Adam（betas=.9/.999，无 weight_decay，无任何读取 L_W 的线搜）；
  Part 2 用固定步长 L-BFGS（PyTorch line_search_fn=None，不做 Armijo 回溯；
         割线阵仅由历史加权梯度构造，属信息集 I_k 内间接量，不读取标量损失 L_W）。
  两者均在准入类内，引理 4.1 覆盖；带线搜（读 L_W）的 L-BFGS 在类外，本文不使用。

三个 Part：
  B1b  lr 裕度配对：ν=.005, clip=1, 4族 × 8 lr(5e-4..1e-1) × 11种子，同 Gate 点配对，500步 Adam。
  B2b  L-BFGS 配对：ν∈{.001,.005}, 4族 × 11种子，Adam 200预热 + 固定步长 L-BFGS。
  GC   逐步梯度曲线：ν=.005, clip=1, lr=1e-3, 4族 × 11种子，逐50步记 ‖g‖₂/裁剪。

判据（pre-registered，跑完照此判，不事后挪阈值）：
  H-lr：四族最大稳定 lr 配对 Wilcoxon p>.05 且中位相同 → 无步长裕度（支持A、否定B）。
  H-lbfgs：L-BFGS 下四族终态 L2 配对 Wilcoxon p>.05 → 二阶法下无差异（支持A）。
  H-curve：早期 ‖g‖₂ 峰值 down_inv<down_lin<uniform≪rational（描述性，已由 Stage-E n=11 支持）。

规范：不改公共代码；规则等距种子 SEEDS_EXT（11个）；逐条落盘、可中断续跑；真解仅离线评估。
运行：python exp_V2_GstabB_stabilizer.py
      V8_SMOKE=1 python exp_V2_GstabB_stabilizer.py  （冒烟）
"""
import os, sys, csv, math
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_V1EXP = _HERE.parent.parent / "v1"        # v7/experiments/v1
_V7ROOT = _V1EXP.parent.parent            # v7
for _p in (str(_V1EXP), str(_V7ROOT), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
import exp_Gstab_grad_stability as gstab

# ---------------- 配置 ----------------
DEVICE = mc.DEVICE
# 代码在 源代码/v7/experiments/V2/，结果在 源代码/results/v7/V2/（v7 与 results 同级）
_SRC_ROOT = _HERE.parent.parent.parent.parent          # 源代码/
OUT = _SRC_ROOT / "results" / "v7" / "V2" / "exp_V2_GstabB_stabilizer"
OUT.mkdir(parents=True, exist_ok=True)

FAMS = ["uniform", "down_inv", "down_lin", "rational"]
CONTRAST = 10.0
SEEDS = mc.SEEDS_EXT              # 11 个等距种子 [0,5,...,50]
NU_B1 = 0.005                     # B1b 代表性刚性端
NUS_B2 = [0.001, 0.005]
CLIP_B1 = 1.0                     # 默认裁剪开启
LR_GRID_B1 = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]  # 8档：低端锚点+高端探索至1e-1
N_B1 = 500                        # B1b 短跑步数
N_B2_ADAM = 200                   # B2b Adam 预热
N_B2_LBFGS = 500                  # B2b 固定步长 L-BFGS 步数（每 outer step max_iter=20）
GC_LR = 1e-3
GC_RECORD_FREQ = 50

SMOKE = bool(os.environ.get("V8_SMOKE"))
if SMOKE:
    SEEDS = [0, 10]
    LR_GRID_B1 = [5e-4, 1e-3, 5e-2]
    NUS_B2 = [0.005]
    N_B1 = 80; N_B2_ADAM = 40; N_B2_LBFGS = 5

B1_CSV = OUT / "gstabB_lr_runs.csv"
B2_CSV = OUT / "gstabB_lbfgs_runs.csv"
GC_CSV = OUT / "gstabB_grad_curve.csv"

B1_FIELDS = ["nu", "seed", "family", "lr", "clip", "blown", "blow_step",
             "g_peak", "frac_clip", "L2", "l2_band"]
B2_FIELDS = ["nu", "seed", "family", "pre_blown", "pre_gpeak",
             "lbfgs_blown", "lbfgs_blow_it", "L2", "l2_band"]
GC_FIELDS = ["nu", "seed", "family", "step", "g_pre", "is_clip"]


def _load(p, fields):
    if not p.exists(): return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def _write(p, fields, rows):
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def _key(*vals):
    out = []
    for v in vals:
        try: out.append(f"{float(v):g}")
        except Exception: out.append(str(v))
    return tuple(out)


# ---------------- Part B1b：lr 裕度配对 ----------------
def collect_b1b():
    """固定步长 Adam（无 L_W 线搜），同 Gate 点配对，加密 lr 网格，n=11。"""
    nu = NU_B1
    xt, ut = r2s.build_test(nu)
    rows = _load(B1_CSV, B1_FIELDS)
    done = {_key(r["nu"], r["seed"], r["family"], r["lr"], r["clip"]) for r in rows}

    for seed in SEEDS:
        basin = mc.enter_basin_cached(nu, seed, xt, ut, sigma_hi=gstab.SIG_HI,
                                      k=4, phase0_epochs=r2s.PHASE0_EPOCHS)
        if (not SMOKE) and basin["l2"] >= mc.GATE_L2_MAX:
            print(f"[B1b] ν={nu:g} s{seed}: 无好盆地(L2={basin['l2']:.3f})，跳过", flush=True)
            continue
        fl = (basin["state"], basin["t_field"]); sigma = float(basin["sigma_end"])
        for fam in FAMS:
            contrast = 1.0 if fam == "uniform" else CONTRAST
            for lr in LR_GRID_B1:
                kk = _key(nu, seed, fam, lr, CLIP_B1)
                if kk in done: continue
                r = gstab.raw_adam_run(
                    nu, seed, fam, contrast, N_B1, lr, xt, ut,
                    grad_clip=CLIP_B1, sigma_course=False, freeze_load=fl,
                    hess_steps=(), tag="GstabB1b", sigma=sigma)
                row = dict(nu=nu, seed=seed, family=fam, lr=lr, clip=CLIP_B1,
                           blown=r["blown"], blow_step=r["blow_step"],
                           g_peak=r["g_peak"], frac_clip=r["frac_clip"],
                           L2=r["L2"], l2_band=r["l2_band"])
                rows.append(row); done.add(kk); _write(B1_CSV, B1_FIELDS, rows)
                print(f"[B1b] s{seed} {fam:9s} lr={lr:g}: blown={r['blown']} "
                      f"g_peak={r['g_peak']:.2e} L2={r['L2']:.2e}", flush=True)
    print(f"[B1b] 完成，{len(rows)} 行 -> {B1_CSV.name}", flush=True)
    return rows


# ---------------- Part B2b：固定步长 L-BFGS 配对 ----------------
def lbfgs_run_fixed(nu, seed, family, contrast, xt, ut, basin,
                    n_adam=N_B2_ADAM, n_lbfgs=N_B2_LBFGS):
    """Adam 预热 + 固定步长 L-BFGS（line_search_fn=None，不读 L_W）。

    与 gstab.lbfgs_run 的区别：显式 line_search_fn=None 并加长 L-BFGS 步数到 n_lbfgs，
    确保在准入类内（割线阵仅由历史加权梯度构造）。
    """
    tag = "GstabB2b"
    sigma = float(basin["sigma_end"]); fl = (basin["state"], basin["t_field"])
    pre = gstab.raw_adam_run(nu, seed, family, contrast, n_adam, r2s.LR_PHASE1, xt, ut,
                            grad_clip=0.0, sigma_course=False, freeze_load=fl,
                            hess_steps=(), tag=tag, sigma=sigma)
    model, pde, crit, samp, cfg = pre["model"], pre["pde"], pre["crit"], pre["samp"], pre["cfg"]
    if not crit.field_frozen: crit.freeze_residual_field(model, pde)
    data = samp.sample(cfg["n_pde"], mode="fixed")
    # 关键：line_search_fn=None（固定步长），不做读取 L_W 的 Armijo 回溯 → 准入类内
    optL = torch.optim.LBFGS(model.parameters(),
                             lr=float(cfg.get("lbfgs_lr", 1.0)), max_iter=20,
                             tolerance_grad=1e-7, tolerance_change=1e-9,
                             history_size=50, line_search_fn=None)
    blown, blow_it = 0, -1
    for it in range(n_lbfgs):
        def closure():
            optL.zero_grad()
            rr = pde.compute_residual(model, data["x_pde"])
            ll, _ = crit(rr, x_pde=data["x_pde"]); ll.backward()
            return ll
        try:
            vv = float(optL.step(closure).detach())
            if not math.isfinite(vv): blown, blow_it = 1, it; break
        except Exception:
            blown, blow_it = 1, it; break
    L2 = r2s.quick_l2(model, xt, ut) if not blown else float("nan")
    if not blown and (not math.isfinite(L2) or L2 > gstab.DIVERGE_L2): blown = 1
    return dict(pre_blown=pre["blown"], pre_gpeak=pre["g_peak"],
                lbfgs_blown=blown, lbfgs_blow_it=blow_it, L2=L2,
                l2_band=gstab._band_l2(model, xt, ut, nu) if not blown else float("nan"))


def collect_b2b():
    rows = _load(B2_CSV, B2_FIELDS)
    done = {_key(r["nu"], r["seed"], r["family"]) for r in rows}
    for nu in NUS_B2:
        xt, ut = r2s.build_test(nu)
        for seed in SEEDS:
            basin = mc.enter_basin_cached(nu, seed, xt, ut, sigma_hi=gstab.SIG_HI,
                                          k=4, phase0_epochs=r2s.PHASE0_EPOCHS)
            if (not SMOKE) and basin["l2"] >= mc.GATE_L2_MAX:
                print(f"[B2b] ν={nu:g} s{seed}: 无好盆地，跳过", flush=True); continue
            for fam in FAMS:
                kk = _key(nu, seed, fam)
                if kk in done: continue
                contrast = 1.0 if fam == "uniform" else CONTRAST
                r = lbfgs_run_fixed(nu, seed, fam, contrast, xt, ut, basin)
                rows.append(dict(nu=nu, seed=seed, family=fam, **r))
                done.add(kk); _write(B2_CSV, B2_FIELDS, rows)
                print(f"[B2b] ν={nu:g} s{seed} {fam:9s}: blown={r['lbfgs_blown']} "
                      f"L2={r['L2']:.2e}", flush=True)
    print(f"[B2b] 完成，{len(rows)} 行 -> {B2_CSV.name}", flush=True)
    return rows


# ---------------- Part GC：逐步梯度曲线 ----------------
def grad_curve_run(nu, seed, family, contrast, xt, ut, basin, lr=GC_LR,
                   n_step=N_B1, record_freq=GC_RECORD_FREQ, grad_clip=CLIP_B1):
    """固定步长 Adam，逐 record_freq 步记录裁剪前 ‖g‖₂ 与裁剪触发。准入类内。"""
    sigma = float(basin["sigma_end"])
    model, pde, crit, opt, samp, cfg = gstab._build(
        nu, seed, family, contrast, lr, sigma, f"GstabGC_{family}_s{seed}",
        grad_clip, freeze_load=(basin["state"], basin["t_field"]))
    recs = []
    blown, blow_step = 0, -1
    for ep in range(n_step):
        model.set_sigma(float(sigma))
        if (not crit.field_frozen) and ep % cfg["update_field_freq"] == 0:
            crit.update_residual_field(model, pde)
        data = samp.sample(cfg["n_pde"], mode="random"); x = data["x_pde"]
        res = pde.compute_residual(model, x); loss, _ = crit(res, x_pde=x)
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            gsq = sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None)
            gpre = float(torch.sqrt(gsq)) if gsq is not None else float("nan")
        lv = float(loss.detach())
        if not math.isfinite(gpre) or not math.isfinite(lv):
            blown, blow_step = 1, ep; break
        is_clip = 0
        if grad_clip and grad_clip > 0:
            is_clip = int(gpre > grad_clip * 1.001)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        if ep % record_freq == 0 or ep == n_step - 1:
            recs.append(dict(nu=nu, seed=seed, family=family, step=ep,
                             g_pre=gpre, is_clip=is_clip))
        opt.step()
    return recs, blown, blow_step


def collect_gc():
    nu = NU_B1
    xt, ut = r2s.build_test(nu)
    rows = _load(GC_CSV, GC_FIELDS)
    done = {(r["seed"], r["family"]) for r in rows}
    for seed in SEEDS:
        basin = mc.enter_basin_cached(nu, seed, xt, ut, sigma_hi=gstab.SIG_HI,
                                      k=4, phase0_epochs=r2s.PHASE0_EPOCHS)
        if (not SMOKE) and basin["l2"] >= mc.GATE_L2_MAX:
            print(f"[GC] ν={nu:g} s{seed}: 无好盆地，跳过", flush=True); continue
        for fam in FAMS:
            if (str(seed), fam) in done: continue
            contrast = 1.0 if fam == "uniform" else CONTRAST
            recs, blown, blow_step = grad_curve_run(nu, seed, fam, contrast, xt, ut, basin)
            rows.extend(recs); _write(GC_CSV, GC_FIELDS, rows)
            peak = max((r["g_pre"] for r in recs), default=float("nan"))
            print(f"[GC] s{seed} {fam:9s}: g_peak={peak:.2e} blown={blown} "
                  f"({len(recs)}记录点)", flush=True)
    print(f"[GC] 完成，{len(rows)} 行 -> {GC_CSV.name}", flush=True)
    return rows


def main():
    print(f"=== Gstab-B 结果目录: {OUT} ===", flush=True)
    print(f"种子 n={len(SEEDS)}: {SEEDS}", flush=True)
    print(f"优化器：Part1 固定步长Adam(无L_W线搜)；Part2 固定步长L-BFGS(line_search_fn=None)，均在准入类内", flush=True)
    print("\n--- Part B1b：lr 裕度配对 ---", flush=True)
    collect_b1b()
    print("\n--- Part GC：逐步梯度曲线 ---", flush=True)
    collect_gc()
    print("\n--- Part B2b：固定步长 L-BFGS 配对 ---", flush=True)
    collect_b2b()
    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()
