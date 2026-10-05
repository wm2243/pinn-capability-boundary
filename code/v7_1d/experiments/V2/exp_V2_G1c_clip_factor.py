# -*- coding: utf-8 -*-
"""实验 G1c：Phase0 盆地进入率的 clip 因子（减权 vs 梯度裁剪的直接对照）
================================================================================
目的：把空间减权的作用拆成"尺度成分"和"方向成分"——
  尺度成分：压低激波区梯度幅值（Gstab 实测 g_peak 14.2→2.2），与梯度裁剪做的事重叠；
  方向成分：对不同空间位置梯度分量做不均匀缩放，改变梯度方向/区域能量分配，裁剪不做。

梯度裁剪（torch.nn.utils.clip_grad_norm_）是【各向同性】的范数归一化，不改方向；
空间减权是【各向异性】的，既压尺度又改方向。本实验用 族×clip 因子设计分离两者：

  因子A：权重族 {uniform, down_inv, down_lin, rational}
  因子B：梯度裁剪 {clip=0（关）, clip=1（开，base_cfg 默认）}
  响应：Phase0 好盆地进入率 p_entry（L2<0.06）+ 训练爆炸率 blown

关键对比（pre-registered）：
  ① uniform clip=0 vs clip=1：纯裁剪（各向同性尺度控制）能救回多少盆地；
  ② down clip=0 vs uniform clip=0：减权作为"自带软裁剪"能否替代硬裁剪救命；
  ③ down clip=1 vs uniform clip=1（已有 G1b 数据）：尺度被裁剪兜底后减权仍损害进入
     → 这部分损害只能归因于方向偏置。

最可能结局：down clip=0 追近 uniform clip=1（软裁剪救命），但 down clip=1 仍低于 uniform clip=1
→ 减权的全部益处来自尺度控制且可被裁剪替代；它独有的方向偏置在裁剪兜底时反而有害。

【clip=1 数据来源】本脚本默认只跑 clip=0（CLIP_VALUES=[0.0]）。
clip=1 的同协议数据已由 G1b（exp_G1b_pentry，base_cfg 默认 grad_clip=1.0、K=4 multistart、
truth 选择、P0=2000）跑完 n=80（4ν），分析时取 ν∈{.005,.001} 子集直接合并，无需重跑。
如需交叉验证，把 CLIP_VALUES 改成 [0.0, 1.0] 即可。

【与 Gstab-B 的分工】本实验是 Phase0（找盆地），Gstab-B 是 Phase1（好盆地内）：
  G1c：找盆地时减权的尺度成分能否被裁剪替代、方向成分是否有害；
  Gstab-B：盆地内减权是不是预条件器（无 lr 裕度、L-BFGS 无差异 → 只是尺度稳定器）。

规范：不改公共代码；规则等距种子 SEEDS_WIDE（20个）；K=4 multistart + truth 选择（与 G1b 同口径）；
逐条落盘、可中断续跑；真解仅离线评估；NaN/爆炸记 blown=1，不中断整体采集。
运行：python exp_V2_G1c_clip_factor.py
      V8_SMOKE=1 python exp_V2_G1c_clip_factor.py
"""
import os, sys, csv, copy, math
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_V1EXP = _HERE.parent.parent / "v1"        # 源代码/v7/experiments/v1
_V7ROOT = _HERE.parent.parent.parent        # 源代码/v7/（models/losses/trainers 在此）
_SRC_ROOT = _HERE.parent.parent.parent.parent  # 源代码/（results 在此）
for _p in (str(_V1EXP), str(_V7ROOT), str(_SRC_ROOT), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from trainers.pinn_trainer import PINNTrainer
from losses.family_weighting_loss import FamilyWeightingLoss

DEVICE = mc.DEVICE
OUT = _SRC_ROOT / "results" / "v7" / "V2" / "exp_V2_G1c_clip_factor"
OUT.mkdir(parents=True, exist_ok=True)

RUNS_CSV = OUT / "g1c_clip_runs.csv"
AGG_CSV = OUT / "g1c_clip_agg.csv"

# ----------------------------- 固定配置（与 G1b 同口径，实验前锁死）-----------------------------
NUS = [0.005, 0.001]                 # 代表性粘度：中间刚性端 + 最硬端（G1b 有 4ν，此处取 2 个）
SEEDS_WIDE = list(range(0, 100, 5))  # 20 个等距种子，G1/G1b 同口径
FAMS = ["uniform", "down_inv", "down_lin", "rational"]
CONTRAST = 10.0
CLIP_VALUES = [0.0]                   # 默认只跑 clip=0；clip=1 复用 G1b 数据（见头部说明）
GATE = mc.GATE_L2_MAX                 # 0.06，好/坏盆地阈值，与 G1/G1b 一致
K_MULTI = 4                            # 每起点 4 个 multistart，truth 选择（与 G1b 一致）
P0_EP = 2000                          # Phase0 步数，与 G1b 一致

SMOKE = bool(os.environ.get("V8_SMOKE"))
if SMOKE:
    NUS = [0.005]; SEEDS_WIDE = [0, 10]; K_MULTI = 1; P0_EP = 240
    FAMS = ["uniform", "down_inv"]

RUN_FIELDS = ["nu", "seed", "family", "contrast", "clip",
              "phase0_L2", "proxy", "gate_epoch", "sigma_end", "blown", "good"]
AGG_FIELDS = ["nu", "family", "clip", "n", "good_k", "p_entry",
              "wilson_lo", "wilson_hi", "blown_k", "blown_rate",
              "L2_med", "L2_q1", "L2_q3"]


def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k / n; den = 1 + z*z/n
    cen = (p + z*z/(2*n))/den
    half = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)))/den
    return max(0., cen-half), min(1., cen+half)


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


# ============ Phase0（复刻 G1b run_phase0_family，新增 grad_clip 参数）============
def run_phase0_clip(nu, base_seed, xt, ut, family, contrast, clip_val,
                    sigma_lo=r2s.SIGMA_LO, sigma_hi=r2s.SIGMA_HI_CFG, sigma_T=r2s.SIGMA_T):
    """同 seed 的 B0 随机方向由 set_seed 保证一致，跨族/跨 clip 只改权重或裁剪阈值，不改随机方向。

    与 G1b run_phase0_family 的唯一区别：cfg.update 显式传入 grad_clip=clip_val
    （base_cfg 默认 1.0；clip_val=0.0 关闭裁剪）。NaN/爆炸记 blown=1，不抛异常。
    """
    cands = []
    for k in range(K_MULTI):
        seed = base_seed + k
        r2s.set_seed(seed)
        cfg = r2s.base_cfg(nu, f"g1c_{family}_clip{clip_val:g}_s{base_seed}_{k}")
        cfg.update(dict(lr=r2s.LR_PHASE0, use_gate=True, gate_patience=3, gate_lr_gamma=0.3,
                        sigma_lo=float(sigma_lo), sigma_hi=float(sigma_hi), sigma_anneal_T=int(sigma_T),
                        fourier_scale=float(sigma_hi),
                        beta_schedule="const", loss_beta=float(contrast), beta_init=float(contrast),
                        adam_epochs=P0_EP, lbfgs_epochs=0, use_best_ckpt=True,
                        best_metric="max_L2_band",
                        grad_clip=float(clip_val)))   # 关键：覆盖默认 clip=1.0
        model, pde, _crit0, opt, samp, vis, cfg = r2s.build_all(cfg, r2s.LR_PHASE0, float(contrast))
        crit = FamilyWeightingLoss(cfg, DEVICE, family=family)
        trainer = PINNTrainer(model, pde, crit, opt, samp, vis, DEVICE, cfg, {"x": xt, "u_true": ut})
        blown = 0
        try:
            trainer.train(cfg, {"x": xt, "u_true": ut}, P0_EP, 0, adaptive_freq=500)
            if not crit.field_frozen:
                crit.freeze_residual_field(model, pde)
            l2 = r2s.quick_l2(model, xt, ut)
            if not math.isfinite(l2):
                blown = 1; l2 = float("nan")
            proxy = float(np.mean(trainer.loss_history[-200:])) if trainer.loss_history else float("nan")
            if not math.isfinite(proxy): proxy = float("nan")
            gate_ep = trainer.gate_epoch
            sig_end = float(model.get_sigma())
        except Exception as e:
            blown = 1; l2 = float("nan"); proxy = float("nan"); gate_ep = -1; sig_end = float("nan")
            print(f"    [blown] ν={nu:g} s{base_seed} k{k} {family} clip={clip_val:g}: {type(e).__name__}: {e}", flush=True)
        try: trainer.log_file.close()
        except Exception: pass
        cands.append(dict(seed=seed, l2=l2, proxy=proxy, gate=gate_ep, sigma_end=sig_end, blown=blown))
    # truth 选择：在未爆炸的候选中选 L2 最小；全炸则取第一个（blown=1）
    alive = [c for c in cands if c["blown"] == 0 and math.isfinite(c["l2"])]
    if alive:
        best = min(alive, key=lambda z: z["l2"])
        best["blown"] = 0
    else:
        best = cands[0]; best["blown"] = 1; best["l2"] = float("nan")
    return best


def collect():
    rows = _load(RUNS_CSV, RUN_FIELDS)
    done = {_key(r["nu"], r["seed"], r["family"], r["clip"]) for r in rows}
    total = len(NUS) * len(SEEDS_WIDE) * len(FAMS) * len(CLIP_VALUES)
    done_count = 0
    for nu in NUS:
        xt, ut = r2s.build_test(nu)
        for seed in SEEDS_WIDE:
            for fam in FAMS:
                contrast = 1.0 if fam == "uniform" else CONTRAST
                for clip_val in CLIP_VALUES:
                    kk = _key(nu, seed, fam, clip_val)
                    if kk in done:
                        done_count += 1; continue
                    b = run_phase0_clip(nu, seed, xt, ut, fam, contrast, clip_val)
                    good = int((b["blown"] == 0) and math.isfinite(b["l2"]) and b["l2"] < GATE)
                    row = dict(nu=nu, seed=seed, family=fam, contrast=contrast, clip=clip_val,
                               phase0_L2=b["l2"], proxy=b["proxy"], gate_epoch=b["gate"],
                               sigma_end=b["sigma_end"], blown=b["blown"], good=good)
                    rows.append(row); done.add(kk); _write(RUNS_CSV, RUN_FIELDS, rows)
                    print(f"[G1c] ν={nu:g} s{seed} {fam:9s} clip={clip_val:g}: "
                          f"L2={b['l2']:.3e} good={good} blown={b['blown']}", flush=True)
    print(f"\n[G1c] 采集完成，{len(rows)} 行（新增 {len(rows)-done_count}）-> {RUNS_CSV.name}", flush=True)
    return rows


def aggregate(rows):
    out = []
    for nu in sorted({float(r["nu"]) for r in rows}):
        for fam in FAMS:
            for clip_val in sorted({float(r["clip"]) for r in rows}):
                z = [r for r in rows if float(r["nu"]) == nu and r["family"] == fam and float(r["clip"]) == clip_val]
                if not z: continue
                n = len(z)
                good_k = sum(int(r["good"]) for r in z)
                blown_k = sum(int(r["blown"]) for r in z)
                lo, hi = wilson(good_k, n)
                L2s = np.asarray([float(r["phase0_L2"]) for r in z if r["phase0_L2"] not in ("", "nan") and math.isfinite(float(r["phase0_L2"]))], float)
                out.append(dict(nu=nu, family=fam, clip=clip_val, n=n, good_k=good_k,
                                p_entry=good_k/n if n else float("nan"),
                                wilson_lo=lo, wilson_hi=hi,
                                blown_k=blown_k, blown_rate=blown_k/n if n else float("nan"),
                                L2_med=float(np.median(L2s)) if L2s.size else float("nan"),
                                L2_q1=float(np.quantile(L2s, .25)) if L2s.size else float("nan"),
                                L2_q3=float(np.quantile(L2s, .75)) if L2s.size else float("nan")))
    _write(AGG_CSV, AGG_FIELDS, out)
    print(f"[G1c] 聚合完成 -> {AGG_CSV.name}", flush=True)
    # 打印 2×2 摘要表
    print("\n=== Phase0 盆地进入率 p_entry（ν 分开）===", flush=True)
    for nu in sorted({float(r["nu"]) for r in rows}):
        print(f"\nν={nu:g}:", flush=True)
        print(f"  {'family':10s} {'clip':>5s} {'n':>4s} {'p_entry':>8s} {'Wilson95':>16s} {'blown%':>7s}", flush=True)
        for r in out:
            if float(r["nu"]) == nu:
                print(f"  {r['family']:10s} {float(r['clip']):5.1f} {int(r['n']):4d} "
                      f"{float(r['p_entry']):8.3f} [{float(r['wilson_lo']):.3f},{float(r['wilson_hi']):.3f}] "
                      f"{float(r['blown_rate']):7.3f}", flush=True)
    return out


def main():
    print(f"=== G1c clip 因子 结果目录: {OUT} ===", flush=True)
    print(f"配置: NUS={NUS}, seeds={len(SEEDS_WIDE)}, FAMS={FAMS}, CLIP={CLIP_VALUES}, "
          f"K_MULTI={K_MULTI}, P0_EP={P0_EP}, GATE={GATE}", flush=True)
    print(f"clip=1 数据复用 G1b（同协议），本脚本只补 clip=0；分析时合并。", flush=True)
    rows = collect()
    aggregate(rows)
    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()
