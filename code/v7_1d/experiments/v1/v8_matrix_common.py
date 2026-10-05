# -*- coding: utf-8 -*-
"""
V9 参数矩阵实验共享层（E1 β×ν 精细 / E2 σ×β / E3 权函数族·增减权对照）
--------------------------------------------------------------------------------
* 只做【薄封装 + 缓存 + 指标/聚合工具】，训练内循环仍复用 PINNTrainer（不复制）。
* Phase0 盆地按 (nu,seed,sigma_hi,K,epochs) 缓存到 results/v6/_gate_cache，
  同一盆地的全部 β/族共享，保证“同 gate 配对”这一控制变量，且省算力。
* 种子规范（防“挑种子”质疑）：主结果用规则等距、实验前固定的 SEEDS_NEW，
  不预贴好/坏标签、全量报告；旧批次种子保留为 OLD_SEEDS 仅用于复现。
"""
import os, sys, csv, copy
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_PKG = _HERE.parent.parent           # experiments/
_PROJ = _PKG.parent                  # v7_1d/（models/trainers/losses 等包所在根）
for _p in (str(_PROJ), str(_PKG), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
from trainers.pinn_trainer import PINNTrainer
from losses.family_weighting_loss import FamilyWeightingLoss, FAMILIES, stability_factor, w_min_theory
from utils.spectral_estimator import HessianSpectralEstimator

DEVICE = r2s.DEVICE
RES_ROOT = _PKG.parent / "results" / "v6"
os.makedirs(RES_ROOT, exist_ok=True)
# 发布版：优先读随仓库发布的 gate cache（data/raw/v7_v1/_gate_cache），
# 避免全新 clone 重训 Phase0；开发环境无该目录时回退到运行时 results/v6/_gate_cache。
_RELEASE_GATE = _HERE.parents[4] / "data" / "raw" / "v7_v1" / "_gate_cache"
if _RELEASE_GATE.exists():
    GATE_CACHE = _RELEASE_GATE
else:
    GATE_CACHE = RES_ROOT / "_gate_cache"
    os.makedirs(GATE_CACHE, exist_ok=True)

# 规则等距、预固定：K=4 时内部 start=s..s+3，间隔 10 保证不重叠
SEEDS_NEW = [0, 10, 20, 30, 40]
OLD_SEEDS = [42, 123, 2024, 99, 7]
# 扩展规则种子（11 个，间隔 5：K=4 内部 s..s+3 与 s+5..s+8 不重叠）；是 SEEDS_NEW 的超集，
# 供需要报成功率/中位、对盆地抽样敏感的统计性主结果（E1/E3b/F2b）扩样，好/坏盆地各凑到 5-8。
SEEDS_EXT = list(range(0, 55, 5))
assert set(SEEDS_NEW) <= set(SEEDS_EXT), "SEEDS_EXT 必须是原 SEEDS_NEW 的超集（保留原批次种子）"
SMOKE = os.environ.get("V8_SMOKE", "0") == "1"

# Phase0 进入“合理盆地”的【固定、与 β/族无关】纳入阈值（诊断口径；truth 只作上界）
GATE_L2_MAX = 0.06


# ============================ Phase0 盆地缓存 ============================
def enter_basin_cached(nu, seed, xt, ut, sigma_hi=15.0, k=4, phase0_epochs=2000, reuse=True):
    tag = ("smoke_" if SMOKE else "") + f"nu{nu}_s{seed}_sig{sigma_hi}_k{k}_ep{phase0_epochs}.pt"
    path = GATE_CACHE / tag
    # 无条件对齐全局带宽/规模：即使命中缓存也要回设，避免上一组 σ（或别的实验）残留污染
    # base_cfg 的 fourier_scale —— trainer 在 phase1 每一步都会用它 set_sigma（见 PINNTrainer._sigma_at）。
    r2s.MULTISTART_K = 1 if SMOKE else int(k)
    r2s.SELECT_BY = "truth"
    r2s.SIG_HI = float(sigma_hi)          # base_cfg 的固定带宽
    r2s.SIGMA_HI_CFG = float(sigma_hi)    # σ 课程终点
    r2s.PHASE0_EPOCHS = 60 if SMOKE else int(phase0_epochs)
    if reuse and path.exists():
        d = torch.load(path, map_location=DEVICE)
        print(f"[GateCache] hit {tag}  L2={d['l2']:.3e}（全局带宽已对齐 σ={sigma_hi:g}）")
        return d
    basin = r2s.run_phase0(nu, seed, xt, ut)
    torch.save(basin, path)
    print(f"[GateCache] saved {tag}")
    return basin


# ===================== Phase1：同盆地、指定权函数族训练 =====================
def train_phase1_family(basin, nu, contrast, seed, xt, ut,
                        family="rational", shuffled=False, epochs=None,
                        tag="E", lr=5e-4, grad_probe=None, beta_kw=None):
    """grad_probe=(x_probe,nu) 时，在【训练前 gate 点】与【训练后 best 点】各测一次参数梯度
    空间分区能量（机制 A 直接证据），结果挂到 trainer.grad_probe；不传则为 None，行为不变。
    beta_kw=None 时 β 恒定（旧行为不变）；F2b 课程缓启动可传
    dict(beta_schedule='cosine', beta_init=1.0, beta_start=0, beta_end=N) 做切换后 1→contrast 缓升。"""
    if epochs is not None:
        r2s.PHASE1_EPOCHS = int(epochs)
    r2s.set_seed(int(seed))
    cfg = r2s.base_cfg(nu, f"{tag}_s{seed}_c{contrast}_{family}")
    bk = beta_kw or {}
    # 权威锁定 phase1 前向带宽=盆地自身带宽：trainer 每步 set_sigma(_sigma_at)，phase1 无 σ 课程
    # 时 _sigma_at 恒返回 cfg['fourier_scale']。显式写入可杜绝全局 r2s.SIG_HI 残留污染（E2 教训）。
    basin_sigma = float(basin.get("sigma_end", r2s.SIG_HI))
    cfg.update(dict(lr=lr, use_gate=False, adam_epochs=r2s.PHASE1_EPOCHS, lbfgs_epochs=0,
                    use_best_ckpt=True, best_metric="max_L2_band",
                    beta_schedule=bk.get("beta_schedule", "const"), loss_beta=float(contrast),
                    beta_init=float(bk.get("beta_init", 1.0)),
                    beta_start_step=int(bk.get("beta_start", 0)),
                    beta_end_step=int(bk.get("beta_end", 0)),
                    fourier_scale=basin_sigma, sigma_lo=None, sigma_hi=None, sigma_anneal_T=0))
    # 复用 build_all 的全部装配，只把 criterion 换成指定权函数族（训练内循环不动）
    model, pde, _crit0, opt, samp, vis, cfg = r2s.build_all(cfg, lr, float(contrast))
    crit = FamilyWeightingLoss(cfg, DEVICE, family=family, shuffled=shuffled)
    model.load_state_dict(basin["state"]); model.set_sigma(basin["sigma_end"])
    crit.load_frozen_field(basin["t_field"], beta=float(contrast))
    trainer = PINNTrainer(model, pde, crit, opt, samp, vis, DEVICE, cfg,
                          {"x": xt, "u_true": ut})
    gp_gate = grad_partition_energy(model, pde, crit, grad_probe[0], grad_probe[1]) \
        if grad_probe is not None else None
    trainer.train(cfg, {"x": xt, "u_true": ut}, r2s.PHASE1_EPOCHS, 0, adaptive_freq=500)
    gp_end = grad_partition_energy(model, pde, crit, grad_probe[0], grad_probe[1]) \
        if grad_probe is not None else None
    trainer.grad_probe = {"gate": gp_gate, "end": gp_end}
    m = trainer.evaluate({"x": xt, "u_true": ut}); trainer.log_file.close()
    return m, trainer.dynamics_history, model, crit, trainer


# ===================== 探针1：参数梯度的空间分区能量（机制A直接证据）=====================
def _flat_grad(loss, params, retain=False):
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(gs, params)])


def grad_partition_energy(model, pde, crit, x, nu, band_c=3.0):
    """固定参数点 θ 与固定配点 x，把加权损失按激波带/光滑区分开，分别反传参数梯度：
      g_band=∇θ mean_{i∈band} w_i r_i²，g_smooth=∇θ mean_{i∉band} w_i r_i²
    返回二者范数平方、能量比 eta=g_band²/g_smooth²、方向夹角 cos。
    同一 θ 跨 β 比较：β↑→eta↑ 即“梯度能量被再分配到激波方向”的直接测量（非输出梯度代理）。"""
    params = [p for p in model.parameters() if p.requires_grad]
    band = float(band_c) * float(nu)
    model.zero_grad()
    res = pde.compute_residual(model, x)                       # 共享计算图
    with torch.no_grad():
        w = crit.get_current_weights(res.detach(), x_pde=x.detach()).reshape(-1)
    r = res.reshape(-1); msk = (x.reshape(-1).abs() < band)
    out = dict(g_band=float("nan"), g_smooth=float("nan"),
               eta=float("nan"), cos=float("nan"))
    if msk.any() and (~msk).any():
        lb = (w[ msk ] * r[ msk ] ** 2).mean()
        ls = (w[ ~msk ] * r[ ~msk ] ** 2).mean()
        gb = _flat_grad(lb, params, retain=True)
        gs = _flat_grad(ls, params, retain=False)
        nb, ns = float(gb.norm() ** 2), float(gs.norm() ** 2)
        out.update(g_band=nb, g_smooth=ns, eta=nb / max(ns, 1e-30),
                   cos=float(torch.dot(gb, gs) / max(gb.norm() * gs.norm(), 1e-30)))
    model.zero_grad()
    return out


# ===================== 探针2：参数 Hessian 最小曲率（U型右支负曲率）=====================
def hessian_min_curvature(model, pde, crit, x, num_lanczos=40):
    """参数空间 H=∇²L 的极端谱：λmin<0 即负曲率。近解点跨 β 比较，右支应见 λmin 走低/负曲率维数↑。"""
    kappa, lam_max, lam_min, n_neg, _sp = HessianSpectralEstimator.estimate_condition_number(
        model, loss_fn=crit, pde_engine=pde, x_spectral=x, num_lanczos=num_lanczos, k_min=2)
    return dict(kappa=kappa, lam_max=lam_max, lam_min=lam_min, n_neg=int(n_neg))


# ============================== 轨迹指标 ==============================
def first_hit_step(dyn, thr):
    es = np.asarray(dyn["eval_step"], float); l2 = np.asarray(dyn["L2_traj"], float)
    idx = np.where(np.isfinite(l2) & (l2 < thr))[0]
    return float(es[idx[0]]) if idx.size else float("nan")


def terminal_med(seq, k=5):
    a = np.asarray(seq, float); a = a[np.isfinite(a)]
    if a.size == 0: return float("nan")
    return float(np.median(a[-k:]))


def final_minus_best(dyn):
    l2 = np.asarray(dyn["L2_traj"], float); l2 = l2[np.isfinite(l2)]
    if l2.size == 0: return float("nan")
    return float(l2[-1] - l2.min())     # >0 且偏大 = 末段回摆/过冲（U 型右支证据）


def mechanism_row(dyn):
    return dict(
        focus_t=terminal_med(dyn.get("focus_ratio", [])),
        grad_energy_t=terminal_med(dyn.get("grad_energy_ratio", [])),
        wmean_shock_t=terminal_med(dyn.get("weight_mean_shock", [])),
        grad_norm_t=terminal_med(dyn.get("grad_norm", [])),
    )


# ============================== 聚合/落盘 ==============================
def write_csv(path, rows, fieldnames=None):
    if not rows:
        print("[warn] empty rows ->", path); return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _iqr(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0: return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    return (float(np.median(v)), float(np.quantile(v, .25)), float(np.quantile(v, .75)),
            float(v.min()), float(v.max()))


def aggregate(rows, group_keys, val_cols, success_col="L2", success_thr=1e-2):
    out = []
    keys = sorted({tuple(r[g] for g in group_keys) for r in rows},
                  key=lambda z: tuple(float(a) for a in z))
    for kv in keys:
        sub = [r for r in rows if all(r[g] == v for g, v in zip(group_keys, kv))]
        row = {g: v for g, v in zip(group_keys, kv)}; row["n"] = len(sub)
        for c in val_cols:
            med, q1, q3, lo, hi = _iqr([r.get(c, float("nan")) for r in sub])
            row[f"{c}_med"] = med; row[f"{c}_q1"] = q1; row[f"{c}_q3"] = q3
            row[f"{c}_min"] = lo; row[f"{c}_max"] = hi
        sc = [r.get(success_col, float("nan")) for r in sub]
        row["success"] = float(np.nanmean(np.asarray(sc) < success_thr)) if len(sc) else float("nan")
        out.append(row)
    return out


def stable_band(nu, c=3.0):
    return c * float(nu)
