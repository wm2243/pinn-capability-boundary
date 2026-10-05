# -*- coding: utf-8 -*-
"""
================================================================================
V6 两阶段训练 / 受控 β 扫描（新脚本，不修改旧 run_sweep_beta_nu1.py）
范式： Phase 0（σ 表示课程 + multi-start，决定进入哪个盆地）
        -> Gate（损失平台 + 残差峰值稳定 + 参数漂移增速回落，自动冻结残差对比度场 W）
        -> Phase 1（在【同一好盆地】上扫 β，检验盆地内动力学 / β*(ν) / 静态冻结 κ(β)）

三种模式（MODE）：
  'twostage'  : 单次完整两阶段（σ课程->Gate->β课程，可接 LBFGS）
  'beta_scan' : Phase0 只跑一次拿到好盆地 θ_gate 与冻结场，随后对 β 列表【共享该盆地】跑 Phase1，
                输出 E(β) 曲线（这才是 β*(ν) 的正确标定，旧脚本在混合盆地上扫 β 看不到 U 型）
  'c1_static' : 同一冻结场、同一 θ_gate，只改 β 算【冻结 Hessian】的 κ/λmin（检验定理3 κ/β，
                与训练中动态 Hessian 严格区分）

参数范围（建议，跟实验结果走；标度率 s=-(2γ-1)/(1+a) 仅作模型预测，先验单调区间）：
  ν∈{0.01,0.005,0.003,0.001}；ν=0.1 属宽激波光滑区，σ 需降到 3~5（σ=30 高频错配），默认不做
  σ 课程：σ_lo=5, σ_hi=15, σ_anneal_T=1200（diag 已验证 12~18 是好窗，40 过宽）
  β 受控扫描：[1,3,5,10,15,25,40,70]（先粗定位 U 型，再在最优点 ±0.25 对数量级细化）
  multi-start K=4（单次 p_entry≈0.6 时覆盖率 1-(.4)^4≈0.97）
产物 results/v6/two_stage_v6/。运行：python run_two_stage_v6.py
================================================================================
"""
# ==== 路径自举(自动生成)：不依赖运行目录；产物统一到 源代码/results/v6/two_stage_v6/ ====
import sys as _sys, os as _os
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent          # v6/
_PROJ_ROOT = _HERE.parent.parent.parent  # 源代码/
for _p in (str(_PROJ_ROOT), str(_PKG_ROOT), str(_HERE.parent)):
    if _p not in _sys.path: _sys.path.insert(0, _p)
OUT = str(_PROJ_ROOT / "results" / "v6" / "two_stage_v6"); _os.makedirs(OUT, exist_ok=True)
# ======================================================================

import os, json, csv, copy, random, itertools
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from models.pinn_FourierFeatures_model import HardBCPINN
from physics.burgers_pde import SteadyBurgersPDE
from losses.rational_weighting_loss1 import RationalWeightingLoss
from samplers.pinn_sampler import PINNSampler
from utils.visualizer import PINNVisualizer
from trainers.pinn_trainer import PINNTrainer
from utils.spectral_estimator import HessianSpectralEstimator
from utils.metrics import PINNMetrics

for c in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(c):
        font_manager.fontManager.addfont(c); plt.rcParams["font.family"] = font_manager.FontProperties(fname=c).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

# ============================== 配置区 ==============================
MODE = "beta_scan"           # twostage / beta_scan / c1_static
NUS = [0.005]
SEEDS = [42, 123, 2024, 7, 99]
MULTISTART_K = 1             # Phase0 多起点数（1=关闭；建议正式 4）
SELECT_BY = "truth"          # truth=按粗解L2选(诊断) / proxy=残差损失+峰值集中度(可实现)

# 规模（冒烟可改小；正式值见注释）
PHASE0_EPOCHS = 2000         # 正式 3000~4000（到 Gate 即可）
PHASE1_EPOCHS = 3000         # 正式 6000~10000
N_PDE = 10000
HIDDEN, NLAYERS, MAPSIZE = 64, 4, 256
SIG_HI = 15.0                # 固定满带宽 / σ课程终点
SIGMA_LO, SIGMA_HI_CFG, SIGMA_T = 5.0, 15.0, 1200
LR_PHASE0 = 1e-3
LR_PHASE1 = 5e-4
BETAS_SCAN = [1.0, 3.0, 5.0, 10.0, 15.0, 25.0, 40.0, 70.0]
BETA_SCHEDULE = "const"      # beta_scan 用 const（盆地内固定 β）；twostage 可 'cosine'/'gate'
BAND_C = 3.0
EVAL_FREQ = 200
USE_LBFGS = False
SPATIAL_EMA = 0.9            # 空间权重场 EMA 衰减（0=关闭）；缓变 W 抑锯齿
WEIGHT_NORM = "raw"          # raw / selfnorm（尺度效应消融）
SUCCESS_L2 = 1e-2
# ===================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def build_test(nu, n=2048):
    x = torch.linspace(-1, 1, n, device=DEVICE).view(-1, 1)
    u = -np.tanh(x.cpu().numpy() / (2 * nu)).flatten()
    return x, u


def base_cfg(nu, seed_tag):
    return dict(
        domain_x=(-1, 1), hidden_dim=HIDDEN, num_layers=NLAYERS, mapping_size=MAPSIZE,
        fourier_scale=SIG_HI, nu=nu, n_pde=N_PDE, buffer_size=5000,
        grad_clip=1.0, shock_band_c=BAND_C, run_tag=seed_tag, out_dir=OUT,
        eval_freq=EVAL_FREQ, dynamics_log_freq=200, spectral_log_freq=10 ** 9,  # 阶段内不自动探谱，c1 手动
        spatial_ema_decay=SPATIAL_EMA, weight_norm=WEIGHT_NORM,
        update_field_freq=100, weight_ref_grid=2048,
    )


def build_all(cfg, lr, beta, weight_mode="adaptive"):
    model = HardBCPINN(cfg).to(DEVICE)
    pde = SteadyBurgersPDE(cfg)
    cfg = copy.deepcopy(cfg); cfg["loss_beta"] = beta; cfg["weight_mode"] = weight_mode
    crit = RationalWeightingLoss(cfg, DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    samp = PINNSampler(cfg["domain_x"], DEVICE, use_adaptive=False, buffer_size=cfg["buffer_size"])
    vis = PINNVisualizer(save_dir=OUT)
    return model, pde, crit, opt, samp, vis, cfg


def quick_l2(model, xt, ut):
    with torch.no_grad():
        up = model(xt).detach().cpu().numpy().flatten()
    return float(np.linalg.norm(up - ut) / np.linalg.norm(ut))


# ---------------- Phase 0：σ 课程 + multi-start，返回好盆地 ----------------
def run_phase0(nu, base_seed, xt, ut, return_all=False):
    cands = []
    for k in range(MULTISTART_K):
        seed = base_seed + k
        set_seed(seed)
        cfg = base_cfg(nu, f"p0_{base_seed}_{k}")
        cfg.update(dict(lr=LR_PHASE0, use_gate=True, gate_patience=3, gate_lr_gamma=0.3,
                        sigma_lo=SIGMA_LO, sigma_hi=SIGMA_HI_CFG, sigma_anneal_T=SIGMA_T,
                        beta_schedule="const", loss_beta=1.0, beta_init=1.0,
                        adam_epochs=PHASE0_EPOCHS, lbfgs_epochs=0, use_best_ckpt=True,
                        best_metric="max_L2_band"))
        model, pde, crit, opt, samp, vis, cfg = build_all(cfg, LR_PHASE0, 1.0)
        trainer = PINNTrainer(model, pde, crit, opt, samp, vis, DEVICE, cfg, {"x": xt, "u_true": ut})
        trainer.train(cfg, {"x": xt, "u_true": ut}, PHASE0_EPOCHS, 0, adaptive_freq=500)
        # Gate 通过则冻结场已生成；否则在终点强制冻结一次
        if not crit.field_frozen:
            crit.freeze_residual_field(model, pde)
        l2 = quick_l2(model, xt, ut)
        proxy = float(np.mean(trainer.loss_history[-200:]))
        cands.append(dict(seed=seed, l2=l2, proxy=proxy, gate=trainer.gate_epoch,
                          state=copy.deepcopy(model.state_dict()),
                          t_field=crit._frozen_t_field.detach().clone(),
                          sigma_end=model.get_sigma()))
        print(f"  [Phase0] start {seed}: L2={l2:.3e} gate@{trainer.gate_epoch}")
    # 选盆地
    if SELECT_BY == "truth":
        best = min(cands, key=lambda z: z["l2"])
    else:
        best = min(cands, key=lambda z: z["proxy"])
    print(f"[Phase0] 选用 start {best['seed']}（{SELECT_BY}）, L2={best['l2']:.3e}, gate@{best['gate']}")
    if return_all:
        return best, cands
    return best


# ---------------- Phase 1：从指定好盆地 + 冻结场，固定 β 训练 ----------------
def run_phase1(nu, beta, basin, base_seed, xt, ut):
    set_seed(base_seed)
    cfg = base_cfg(nu, f"p1_{base_seed}_b{beta}")
    cfg.update(dict(lr=LR_PHASE1, use_gate=False, adam_epochs=PHASE1_EPOCHS, lbfgs_epochs=0,
                    use_best_ckpt=True, best_metric="max_L2_band",
                    beta_schedule="const", loss_beta=beta))
    model, pde, crit, opt, samp, vis, cfg = build_all(cfg, LR_PHASE1, beta)
    model.load_state_dict(basin["state"]); model.set_sigma(basin["sigma_end"])
    crit.load_frozen_field(basin["t_field"], beta=beta)     # 共享同一冻结残差对比度场
    trainer = PINNTrainer(model, pde, crit, opt, samp, vis, DEVICE, cfg, {"x": xt, "u_true": ut})
    trainer.train(cfg, {"x": xt, "u_true": ut}, PHASE1_EPOCHS, USE_LBFGS * 300, adaptive_freq=500)
    m = trainer.evaluate({"x": xt, "u_true": ut}); trainer.log_file.close()
    return m, trainer.dynamics_history


# ---------------- C1：同一冻结场、只改 β 算静态冻结 Hessian ----------------
def c1_static_spectrum(nu, basin, xt, ut):
    rows = []
    for beta in BETAS_SCAN:
        set_seed(0)
        cfg = base_cfg(nu, f"c1_b{beta}"); cfg.update(loss_beta=beta, weight_mode="adaptive", num_lanczos=40)
        model, pde, crit, opt, samp, vis, cfg = build_all(cfg, LR_PHASE1, beta)
        model.load_state_dict(basin["state"]); model.set_sigma(basin["sigma_end"])
        crit.load_frozen_field(basin["t_field"], beta=beta)
        probe = samp.sample(2000, mode="random")["x_pde"].detach().to(DEVICE)
        kappa, lmax, lmin, nneg, _ = HessianSpectralEstimator.estimate_condition_number(
            model=model, loss_fn=crit, pde_engine=pde, x_spectral=probe, num_lanczos=40)
        rows.append(dict(beta=beta, kappa_static=kappa, lam_max=lmax, lam_min=lmin, neg_dim=nneg))
        print(f"  [C1-static] β={beta:g}: κ={kappa:.3e} λmin={lmin:.3e} neg={nneg}")
    return rows


def main():
    all_runs, spec_rows = [], []
    e_curves = {}
    for nu in NUS:
        xt, ut = build_test(nu)
        for seed in SEEDS:
            basin = run_phase0(nu, seed, xt, ut)
            if MODE == "c1_static":
                for r in c1_static_spectrum(nu, basin, xt, ut):
                    r.update(nu=nu, seed=seed); spec_rows.append(r)
                continue
            betas = [15.0] if MODE == "twostage" else BETAS_SCAN
            curve = []
            for beta in betas:
                m, dyn = run_phase1(nu, float(beta), basin, seed, xt, ut)
                row = dict(nu=nu, seed=seed, beta=float(beta), gate=basin["gate"],
                           phase0_L2=basin["l2"], **{k: m.get(k) for k in
                           ["L2_Error", "L_inf_Error", "TV_Error", "Shock_Width_Ratio",
                            "Nu_Eff_Ratio", "Oscillation_Count", "l2_band", "best_epoch",
                            "weight_mean_shock", "nonmonotone_frac"]})
                all_runs.append(row); curve.append(row)
                print(f"[Phase1] nu={nu} seed{seed} β={beta:g}: L2={m['L2_Error']:.3e} band={m.get('l2_band',0):.3e}")
            e_curves[(nu, seed)] = curve

    # 落盘
    if all_runs:
        with open(os.path.join(OUT, "v6_runs.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(all_runs[0].keys())); w.writeheader(); w.writerows(all_runs)
        # 聚合：每个 (nu,beta) 跨种子 median/best/worst/成功率
        agg = []
        for nu in NUS:
            for beta in sorted({r["beta"] for r in all_runs}):
                z = [r["L2_Error"] for r in all_runs if r["nu"] == nu and r["beta"] == beta]
                if not z: continue
                z = np.array(z)
                agg.append(dict(nu=nu, beta=beta, n=len(z), median=float(np.median(z)),
                                best=float(z.min()), worst=float(z.max()), std=float(z.std()),
                                success=float(np.mean(z < SUCCESS_L2))))
        with open(os.path.join(OUT, "v6_aggregated.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
        print("\n===== 受控 β 扫描（同一好盆地，跨种子聚合）=====")
        for z in sorted(agg, key=lambda q: (q["nu"], q["beta"])):
            print(f"ν={z['nu']} β={z['beta']:>5g}: median={z['median']:.3e} best={z['best']:.2e} "
                  f"worst={z['worst']:.2e} 成功率={z['success']:.0%}")
        _plot_e_curves(agg)
    if spec_rows:
        with open(os.path.join(OUT, "c1_static_spectrum.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(spec_rows[0].keys())); w.writeheader(); w.writerows(spec_rows)
        _plot_c1(spec_rows)
    print("\n完成。产物在", OUT)


def _plot_e_curves(agg):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for nu in sorted({z["nu"] for z in agg}):
        z = sorted([q for q in agg if q["nu"] == nu], key=lambda q: q["beta"])
        bs = [q["beta"] for q in z]
        ax[0].plot(bs, [q["median"] for q in z], "o-", label=f"ν={nu} median")
        ax[0].fill_between(bs, [q["best"] for q in z], [q["worst"] for q in z], alpha=.15)
        ax[1].plot(bs, [q["success"] for q in z], "s-", label=f"ν={nu}")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("β"); ax[0].set_ylabel("Phase1 L2")
    ax[0].set_title("盆地内 E(β)：定位 β*(ν) 单调/U型区间"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xscale("log"); ax[1].set_xlabel("β"); ax[1].set_ylabel("成功率"); ax[1].set_title("达标率"); ax[1].grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(OUT, "fig_beta_scan.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)


def _plot_c1(rows):
    fig, a = plt.subplots(1, 2, figsize=(11, 4.2))
    for seed in sorted({r["seed"] for r in rows}):
        z = sorted([r for r in rows if r["seed"] == seed], key=lambda q: q["beta"])
        a[0].plot([q["beta"] for q in z], [q["kappa_static"] for q in z], "o-", label=f"seed{seed}")
        a[1].plot([q["beta"] for q in z], [q["lam_min"] for q in z], "s-", label=f"seed{seed}")
    for ax_, t, y in [(a[0], "C1 冻结 Hessian κ(β)（定理3 预期 ∝1/β）", "κ"),
                      (a[1], "冻结 λmin(β)", "λmin")]:
        ax_.set_xscale("log"); ax_.set_yscale("symlog"); ax_.set_xlabel("β"); ax_.set_ylabel(y)
        ax_.set_title(t); ax_.legend(fontsize=8); ax_.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(OUT, "fig_c1_static.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)


if __name__ == "__main__":
    main()
