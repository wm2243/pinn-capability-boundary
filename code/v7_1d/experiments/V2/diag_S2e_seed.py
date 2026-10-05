# -*- coding: utf-8 -*-
"""
S2e 诊断：粗峰种子（拓扑种子）探针——回答“内部尖峰是否只需拓扑、不需精确形状”。
================================================================================
对照（主 eps=0.0044，配 Burgers ν≈.005；σ课程 5→15，multistart K=4，P0=2000，P1=8000，β=1）：
  none       无种子 A=0（已知 36/36 塌零，此处不重跑）
  sech       精确 sech 提升 A=√2 sech(x/eps)（机制上界：把答案放进 ansatz，网络学零）
  gauss_std  高斯粗峰 A=a0 exp(-(x-x0)^2/2w^2)，a0=√2, w=1.5eps（只给拓扑，形状非 sech）
  gauss_low  幅值不足 a0=0.8（网络须把峰抬高）
  gauss_high 幅值过大 a0=2.0（网络须压低）
  gauss_wide 宽度失准 w=3eps
  gauss_shift 位置偏移 x0=0.15（方程真峰在 0，检验位置是否须精确）
高斯在端点 x=±1 对 w~eps 指数下溢为 0，严格满足齐次 Dirichlet；内部 (1-x²)≈1，网络有修正力。
判据：Phase0 盆地分类（peak/neg/zero/other、峰幅）＋Phase1 全域/峰区 L2、峰幅是否收敛 √2。
不修改公共代码；结果仅打印 + 写 diag_seed.csv（不污染正式 s2e_runs.csv）。
"""
import os, sys, csv, copy, math
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve()
for _p in (str(_HERE.parents[2]), str(_HERE.parents[2] / "experiments" / "v1"), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
import exp_V2_S2e_spike as S2e
from models.pinn_FourierFeatures_model import HardBCPINN
from losses.family_weighting_loss import FamilyWeightingLoss
from losses.rational_weighting_loss1 import RationalWeightingLoss
from samplers.pinn_sampler import PINNSampler
from trainers.pinn_trainer import PINNTrainer
from utils.visualizer import PINNVisualizer

DEV = mc.DEVICE
OUT = os.path.join(S2e.OUT)
CSVP = os.path.join(OUT, "diag_seed.csv")


class SeededPINN(HardBCPINN):
    """在标准硬约束上支持 gauss_seed / sech_seed 两种提升（不引入新可训参数）。"""
    def __init__(self, config):
        super().__init__(config)
        self.seed_a0 = float(config.get("seed_a0", math.sqrt(2)))
        self.seed_w = float(config.get("seed_w", 1.0))
        self.seed_x0 = float(config.get("seed_x0", 0.0))
        self.spike_eps = float(config["spike_eps"])

    def boundary_lift(self, x):
        if self.bc_type == "gauss_seed":
            return self.seed_a0 * torch.exp(-((x - self.seed_x0) ** 2) / (2.0 * self.seed_w ** 2))
        if self.bc_type == "sech_seed":
            # clamp 避免端点 x/eps~227 时 float32 cosh overflow；|z|=12 处 sech≈1.2e-5，边界误差可接受
            _z = ((x - self.seed_x0) / self.spike_eps).clamp(-12.0, 12.0)
            return math.sqrt(2.0) / torch.cosh(_z)
        return super().boundary_lift(x)


def build_cfg(eps, tag, kind, a0, w, x0):
    cfg = S2e.sp_base_cfg(eps, tag)
    if kind == "sech":
        cfg.update(bc_type="sech_seed", seed_x0=x0)
    elif kind == "gauss":
        cfg.update(bc_type="gauss_seed", seed_a0=a0, seed_w=w, seed_x0=x0)
    return cfg


def run_group(eps, name, kind, a0=np.sqrt(2), w=None, x0=0.0, k=4, p0=2000, p1=8000):
    xt, un = S2e.test_tensors(eps)
    w = (1.5 * eps) if w is None else w
    td = {"x": xt, "u_true": un.flatten()}
    r2s.MULTISTART_K = 1 if mc.SMOKE else k
    r2s.PHASE0_EPOCHS = 60 if mc.SMOKE else p0
    cands = []
    for kk in range(r2s.MULTISTART_K):
        seed = kk
        r2s.set_seed(seed)
        cfg = build_cfg(eps, f"{name}_{kk}", kind, a0, w, x0)
        cfg.update(dict(lr=r2s.LR_PHASE0, use_gate=True, gate_patience=3, gate_lr_gamma=0.3,
                        sigma_lo=r2s.SIGMA_LO, sigma_hi=r2s.SIGMA_HI_CFG if False else 15.0,
                        sigma_anneal_T=r2s.SIGMA_T, beta_schedule="const", loss_beta=1.0, beta_init=1.0,
                        adam_epochs=r2s.PHASE0_EPOCHS, lbfgs_epochs=0, use_best_ckpt=True,
                        best_metric="max_L2_band", weight_mode="adaptive"))
        model = SeededPINN(cfg).to(DEV)
        pde = S2e.SpikePDE1D(cfg)
        cfg2 = copy.deepcopy(cfg); cfg2["loss_beta"] = 1.0; cfg2["weight_mode"] = "adaptive"
        crit = RationalWeightingLoss(cfg2, DEV)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        samp = PINNSampler(cfg["domain_x"], DEV, use_adaptive=False, buffer_size=cfg["buffer_size"])
        vis = PINNVisualizer(save_dir=OUT)
        tr = PINNTrainer(model, pde, crit, opt, samp, vis, DEV, cfg2, test_data=td)
        tr.train(cfg2, td, r2s.PHASE0_EPOCHS, 0, adaptive_freq=500)
        if not crit.field_frozen:
            crit.freeze_residual_field(model, pde)
        ev0 = S2e.eval_pred(model, xt, un, eps)
        cands.append(dict(l2=ev0["L2"], cls=ev0["cls"], amp=ev0["peak_amp"], gate=tr.gate_epoch,
                          state=copy.deepcopy(model.state_dict()),
                          t_field=crit._frozen_t_field.detach().clone(), sigma_end=model.get_sigma(),
                          cfg=cfg))
        print(f"    [{name}] start {seed} P0: L2={ev0['L2']:.2e} cls={ev0['cls']} 峰幅={ev0['peak_amp']:+.3f}")
        tr.log_file.close()
    counts = {"peak": 0, "negpeak": 0, "zero": 0, "other": 0}
    for c in cands: counts[c["cls"]] += 1
    best = min(cands, key=lambda z: z["l2"])

    # Phase1 精修选中盆地
    p1e = 60 if mc.SMOKE else p1
    r2s.set_seed(0)
    cfg = copy.deepcopy(best["cfg"])
    cfg.update(dict(lr=r2s.LR_PHASE1, use_gate=False, adam_epochs=p1e, lbfgs_epochs=0,
                    use_best_ckpt=True, best_metric="max_L2_band", beta_schedule="const",
                    loss_beta=1.0, beta_init=1.0, beta_start_step=0, beta_end_step=0,
                    fourier_scale=float(best["sigma_end"]), sigma_lo=None, sigma_hi=None, sigma_anneal_T=0))
    model = SeededPINN(cfg).to(DEV)
    pde = S2e.SpikePDE1D(cfg)
    cfg2 = copy.deepcopy(cfg); cfg2["loss_beta"] = 1.0; cfg2["weight_mode"] = "adaptive"
    crit = FamilyWeightingLoss(cfg2, DEV, family="rational")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    samp = PINNSampler(cfg["domain_x"], DEV, use_adaptive=False, buffer_size=cfg["buffer_size"])
    vis = PINNVisualizer(save_dir=OUT)
    tr = PINNTrainer(model, pde, crit, opt, samp, vis, DEV, cfg2, test_data=td)
    model.load_state_dict(best["state"]); model.set_sigma(best["sigma_end"])
    crit.load_frozen_field(best["t_field"], beta=1.0)
    tr.train(cfg2, td, p1e, 0, adaptive_freq=500)
    ev1 = S2e.eval_pred(model, xt, un, eps)
    tr.log_file.close()
    print(f"  == [{name}] {kind} a0={a0:.3f} w={w:.5f} x0={x0}: P0计数={counts} | "
          f"Phase1 L2={ev1['L2']:.2e} 峰区={ev1['L2_peak']:.2e} 峰幅={ev1['peak_amp']:+.3f} cls={ev1['cls']}")
    return dict(name=name, kind=kind, a0=float(a0), w=float(w), x0=float(x0), eps=eps,
                n_peak=counts["peak"], n_neg=counts["negpeak"], n_zero=counts["zero"], n_other=counts["other"],
                p0_best=float(best["l2"]), L2=ev1["L2"], L2_peak=ev1["L2_peak"],
                peak_amp=ev1["peak_amp"], cls=ev1["cls"])


def main():
    eps = 0.0044
    groups = [
        ("sech",      "sech", np.sqrt(2), None, 0.0),
        ("gauss_std", "gauss", np.sqrt(2), None, 0.0),
        ("gauss_low", "gauss", 0.8,       None, 0.0),
        ("gauss_high","gauss", 2.0,       None, 0.0),
        ("gauss_wide","gauss", np.sqrt(2), 3.0 * eps, 0.0),
        ("gauss_shift","gauss", np.sqrt(2), None, 0.15),
    ]
    rows = []
    for name, kind, a0, wfac, x0 in groups:
        w = wfac  # None -> 1.5eps inside
        rows.append(run_group(eps, name, kind, a0=a0, w=w, x0=x0))
    cols = ["name", "kind", "eps", "a0", "w", "x0", "n_peak", "n_neg", "n_zero", "n_other",
            "p0_best", "L2", "L2_peak", "peak_amp", "cls"]
    with open(CSVP, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        for r in rows: wr.writerow({k: r[k] for k in cols})
    print("\n===== 粗峰种子探针汇总（eps=%.4f, K=4）=====" % eps)
    print("无种子对照(none)：36/36 塌零（baseline+K8）")
    for r in rows:
        print(f"  {r['name']:<12} P0 正峰{r['n_peak']}/4 负{r['n_neg']} 零{r['n_zero']} 其他{r['n_other']} "
              f"| Phase1 L2={r['L2']:.2e} 峰区={r['L2_peak']:.2e} 峰幅={r['peak_amp']:+.3f} ({r['cls']})")
    print("csv:", CSVP)


if __name__ == "__main__":
    main()
