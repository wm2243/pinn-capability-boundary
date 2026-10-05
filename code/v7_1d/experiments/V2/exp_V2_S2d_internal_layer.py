# -*- coding: utf-8 -*-
"""
V2 实验 S2d：内部层（转向点奇摄动）——Burgers 两阶段配置向【线性椭圆、域内陡变】的迁移检验
================================================================================
对照定位：
  * Burgers 激波 / 本算例内部层：陡变居中 x=0，(1-x²)≈1，提升 A=½(1+x) 两端匹配渐近值，
    N*=(u*-A)/(1-x²) 光滑有界 -> 预期 σ课程+multistart+两阶段 可直接迁移（好盆地可命中）。
  * S2b/S2c 贴边边界层：陡变在 x=1、(1-x²)→0、N* 发散/大幅值 -> K=16 零命中（能力边界对照）。
方程：-eps u'' - x u' = 0, u(-1)=0,u(1)=1；u*=½[1+erf(x/√(2eps))/erf(1/√(2eps))]，层厚 O(√eps)。
厚度配对（10–90 全宽 2.563√eps = 4.394ν）：ν_eq = 0.5832√eps；eps=1e-4 ↔ ν_eq≈5.8e-3（≈Burgers .005）。

装配逐行复刻 r2s.run_phase0 / mc.train_phase1_family，仅替换：
  PDE=TurningPointPDE1D(tp_eps=eps)、cfg['nu']=√eps（居中带/Gate 尺度）、bc=lift_linear(0→1)；
内部层居中，直接用标准 PINNTrainer（|x|<c√eps 内层掩膜天然正确），不做任何掩膜子类化。
不修改公共代码；盆地缓存于本目录 _cache。

运行：
  set V8_SMOKE=1 && python exp_V2_S2d_internal_layer.py --baseline
  python exp_V2_S2d_internal_layer.py --baseline
  python exp_V2_S2d_internal_layer.py --eps 1e-4 --seeds 0,5,10 \
      --families uniform:1,rational:1,rational:3,rational:10,down_inv:1,down_inv:3,down_inv:10
================================================================================
"""
import os, sys, csv, argparse, copy
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve(); _V7 = _HERE.parents[2]; _V1EXP = _V7 / "experiments" / "v1"
for _p in (str(_V7), str(_V1EXP), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from physics.turningpoint_pde import TurningPointPDE1D
from models.pinn_FourierFeatures_model import HardBCPINN
from losses.family_weighting_loss import FamilyWeightingLoss
from losses.rational_weighting_loss1 import RationalWeightingLoss
from samplers.pinn_sampler import PINNSampler
from trainers.pinn_trainer import PINNTrainer
from utils.visualizer import PINNVisualizer

OUT = str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S2d_internal_layer")
CACHE = os.path.join(OUT, "_cache"); os.makedirs(CACHE, exist_ok=True)
r2s.OUT = OUT
DEV = mc.DEVICE
N_TEST = 2048
BAND_C = 3.0
FIELDS = ["eps", "nu_eff", "nu_eq_width", "seed", "family", "beta", "p0", "p1",
          "phase0_L2", "L2", "Linf", "L2_layer", "gate_epoch", "best_epoch", "k"]

_orig_base_cfg = r2s.base_cfg
def _bc_base_cfg(nu, seed_tag):
    c = _orig_base_cfg(nu, seed_tag)
    c["bc_amp"] = float(np.tanh(1.0 / (2.0 * max(float(nu), 1e-8))))
    return c
r2s.base_cfg = _bc_base_cfg


def tp_base_cfg(eps, tag):
    nu_eff = float(np.sqrt(eps))
    c = r2s.base_cfg(nu_eff, tag)
    c.update(dict(shock_band_c=BAND_C, bc_type="lift_linear", bc_left=0.0, bc_right=1.0,
                  tp_eps=float(eps)))
    return c, nu_eff


def test_tensors(eps):
    pde = TurningPointPDE1D(dict(tp_eps=eps))
    xn = np.linspace(-1, 1, N_TEST, dtype=np.float64).reshape(-1, 1)
    un = pde.exact_np(xn).astype(np.float64)
    return torch.tensor(xn, dtype=torch.float32, device=DEV), un, nu_band(eps)


def nu_band(eps):
    return float(np.sqrt(eps))


def quick_l2(model, xt, un):
    model.eval()
    with torch.no_grad():
        up = model(xt).detach().cpu().numpy().astype(np.float64).flatten()
    model.train()
    return float(np.linalg.norm(up - un.flatten()) / max(np.linalg.norm(un), 1e-12))


def final_eval(model, xt, un, nu_eff):
    model.eval()
    with torch.no_grad():
        pred = model(xt).detach().cpu().numpy().astype(np.float64).flatten()
    x = xt.detach().cpu().numpy().flatten(); u = un.flatten(); d = pred - u
    m = np.abs(x) < BAND_C * nu_eff
    g = max(np.linalg.norm(u), 1e-12)
    model.train()
    return dict(L2=float(np.linalg.norm(d) / g), Linf=float(np.max(np.abs(d))),
                L2_layer=float(np.linalg.norm(d[m]) / max(np.linalg.norm(u[m]), 1e-12)))


def enter_basin(eps, base_seed, xt, un, k, p0, sigma_hi=15.0, reuse=True):
    _, nu_eff = tp_base_cfg(eps, "x")
    tag = f"tp{eps:g}_s{base_seed}_sig{sigma_hi:g}_k{k}_p0{p0}.pt"
    path = os.path.join(CACHE, tag)
    r2s.MULTISTART_K = 1 if mc.SMOKE else int(k)
    r2s.SELECT_BY = "truth"; r2s.SIG_HI = float(sigma_hi); r2s.SIGMA_HI_CFG = float(sigma_hi)
    r2s.PHASE0_EPOCHS = 60 if mc.SMOKE else int(p0)
    if reuse and os.path.exists(path):
        d = torch.load(path, map_location=DEV)
        print(f"[Cache] hit {tag} Phase0_L2={d['l2']:.3e}")
        return d
    td = {"x": xt, "u_true": un.flatten()}
    cands = []
    for kk in range(r2s.MULTISTART_K):
        seed = base_seed + kk
        r2s.set_seed(seed)
        cfg, _ = tp_base_cfg(eps, f"p0_tp{eps:g}_{base_seed}_{kk}")
        cfg.update(dict(lr=r2s.LR_PHASE0, use_gate=True, gate_patience=3, gate_lr_gamma=0.3,
                        sigma_lo=r2s.SIGMA_LO, sigma_hi=r2s.SIGMA_HI_CFG, sigma_anneal_T=r2s.SIGMA_T,
                        beta_schedule="const", loss_beta=1.0, beta_init=1.0,
                        adam_epochs=r2s.PHASE0_EPOCHS, lbfgs_epochs=0, use_best_ckpt=True,
                        best_metric="max_L2_band", weight_mode="adaptive"))
        model = HardBCPINN(cfg).to(DEV)
        pde = TurningPointPDE1D(cfg)
        cfg2 = copy.deepcopy(cfg); cfg2["loss_beta"] = 1.0; cfg2["weight_mode"] = "adaptive"
        crit = RationalWeightingLoss(cfg2, DEV)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        samp = PINNSampler(cfg["domain_x"], DEV, use_adaptive=False, buffer_size=cfg["buffer_size"])
        vis = PINNVisualizer(save_dir=OUT)
        tr = PINNTrainer(model, pde, crit, opt, samp, vis, DEV, cfg2, test_data=td)
        tr.train(cfg2, td, r2s.PHASE0_EPOCHS, 0, adaptive_freq=500)
        if not crit.field_frozen:
            crit.freeze_residual_field(model, pde)
        l2 = quick_l2(model, xt, un)
        cands.append(dict(seed=seed, l2=l2, gate=tr.gate_epoch,
                          state=copy.deepcopy(model.state_dict()),
                          t_field=crit._frozen_t_field.detach().clone(), sigma_end=model.get_sigma()))
        print(f"  [Phase0 tp{eps:g}] start {seed}: L2={l2:.3e} gate@{tr.gate_epoch}")
        tr.log_file.close()
    best = min(cands, key=lambda z: z["l2"])
    print(f"[Phase0 tp] 选用 start {best['seed']}, L2={best['l2']:.3e}")
    torch.save(best, path)
    return best


def phase1(eps, basin, family, beta, base_seed, xt, un, p1):
    _, nu_eff = tp_base_cfg(eps, "x")
    r2s.set_seed(int(base_seed))
    cfg, _ = tp_base_cfg(eps, f"p1_tp{eps:g}_s{base_seed}_{family}b{beta:g}")
    cfg.update(dict(lr=r2s.LR_PHASE1, use_gate=False, adam_epochs=p1, lbfgs_epochs=0,
                    use_best_ckpt=True, best_metric="max_L2_band", beta_schedule="const",
                    loss_beta=float(beta), beta_init=1.0, beta_start_step=0, beta_end_step=0,
                    fourier_scale=float(basin["sigma_end"]),
                    sigma_lo=None, sigma_hi=None, sigma_anneal_T=0))
    model = HardBCPINN(cfg).to(DEV)
    pde = TurningPointPDE1D(cfg)
    cfg2 = copy.deepcopy(cfg); cfg2["loss_beta"] = float(beta); cfg2["weight_mode"] = "adaptive"
    crit = FamilyWeightingLoss(cfg2, DEV, family=family)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    samp = PINNSampler(cfg["domain_x"], DEV, use_adaptive=False, buffer_size=cfg["buffer_size"])
    vis = PINNVisualizer(save_dir=OUT)
    td = {"x": xt, "u_true": un.flatten()}
    tr = PINNTrainer(model, pde, crit, opt, samp, vis, DEV, cfg2, test_data=td)
    model.load_state_dict(basin["state"]); model.set_sigma(basin["sigma_end"])
    crit.load_frozen_field(basin["t_field"], beta=float(beta))
    tr.train(cfg2, td, p1, 0, adaptive_freq=500)
    ev = final_eval(model, xt, un, nu_eff)
    best_ep = int(getattr(tr, "best_epoch", -1)); tr.log_file.close()
    return ev, int(basin.get("gate", -1)), best_ep


def parse_families(s):
    return [(lambda t: (t.split(":")[0], float(t.split(":")[1])))(t) for t in s.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--eps", default="")
    ap.add_argument("--families", default="rational:1")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--p0", type=int, default=2000)
    ap.add_argument("--p1", type=int, default=8000)
    a = ap.parse_args()

    if a.baseline:
        eps_list = [0.01, 0.001, 1e-4]
        seeds, fams = [0], [("rational", 1.0)]
        k, p0, p1 = (1, 60, 120) if mc.SMOKE else (a.k, a.p0, a.p1)
    else:
        eps_list = [float(x) for x in a.eps.split(",") if x.strip()] or [1e-4]
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
        fams = parse_families(a.families)
        k, p0, p1 = (1, 60, 120) if mc.SMOKE else (a.k, a.p0, a.p1)

    csvp = os.path.join(OUT, "s2d_runs.csv")
    done = set()
    if os.path.exists(csvp):
        for r in csv.DictReader(open(csvp, encoding="utf-8-sig")):
            done.add((float(r["eps"]), int(r["seed"]), r["family"], float(r["beta"])))
    newfile = not os.path.exists(csvp)
    fout = open(csvp, "a", newline="", encoding="utf-8-sig")
    wcsv = csv.DictWriter(fout, fieldnames=FIELDS)
    if newfile: wcsv.writeheader()
    n_new = 0
    try:
        for eps in eps_list:
            xt, un, nu_eff = test_tensors(eps)
            for seed in seeds:
                basin = enter_basin(eps, seed, xt, un, k, p0)
                for family, beta in fams:
                    key = (float(eps), int(seed), family, float(beta))
                    if key in done:
                        print("skip", key); continue
                    ev, gate, best_ep = phase1(eps, basin, family, beta, seed, xt, un, p1)
                    nu_eq = 0.5832 * np.sqrt(eps)
                    row = dict(eps=float(eps), nu_eff=nu_eff, nu_eq_width=float(nu_eq),
                               seed=seed, family=family, beta=float(beta), p0=p0, p1=p1,
                               phase0_L2=float(basin["l2"]), **ev,
                               gate_epoch=gate, best_epoch=best_ep, k=k)
                    wcsv.writerow(row); fout.flush(); done.add(key); n_new += 1
                    print(f"[S2d] eps={eps:g}(√ε={nu_eff:.4f},ν_eq={nu_eq:.4f}) s{seed} "
                          f"{family}β{beta:g}: P0L2={basin['l2']:.2e} -> L2={ev['L2']:.2e} "
                          f"内层={ev['L2_layer']:.2e} gate@{gate}")
    finally:
        fout.close()
    print(f"\n[S2d] 新训 {n_new} 组，结果 {csvp}")
    print("\n===== 内部层 baseline（β=1，全域 L2 / 居中内层 L2）=====")
    allrows = list(csv.DictReader(open(csvp, encoding="utf-8-sig")))
    for eps in sorted({float(r["eps"]) for r in allrows}):
        sub = [r for r in allrows if abs(float(r["eps"]) - eps) < 1e-15]
        if not sub: continue
        L2 = np.median([float(r["L2"]) for r in sub]); Ll = np.median([float(r["L2_layer"]) for r in sub])
        p0l = np.median([float(r["phase0_L2"]) for r in sub])
        print(f"  eps={eps:<8g} √ε={np.sqrt(eps):.4f} ν_eq={.5832*np.sqrt(eps):.4f}: "
              f"Phase0={p0l:.2e}  全域L2={L2:.3e}  内层L2={Ll:.3e}  (n={len(sub)})")


if __name__ == "__main__":
    main()
