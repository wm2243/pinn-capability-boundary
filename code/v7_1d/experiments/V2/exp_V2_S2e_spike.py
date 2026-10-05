# -*- coding: utf-8 -*-
"""
V2 实验 S2e：内部尖峰（反应-扩散 homoclinic spike）——服务理论的盆地选择迁移检验
================================================================================
主输出不是最低 L2，而是 multistart 的【盆地形态分类命中率】：
  peak    : 正峰 u*(0)≈+sqrt2（目标带峰盆地）
  negpeak : 负峰 -u*（符号简并的另一精确解）
  zero    : 塌到平凡解 u≡0（竞争盆地，归一化 L2≈1）
  other   : 其余杂峰/半成品
检验主张：盆地选择由 σ 表示课程 + multistart 决定；减权不能把塌零解拉成带峰解（与 G1b 同构）。
方程 -eps^2 u''+u-u^3=0，u*=sqrt2 sech((x-x0)/eps)，x0=0，宽度 O(eps)（10–90 全宽 5.05eps）。
厚度配对 Burgers（10–90 全宽 4.394ν）：eps≈0.87ν；ν=.005 ↔ eps≈.0044。

装配逐行复刻 r2s/mc 两阶段，仅替换 PDE=SpikePDE1D、cfg nu=eps、shock_band_c=5、
bc=lift_linear（两端精确指数小尾值，eps 小时为 0）；尖峰居中，标准 PINNTrainer。
不修改公共代码；盆地缓存于本目录 _cache。

运行：
  set V8_SMOKE=1 && python exp_V2_S2e_spike.py --baseline
  python exp_V2_S2e_spike.py --baseline
  python exp_V2_S2e_spike.py --eps 0.0044 --seeds 0,5,10 --k 8 \
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
from physics.spike_pde import SpikePDE1D
from models.pinn_FourierFeatures_model import HardBCPINN
from losses.family_weighting_loss import FamilyWeightingLoss
from losses.rational_weighting_loss1 import RationalWeightingLoss
from samplers.pinn_sampler import PINNSampler
from trainers.pinn_trainer import PINNTrainer
from utils.visualizer import PINNVisualizer

OUT = str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S2e_spike")
CACHE = os.path.join(OUT, "_cache"); os.makedirs(CACHE, exist_ok=True)
r2s.OUT = OUT
DEV = mc.DEVICE
N_TEST = 2048
BAND_C = 5.0
AMP = float(np.sqrt(2.0))
FIELDS = ["eps", "nu_eq_width", "seed", "family", "beta", "p0", "p1",
          "phase0_L2", "cls0", "L2", "Linf", "L2_peak", "peak_amp", "cls1",
          "gate_epoch", "best_epoch", "k", "n_peak", "n_negpeak", "n_zero", "n_other"]

_orig_base_cfg = r2s.base_cfg
def _bc_base_cfg(nu, seed_tag):
    c = _orig_base_cfg(nu, seed_tag)
    c["bc_amp"] = float(np.tanh(1.0 / (2.0 * max(float(nu), 1e-8))))
    return c
r2s.base_cfg = _bc_base_cfg


def sp_base_cfg(eps, tag):
    c = r2s.base_cfg(float(eps), tag)
    bl, br = SpikePDE1D(dict(spike_eps=eps)).bc_values()
    c.update(dict(shock_band_c=BAND_C, bc_type="lift_linear", bc_left=bl, bc_right=br,
                  spike_eps=float(eps), spike_x0=0.0))
    return c


def test_tensors(eps):
    pde = SpikePDE1D(dict(spike_eps=eps))
    xn = np.linspace(-1, 1, N_TEST, dtype=np.float64).reshape(-1, 1)
    un = pde.exact_np(xn).astype(np.float64)
    return torch.tensor(xn, dtype=torch.float32, device=DEV), un


def classify(pred, x, center):
    """盆地形态分类。center=pred(x0)；按峰幅与最大幅值区分正峰/负峰/塌零/其他。"""
    c = float(pred[np.argmin(np.abs(x - 0.0))])
    mx = float(np.max(np.abs(pred)))
    if c > 1.0:
        return "peak", c, mx
    if c < -1.0:
        return "negpeak", c, mx
    if mx < 0.25:
        return "zero", c, mx
    return "other", c, mx


def eval_pred(model, xt, un, eps):
    model.eval()
    with torch.no_grad():
        pred = model(xt).detach().cpu().numpy().astype(np.float64).flatten()
    x = xt.detach().cpu().numpy().flatten(); u = un.flatten(); d = pred - u
    m = np.abs(x) < BAND_C * eps
    g = max(np.linalg.norm(u), 1e-12)
    cls, camp, mx = classify(pred, x, 0.0)
    model.train()
    return dict(L2=float(np.linalg.norm(d) / g), Linf=float(np.max(np.abs(d))),
                L2_peak=float(np.linalg.norm(d[m]) / max(np.linalg.norm(u[m]), 1e-12)),
                peak_amp=camp, cls=cls, maxabs=mx)


def enter_basin(eps, base_seed, xt, un, k, p0, sigma_hi=15.0, reuse=True):
    tag = f"sp{eps:g}_s{base_seed}_sig{sigma_hi:g}_k{k}_p0{p0}.pt"
    path = os.path.join(CACHE, tag)
    r2s.MULTISTART_K = 1 if mc.SMOKE else int(k)
    r2s.SELECT_BY = "truth"; r2s.SIG_HI = float(sigma_hi); r2s.SIGMA_HI_CFG = float(sigma_hi)
    r2s.PHASE0_EPOCHS = 60 if mc.SMOKE else int(p0)
    if reuse and os.path.exists(path):
        d = torch.load(path, map_location=DEV)
        print(f"[Cache] hit {tag} Phase0_L2={d['l2']:.3e} cls={d['cls']} "
              f"counts={d['counts']}")
        return d
    td = {"x": xt, "u_true": un.flatten()}
    cands, counts = [], {"peak": 0, "negpeak": 0, "zero": 0, "other": 0}
    for kk in range(r2s.MULTISTART_K):
        seed = base_seed + kk
        r2s.set_seed(seed)
        cfg = sp_base_cfg(eps, f"p0_sp{eps:g}_{base_seed}_{kk}")
        cfg.update(dict(lr=r2s.LR_PHASE0, use_gate=True, gate_patience=3, gate_lr_gamma=0.3,
                        sigma_lo=r2s.SIGMA_LO, sigma_hi=r2s.SIGMA_HI_CFG, sigma_anneal_T=r2s.SIGMA_T,
                        beta_schedule="const", loss_beta=1.0, beta_init=1.0,
                        adam_epochs=r2s.PHASE0_EPOCHS, lbfgs_epochs=0, use_best_ckpt=True,
                        best_metric="max_L2_band", weight_mode="adaptive"))
        model = HardBCPINN(cfg).to(DEV)
        pde = SpikePDE1D(cfg)
        cfg2 = copy.deepcopy(cfg); cfg2["loss_beta"] = 1.0; cfg2["weight_mode"] = "adaptive"
        crit = RationalWeightingLoss(cfg2, DEV)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        samp = PINNSampler(cfg["domain_x"], DEV, use_adaptive=False, buffer_size=cfg["buffer_size"])
        vis = PINNVisualizer(save_dir=OUT)
        tr = PINNTrainer(model, pde, crit, opt, samp, vis, DEV, cfg2, test_data=td)
        tr.train(cfg2, td, r2s.PHASE0_EPOCHS, 0, adaptive_freq=500)
        if not crit.field_frozen:
            crit.freeze_residual_field(model, pde)
        ev = eval_pred(model, xt, un, eps)
        counts[ev["cls"]] += 1
        cands.append(dict(seed=seed, l2=ev["L2"], cls=ev["cls"], gate=tr.gate_epoch,
                          state=copy.deepcopy(model.state_dict()),
                          t_field=crit._frozen_t_field.detach().clone(), sigma_end=model.get_sigma()))
        print(f"  [Phase0 sp{eps:g}] start {seed}: L2={ev['L2']:.3e} cls={ev['cls']} "
              f"峰幅={ev['peak_amp']:+.3f} gate@{tr.gate_epoch}")
        tr.log_file.close()
    best = min(cands, key=lambda z: z["l2"])
    print(f"[Phase0 sp] 选用 start {best['seed']} cls={best['cls']} L2={best['l2']:.3e}; "
          f"盆地计数 {counts}")
    best["counts"] = counts
    torch.save(best, path)
    return best


def phase1(eps, basin, family, beta, base_seed, xt, un, p1):
    r2s.set_seed(int(base_seed))
    cfg = sp_base_cfg(eps, f"p1_sp{eps:g}_s{base_seed}_{family}b{beta:g}")
    cfg.update(dict(lr=r2s.LR_PHASE1, use_gate=False, adam_epochs=p1, lbfgs_epochs=0,
                    use_best_ckpt=True, best_metric="max_L2_band", beta_schedule="const",
                    loss_beta=float(beta), beta_init=1.0, beta_start_step=0, beta_end_step=0,
                    fourier_scale=float(basin["sigma_end"]),
                    sigma_lo=None, sigma_hi=None, sigma_anneal_T=0))
    model = HardBCPINN(cfg).to(DEV)
    pde = SpikePDE1D(cfg)
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
    ev = eval_pred(model, xt, un, eps)
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
        eps_list = [0.01, 0.0044, 0.002]
        seeds, fams = [0], [("rational", 1.0)]
        k, p0, p1 = (1, 60, 120) if mc.SMOKE else (a.k, a.p0, a.p1)
    else:
        eps_list = [float(x) for x in a.eps.split(",") if x.strip()] or [0.0044]
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
        fams = parse_families(a.families)
        k, p0, p1 = (1, 60, 120) if mc.SMOKE else (a.k, a.p0, a.p1)

    csvp = os.path.join(OUT, "s2e_runs.csv")
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
            xt, un = test_tensors(eps)
            for seed in seeds:
                basin = enter_basin(eps, seed, xt, un, k, p0)
                co = basin["counts"]
                for family, beta in fams:
                    key = (float(eps), int(seed), family, float(beta))
                    if key in done:
                        print("skip", key); continue
                    ev, gate, best_ep = phase1(eps, basin, family, beta, seed, xt, un, p1)
                    row = dict(eps=float(eps), nu_eq_width=float(eps / 0.87), seed=seed,
                               family=family, beta=float(beta), p0=p0, p1=p1,
                               phase0_L2=float(basin["l2"]), cls0=basin["cls"], L2=ev["L2"], Linf=ev["Linf"], L2_peak=ev["L2_peak"], peak_amp=ev["peak_amp"], cls1=ev["cls"],
                               gate_epoch=gate, best_epoch=best_ep, k=k,
                               n_peak=co["peak"], n_negpeak=co["negpeak"],
                               n_zero=co["zero"], n_other=co["other"])
                    wcsv.writerow(row); fout.flush(); done.add(key); n_new += 1
                    print(f"[S2e] eps={eps:g}(ν_eq={eps/0.87:.4f}) s{seed} {family}β{beta:g}: "
                          f"P0={basin['l2']:.2e}({basin['cls']}) -> L2={ev['L2']:.2e} "
                          f"峰幅={ev['peak_amp']:+.3f} cls={ev['cls']} 盆地计数={co}")
    finally:
        fout.close()
    print(f"\n[S2e] 新训 {n_new} 组，结果 {csvp}")
    print("\n===== 尖峰盆地分类（Phase0 multistart）＋精修 =====")
    allrows = list(csv.DictReader(open(csvp, encoding="utf-8-sig")))
    for eps in sorted({float(r["eps"]) for r in allrows}):
        sub = [r for r in allrows if abs(float(r["eps"]) - eps) < 1e-15]
        if not sub: continue
        pk = int(sub[0]["n_peak"]); ng = int(sub[0]["n_negpeak"]); zr = int(sub[0]["n_zero"]); ot = int(sub[0]["n_other"])
        L2 = np.median([float(r["L2"]) for r in sub]); Lp = np.median([float(r["L2_peak"]) for r in sub])
        print(f"  eps={eps:<7g} ν_eq={eps/0.87:.4f}: 正峰 {pk}/{int(sub[0]['k'])}  负峰 {ng}  "
              f"塌零 {zr}  其他 {ot} | 精修全域L2中位={L2:.2e} 峰区={Lp:.2e} (行={len(sub)})")


if __name__ == "__main__":
    main()
