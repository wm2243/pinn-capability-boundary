# -*- coding: utf-8 -*-
"""
V2 实验 S2d-main：内部转向点层【全量主结果】——3 厚度 × n=11 × K=4 两阶段
================================================================================
定位：S2d 跨方程算例的能力主结果（不是减权实验——basin_diag 已证该配置无坏盆地，
减权无作用对象）。在与 Burgers 完全对齐的配置下（σ 课程 5→15、64×4、N_PDE=10000、
lr 1e-3/5e-4、band_c=3、Phase0=2000 + Phase1=6000=8000），报告：
  · 逐起点（3 厚度 × 11 base × K=4 = 132 个独立 Phase0）几何好盆地命中率（几何判据，
    跨表示器/跨厚度公平；不搬 Burgers 的 L2<.06）；
  · K=4 multistart 真解选优（能力上界诊断口径，非可部署判据，论文写明）的选中率；
  · 选中 Gate 点经 Phase1（rational β=1，即等权）精修后的全域/层内相对 L2（n=11，中位/IQR）。
同时把每个 (eps,base) 的选中 Gate 点权重快照存入 _cache/，供 exp_V2_S2d_beta_effect 复用
（机制 A 冻结探针与 β 档配对精修），不重复 Phase0。

方程 -eps u''-x u'=0, u(-1)=0,u(1)=1；厚度 eps∈{.01,.001,1e-4}（等效 ν_eq=.5832√eps）。
断点续跑（以 (eps,base) 为键）；不修改公共代码；结果 results/v7/V2/exp_V2_S2d_main_accuracy/。

运行：
  set V8_SMOKE=1 && python exp_V2_S2d_main_accuracy.py
  python exp_V2_S2d_main_accuracy.py
================================================================================
"""
import os, sys, csv, copy, argparse
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve(); _V7 = _HERE.parents[2]; _V1EXP = _V7 / "experiments" / "v1"
for _p in (str(_V7), str(_V1EXP), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
import exp_V2_S2d_basin_diag as diag
from physics.turningpoint_pde import TurningPointPDE1D
from models.pinn_FourierFeatures_model import HardBCPINN
from losses.rational_weighting_loss1 import RationalWeightingLoss
from samplers.pinn_sampler import PINNSampler
from trainers.pinn_trainer import PINNTrainer
from utils.visualizer import PINNVisualizer

OUT = str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S2d_main_accuracy")
CACHE = os.path.join(OUT, "_cache")
os.makedirs(CACHE, exist_ok=True)
DEV = mc.DEVICE
P0, P1 = 2000, 6000
EPS_ALL = [0.01, 0.001, 1e-4]
SIGMA = 15.0
BASES_ALL = list(mc.SEEDS_EXT)
K_ALL = 4
N_TRAIN_TEST = 2048
if mc.SMOKE:
    EPS_ALL = [1e-4]; BASES_ALL = [0]; K_ALL = 2; P0 = P1 = 60

F_START = ["eps", "base_seed", "k_idx", "start_seed", "p0",
           "L2_phase0", "L2_layer_phase0", "u_center", "slope_ratio", "jump05", "plat_err", "geom_good"]
F_BASE = ["eps", "base_seed", "chosen_seed", "gate_geom_good", "p0", "p1",
          "L2_final", "L2_layer_final", "u_center_f", "slope_f", "jump_f", "plat_f"]


def _phase0_candidate(eps, seed, p0, x_eval, u_eval, x_td, u_td):
    r2s.set_seed(seed)
    cfg, nu_eff = diag.tp_cfg(eps, f"main_tp{eps:g}_s{seed}", "fourier", SIGMA)
    cfg.update(dict(lr=r2s.LR_PHASE0, use_gate=True, gate_patience=3, gate_lr_gamma=0.3,
                    sigma_lo=r2s.SIGMA_LO, sigma_hi=float(SIGMA), sigma_anneal_T=r2s.SIGMA_T,
                    beta_schedule="const", loss_beta=1.0, beta_init=1.0,
                    adam_epochs=int(p0), lbfgs_epochs=0, use_best_ckpt=True,
                    best_metric="max_L2_band", weight_mode="adaptive"))
    model = HardBCPINN(cfg).to(DEV)
    pde = TurningPointPDE1D(cfg)
    cfg2 = copy.deepcopy(cfg); cfg2["loss_beta"] = 1.0; cfg2["weight_mode"] = "adaptive"
    crit = RationalWeightingLoss(cfg2, DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    samp = PINNSampler(cfg["domain_x"], DEV, use_adaptive=False, buffer_size=cfg["buffer_size"])
    vis = PINNVisualizer(save_dir=OUT)
    td = {"x": torch.tensor(x_td.reshape(-1, 1), dtype=torch.float32, device=DEV),
          "u_true": u_td.flatten()}
    tr = PINNTrainer(model, pde, crit, opt, samp, vis, DEV, cfg2, test_data=td)
    tr.train(cfg2, td, int(p0), 0, adaptive_freq=500)
    m, _ = diag.eval_start(model, x_eval, u_eval, eps)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tr.log_file.close()
    del model, crit, opt, tr
    if DEV.type == "cuda": torch.cuda.empty_cache()
    return dict(seed=seed, state=state, cfg=cfg, m=m)


def _phase1_refine(eps, cand, beta, p1, x_eval, u_eval, x_td, u_td, tag):
    cfg = copy.deepcopy(cand["cfg"])
    r2s.set_seed(int(cand["seed"]))   # 配对精修：三档 β 同种子/同 Gate，仅 β 不同
    model = HardBCPINN(cfg).to(DEV); model.load_state_dict(cand["state"])
    pde = TurningPointPDE1D(cfg)
    cfg.update(dict(lr=r2s.LR_PHASE1, adam_epochs=int(p1), lbfgs_epochs=0,
                    weight_mode="adaptive", loss_beta=float(beta), beta_init=float(beta),
                    beta_schedule="const", use_best_ckpt=True, best_metric="max_L2_band"))
    crit = RationalWeightingLoss(cfg, DEV)
    crit.freeze_residual_field(model, pde)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    samp = PINNSampler(cfg["domain_x"], DEV, use_adaptive=False, buffer_size=cfg["buffer_size"])
    vis = PINNVisualizer(save_dir=OUT)
    td = {"x": torch.tensor(x_td.reshape(-1, 1), dtype=torch.float32, device=DEV),
          "u_true": u_td.flatten()}
    tr = PINNTrainer(model, pde, crit, opt, samp, vis, DEV, cfg, test_data=td)
    tr.train(cfg, td, int(p1), 0, adaptive_freq=500)
    mf, _ = diag.eval_start(model, x_eval, u_eval, eps)
    tr.log_file.close()
    del model, crit, opt, tr
    if DEV.type == "cuda": torch.cuda.empty_cache()
    return mf


def run_base(eps, base, K, p0, p1, ws, wb, fs, fb):
    x_eval, u_eval = diag.test_arrays(eps)
    x_td, u_td = diag.test_arrays_n(eps, N_TRAIN_TEST)
    cands = []
    for ki in range(K):
        seed = int(base) + ki
        c = _phase0_candidate(eps, seed, p0, x_eval, u_eval, x_td, u_td)
        cands.append(c)
        good = int(diag.geom_good(c["m"]))
        ws.writerow(dict(eps=float(eps), base_seed=int(base), k_idx=ki, start_seed=seed, p0=int(p0),
                         L2_phase0=c["m"]["L2"], L2_layer_phase0=c["m"]["L2_layer"],
                         u_center=c["m"]["u_center"], slope_ratio=c["m"]["slope_ratio"],
                         jump05=c["m"]["jump05"], plat_err=c["m"]["plat_err"], geom_good=good))
        fs.flush()
        print(f"  [p0] eps{eps:g} base{base} start{seed}: L2={c['m']['L2']:.2e} "
              f"slope={c['m']['slope_ratio']:.3f} jump={c['m']['jump05']:.3f} good={good}")
    kk = int(np.argmin([c["m"]["L2"] for c in cands]))
    chosen = cands[kk]
    gate_good = int(diag.geom_good(chosen["m"]))
    torch.save(dict(eps=float(eps), base=int(base), chosen_seed=int(chosen["seed"]),
                    state=chosen["state"], cfg=chosen["cfg"], gate_m=chosen["m"],
                    gate_geom_good=gate_good),
               os.path.join(CACHE, f"gate_eps{eps:g}_base{int(base)}.pt"))
    mf = _phase1_refine(eps, chosen, 1.0, p1, x_eval, u_eval, x_td, u_td, "main")
    wb.writerow(dict(eps=float(eps), base_seed=int(base), chosen_seed=int(chosen["seed"]),
                     gate_geom_good=gate_good, p0=int(p0), p1=int(p1),
                     L2_final=mf["L2"], L2_layer_final=mf["L2_layer"], u_center_f=mf["u_center"],
                     slope_f=mf["slope_ratio"], jump_f=mf["jump05"], plat_f=mf["plat_err"]))
    fb.flush()
    print(f"[main] eps{eps:g} base{base} chosen{chosen['seed']} good{gate_good}: "
          f"L2fin={mf['L2']:.2e} layer={mf['L2_layer']:.2e}")


def _wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0., c - h), min(1., c + h)


def _iqr(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.median(v), np.quantile(v, .25), np.quantile(v, .75)


def aggregate_and_plot():
    sp = os.path.join(OUT, "s2d_main_starts.csv"); bp = os.path.join(OUT, "s2d_main_bases.csv")
    if not (os.path.exists(sp) and os.path.exists(bp)): return
    starts = list(csv.DictReader(open(sp, encoding="utf-8-sig")))
    bases = list(csv.DictReader(open(bp, encoding="utf-8-sig")))
    agg = []
    print("\n===== S2d 主结果聚合 =====")
    for eps in sorted({float(r["eps"]) for r in bases}):
        st = [r for r in starts if abs(float(r["eps"]) - eps) < 1e-15]
        bs = [r for r in bases if abs(float(r["eps"]) - eps) < 1e-15]
        kg = sum(int(r["geom_good"]) for r in st); n = len(st)
        kms = sum(int(r["gate_geom_good"]) for r in bs); nb = len(bs)
        lo, hi = _wilson(kg, n); lom, him = _wilson(kms, nb)
        m, q1, q3 = _iqr([float(r["L2_final"]) for r in bs])
        ml, q1l, q3l = _iqr([float(r["L2_layer_final"]) for r in bs])
        agg.append(dict(eps=eps, n_starts=n, single_hit=kg / n, single_lo=lo, single_hi=hi,
                        n_bases=nb, k4_hit=kms / nb, k4_lo=lom, k4_hi=him,
                        L2_med=m, L2_q1=q1, L2_q3=q3, L2layer_med=ml, L2layer_q1=q1l, L2layer_q3=q3l))
        print(f"  eps={eps:g}: 单起点几何命中 {kg}/{n}={kg/n:.2f}[{lo:.2f},{hi:.2f}] | "
              f"K4选优 {kms}/{nb}={kms/nb:.2f} | 精修全域L2 {m:.2e}[{q1:.2e},{q3:.2e}] 层内 {ml:.2e}")
    mc.write_csv(os.path.join(OUT, "s2d_main_agg.csv"), agg)
    if len(agg) < 1: return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    fp = r"C:\Windows\Fonts\msyh.ttc"
    for lang in ["zh", "en"]:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp); plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name() if lang == "zh" else "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False
        L = lang == "zh"
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        es = [a["eps"] for a in agg]; xs = np.arange(len(es)); w = 0.35
        yerr_g = np.array([[a["L2_med"] - a["L2_q1"] for a in agg], [a["L2_q3"] - a["L2_med"] for a in agg]])
        yerr_l = np.array([[a["L2layer_med"] - a["L2layer_q1"] for a in agg], [a["L2layer_q3"] - a["L2layer_med"] for a in agg]])
        ax[0].bar(xs - w / 2, [a["L2_med"] for a in agg], w, yerr=yerr_g,
                  capsize=3, color="#4B86B4", label=("全域 L2" if L else "global L2"))
        ax[0].bar(xs + w / 2, [a["L2layer_med"] for a in agg], w, yerr=yerr_l,
                  capsize=3, color="#E8A24B", label=("层内 L2" if L else "in-layer L2"))
        ax[0].set_yscale("log"); ax[0].set_xticks(xs); ax[0].set_xticklabels([f"{e:g}" for e in es])
        ax[0].set_xlabel("ε" if L else "ε"); ax[0].set_ylabel(("精修后相对 L2（中位，n=11）" if L else "post-refine rel. L2 (median, n=11)"))
        ax[0].set_title(("三厚度精度" if L else "Accuracy across layer widths")); ax[0].grid(alpha=.3, axis="y"); ax[0].legend(fontsize=8)
        ax[1].bar(xs - w / 2, [a["single_hit"] for a in agg], w, color="#7FB3D5",
                  label=("单起点命中" if L else "single-start"))
        ax[1].bar(xs + w / 2, [a["k4_hit"] for a in agg], w, color="#2E8B57", label="K=4")
        ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{e:g}" for e in es]); ax[1].set_ylim(0, 1.05)
        ax[1].set_xlabel("ε" if L else "ε"); ax[1].set_ylabel(("几何好盆地命中率" if L else "geometric basin-hit rate"))
        ax[1].set_title(("Phase0 盆地命中" if L else "Phase0 basin hit")); ax[1].grid(alpha=.3, axis="y"); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"fig_s2d_main_{lang}.png"), dpi=150)
        fdir = _V7.parent / "theory" / "final" / "figs" / lang; fdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fdir / "fig_s2d_main.png", dpi=150)
        plt.close(fig)
    print("图与 s2d_main_agg.csv 已存。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default=""); ap.add_argument("--bases", default="")
    a = ap.parse_args()
    eps_list = [float(x) for x in a.eps.split(",") if x] or EPS_ALL
    bases = [int(x) for x in a.bases.split(",") if x] or BASES_ALL
    sp = os.path.join(OUT, "s2d_main_starts.csv"); bp = os.path.join(OUT, "s2d_main_bases.csv")
    done = set()
    if os.path.exists(bp):
        for r in csv.DictReader(open(bp, encoding="utf-8-sig")): done.add((float(r["eps"]), int(r["base_seed"])))
    fs = open(sp, "a", newline="", encoding="utf-8-sig"); fb = open(bp, "a", newline="", encoding="utf-8-sig")
    ws = csv.DictWriter(fs, fieldnames=F_START); wb = csv.DictWriter(fb, fieldnames=F_BASE)
    if not os.path.exists(sp) or os.path.getsize(sp) == 0: ws.writeheader()
    if not os.path.exists(bp) or os.path.getsize(bp) == 0: wb.writeheader()
    try:
        for eps in eps_list:
            for base in bases:
                if (float(eps), int(base)) in done: print("skip", eps, base); continue
                run_base(eps, base, K_ALL, P0, P1, ws, wb, fs, fb)
    finally:
        fs.close(); fb.close()
    aggregate_and_plot()


if __name__ == "__main__":
    main()
