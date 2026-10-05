# -*- coding: utf-8 -*-
"""
V2 实验 S2：光滑椭圆（Poisson）退化对照 —— 验证“空间加权只被局域刚性对比度分层激活”
================================================================================
理论预期（全部可证伪，若不成立则反向修正框架）：
  D1 对比度均匀：光滑解残差对比度场 t=|r|/bg 空间近似均匀（CV 小、max/min 接近 1），
                 不存在 Burgers 式激波带/光滑带 S/O 强分离（C-Sep 不成立）。
  D2 加权退化：六族中取 uniform / rational(增权) / down_inv(减权)，β∈{1,3,10}，
                 终值 L2 对族/β 不敏感、E(β) 平坦、无 U 型、不存在 β*；β=1 时三族恒等（内部一致性）。
  D3 表示退化：光滑解是普通 PINN 好球区，普通 MLP（无 Fourier）或低 σ 即可，高 σ 无优势；
                 单阶段单起点即可，无需 multistart/表示课程。
  D4 梯度尺度：无局域梯度峰爆炸，各权族原生梯度峰同量级（对照 Burgers rational 达数百）。
本脚本先落 D1/D2/D3（终值 + 对比度 + 表示器），D4 梯度峰由独立探针脚本另做（与 Gstab 同口径）。

算例（MMS，域[-1,1]，齐次 Dirichlet，bc_type='dirichlet0' 精确硬边界）：
  single : u*=sin(π(x+1)/2)                       光滑单尺度
  multi  : u*=sin(π(x+1)/2)+0.5 sin(3π(x+1)/2)    含全局高频但无局域窄层（解耦“表示需求σ”与“空间加权W”）

单阶段训练（光滑、不做两阶段/盆地选择——这本身即退化证据）。增量断点续跑、规则种子、median/IQR+n。
运行：
  python exp_V2_poisson_degenerate.py
  V8_SMOKE=1 python exp_V2_poisson_degenerate.py
================================================================================
"""
import os, sys, csv, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager

_HERE = Path(__file__).resolve(); _V7 = _HERE.parents[2]; _V1EXP = _V7 / "experiments" / "v1"
for _p in (str(_V7), str(_V1EXP), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from physics.poisson_pde import PoissonPDE1D
from models.pinn_FourierFeatures_model import HardBCPINN
from losses.family_weighting_loss import FamilyWeightingLoss
from samplers.pinn_sampler import PINNSampler
from trainers.pinn_trainer import PINNTrainer
from utils.visualizer import PINNVisualizer
from utils.metrics import PINNMetrics

OUT = str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S2a_poisson_degenerate"); os.makedirs(OUT, exist_ok=True)
r2s.OUT = OUT
for _c in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(_c):
        font_manager.fontManager.addfont(_c); plt.rcParams["font.family"] = font_manager.FontProperties(fname=_c).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

# (feature, fourier_scale)；mlp 关闭随机傅里叶
FEATURES_ALL = [("mlp", 0.0), ("fourier", 1.0), ("fourier", 15.0)]
CASES_ALL = ["single", "multi"]
# 族 -> β 列表；β=1 三族恒等，作为内部一致性检查
FAM_BETA = {"uniform": [1], "rational": [1, 3, 10], "down_inv": [1, 3, 10]}
SEEDS_ALL = mc.SEEDS_EXT
EP_ALL = 3000  # 光滑退化用例预算：~200步即达1e-4量级，3000给 multi/fourier15 留余量；
# 奇摄动/边界层用例另立 S2b，预算单独定，不在此一刀切
if mc.SMOKE:
    FEATURES_ALL = [("mlp", 0.0), ("fourier", 15.0)]
    CASES_ALL = ["single"]; FAM_BETA = {"uniform": [1], "rational": [1, 10]}
    SEEDS_ALL = [0]; EP_ALL = 120

FIELDS = ["case", "feature", "sigma", "family", "beta", "seed", "epochs",
          "L2", "Linf", "TV", "contrast_cv", "contrast_q95q05"]


def build_poisson(case, feature, sigma, family, beta, seed, epochs):
    r2s.set_seed(seed)
    cfg = r2s.base_cfg(0.0, f"s2_{case}_{feature}{sigma:g}_{family}b{beta}_s{seed}")
    cfg.update(dict(
        pois_mode=case, bc_type="dirichlet0", bc_amp=0.0, nu=0.0, shock_band_c=0.0,
        feature=feature, fourier_scale=float(sigma),
        use_gate=False, weight_mode="adaptive", weight_family=family,
        beta_schedule="const", loss_beta=float(beta), beta_init=float(beta),
        adam_epochs=int(epochs), lbfgs_epochs=0, lr=1e-3,
        use_best_ckpt=True, best_metric="L2", n_pde=8192))
    pde = PoissonPDE1D(cfg)
    model = HardBCPINN(cfg).to(mc.DEVICE)
    cfg.setdefault("weight_bg_mode", "ema")
    crit = FamilyWeightingLoss(cfg, mc.DEVICE, family=family)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    samp = PINNSampler(cfg["domain_x"], mc.DEVICE, use_adaptive=False, buffer_size=cfg["n_pde"])
    vis = PINNVisualizer(save_dir=OUT)
    trainer = PINNTrainer(model, pde, crit, opt, samp, vis, mc.DEVICE, cfg, test_data=None)
    return cfg, pde, model, crit, trainer


def contrast_stats(pde, model, n=2048):
    """对比度场 t=|r|/mean|r| 的变异系数与 max/min（D1）。"""
    x = torch.linspace(-1, 1, n, device=mc.DEVICE).reshape(-1, 1)
    model.eval()
    with torch.enable_grad():
        r = pde.compute_residual(model, x).detach().abs().flatten()
    m = r.mean().clamp_min(1e-12)
    t = r / m
    cv = float(t.std() / (t.mean() + 1e-12))
    q = torch.quantile(r.flatten(), torch.tensor([0.05, 0.5, 0.95], device=r.device))
    q95q05 = float(q[2] / q[0].clamp_min(1e-10))   # 稳健对比度比（不被近零点打爆）
    return cv, q95q05


def run_unit(case, feature, sigma, family, beta, seed, epochs, xt, ut):
    cfg, pde, model, crit, trainer = build_poisson(case, feature, sigma, family, beta, seed, epochs)
    # u_true 必须为 1D(n,)：trainer._eval_test 对预测做了 flatten，若这里传 (n,1) 列向量，
    # numpy 会把差广播成 (n,n)，得到 ≈0.615√n≈27.9 的“伪 L2”、伪 Linf=1（Burgers 的 build_test
    # 本就返回 1D 故主线无此问题）。下方最终指标仍用列向量 ut 与 (n,1) pred 对齐，不受影响。
    _td = {"x": xt.to(mc.DEVICE), "u_true": ut.view(-1).cpu().numpy()}
    trainer.train(cfg, _td, epochs, 0, adaptive_freq=1000)
    model.eval()
    with torch.no_grad():
        pred = model(xt.to(mc.DEVICE)).detach().cpu().numpy()
    met = PINNMetrics()
    xn = xt.detach().cpu().numpy()
    full = met.compute_full(pred, ut, x=xn, nu=0.0, shock_band_c=0.0)  # 光滑椭圆：全域、无激波带
    l2 = float(full["L2_Error"]); linf = float(full["L_inf_Error"])
    tv = float(full.get("TV_Error", np.nan))
    cv, q95q05 = contrast_stats(pde, model)
    trainer.log_file.close()
    return dict(case=case, feature=feature, sigma=float(sigma), family=family, beta=float(beta),
                seed=int(seed), epochs=int(epochs), L2=l2, Linf=linf,
                TV=tv, contrast_cv=cv, contrast_q95q05=q95q05)


def _done(path):
    s = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            s.add((r["case"], r["feature"], float(r["sigma"]), r["family"], float(r["beta"]), int(r["seed"])))
    return s


def read_rows(path):
    if not os.path.exists(path): return []
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=""); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cases = [x for x in a.cases.split(",") if x] or CASES_ALL
    csvp = os.path.join(OUT, "poisson_runs.csv")
    done = _done(csvp); newfile = not os.path.exists(csvp); n_new = 0
    fout = open(csvp, "a", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if newfile: w.writeheader(); fout.flush()
    try:
        for case in cases:
            pde0 = PoissonPDE1D(dict(pois_mode=case))
            xn = np.linspace(-1, 1, 2048).reshape(-1, 1); un = pde0.exact_np(xn)
            xt = torch.tensor(xn, dtype=torch.float32); ut = torch.tensor(un, dtype=torch.float32)
            for feature, sigma in FEATURES_ALL:
                for family, betas in FAM_BETA.items():
                    for beta in betas:
                        for seed in SEEDS_ALL:
                            key = (case, feature, float(sigma), family, float(beta), int(seed))
                            if key in done: continue
                            row = run_unit(case, feature, sigma, family, beta, seed, EP_ALL, xt, ut)
                            w.writerow(row); fout.flush(); done.add(key); n_new += 1
                            print(f"[S2] {case} {feature}σ{sigma:g} {family}β{beta:g} s{seed}: "
                                  f"L2={row['L2']:.2e} CV={row['contrast_cv']:.2f} p95/p05={row['contrast_q95q05']:.1f}")
                            if a.limit and n_new >= a.limit: raise StopIteration
    except StopIteration:
        pass
    finally:
        fout.close()
    aggregate_and_plot(read_rows(csvp))
    print("\n[S2] 完成，产物在", OUT)


def _med_iqr(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(np.median(v)), float(np.quantile(v, .25)), float(np.quantile(v, .75)), len(v))


def aggregate_and_plot(rows):
    if not rows: return
    for r in rows:
        for k in ("sigma", "beta", "L2", "Linf", "TV", "contrast_cv", "contrast_q95q05"): r[k] = float(r[k])
        r["seed"] = int(r["seed"])
    agg = []
    keys = {}
    for r in rows:
        k = (r["case"], r["feature"], r["sigma"], r["family"], r["beta"]); keys.setdefault(k, []).append(r)
    for k, sub in keys.items():
        m, q1, q3, n = _med_iqr([x["L2"] for x in sub])
        cv = np.median([x["contrast_cv"] for x in sub]); qq = np.median([x["contrast_q95q05"] for x in sub])
        agg.append(dict(case=k[0], feature=k[1], sigma=k[2], family=k[3], beta=k[4],
                        n=n, L2_med=m, L2_q1=q1, L2_q3=q3, contrast_cv_med=cv, contrast_q95q05_med=qq))
    mc.write_csv(os.path.join(OUT, "poisson_agg.csv"), agg)
    # 图1：β 平坦性（每 case×feature 一张子图，三族 L2-β）
    for case in sorted({r["case"] for r in agg}):
        feats = sorted({(r["feature"], r["sigma"]) for r in agg if r["case"] == case}, key=lambda z: str(z))
        fig, axes = plt.subplots(1, len(feats), figsize=(5.2 * len(feats), 4.2), squeeze=False)
        fcol = {"uniform": "#9A9A9A", "rational": "#E8A24B", "down_inv": "#4B86B4"}
        for ax, (feat, sg) in zip(axes[0], feats):
            for fam in ["uniform", "rational", "down_inv"]:
                z = sorted((r for r in agg if r["case"] == case and r["feature"] == feat
                            and r["sigma"] == sg and r["family"] == fam), key=lambda r: r["beta"])
                if not z: continue
                ax.plot([r["beta"] for r in z], [r["L2_med"] for r in z], "o-",
                        color=fcol[fam], label=fam)
                ax.fill_between([r["beta"] for r in z], [r["L2_q1"] for r in z], [r["L2_q3"] for r in z],
                                color=fcol[fam], alpha=.12)
            ax.set_xscale("symlog", linthresh=1.05); ax.set_yscale("log")
            ax.set_xlabel("β（1=无加权基线）"); ax.set_ylabel("终值 L2（中位/IQR）")
            ax.set_title(f"{case} | {feat} σ={sg:g}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.suptitle("D2 加权退化：光滑椭圆上 L2 对族/β 应平坦（无 β*）")
        fig.tight_layout(); p = os.path.join(OUT, f"fig_{case}_beta_flat.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)
    # 图2：表示器对比（uniform β=1 下各 feature 的 L2）
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    base = [r for r in agg if r["family"] == "uniform"]
    labels = sorted({f"{r['case']}\n{r['feature']}σ{r['sigma']:g}" for r in base})
    xpos = range(len(labels)); vals = []
    ld = {f"{r['case']}\n{r['feature']}σ{r['sigma']:g}": r["L2_med"] for r in base}
    vals = [ld.get(l, np.nan) for l in labels]
    ax.bar(list(xpos), vals, color="#7FB3D5"); ax.set_xticks(list(xpos)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("uniform 终值 L2（中位）")
    ax.set_title("D3 表示退化：普通 MLP/低σ 即可，高σ 无优势"); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); p = os.path.join(OUT, "fig_feature_compare.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


if __name__ == "__main__":
    main()
