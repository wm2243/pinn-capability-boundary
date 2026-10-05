# -*- coding: utf-8 -*-
"""
实验 E3（V9）：权函数族 × 增权/减权 —— “双分离”与可推广判据的受控检验
--------------------------------------------------------------------------------
(a) 冻结谱（不训练、便宜）：固定同一冻结点的残差 Gram K_r 与同一配点，只换权重族，
    用 spectral_alignment(w_override=...) 单变量检验：增权族 improve=κ_new/κ_orig>1（反
    预处理），减权族预期 improve<1（顺预处理）、ŵ-λ 相关 a 符号反转。
(b) 训练对照：同一好盆地，换族训练到终点，比较 L2/Linf/激波带误差与稳定性常数 Cw。
核心命题（双分离）：减权可能改善条件数却抹平激波（精度更差），增权恶化条件数却提升激波
精度 —— 条件数既不解释优化增益、也不预测最终精度；同时减权 w_min=1/β 使稳定性常数放大
sqrt(β)（Cw 列直接量化）。

产出 results/v6/exp_E3_family_sa/：
  e3_spectral.csv / e3_spectral_a.csv / e3_family_runs.csv
  fig_E3_sa_family.png / fig_E3_spec_a.png / fig_E3_double_separation.png / fig_E3_train_family.png
SMOKE：V8_SMOKE=1
"""
import os, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve(); _PKG = _HERE.parent.parent
for _p in (str(_PKG), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from utils import residual_spectral as rs
from losses.family_weighting_loss import family_weight, FAMILIES, UP_FAMILIES, DOWN_FAMILIES

OUT = str(mc.RES_ROOT / "exp_E3_family_sa"); os.makedirs(OUT, exist_ok=True)
r2s.OUT = OUT
NU = 0.005
FAM_SPEC = ["rational", "linear", "band", "down_lin", "down_inv"]          # 冻结谱五族
FAM_TRAIN = ["rational", "down_inv", "down_lin", "linear", "band", "uniform"]
ORDER = FAM_TRAIN
UP_COLOR, DOWN_COLOR, UNI_COLOR, SHUF_COLOR = "#E8A24B", "#4B86B4", "#9A9A9A", "#8E5EA1"


def fam_color(f, shuffled=False):
    if shuffled: return SHUF_COLOR
    if f in UP_FAMILIES: return UP_COLOR
    if f in DOWN_FAMILIES: return DOWN_COLOR
    return UNI_COLOR


def frozen_from_basin(basin, nu, beta=1.0):
    """gate 点：装载盆地状态并加载其冻结对比度场（rational crit 仅作谱测量载体）。"""
    cfg = r2s.base_cfg(nu, "E3spec"); cfg.update(weight_normalize="raw")
    model, pde, crit, opt, samp, vis, cfg = r2s.build_all(cfg, 5e-4, float(beta))
    model.load_state_dict(basin["state"]); model.set_sigma(basin["sigma_end"])
    crit.load_frozen_field(basin["t_field"], beta=float(beta))
    return model, pde, crit


def conv_frozen(basin, seed, xt, ut, conv_ep):
    """近解点：β=1 从 gate 续训，再用【当前瞬时残差】现场冻结（与 C2 同口径）。"""
    _m, _d, model_c, crit_c, tr = mc.train_phase1_family(
        basin, NU, 1.0, seed, xt, ut, family="rational", epochs=conv_ep, tag="E3conv")
    pde_c = tr.pde_engine
    crit_c.freeze_residual_field(model_c, pde_c)
    return model_c, pde_c, crit_c


# ============================ (a) 冻结谱 ============================
def collect_spectral():
    seeds = [mc.SEEDS_NEW[ 0 ]] if mc.SMOKE else mc.SEEDS_NEW[ :3 ]
    ns = [48] if mc.SMOKE else [128, 256]
    contrasts = [15.0] if mc.SMOKE else [5.0, 15.0, 40.0]
    points = ["gate"] if mc.SMOKE else ["gate", "conv"]
    conv_ep = 120 if mc.SMOKE else 6000
    rows, arows = [], []
    xt, ut = r2s.build_test(NU)
    for seed in seeds:
        basin = mc.enter_basin_cached(NU, seed, xt, ut, sigma_hi=r2s.SIGMA_HI_CFG, k=4, phase0_epochs=2000)
        holders = {"gate": frozen_from_basin(basin, NU)}
        if "conv" in points: holders["conv"] = conv_frozen(basin, seed, xt, ut, conv_ep)
        for point, (model, pde, crit) in holders.items():
            for n in ns:
                x_col = torch.linspace(-1.0, 1.0, n, device=mc.DEVICE).view(-1, 1)
                x_flat = x_col.reshape(-1)
                t = crit._interp_field(crit._frozen_t_field, x_flat).reshape(-1)
                for c in contrasts:
                    for fam in FAM_SPEC + ["rational_shuf"]:
                        base_fam = "rational" if fam == "rational_shuf" else fam
                        w = crit._normalize(family_weight(base_fam, t, c))
                        if fam == "rational_shuf":
                            g = torch.Generator(device=w.device); g.manual_seed(1000 + n)
                            w = w[ torch.randperm(w.numel(), generator=g, device=w.device) ]
                        sc, ar = rs.spectral_alignment(model, pde, crit, x_col, beta=c, w_override=w)
                        aa = rs.estimate_spectral_a(ar["lam"], ar["what"])
                        rows.append(dict(seed=seed, point=point, n=n, family=fam, contrast=c,
                                         E_off=sc["E_off"], kappa_orig=sc["kappa_orig"],
                                         kappa_new=sc["kappa_new"], improve_ratio=sc["improve_ratio"],
                                         beta_sat=sc["beta_sat"], eps_star=sc["eps_star"],
                                         SA_holds=int(bool(sc["SA_holds"])),
                                         w_min=sc["w_min"], w_max=sc["w_max"]))
                        arows.append(dict(seed=seed, point=point, n=n, family=fam, contrast=c,
                                          spec_a=aa["a"], spec_r2=aa["r2"]))
                        print(f"[E3a] s{seed} {point} n{n} {fam} c{c}: "
                              f"Eoff={sc['E_off']:.3f} improve={sc['improve_ratio']:.3f} a={aa['a']:+.3f}")
    mc.write_csv(os.path.join(OUT, "e3_spectral.csv"), rows)
    mc.write_csv(os.path.join(OUT, "e3_spectral_a.csv"), arows)
    return rows, arows


# ============================ (b) 训练对照 ============================
def collect_train():
    seeds = [mc.SEEDS_NEW[ 0 ]] if mc.SMOKE else mc.SEEDS_NEW
    contrasts = [15.0] if mc.SMOKE else [15.0, 40.0]
    ep = 240 if mc.SMOKE else 8000
    xt, ut = r2s.build_test(NU)
    rows = []
    for seed in seeds:
        basin = mc.enter_basin_cached(NU, seed, xt, ut, sigma_hi=r2s.SIGMA_HI_CFG, k=4, phase0_epochs=2000)
        jobs = []
        for fam in FAM_TRAIN:
            cs = [contrasts[ 0 ]] if fam == "uniform" else contrasts
            for c in cs: jobs.append((fam, False, c))
        if not mc.SMOKE:
            for c in contrasts: jobs.append(("rational", True, c))   # 训练侧置换对照
        for fam, shuf, c in jobs:
            m, dyn, _mm, _cc, _tt = mc.train_phase1_family(
                basin, NU, c, seed, xt, ut, family=fam, shuffled=shuf, epochs=ep, tag="E3b")
            mech = mc.mechanism_row(dyn)
            rows.append(dict(seed=seed, family=fam, shuffled=int(shuf), contrast=c,
                             L2=m.get("L2_Error", np.nan), Linf=m.get("L_inf_Error", np.nan),
                             l2_band=m.get("l2_band", np.nan), l2_smooth=m.get("l2_smooth", np.nan),
                             T01=mc.first_hit_step(dyn, .01), best_epoch=m.get("best_epoch", -1),
                             wmin_th=(1.0 / c if fam in DOWN_FAMILIES else 1.0),
                             Cw=(c ** 0.5 if fam in DOWN_FAMILIES else 1.0),
                             focus_t=mech["focus_t"], gap=mc.final_minus_best(dyn)))
            print(f"[E3b] s{seed} {fam}{'(shuf)' if shuf else ''} c{c}: "
                  f"L2={rows[-1]['L2']:.3e} band={rows[-1]['l2_band']:.3e} Cw={rows[-1]['Cw']:.2f}")
    mc.write_csv(os.path.join(OUT, "e3_family_runs.csv"), rows)
    return rows


# ============================== 图 ==============================
def _med(vals):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def plot_sa_family(spec):
    fams = FAM_SPEC + ["rational_shuf"]
    sub = [r for r in spec if r["contrast"] == 15.0 and r["n"] == max({z["n"] for z in spec})]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    eoff, impr, cols = [], [], []
    for fam in fams:
        eoff.append(_med([r["E_off"] for r in sub if r["family"] == fam]))
        impr.append(_med([r["improve_ratio"] for r in sub if r["family"] == fam]))
        cols.append(fam_color("rational" if fam == "rational_shuf" else fam, fam == "rational_shuf"))
    xs = np.arange(len(fams))
    ax[ 0 ].bar(xs, eoff, color=cols)
    eps = _med([r["eps_star"] for r in sub])
    if np.isfinite(eps): ax[ 0 ].axhline(eps, color="k", ls="--", lw=1, label=f"ε*≈{eps:.2e}")
    ax[ 0 ].set_xticks(xs, fams, rotation=30, ha="right", fontsize=8); ax[ 0 ].set_title("非对角能量 E_off")
    ax[ 0 ].legend(fontsize=8); ax[ 0 ].grid(alpha=.3, axis="y")
    ax[ 1 ].bar(xs, impr, color=cols); ax[ 1 ].axhline(1.0, color="k", ls="--", lw=1)
    ax[ 1 ].set_xticks(xs, fams, rotation=30, ha="right", fontsize=8)
    ax[ 1 ].set_title("κ_new/κ_orig（>1 反预处理，<1 顺预处理）"); ax[ 1 ].grid(alpha=.3, axis="y")
    fig.suptitle("(a) 冻结谱：增权族恶化条件数，减权族改善，置换对照失去差异")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E3_sa_family.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def plot_spec_a(arows):
    fams = FAM_SPEC
    sub = [r for r in arows if r["contrast"] == 15.0 and r["n"] == max({z["n"] for z in arows})]
    aa = [_med([r["spec_a"] for r in sub if r["family"] == f]) for f in fams]
    fig, a = plt.subplots(figsize=(7.4, 4.2))
    a.bar(np.arange(len(fams)), aa, color=[fam_color(f) for f in fams]); a.axhline(0, color="k", lw=1)
    a.set_xticks(np.arange(len(fams)), fams); a.set_ylabel("ŵ-λ 谱形指数 a")
    a.set_title("减权使 ŵ-λ 相关符号反转（a：增权≈0/负，减权预期正）"); a.grid(alpha=.3, axis="y")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E3_spec_a.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def plot_double_separation(spec, train):
    fams = [f for f in ORDER]
    ns = max({z["n"] for z in spec})
    impr, band, cols = [], [], []
    for fam in fams:
        impr.append(_med([r["improve_ratio"] for r in spec
                          if r["family"] == fam and r["contrast"] == 15.0 and r["n"] == ns]))
        band.append(_med([r["l2_band"] for r in train
                          if r["family"] == fam and r["contrast"] == 15.0 and r["shuffled"] == 0]))
        cols.append(fam_color(fam))
    xs = np.arange(len(fams)); w = .38
    fig, a1 = plt.subplots(figsize=(9.4, 4.8))
    a1.bar(xs - w / 2, impr, w, color="#B0BEC5", label="κ_new/κ_orig（左，越小条件数越好）")
    a1.axhline(1, color="#607D8B", ls="--", lw=1)
    a1.set_ylabel("条件数比 κ_new/κ_orig"); a1.set_xticks(xs, fams)
    a2 = a1.twinx()
    a2.bar(xs + w / 2, band, w, color=cols, label="激波带误差 l2_band（右，越小精度越好）")
    a2.set_ylabel("l2_band（对数）")
    if any(np.isfinite(v) and v > 0 for v in band): a2.set_yscale("log")
    a1.set_title("双分离：减权改善条件数却抹平激波，增权反之（条件数不预测精度）")
    l1, lb1 = a1.get_legend_handles_labels(); l2, lb2 = a2.get_legend_handles_labels()
    a1.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper center")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E3_double_separation.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def plot_train_family(train):
    fams = ORDER
    cons = sorted({r["contrast"] for r in train if r["family"] != "uniform"})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    vals = ["L2", "Linf", "l2_band"]
    xs = np.arange(len(fams)); w = .8 / max(len(cons), 1)
    cmap = {15.0: "#2E6E8E", 40.0: "#E8A24B"}
    for k, val in enumerate(vals):
        any_pos = False
        for j, c in enumerate(cons):
            yy = [_med([r[ val ] for r in train if r["family"] == f and r["contrast"] == c
                        and r["shuffled"] == 0]) for f in fams]
            any_pos = any_pos or any(np.isfinite(q) and q > 0 for q in yy)
            ax[ k ].bar(xs + (j - (len(cons) - 1) / 2) * w, yy, w,
                        color=cmap.get(c, "#888"), label=f"contrast={c:g}")
        ax[ k ].set_xticks(xs, fams, rotation=30, ha="right", fontsize=8)
        if any_pos: ax[ k ].set_yscale("log")
        ax[ k ].set_title(val); ax[ k ].grid(alpha=.3, axis="y")
    ax[ 0 ].legend(fontsize=8)
    fig.suptitle("(b) 训练终点：增权族激波精度占优，减权族付出精度代价（Cw=√β 已在 CSV）")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E3_train_family.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def main():
    spec, arows = collect_spectral()
    train = collect_train()
    plot_sa_family(spec); plot_spec_a(arows)
    plot_double_separation(spec, train); plot_train_family(train)
    print("\n[E3] 完成。产物在", OUT)


if __name__ == "__main__":
    main()
