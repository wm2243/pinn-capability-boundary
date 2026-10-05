# -*- coding: utf-8 -*-
"""
V2 实验 S2d-beta：好盆地内 β 的作用——机制 A 冻结探针 + β 档配对精修（内部转向点层）
================================================================================
前提：exp_V2_S2d_main_accuracy 已为 eps=1e-4 的 n=11 个 base 各选出 K=4 multistart 的
Gate 点（几何好盆地），权重快照存于其 _cache/gate_eps0.0001_base{b}.pt。本脚本在【同一
Gate 点】上做两件事（只改 β，其余完全对齐）：

  ③ 冻结探针（零训练）：θ、配点 x 固定，对 β∈{1,3,8} 分别取 w=1+(β-1)q(t)，用
     v8_matrix_common.grad_partition_energy 把加权梯度拆成层带 |x|<3√eps 与光滑区，
     报能量比 η=‖g_layer‖²/‖g_smooth‖² 与方向夹角 cos。机制 A 预测 β↑→η↑（梯度能量被
     再分配到内部层方向），这是直接测参数梯度（非输出代理）。
  ④ 配对精修：同一 Gate 快照分别用 β∈{1,3,8} 的 rational 权各 Phase1=6000 步（同种子、
     同冻结残差场），报全域/层内相对 L2；配对检验 β 是否在好盆地内提升层内精度（收益面）。

不做减权档（down_*）：basin_diag 已证该配置无坏盆地，减权无作用对象；本实验只问好盆地内
增权 β 的能量再分配与精修收益。配对优先 Wilcoxon 符号秩，n=11，多重比较 Holm。
结果 results/v7/V2/exp_V2_S2d_beta_effect/。

  set V8_SMOKE=1 && python exp_V2_S2d_beta_effect.py
  python exp_V2_S2d_beta_effect.py
================================================================================
"""
import os, sys, csv, copy, argparse
from pathlib import Path
import numpy as np
import torch
from scipy import stats

_HERE = Path(__file__).resolve(); _V7 = _HERE.parents[2]; _V1EXP = _V7 / "experiments" / "v1"
for _p in (str(_V7), str(_V1EXP), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
import exp_V2_S2d_basin_diag as diag
import exp_V2_S2d_main_accuracy as mainmod
from physics.turningpoint_pde import TurningPointPDE1D
from models.pinn_FourierFeatures_model import HardBCPINN
from losses.rational_weighting_loss1 import RationalWeightingLoss

OUT = str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S2d_beta_effect")
CACHE = os.path.join(str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S2d_main_accuracy"), "_cache")
os.makedirs(OUT, exist_ok=True)
DEV = mc.DEVICE
EPS = 1e-4
BETAS = [1.0, 3.0, 8.0]
BASES_ALL = list(mc.SEEDS_EXT)
P1 = 6000
N_PROBE = 10000
if mc.SMOKE:
    BASES_ALL = [0]; BETAS = [1.0, 8.0]; P1 = 60; N_PROBE = 2000

F_PROBE = ["eps", "base_seed", "beta", "g_band", "g_smooth", "eta", "cos"]
F_REF = ["eps", "base_seed", "beta", "L2", "L2_layer", "slope_ratio", "jump05"]


def ensure_gate(eps, base):
    """加载主结果 Gate 快照；缺失则现场用 K=4 Phase0 选优并写缓存（fallback）。"""
    p = os.path.join(CACHE, f"gate_eps{eps:g}_base{int(base)}.pt")
    if os.path.exists(p):
        d = torch.load(p, map_location=DEV, weights_only=False)
        return dict(seed=int(d["chosen_seed"]), state=d["state"], cfg=d["cfg"], m=d["gate_m"])
    K = 2 if mc.SMOKE else mainmod.K_ALL
    p0 = 60 if mc.SMOKE else mainmod.P0
    x_eval, u_eval = diag.test_arrays(eps)
    x_td, u_td = diag.test_arrays_n(eps, mainmod.N_TRAIN_TEST)
    cands = [mainmod._phase0_candidate(eps, int(base) + ki, p0, x_eval, u_eval, x_td, u_td)
             for ki in range(K)]
    kk = int(np.argmin([c["m"]["L2"] for c in cands])); ch = cands[kk]
    os.makedirs(CACHE, exist_ok=True)
    torch.save(dict(eps=float(eps), base=int(base), chosen_seed=int(ch["seed"]),
                    state=ch["state"], cfg=ch["cfg"], gate_m=ch["m"],
                    gate_geom_good=int(diag.geom_good(ch["m"]))), p)
    return ch


def frozen_probe(eps, cand):
    """③ 零训练机制 A 探针：同 θ、同配点，仅 β 变。"""
    cfg = cand["cfg"]
    model = HardBCPINN(cfg).to(DEV); model.load_state_dict(cand["state"]); model.eval()
    pde = TurningPointPDE1D(cfg)
    xp = torch.tensor(np.linspace(-1, 1, N_PROBE).reshape(-1, 1), dtype=torch.float32, device=DEV)
    rows = []
    for b in BETAS:
        cb = copy.deepcopy(cfg)
        cb.update(loss_beta=float(b), beta_init=float(b), weight_mode="adaptive", beta_schedule="const")
        crit = RationalWeightingLoss(cb, DEV)
        o = mc.grad_partition_energy(model, pde, crit, xp, float(np.sqrt(eps)), band_c=3.0)
        rows.append(dict(eps=float(eps), base_seed=int(cand["seed"]), beta=float(b),
                         g_band=o["g_band"], g_smooth=o["g_smooth"], eta=o["eta"], cos=o["cos"]))
        del crit
    del model
    if DEV.type == "cuda": torch.cuda.empty_cache()
    return rows


def _iqr(v):
    v = np.asarray(v, float); return np.median(v), np.quantile(v, .25), np.quantile(v, .75)


def _holm(pvals):
    m = len(pvals); order = np.argsort(pvals); adj = {}
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx]); adj[idx] = min(1.0, run)
    return [adj[i] for i in range(m)]


def aggregate_and_plot():
    pp = os.path.join(OUT, "s2d_beta_probe.csv"); rp = os.path.join(OUT, "s2d_beta_refine.csv")
    if not (os.path.exists(pp) and os.path.exists(rp)): return
    probe = list(csv.DictReader(open(pp, encoding="utf-8-sig")))
    ref = list(csv.DictReader(open(rp, encoding="utf-8-sig")))
    print("\n===== S2d β 机制聚合 =====")
    # ③ 探针
    by_b = {b: [float(r["eta"]) for r in probe if abs(float(r["beta"]) - b) < 1e-9] for b in BETAS}
    print("  ③ 冻结梯度能量比 η（中位[IQR]）：")
    for b in BETAS:
        if by_b[b]:
            m, q1, q3 = _iqr(by_b[b]); print(f"     β={b:g}: {m:.3e} [{q1:.3e},{q3:.3e}]")
    seeds = sorted({int(r["base_seed"]) for r in probe})
    e1 = [float(r["eta"]) for r in probe if abs(float(r["beta"]) - 1) < 1e-9]
    e8 = [float(r["eta"]) for r in probe if abs(float(r["beta"]) - 8) < 1e-9]
    if len(e1) == len(e8) and len(e1) >= 5:
        d = np.array(e8) - np.array(e1)
        w = stats.wilcoxon(e8, e1, alternative="greater", zero_method="wilcox")
        print(f"     η(8)>η(1)：{int((d>0).sum())}/{len(d)} 种子，Wilcoxon 单侧 p={w.pvalue:.4f}")
    cosmed = np.median([float(r["cos"]) for r in probe if abs(float(r["beta"]) - 1) < 1e-9]) if e1 else float("nan")
    print(f"     β=1 层带/光滑梯度夹角 cos 中位={cosmed:.3f}")
    # ④ 配对精修
    print("  ④ 配对精修层内 L2（中位[IQR]）：")
    rl = {b: [float(r["L2_layer"]) for r in ref if abs(float(r["beta"]) - b) < 1e-9] for b in BETAS}
    for b in BETAS:
        if rl[b]:
            m, q1, q3 = _iqr(rl[b]); print(f"     β={b:g}: {m:.3e} [{q1:.3e},{q3:.3e}]")
    rseeds = sorted({int(r["base_seed"]) for r in ref})
    base1 = {int(r["base_seed"]): float(r["L2_layer"]) for r in ref if abs(float(r["beta"]) - 1) < 1e-9}
    cmp_rows = []
    pvals = []
    for b in [3.0, 8.0]:
        db = {int(r["base_seed"]): float(r["L2_layer"]) for r in ref if abs(float(r["beta"]) - b) < 1e-9}
        common = sorted(set(base1) & set(db))
        if len(common) >= 5:
            x = np.array([base1[s] for s in common]); y = np.array([db[s] for s in common])
            diff = y - x  # 负=β 档更好
            w = stats.wilcoxon(y, x, alternative="two-sided", zero_method="wilcox"); pvals.append(w.pvalue)
            cmp_rows.append(dict(cmp=f"β{b:g} vs β1", n=len(common), n_better=int((diff < 0).sum()),
                                 med_diff=float(np.median(diff)), p_raw=float(w.pvalue)))
    if pvals:
        adj = _holm(pvals)
        for c, pa in zip(cmp_rows, adj): c["p_holm"] = float(pa); print(f"     {c['cmp']}: 更优 {c['n_better']}/{c['n']}，"
                                                                          f"中位差 {c['med_diff']:.3e}，Holm p={pa:.4f}")
    mc.write_csv(os.path.join(OUT, "s2d_beta_refine_cmp.csv"), cmp_rows)
    # ---- 双语图 ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    fp = r"C:\Windows\Fonts\msyh.ttc"
    for lang in ["zh", "en"]:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name() if lang == "zh" else "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False; L = lang == "zh"
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
        # ③ η-β
        for s in seeds:
            rr = sorted((r for r in probe if int(r["base_seed"]) == s), key=lambda r: float(r["beta"]))
            ax[0].plot([float(r["beta"]) for r in rr], [float(r["eta"]) for r in rr],
                       "-", color="#9A9A9A", alpha=.35, lw=.8)
        meds = [np.median(by_b[b]) for b in BETAS if by_b[b]]
        q1s = [np.quantile(by_b[b], .25) for b in BETAS if by_b[b]]
        q3s = [np.quantile(by_b[b], .75) for b in BETAS if by_b[b]]
        bx = [b for b in BETAS if by_b[b]]
        ax[0].errorbar(bx, meds, yerr=[np.array(meds) - np.array(q1s), np.array(q3s) - np.array(meds)],
                       fmt="o-", color="#E8A24B", capsize=4, lw=2, ms=7)
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel("β"); ax[0].set_ylabel(("层带/光滑梯度能量比 η" if L else "energy ratio η (layer/smooth)"))
        ax[0].set_title(("③ 冻结探针：β↑→η↑（机制A）" if L else "③ Frozen probe: β↑→η↑ (Mech. A)"))
        ax[0].grid(alpha=.3, which="both")
        # ④ 层内 L2 配对
        for s in rseeds:
            rr = sorted((r for r in ref if int(r["base_seed"]) == s), key=lambda r: float(r["beta"]))
            ax[1].plot([float(r["beta"]) for r in rr], [float(r["L2_layer"]) for r in rr],
                       "-", color="#9A9A9A", alpha=.35, lw=.8)
        for b, c in zip(BETAS, ["#9A9A9A", "#7FB3D5", "#E8A24B"]):
            if rl[b]:
                m, q1, q3 = _iqr(rl[b])
                ax[1].errorbar(b, m, yerr=[[m - q1], [q3 - m]], fmt="o", color=c, capsize=4, ms=8)
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("β"); ax[1].set_ylabel(("精修后层内相对 L2" if L else "post-refine in-layer rel. L2"))
        ax[1].set_title(("④ 好盆地内 β 档配对精修" if L else "④ Paired β refinement within basin"))
        ax[1].grid(alpha=.3, which="both")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_s2d_beta_{lang}.png"), dpi=150)
        fdir = _V7.parent / "theory" / "final" / "figs" / lang; fdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fdir / "fig_s2d_beta.png", dpi=150); plt.close(fig)
    print("图已存。")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bases", default=""); a = ap.parse_args()
    bases = [int(x) for x in a.bases.split(",") if x] or BASES_ALL
    pp = os.path.join(OUT, "s2d_beta_probe.csv"); rp = os.path.join(OUT, "s2d_beta_refine.csv")
    probe_done = set(); ref_done = set()
    if os.path.exists(pp):
        for r in csv.DictReader(open(pp, encoding="utf-8-sig")): probe_done.add((int(r["base_seed"]), float(r["beta"])))
    if os.path.exists(rp):
        for r in csv.DictReader(open(rp, encoding="utf-8-sig")): ref_done.add((int(r["base_seed"]), float(r["beta"])))
    fp = open(pp, "a", newline="", encoding="utf-8-sig"); fr = open(rp, "a", newline="", encoding="utf-8-sig")
    wp = csv.DictWriter(fp, fieldnames=F_PROBE); wr = csv.DictWriter(fr, fieldnames=F_REF)
    if not os.path.exists(pp) or os.path.getsize(pp) == 0: wp.writeheader()
    if not os.path.exists(rp) or os.path.getsize(rp) == 0: wr.writeheader()
    try:
        for base in bases:
            cand = ensure_gate(EPS, base)
            if not all((int(cand["seed"]), float(b)) in probe_done for b in BETAS):
                for row in frozen_probe(EPS, cand):
                    if (int(cand["seed"]), float(row["beta"])) not in probe_done:
                        wp.writerow(row); probe_done.add((int(cand["seed"]), float(row["beta"])))
                fp.flush()
            todo_b = [b for b in BETAS if (int(cand["seed"]), float(b)) not in ref_done]
            if todo_b:
                x_eval, u_eval = diag.test_arrays(EPS); x_td, u_td = diag.test_arrays_n(EPS, mainmod.N_TRAIN_TEST)
                for b in todo_b:
                    mf = mainmod._phase1_refine(EPS, cand, float(b), P1, x_eval, u_eval, x_td, u_td, "beta")
                    wr.writerow(dict(eps=float(EPS), base_seed=int(cand["seed"]), beta=float(b),
                                     L2=mf["L2"], L2_layer=mf["L2_layer"],
                                     slope_ratio=mf["slope_ratio"], jump05=mf["jump05"]))
                    fr.flush()
                    print(f"[refine] gate{int(cand['seed'])} β={b:g}: L2={mf['L2']:.3e} layer={mf['L2_layer']:.3e}")
    finally:
        fp.close(); fr.close()
    aggregate_and_plot()


if __name__ == "__main__":
    main()
