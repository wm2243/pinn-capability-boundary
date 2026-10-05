# -*- coding: utf-8 -*-
"""
V2 实验 S2d-basin-diag：内部转向点层【逐起点盆地命中率】诊断（不做 truth 选优、不跑 Phase1）
================================================================================
目的（回答“为什么 S2d 没观测到坏盆地”）：
  原 exp_V2_S2d_internal_layer.py 的 enter_basin 对 K=4 起点用真解 L2 只保留最好的一个，
  落选 K-1 起点不落盘，csv 无 basin_hit —— 那是“选优后能力上界”，会系统性藏起坏起点。
  本脚本对每个【独立起点】各自跑同一份 Phase0（与 Burgers/S2d 完全相同的 2000 步协议），
  不选优、不精修，逐个记录误差与“层是否被捕捉”的几何量，再：
    (1) 给各几何量的分布与一维自然间隙（学 G2，阈值取间隙、不拍 .06）；
    (2) 从同一份数据现算【单起点命中率】与【K=4 multistart 选优命中率】；
    (3) 给命中率的 Wilson 区间 / 0 事件 Clopper–Pearson 上界（不写“无坏盆地”全称句）。

  几何量为何不能只看 u(0)：未 sharpen 的塌缩态 ≈ 线性提升 A=½(1+x)，其 u(0) 恰为 0.5，
  故同时记录中心斜率比、跨层跳变、平台误差，才能把“线性斜线”与“真内部层”分开。

方程：-eps u'' - x u' = 0, u(-1)=0,u(1)=1；u*=½[1+erf(x/√(2eps))/erf(1/√(2eps))]，层在 x=0。
默认 eps=1e-4（等效 Burgers ν_eq≈5.8e-3），base seeds=SEEDS_EXT 11 颗 × K=4 = 44 独立起点。
rep：fourier=对齐带宽 σ=15（默认，与 S2d baseline 同）；mlp=关闭随机傅里叶（归因对照）。

不修改公共代码；结果存 results/v7/V2/exp_V2_S2d_basin_diag/，按 (rep,eps,start_seed) 断点续跑。

运行：
  set V8_SMOKE=1 && python exp_V2_S2d_basin_diag.py            # 冒烟（2 起点、60 步）
  python exp_V2_S2d_basin_diag.py --rep fourier                # 正式 44 起点（对齐 Fourier）
  python exp_V2_S2d_basin_diag.py --rep mlp                    # 归因对照（普通 MLP，另 44 起点）
================================================================================
"""
import os, sys, csv, argparse, copy
from pathlib import Path
import numpy as np
import torch
from scipy.special import erf

_HERE = Path(__file__).resolve(); _V7 = _HERE.parents[2]; _V1EXP = _V7 / "experiments" / "v1"
for _p in (str(_V7), str(_V1EXP), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from physics.turningpoint_pde import TurningPointPDE1D
from models.pinn_FourierFeatures_model import HardBCPINN
from losses.rational_weighting_loss1 import RationalWeightingLoss
from samplers.pinn_sampler import PINNSampler
from trainers.pinn_trainer import PINNTrainer
from utils.visualizer import PINNVisualizer

OUT = str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S2d_basin_diag")
os.makedirs(OUT, exist_ok=True)
DEV = mc.DEVICE
N_TEST = 2048
BAND_C = 3.0
FIELDS = ["rep", "eps", "base_seed", "k_idx", "start_seed", "p0", "sigma", "gate_epoch",
          "L2", "L2_layer", "u_center", "slope_ratio", "jump05", "plat_err"]

# 与 S2d 相同的 bc_amp monkeypatch（lift_linear 下虽不参与 A，保持口径一致）
_orig_base_cfg = r2s.base_cfg
def _bc_base_cfg(nu, seed_tag):
    c = _orig_base_cfg(nu, seed_tag)
    c["bc_amp"] = float(np.tanh(1.0 / (2.0 * max(float(nu), 1e-8))))
    return c
r2s.base_cfg = _bc_base_cfg


def tp_cfg(eps, tag, rep, sigma):
    nu_eff = float(np.sqrt(eps))
    c = r2s.base_cfg(nu_eff, tag)
    c.update(dict(shock_band_c=BAND_C, bc_type="lift_linear", bc_left=0.0, bc_right=1.0,
                  tp_eps=float(eps),
                  feature=("mlp" if rep == "mlp" else "fourier"),
                  fourier_scale=(0.0 if rep == "mlp" else float(sigma))))
    return c, nu_eff


def test_arrays_n(eps, n):
    pde = TurningPointPDE1D(dict(tp_eps=eps, bc_left=0.0, bc_right=1.0))
    x = np.linspace(-1, 1, n, dtype=np.float64)
    u = pde.exact_np(x.reshape(-1, 1)).astype(np.float64).flatten()
    return x, u


def test_arrays(eps):
    # 最终几何评估网格随层宽 √eps 自适应：保证 dx ≤ √eps/10（层内≥20 点），下限 N_TEST=2048；
    # ε=1e-4→2048，ε=1e-6→32768（备查更薄层时中心斜率差分才不跨层）。
    n_need = int(np.ceil(2.0 / (max(float(np.sqrt(eps)), 1e-12) / 10.0) + 1))
    n = max(N_TEST, int(2 ** np.ceil(np.log2(max(n_need, 2)))))
    return test_arrays_n(eps, n)


def true_center_slope(eps):
    """u*'(0) = 0.5 * sqrt(2/(pi eps)) / erf(1/sqrt(2eps))。"""
    return 0.5 * np.sqrt(2.0 / (np.pi * eps)) / float(erf(1.0 / np.sqrt(2.0 * eps)))


def eval_start(model, x, u, eps):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=DEV)
        p = model(xt).detach().cpu().numpy().astype(np.float64).flatten()
    model.train()
    nu_eff = float(np.sqrt(eps))
    band = np.abs(x) < BAND_C * nu_eff
    plat = (np.abs(x) > 0.3) & (np.abs(x) < 0.8)
    target_plat = (x[plat] > 0).astype(np.float64)
    # 中心斜率：h≈2 个测试网格、且取层宽 √eps/5，中心差分后比解析真解斜率
    dx = 2.0 / (len(x) - 1)
    h = max(2.0 * dx, 0.2 * nu_eff)
    def vv(x0):
        with torch.no_grad():
            return float(model(torch.tensor([[x0]], dtype=torch.float32, device=DEV)).item())
    slope_pred = (vv(h) - vv(-h)) / (2.0 * h)
    s_true = true_center_slope(eps)
    u0 = float(np.interp(0.0, x, p))
    jump = float(np.interp(0.5, x, p) - np.interp(-0.5, x, p))
    d = p - u
    return dict(
        L2=float(np.linalg.norm(d) / max(np.linalg.norm(u), 1e-12)),
        L2_layer=float(np.linalg.norm(d[band]) / max(np.linalg.norm(u[band]), 1e-12)),
        u_center=u0,
        slope_ratio=float(slope_pred / max(s_true, 1e-12)),
        jump05=jump,
        plat_err=float(np.sqrt(np.mean((p[plat] - target_plat) ** 2))),
    ), p


def run_one(eps, base_seed, k_idx, rep, sigma, p0):
    start_seed = int(base_seed) + int(k_idx)
    r2s.set_seed(start_seed)
    cfg, nu_eff = tp_cfg(eps, f"diag_tp{eps:g}_{rep}_{start_seed}", rep, sigma)
    cfg.update(dict(lr=r2s.LR_PHASE0, use_gate=True, gate_patience=3, gate_lr_gamma=0.3,
                    sigma_lo=r2s.SIGMA_LO, sigma_hi=float(sigma), sigma_anneal_T=r2s.SIGMA_T,
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
    # 训练/Gate 用固定标准测试网格（跨 ε 与 Burgers 同口径 n=2048）；最终几何评估用自适应密网格
    td = {"x": torch.tensor(x_td_global.reshape(-1, 1), dtype=torch.float32, device=DEV),
          "u_true": u_td_global.flatten()}
    tr = PINNTrainer(model, pde, crit, opt, samp, vis, DEV, cfg2, test_data=td)
    tr.train(cfg2, td, int(p0), 0, adaptive_freq=500)
    if not crit.field_frozen:
        crit.freeze_residual_field(model, pde)
    m, pred = eval_start(model, x_global, u_global, eps)
    m.update(gate_epoch=int(getattr(tr, "gate_epoch", -1)))
    tr.log_file.close()
    row = dict(rep=rep, eps=float(eps), base_seed=int(base_seed), k_idx=int(k_idx),
               start_seed=start_seed, p0=int(p0),
               sigma=(0.0 if rep == "mlp" else float(sigma)), **m)
    return row, pred


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def bad_lift_ref(eps):
    """未 sharpen 的塌缩态 ≈ 线性边界提升 A=1/2(1+x) 的理论参考值。"""
    return dict(slope_ratio=0.5 / true_center_slope(eps), jump05=0.5, u_center=0.5)


def geom_good(r):
    """几何好盆地（=可精修盆地）判据：层的拓扑/形状已被正确捕捉。
    阈值取理论好/坏两态巨大自然间隙正中（非临界拍值）：好态 slope≥.98、jump≥.987、
    plat<6e-3；线性提升坏态 slope≈0.5/s_true（eps=1e-4 时≈.0125）、jump=.5、plat=O(.2)。
    跨表示器公平——MLP 粗解 L2 偏大但几何全对，不应被固定 L2 阈值误判。"""
    return (float(r["slope_ratio"]) > 0.5 and abs(float(r["jump05"]) - 1.0) < 0.05
            and abs(float(r["u_center"]) - 0.5) < 0.05 and float(r["plat_err"]) < 2e-2)


def topo_classify(rows, eps, rep):
    """二级分类（备查极薄层用），读预测曲线区分三种态：
      refinable  几何好盆地（中心斜率已充分、可精修）；
      under      拓扑正确但欠 sharpen——单调不减、两端平台对、过零中心在 5√eps 内，
                 仅中心斜率不足（表示带宽/采样受限，仍是正确盆地，非伪解）；
      pseudo     伪解/错拓扑——非单调、错边界平台，或过零中心漂移 >5√eps（错误位置的层）。
    关键：slope_ratio 单点阈值会把“展宽/亚层宽微抖的正确解”误判为坏，故盆地归属以拓扑为准。"""
    p = os.path.join(OUT, f"diag_preds_{rep}_eps{eps:g}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p); x = d["x"]; P = d["preds"]; seeds = [int(s) for s in d["seeds"]]
    rmap = {int(r["start_seed"]): r for r in rows}
    nu = float(np.sqrt(eps)); cnt = {"refinable": 0, "under": 0, "pseudo": 0}; detail = []
    tol = 2e-3 * max(float(np.abs(P).max()), 1e-12)
    for pp, sd in zip(P, seeds):
        r = rmap.get(sd)
        if r is None: continue
        mono = bool(np.all(np.diff(pp) >= -tol))
        bc = abs(pp[0]) < .02 and abs(pp[-1] - 1) < .02
        pm = np.maximum.accumulate(pp)          # 抹平亚网格微小抖动后求过零中心
        xc = float(np.interp(.5, pm, x))
        center = abs(xc) < 5 * nu
        topo_ok = mono and bc and center
        k = "refinable" if geom_good(r) else ("under" if topo_ok else "pseudo")
        cnt[k] += 1; detail.append((sd, k, mono, bc, xc))
    return cnt, detail


def largest_gap(vals, log=False):
    v = np.sort(np.array(vals, dtype=float))
    vv = np.log10(v) if log else v
    if len(vv) < 2: return None
    d = np.diff(vv)
    j = int(np.argmax(d))
    return dict(lo=float(v[j]), hi=float(v[j + 1]),
                mid=(float((vv[j] + vv[j + 1]) / 2.0) if log else float(0.5 * (v[j] + v[j + 1]))),
                gap=float(d[j]))


def aggregate(rows, eps, rep):
    import math
    n = len(rows)
    bases = sorted({r["base_seed"] for r in rows})
    K = int(max(1, max(r["k_idx"] for r in rows) + 1))
    print(f"\n===== S2d 盆地诊断：eps={eps:g} rep={rep}，独立起点 n={n}，base={len(bases)}，K={K} =====")
    metrics = [("L2", True), ("L2_layer", True), ("slope_ratio", False),
               ("u_center", False), ("jump05", False), ("plat_err", False)]
    gaps = {}
    for name, lg in metrics:
        v = [r[name] for r in rows]
        g = largest_gap(v, log=lg)
        gaps[name] = g
        v_s = np.sort(v)
        print(f"  {name:11s} min={np.min(v):.3e} med={np.median(v):.3e} max={np.max(v):.3e} | "
              f"最大间隙: [{g['lo']:.3e},{g['hi']:.3e}]" + (f" (log10 宽 {g['gap']:.2f})" if lg else ""))

    # 一组候选阈值下的单起点命中率 与 K=multistart 选优命中率（阈值让分布/间隙说话，不拍 .06）
    print("\n  候选好盆地阈值（按全域 L2）：单起点命中率 vs K=%d 选优命中率" % K)
    thr_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1]
    by_base = {}
    for r in rows: by_base.setdefault(r["base_seed"], []).append(r["L2"])
    lines = []
    for thr in thr_list:
        k_single = sum(1 for r in rows if r["L2"] < thr)
        lo, hi = wilson(k_single, n)
        k_ms = sum(1 for b, ls in by_base.items() if min(ls) < thr)
        nb = len(by_base)
        lom, him = wilson(k_ms, nb)
        line = (f"   L2<{thr:g}: 单起点 {k_single}/{n}={k_single/n:.2f} (Wilson [{lo:.2f},{hi:.2f}]) | "
                f"K{K}选优 {k_ms}/{nb}={k_ms/nb:.2f} (Wilson [{lom:.2f},{him:.2f}])")
        print(line); lines.append(line)
    # ---- 几何好盆地（可精修）判据：跨表示器公平，阈值取理论好/坏态巨大间隙正中 ----
    br = bad_lift_ref(eps)
    print(f"\n  理论线性提升坏态参考：slope_ratio≈{br['slope_ratio']:.4f}, jump={br['jump05']:.2f}, u(0)=0.50")
    kg = sum(1 for r in rows if geom_good(r)); glo, ghi = wilson(kg, n)
    by_geo = {}
    for r in rows: by_geo.setdefault(r["base_seed"], []).append(geom_good(r))
    kgms = sum(1 for v in by_geo.values() if any(v)); lom, him = wilson(kgms, len(by_geo))
    print(f"  几何好盆地（层已正确捕捉、可精修）：单起点 {kg}/{n}={kg/n:.2f} (Wilson [{glo:.2f},{ghi:.2f}])")
    print(f"                                     K{K}选优 {kgms}/{len(by_geo)}={kgms/len(by_geo):.2f} "
          f"(Wilson [{lom:.2f},{him:.2f}])")
    print(f"   判据 slope_ratio>0.5 且 |jump-1|<.05 且 |u(0)-.5|<.05 且 plat_err<.02；"
          f"好态 slope≥.98/jump≥.987，坏态 slope≈{br['slope_ratio']:.4f}/jump=.5")
    # 0 事件上界（以几何口径为准）
    if kg == n:
        print(f"   注：{n}/{n} 全为几何好盆地，坏盆地率单侧95% CP 上界 ≈ {1-0.05**(1/n):.3f}（不能写 0）")
    # 拓扑正确性二级分类（极薄层备查：区分“无伪解”与“表示能力不足以 sharpen”）
    tc = topo_classify(rows, eps, rep)
    if tc is not None:
        cnt, _ = tc; ntot = sum(cnt.values())
        print(f"  拓扑二级分类：可精修 {cnt['refinable']} | 拓扑对但欠sharpen {cnt['under']} | "
              f"伪解/错拓扑 {cnt['pseudo']}（共 {ntot}）")
        print("   欠sharpen＝单调、两端平台对、过零中心在5√eps内但中心斜率不足（表示带宽受限，仍属正确盆地）；")
        print("   伪解＝非单调/错平台/过零中心漂移>5√eps（错误位置的层）。盆地归属以拓扑为准、与 sharpen 程度分开。")
    make_plots(rows, eps, rep, gaps, thr_list, by_base)
    return gaps


def make_plots(rows, eps, rep, gaps, thr_list, by_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = r"C:\Windows\Fonts\msyh.ttc"
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name()]
    plt.rcParams["axes.unicode_minus"] = False

    # 图1：六指标分布（strip）
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    specs = [("L2", True), ("L2_layer", True), ("slope_ratio", False),
             ("u_center", False), ("jump05", False), ("plat_err", False)]
    for ax, (name, lg) in zip(axes.flat, specs):
        v = np.array([r[name] for r in rows], float)
        order = np.argsort(v)
        x = np.log10(v) if lg else v
        ax.scatter(x, np.zeros_like(x) + np.random.default_rng(0).normal(0, 0.02, len(x)),
                   s=22, color="#4B86B4", alpha=0.8)
        g = gaps[name]
        if g is not None:
            gx = np.log10(0.5 * (g["lo"] * g["hi"]) ** 0.5) if name in ("L2", "L2_layer") \
                else 0.5 * (g["lo"] + g["hi"])
            ax.axvline(gx, color="#E88A24", ls="--", lw=1.2, label="最大自然间隙")
        ax.set_yticks([]); ax.set_title(name + ("（log10）" if lg else ""), fontsize=11)
        ax.grid(alpha=.25, axis="x"); ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"S2d 内部转向点层 逐起点 Phase0 几何量分布（eps={eps:g}, {rep}, n={len(rows)}）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    f1 = os.path.join(OUT, f"s2d_basin_dist_{rep}_eps{eps:g}.png")
    fig.savefig(f1, dpi=160); plt.close(fig)

    # 图2：最好/中位/最差代表解叠真解
    pred_npz = np.load(os.path.join(OUT, f"diag_preds_{rep}_eps{eps:g}.npz"))
    x = pred_npz["x"]; P = pred_npz["preds"]; seeds = pred_npz["seeds"]; u = pred_npz["u"]
    l2 = np.array([r["L2"] for r in rows]); order = np.argsort(l2)
    pick = [order[0], order[len(order) // 2], order[-1]]
    lab = ["最好", "中位", "最差"]
    col = ["#2E8B57", "#E88A24", "#C0392B"]
    fig2, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x, u, "k-", lw=2, label="真解 erf")
    for idx, lb, c in zip(pick, lab, col):
        ax.plot(x, P[idx], "--", color=c, lw=1.6,
                label=f"{lb}（seed {int(seeds[idx])}, L2={l2[idx]:.1e}）")
    ax.set_xlim(-0.35, 0.35); ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.set_title(f"代表性 Phase0 粗解（eps={eps:g}, {rep}）：坏形态是否只是线性提升/未 sharpen")
    ax.grid(alpha=.25); ax.legend(fontsize=9)
    f2 = os.path.join(OUT, f"s2d_basin_shapes_{rep}_eps{eps:g}.png")
    fig2.tight_layout(); fig2.savefig(f2, dpi=160); plt.close(fig2)
    print("  图已存：", f1, "|", f2)


def main():
    global x_global, u_global, x_td_global, u_td_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--base-seeds", default="")  # 默认 SEEDS_EXT 11 颗
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--p0", type=int, default=2000)
    ap.add_argument("--rep", choices=["fourier", "mlp"], default="fourier")
    ap.add_argument("--sigma", type=float, default=15.0)
    a = ap.parse_args()

    p0 = 60 if mc.SMOKE else a.p0
    if a.base_seeds.strip():
        bases = [int(v) for v in a.base_seeds.split(",") if v.strip()]
    else:
        bases = list(mc.SEEDS_EXT)
    if mc.SMOKE:
        bases = bases[:1]; a.k = 2

    x_global, u_global = test_arrays(a.eps)          # 最终几何评估（随 eps 加密）
    x_td_global, u_td_global = test_arrays_n(a.eps, N_TEST)  # 训练/Gate 固定 2048
    csvp = os.path.join(OUT, "s2d_basin_diag.csv")
    done = set()
    if os.path.exists(csvp):
        for r in csv.DictReader(open(csvp, encoding="utf-8-sig")):
            done.add((r["rep"], float(r["eps"]), int(r["start_seed"])))
    newfile = not os.path.exists(csvp)
    fout = open(csvp, "a", newline="", encoding="utf-8-sig")
    wcsv = csv.DictWriter(fout, fieldnames=FIELDS)
    if newfile: wcsv.writeheader()

    preds, seeds_here = [], []
    try:
        for b in bases:
            for ki in range(a.k):
                ss = int(b) + ki
                if (a.rep, float(a.eps), ss) in done:
                    print("skip", a.rep, a.eps, ss); continue
                row, pred = run_one(a.eps, b, ki, a.rep, a.sigma, p0)
                wcsv.writerow(row); fout.flush()
                preds.append(pred); seeds_here.append(ss)
                print(f"[diag {a.rep} eps{a.eps:g}] start {ss}: L2={row['L2']:.2e} "
                      f"层内={row['L2_layer']:.2e} u0={row['u_center']:.3f} "
                      f"slope比={row['slope_ratio']:.3f} jump={row['jump05']:.3f} "
                      f"plat={row['plat_err']:.2e} gate@{row['gate_epoch']}")
    finally:
        fout.close()

    # 汇总（读全量 csv 中本 rep/eps 的行）；预测曲线合并历史 npz 后重存
    rows = [dict(r) for r in csv.DictReader(open(csvp, encoding="utf-8-sig"))
            if r["rep"] == a.rep and abs(float(r["eps"]) - a.eps) < 1e-15]
    for r in rows:
        for k in FIELDS[3:]:
            r[k] = float(r[k])
    # 用本次内存预测 + 必要时占位补齐（图只画当前 rep/eps；历史预测从 npz 合并）
    npz = os.path.join(OUT, f"diag_preds_{a.rep}_eps{a.eps:g}.npz")
    if preds:
        P_new = np.array(preds); S_new = np.array(seeds_here)
        if os.path.exists(npz):
            old = np.load(npz); Po, So = old["preds"], old["seeds"]
            m = {int(s): i for i, s in enumerate(So)}
            Plist, Slist = list(Po), list(So)
            for s, p in zip(S_new, P_new):
                if int(s) in m: Plist[m[int(s)]] = p
                else: Plist.append(p); Slist.append(int(s))
            P_all = np.array(Plist); S_all = np.array(Slist)
        else:
            P_all, S_all = P_new, S_new
        order = np.argsort(S_all)
        np.savez(npz, x=x_global, u=u_global, preds=P_all[order], seeds=S_all[order])
    if len(rows) >= (a.k if mc.SMOKE else a.k * len(bases)):
        aggregate(rows, a.eps, a.rep)
    else:
        print(f"\n[提示] 当前仅 {len(rows)} 行，未集齐 {a.k*len(bases)} 起点；续跑后自动汇总出图。")


if __name__ == "__main__":
    main()
