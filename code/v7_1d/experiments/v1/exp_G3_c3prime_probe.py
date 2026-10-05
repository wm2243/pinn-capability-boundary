# -*- coding: utf-8 -*-
"""
实验 G3（V9.3 审稿 P0-2 补证）：命题 8.3 方向判据的成立条件——C3′ 失配量 ε_S 与对齐间隙
================================================================================
理论背景（framework_v9_3 §8.2，注记 8.2 / 命题 8.3 / 附录 F）：
  弱 C3  ⟨∂β g_S, g_S⟩ ≥ 0 只控制“平行分量非负”，推不出 ∂β g_S 与 g_S 同向；
  方向命题需要更强的 (C3′) 带内准均匀：
        max_{i∈S} |∂βw_i/w_i − c̄β| ≤ ε_S c̄β,  c̄β = mean_{i∈S} ∂βw_i/w_i,
  从而 ∂β g_S = c̄β g_S + v_S，‖v_S‖ ≤ ε_S c̄β ‖g_S‖。
  命题 8.3 只在 |cos α_S| > ε_S 时严格定号（好盆地 cos>c0>ε_S 加速靠近、坏盆地 cos<−c0<−ε_S 加速焊死）。
  对有理族 w=(1+βt)/(1+t)：∂βw = t/(1+t)，∂βw/w = t/(1+βt)，C3′ 等价“激波带内对比度 t 的变异小”。

本实验在【冻结 Gate 点】（同 θ、同配点、同冻结对比度场，仅改 β，与 C1 同一“静态冻结”口径）直接测量：
  * g_S(β) = mean_{i∈S} w_i r_i J_i，  ∂β g_S = mean_{i∈S} (∂βw_i) r_i J_i，  J_i = ∂r_i/∂θ（逐配点参数梯度）；
  * ε_S(β) = ‖∂βg_S − c̄β g_S‖ / (c̄β‖g_S‖)                 —— C3′ 失配量（越小 C3′ 越成立）；
  * C3 内积 ⟨∂βg_S, g_S⟩（应 ≥0，A1 范数定号的弱条件）与 ‖g_S‖² 对 β 的单调性（∂β‖g_S‖²=2⟨∂βg_S,g_S⟩）；
  * cos α_S(β) = −⟨g_S, d⟩/(‖g_S‖‖d‖)，d = θ*−θ_gate（盆地内目标方向，离线 oracle：同一盆地以 β_ref 长训收敛得 θ*）；
  * 定号间隙 gap(β) = |cos α_S| − ε_S（>0 命题 8.3 严格定号；好盆地应 cos>ε_S、坏盆地应 −cos>ε_S）。
  * 实现自检：用逐配点 Gram 子块 G_S=[⟨J_i,J_j⟩] 验证 ⟨∂βg_S,g_S⟩ = q_S^T G_S p_S（两条路径必须相等，附录 F 判据）。

口径声明（防 overclaim）：cos α_S 与好/坏标签都用真解/收敛解，属【离线机制验证】，不是无真解部署量；
  无真解可部署判据是 G2 的 Gate-B。ε_S、C3 内积、‖g_S‖² 本身【不需要真解】（纯冻结残差与权重）。

产出 results/v6/exp_G3_c3prime/：
  g3_probe.csv   每 (nu,seed,beta) 一行：能量/内积/c̄/ε_S/cos/gap + 标签（好/坏由 Gate L2 事后贴）
  g3_agg.csv     按 (truth_good,beta) 跨种子中位+IQR
  fig_g3_energy.png  ‖g_S‖² 与 C3 内积随 β（验证 A1：能量随 β 单调增、内积≥0）
  fig_g3_eps.png     好/坏盆地 ε_S 与 |cos α_S| 对比（验证定号间隙 |cos|>ε_S）
  fig_g3_cos.png     cos α_S(β)：好盆地正、坏盆地负，与 ε_S 带叠加
环境：V8_SMOKE=1 自检；G3_K 多起点（默认4，与主结果一致）；G3_M 激波带内探针点数（默认48）；
      G3_EPREF 参考长训步数（默认4000）；G3_FAST=1 用 5 个规则种子提速。
运行：python exp_G3_c3prime_probe.py     （复用 results/v6/_gate_cache 的 Gate 盆地，不重复 Phase0）
"""
import os, sys, csv, copy
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve(); _PKG = _HERE.parent.parent
for _p in (str(_PKG), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import run_two_stage_v6 as r2s
import v8_matrix_common as mc

OUT = str(mc.RES_ROOT / "exp_G3_c3prime"); os.makedirs(OUT, exist_ok=True)
PROBE_CSV = os.path.join(OUT, "g3_probe.csv"); AGG_CSV = os.path.join(OUT, "g3_agg.csv")

NUS = [0.01, 0.005, 0.003, 0.001]
SEEDS = mc.SEEDS_NEW if os.environ.get("G3_FAST", "0") == "1" else mc.SEEDS_EXT
K_START = int(os.environ.get("G3_K", "4"))
M_PROBE = int(os.environ.get("G3_M", "48"))          # 激波带内探针配点数
EP_GATE = 2000
EP_REF = int(os.environ.get("G3_EPREF", "4000"))    # 盆地内目标 θ* 的参考长训步数
BETA_REF = 20.0                                     # 取 θ* 的加权（E1 最优区间中位）
BETAS = [1.0, 3.0, 10.0, 40.0]
BAND_C = 3.0
if mc.SMOKE: NUS, SEEDS, K_START, M_PROBE, EP_REF, BETAS = [0.005], [0], 1, 10, 120, [1.0, 10.0]


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


def row_jacobian(model, pde, x):
    """逐配点残差 Jacobian J[i]=∂r_i/∂θ（M×P）。M 很小（仅激波带内子采样），逐行一阶反传可承受。"""
    params = [p for p in model.parameters() if p.requires_grad]
    res = pde.compute_residual(model, x).reshape(-1)
    rows = []
    m = res.numel()
    for i in range(m):
        gi = torch.autograd.grad(res[i], params, retain_graph=(i < m - 1), allow_unused=True)
        rows.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                               for g, p in zip(gi, params)]))
    return torch.stack(rows), res.detach()


def build_gate_model(basin, nu, beta):
    """同一 Gate 状态、同一冻结场，仅设 β。"""
    r2s.set_seed(0)
    cfg = r2s.base_cfg(nu, f"g3gate_b{beta}")
    cfg.update(dict(use_gate=False, adam_epochs=1, lbfgs_epochs=0, beta_schedule="const",
                    loss_beta=float(beta), fourier_scale=float(basin["sigma_end"]),
                    sigma_lo=None, sigma_hi=None, sigma_anneal_T=0))
    model, pde, crit, opt, samp, vis, cfg = r2s.build_all(cfg, r2s.LR_PHASE1, float(beta))
    model.load_state_dict(basin["state"]); model.set_sigma(basin["sigma_end"])
    crit.load_frozen_field(basin["t_field"], beta=float(beta))
    model.eval()
    return model, pde, crit


@torch.enable_grad()
def probe_one_beta(basin, nu, beta, J, r, t, theta_gate, d_dir):
    """给定预计算的逐配点 J/r 与冻结对比度 t，算该 β 下的全部 C3′/对齐量。J,r,t 与 β 无关（冻结）。"""
    t = t.clamp_min(0)
    w = (1.0 + beta * t) / (1.0 + t + 1e-12)
    dw = t / (1.0 + t + 1e-12)
    m = J.shape[0]
    wr = (w * r).unsqueeze(1); dwr = (dw * r).unsqueeze(1)
    gS = (J * wr).mean(0)              # = J^T(w r)/m
    dgS = (J * dwr).mean(0)            # = J^T(∂βw r)/m
    gn2 = float(gS.norm() ** 2)
    c3_inner = float(torch.dot(dgS, gS))                 # 弱 C3：应 ≥0
    ratio = dw / w.clamp_min(1e-12)                       # ∂βw/w = t/(1+βt)
    cbar = float(ratio.mean())
    v = dgS - cbar * gS
    eps = float(v.norm() / (abs(cbar) * gS.norm() + 1e-30))
    # cos α_S = −⟨g_S,d⟩/(‖g_S‖‖d‖)，d 已归一化为方向
    cos_a = float(-torch.dot(gS, d_dir) / (gS.norm() + 1e-30))
    # 附录 F 自检：⟨∂βg,g⟩ = q^T G_S p，G_S = J J^T/m²
    G = J @ J.t()  # G_S=[<J_i,J_j>]，不再除 m：q=(∂βw·r)/m、p=(w·r)/m，q^T G p 恰等于内积
    gram_inner = float(((dw * r) / m) @ G @ ((w * r) / m))
    return dict(beta=float(beta), gS_norm2=gn2, c3_inner=c3_inner, cbar=cbar,
                eps_S=eps, cos_S=cos_a, gap=abs(cos_a) - eps,
                gram_check_err=abs(gram_inner - c3_inner) / (abs(c3_inner) + 1e-30))


def reference_direction(basin, nu, seed, xt, ut):
    """离线盆地内目标方向 d=θ*−θ_gate（θ*：同一盆地以 BETA_REF 长训收敛）。仅用于机制验证，非部署量。"""
    _, _, mref, _, _ = mc.train_phase1_family(
        basin, nu, BETA_REF, seed, xt, ut, family="rational",
        epochs=EP_REF, tag="g3ref", lr=r2s.LR_PHASE1)
    theta_star = flat_params(mref)
    gate_model, _, _ = build_gate_model(basin, nu, 1.0)
    theta_gate = flat_params(gate_model)
    d = theta_star - theta_gate
    return theta_gate, (d / (d.norm() + 1e-30)).detach()


def collect():
    done, rows = {}, []
    if os.path.exists(PROBE_CSV):
        with open(PROBE_CSV, "r", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh): rows.append(r); done[(r["nu"], r["seed"])] = r
    for nu in NUS:
        xt, ut = r2s.build_test(nu)
        band = BAND_C * nu
        xp = torch.linspace(-band, band, M_PROBE, device=mc.DEVICE).view(-1, 1).requires_grad_(True)
        for seed in SEEDS:
            tag = (f"{nu:g}", str(seed))
            if tag in done: continue
            basin = mc.enter_basin_cached(nu, seed, xt, ut, sigma_hi=r2s.SIGMA_HI_CFG,
                                          k=K_START, phase0_epochs=EP_GATE)
            truth_good = int(basin["l2"] < mc.GATE_L2_MAX)
            theta_gate, d_dir = reference_direction(basin, nu, seed, xt, ut)
            # 冻结 Gate 点：J/r/t 与 β 无关，只算一次
            gm, pde, crit = build_gate_model(basin, nu, 1.0)
            J, r = row_jacobian(gm, pde, xp)
            with torch.no_grad():
                t = crit._interp_field(crit._frozen_t_field, xp).detach()
            gerr_max = 0.0
            for beta in BETAS:
                rec = probe_one_beta(basin, nu, beta, J, r, t, theta_gate, d_dir)
                rec.update(nu=nu, seed=seed, gate_L2=float(basin["l2"]),
                           truth_good=truth_good, t_cv=float(t.std() / (t.mean() + 1e-30)))
                rows.append(rec); gerr_max = max(gerr_max, rec["gram_check_err"])
            these = [z for z in rows if z["nu"] == nu and z["seed"] == seed]
            _dump(rows)
            print(f"[G3] nu={nu:g} s{seed} good={truth_good} gateL2={basin['l2']:.3e} "
                  f"| eps@1={these[0]['eps_S']:.3f} cos@last={these[-1]['cos_S']:+.3f} "
                  f"Gram自检误差={gerr_max:.1e}")
    for r in rows:
        for c in ["beta", "gS_norm2", "c3_inner", "cbar", "eps_S", "cos_S", "gap",
                  "gram_check_err", "gate_L2", "t_cv"]:
            r[c] = float(r[c])
        r["truth_good"] = int(r["truth_good"]); r["seed"] = int(r["seed"]); r["nu"] = float(r["nu"])
    return rows


COLS = ["nu", "seed", "beta", "truth_good", "gate_L2", "t_cv", "gS_norm2", "c3_inner",
        "cbar", "eps_S", "cos_S", "gap", "gram_check_err"]


def _dump(rows):
    mc.write_csv(PROBE_CSV, [{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows], COLS)


def _iqr(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0: return (np.nan,) * 5
    return float(np.median(v)), float(np.quantile(v, .25)), float(np.quantile(v, .75)), float(v.min()), float(v.max())


def aggregate(rows):
    out = []
    for good in (1, 0):
        for beta in BETAS:
            sub = [r for r in rows if r["truth_good"] == good and abs(r["beta"] - beta) < 1e-9]
            if not sub: continue
            rec = dict(truth_good=good, beta=beta, n=len(sub))
            for c in ["eps_S", "cos_S", "gap", "gS_norm2", "c3_inner", "t_cv"]:
                md, q1, q3, lo, hi = _iqr([r[c] for r in sub])
                rec.update({f"{c}_med": md, f"{c}_q1": q1, f"{c}_q3": q3})
            # 命题 8.3 定号条件达成率：好盆地 cos>eps、坏盆地 cos<−eps
            if good:
                rec["sign_hold_rate"] = float(np.mean([r["cos_S"] > r["eps_S"] for r in sub]))
            else:
                rec["sign_hold_rate"] = float(np.mean([r["cos_S"] < -r["eps_S"] for r in sub]))
            out.append(rec)
    mc.write_csv(AGG_CSV, out)
    return out


def plots(rows):
    cmap = {1: "#3a8a18", 0: "#c0392b"}; lab = {1: "好盆地", 0: "#坏盆地".replace("#", "")}
    # 1) 能量与 C3 内积随 β（A1：单调增、内积≥0）
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for good in (1, 0):
        for nu in sorted({r["nu"] for r in rows}):
            z = sorted([r for r in rows if r["truth_good"] == good and r["nu"] == nu], key=lambda q: q["beta"])
            if not z: continue
            ax[0].plot([q["beta"] for q in z], [q["gS_norm2"] for q in z], "o-",
                       color=cmap[good], alpha=.35 + .15 * sorted({r['nu'] for r in rows}).index(nu))
            ax[1].plot([q["beta"] for q in z], [q["c3_inner"] for q in z], "s-", color=cmap[good], alpha=.5)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("β"); ax[0].set_ylabel(r"$\|g_S\|^2$")
    ax[0].set_title("A1 验证：激波带梯度能量随 β 增大（绿=好/红=坏）"); ax[0].grid(alpha=.3)
    ax[1].axhline(0, color="k", ls="--", lw=1); ax[1].set_xscale("log"); ax[1].set_xlabel("β")
    ax[1].set_ylabel(r"$\langle\partial_\beta g_S,g_S\rangle$"); ax[1].set_title("弱 C3：内积应 ≥0"); ax[1].grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(OUT, "fig_g3_energy.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)

    # 2) ε_S 与 |cos| 对比（定号间隙）
    fig, a = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for good in (1, 0):
        sub = [r for r in rows if r["truth_good"] == good]
        if not sub: continue
        a[0].scatter([r["eps_S"] for r in sub], [abs(r["cos_S"]) for r in sub],
                     s=22, alpha=.6, color=cmap[good], label=lab[good])
    lim = max([max(r["eps_S"], abs(r["cos_S"])) for r in rows] + [1e-3])
    a[0].plot([0, lim], [0, lim], "k--", lw=1); a[0].set_xlabel(r"$\varepsilon_S$（C3′失配）")
    a[0].set_ylabel(r"$|\cos\alpha_S|$"); a[0].set_title("点在对角线上方 ⇔ |cos|>ε_S（命题8.3严格定号）")
    a[0].legend(); a[0].grid(alpha=.3)
    # 3) cos(β) 曲线，叠加 ε 带
    for good in (1, 0):
        for nu in sorted({r["nu"] for r in rows}):
            z = sorted([r for r in rows if r["truth_good"] == good and r["nu"] == nu], key=lambda q: q["beta"])
            if not z: continue
            bs = [q["beta"] for q in z]
            a[1].plot(bs, [q["cos_S"] for q in z], "o-", color=cmap[good], alpha=.55)
            a[1].fill_between(bs, [-q["eps_S"] for q in z], [q["eps_S"] for q in z], color=cmap[good], alpha=.08)
    a[1].axhline(0, color="k", lw=1); a[1].set_xscale("log"); a[1].set_xlabel("β")
    a[1].set_ylabel(r"$\cos\alpha_S$（阴影=±ε_S 带）"); a[1].set_title("好盆地应稳定为正、坏盆地稳定为负"); a[1].grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(OUT, "fig_g3_cos.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)

    # 4) ε_S 分布（C3′ 是否天然近似成立：薄激波带应较小）
    fig, a = plt.subplots(figsize=(6.2, 4.4))
    data = [[r["eps_S"] for r in rows if r["truth_good"] == g] for g in (1, 0)]
    if all(data):
        a.boxplot(data, showfliers=False); a.set_xticklabels(["好盆地", "坏盆地"])
        for j, arr in enumerate(data): a.scatter(np.random.normal(j + 1, .05, len(arr)), arr, s=14, alpha=.6, color=cmap[1 - j])
    a.set_ylabel(r"$\varepsilon_S$"); a.set_title("C3′ 失配量（越小越接近带内准均匀）"); a.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(OUT, "fig_g3_eps.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)


def main():
    rows = collect()
    if len({r["truth_good"] for r in rows}) < 2:
        print("[G3] 好/坏两类未同时出现，先扩 Gate 样（G1）；已落探针数据。")
    agg = aggregate(rows)
    plots(rows)
    gerr = np.max([r["gram_check_err"] for r in rows]) if rows else np.nan
    print("\n===== G3 命题8.3 成立条件（跨种子聚合，β 分组）=====")
    print("类别 β     n   ε_S中位   cos中位   gap中位   定号达成率")
    for z in sorted(agg, key=lambda q: (-q["truth_good"], q["beta"])):
        print(f"  {'好' if z['truth_good'] else '坏'} {z['beta']:>4g} {z['n']:>2}  "
              f"{z['eps_S_med']:.3f}    {z['cos_S_med']:+.3f}    {z['gap_med']:+.3f}    {z['sign_hold_rate']:.0%}")
    print(f"\n附录F 双路径自检 max|qᵀGp−⟨∂βg,g⟩|/|·| = {gerr:.2e}（应 ≈0，验证探针实现正确）")
    print("[判读] 好/坏盆地 gap 中位均 >0 且定号达成率高 ⇒ C3′ 近似成立、命题8.3 阈值 |cos|>ε_S 满足；")
    print("       ‖g_S‖² 随 β 单调增、C3 内积≥0 ⇒ A1 范数结论独立成立（不依赖 C3′）。产物在", OUT)


if __name__ == "__main__":
    main()
