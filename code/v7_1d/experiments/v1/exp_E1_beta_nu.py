# -*- coding: utf-8 -*-
"""
实验 E1（V9）：盆地内 β×ν 精细扫描 —— 坐实 U 型、β*(ν) 迁移与 U 型左右支机制
--------------------------------------------------------------------------------
设计（同 gate 配对）：每个 (nu,seed) 只进一次盆地并缓存，随后在【同一好盆地】上扫 β，
唯一变量是 β。全量用规则等距种子 SEEDS_EXT（11 个：0,5,…,50，是原 SEEDS_NEW=[0,10,20,30,40]
的超集，原 5 个全部保留），不预贴好/坏标签、全量报告（gate_ok 按 Phase0(β=1) 的固定阈值标记，
与 β 无关）。

产出 results/v6/exp_E1_beta_nu/：
  e1_runs.csv / e1_agg.csv
  fig_E1_ucurve.png       各 ν 的 L2/Linf/l2_band/T_hit U 型曲线（median+IQR）
  fig_E1_beta_star_nu.png β*(ν) 迁移（L2 口径与激波带口径，跨种子区间）
  fig_E1_mechanism.png    ν=.005：focus/梯度能量/末段回摆/带内权重 随 β（解释左右支）
  fig_E1_bridge.png       ν=.005、首种子：分 β 的 ||r||_W–||e|| 轨迹（同 β 内单调、跨 β 不共带）
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

OUT = str(mc.RES_ROOT / "exp_E1_beta_nu"); os.makedirs(OUT, exist_ok=True)
r2s.OUT = OUT

# 主 ν=.005 用 12 点密网格定位 U 型底；其余 ν 用 6 点看 β*(ν) 迁移
BETA_GRID = {
    0.005: [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0],
    0.01:  [1.0, 5.0, 10.0, 15.0, 30.0, 60.0],
    0.003: [1.0, 5.0, 10.0, 15.0, 30.0, 60.0],
    0.001: [1.0, 5.0, 10.0, 15.0, 30.0, 60.0],
}
NUS = [0.01, 0.005, 0.003, 0.001]
SEEDS = mc.SEEDS_EXT        # 11 个规则种子（0,5,…,50），统计性主结果扩样
THRS = (0.05, 0.02, 0.01)
if mc.SMOKE:
    NUS = [0.005]; SEEDS = [mc.SEEDS_EXT[ 0 ]]; BETA_GRID = {0.005: [1.0, 15.0]}

# ---- 机制探针（不改训练内循环，只在冻结点反传 / Lanczos）----
PROBE_NU = 0.005                                   # 机制深挖只在主 ν
PROBE_N = 256 if mc.SMOKE else 4096                # 梯度分区能量的固定配点（跨 β 同一批）
MID_EP = 120 if mc.SMOKE else 4000                 # 中段训练步数
MID_BETAS = [1.0, 15.0, 40.0]                      # 演化探针（首种子）
HESS_BETAS = [1.0, 15.0, 100.0]                    # 参数 Hessian 抽测（首种子、近解点）
HESS_N = 64 if mc.SMOKE else 512
HESS_LANCZOS = 8 if mc.SMOKE else 40


def _align_traj(dyn):
    steps = np.asarray(dyn["step"], float); tl = np.asarray(dyn["total_loss"], float)
    estep = np.asarray(dyn["eval_step"], float); l2 = np.asarray(dyn["L2_traj"], float)
    rw, ee = [], []
    for s, z in zip(estep, l2):
        j = int(np.argmin(np.abs(steps - s)))
        if tl[ j ] > 0 and np.isfinite(z):
            rw.append(np.sqrt(tl[ j ])); ee.append(z)
    return np.asarray(rw), np.asarray(ee)


def collect():
    r2s.LR_PHASE1 = 5e-4
    rows, bridges, probe_rows, hess_rows = [], {}, [], []
    for nu in NUS:
        betas = BETA_GRID[ nu ]
        xt, ut = r2s.build_test(nu)
        xp = torch.linspace(-1.0, 1.0, PROBE_N, device=mc.DEVICE).view(-1, 1)  # 跨 β 同一固定批
        for seed in SEEDS:
            basin = mc.enter_basin_cached(nu, seed, xt, ut, sigma_hi=r2s.SIGMA_HI_CFG,
                                          k=4, phase0_epochs=2000)
            for beta in betas:
                ep = 240 if mc.SMOKE else 8000
                m, dyn, model, crit, trainer = mc.train_phase1_family(
                    basin, nu, beta, seed, xt, ut, family="rational",
                    epochs=ep, tag="E1", grad_probe=(xp, nu))
                mech = mc.mechanism_row(dyn); gp = trainer.grad_probe
                row = dict(
                    nu=nu, seed=seed, beta=float(beta),
                    phase0_L2=basin["l2"], gate_ok=int(basin["l2"] < mc.GATE_L2_MAX),
                    L2=m.get("L2_Error", np.nan), Linf=m.get("L_inf_Error", np.nan),
                    TV=m.get("TV_Error", np.nan), l2_band=m.get("l2_band", np.nan),
                    l2_smooth=m.get("l2_smooth", np.nan),
                    loss_shock=m.get("loss_shock", np.nan), loss_smooth=m.get("loss_smooth", np.nan),
                    wmean_shock=m.get("weight_mean_shock", np.nan),
                    osc=m.get("Oscillation_Count", np.nan),
                    width_ratio=m.get("Shock_Width_Ratio", np.nan),
                    best_epoch=m.get("best_epoch", -1),
                    T05=mc.first_hit_step(dyn, THRS[ 0 ]),
                    T02=mc.first_hit_step(dyn, THRS[ 1 ]),
                    T01=mc.first_hit_step(dyn, THRS[ 2 ]),
                    gap_final_best=mc.final_minus_best(dyn),
                    focus_t=mech["focus_t"], grad_energy_t=mech["grad_energy_t"],
                    wshock_t=mech["wmean_shock_t"], Cw=1.0,
                    eta_gate=gp["gate"]["eta"], eta_end=gp["end"]["eta"],
                    cos_gate=gp["gate"]["cos"], cos_end=gp["end"]["cos"])
                rows.append(row)
                for point in ("gate", "end"):
                    d = gp[ point ]
                    probe_rows.append(dict(nu=nu, seed=seed, beta=float(beta), point=point,
                                           g_band=d["g_band"], g_smooth=d["g_smooth"],
                                           eta=d["eta"], cos=d["cos"]))
                # 探针2：近解点参数 Hessian（仅主 ν、首种子、抽测 β）
                if nu == PROBE_NU and seed == SEEDS[ 0 ] and float(beta) in HESS_BETAS:
                    xh = torch.linspace(-1.0, 1.0, HESS_N, device=mc.DEVICE).view(-1, 1)
                    hh = mc.hessian_min_curvature(model, trainer.pde_engine, crit, xh, HESS_LANCZOS)
                    hess_rows.append(dict(nu=nu, seed=seed, beta=float(beta), **hh))
                    print(f"[E1-Hess] β{beta:g}: λmin={hh['lam_min']:.3e} "
                          f"κ={hh['kappa']:.2e} neg={hh['n_neg']}")
                if nu == PROBE_NU and seed == SEEDS[ 0 ]:
                    bridges[float(beta)] = _align_traj(dyn)
                print(f"[E1] ν{nu} s{seed} β{beta:g}: L2={row['L2']:.3e} band={row['l2_band']:.3e} "
                      f"T.01={row['T01']} η_gate={row['eta_gate']:.2f} η_end={row['eta_end']:.2f} "
                      f"gate_ok={row['gate_ok']}")

            # 探针1·中段演化（仅主 ν、首种子、抽测 β，独立短训到 MID_EP）
            if nu == PROBE_NU and seed == SEEDS[ 0 ]:
                for beta in [b for b in MID_BETAS if b in betas]:
                    _mm, _dd, model_m, crit_m, tr_m = mc.train_phase1_family(
                        basin, nu, beta, seed, xt, ut, family="rational",
                        epochs=MID_EP, tag="E1mid")
                    dmid = mc.grad_partition_energy(model_m, tr_m.pde_engine, crit_m, xp, nu)
                    probe_rows.append(dict(nu=nu, seed=seed, beta=float(beta), point="mid",
                                           g_band=dmid["g_band"], g_smooth=dmid["g_smooth"],
                                           eta=dmid["eta"], cos=dmid["cos"]))
                    print(f"[E1-Mid] β{beta:g} @{MID_EP}: η={dmid['eta']:.2f}")
    mc.write_csv(os.path.join(OUT, "e1_runs.csv"), rows)
    agg = mc.aggregate(rows, ["nu", "beta"],
                       ["L2", "Linf", "l2_band", "l2_smooth", "T01", "focus_t",
                        "grad_energy_t", "gap_final_best", "wshock_t", "eta_gate", "eta_end"])
    mc.write_csv(os.path.join(OUT, "e1_agg.csv"), agg)
    mc.write_csv(os.path.join(OUT, "e1_grad_probe.csv"), probe_rows)
    mc.write_csv(os.path.join(OUT, "e1_hessian.csv"), hess_rows)
    return rows, agg, bridges, probe_rows, hess_rows


# ----------------------------- 图 -----------------------------
def plot_ucurve(rows):
    fig, ax = plt.subplots(len(NUS), 4, figsize=(17, 3.1 * len(NUS)), squeeze=False)
    cols = [("L2_med", "L2 相对误差", True), ("Linf_med", "Linf", True),
            ("l2_band_med", "激波带 l2_band", True), ("T01_med", "达阈步数 T(.01)", False)]
    for i, nu in enumerate(NUS):
        bs = sorted({r["beta"] for r in rows if r["nu"] == nu})
        for j, (key, ttl, logy) in enumerate(cols):
            a = ax[ i ][ j ]
            # 直接从原始 rows 重算 median/IQR（稳健、不依赖 agg 键名）
            med, lo, hi = [], [], []
            src = {"L2_med": "L2", "Linf_med": "Linf", "l2_band_med": "l2_band", "T01_med": "T01"}[ key ]
            for b in bs:
                v = np.asarray([r[ src ] for r in rows if r["nu"] == nu and r["beta"] == b], float)
                v = v[np.isfinite(v)]
                med.append(np.median(v) if v.size else np.nan)
                lo.append(np.quantile(v, .25) if v.size else np.nan)
                hi.append(np.quantile(v, .75) if v.size else np.nan)
            a.plot(bs, med, "o-", color="#2E6E8E")
            a.fill_between(bs, lo, hi, alpha=.18, color="#2E6E8E")
            fin = [m_ for m_ in med if np.isfinite(m_) and m_ > 0]
            if fin:
                ib = int(np.nanargmin(med)); a.axvline(bs[ ib ], color="#EA6668", ls="--", lw=1)
                a.annotate(f"β*≈{bs[ib]:g}", (bs[ ib ], med[ ib ]), fontsize=8, color="#b03a3c")
            has_data = any(np.isfinite(m_) for m_ in med)
            if has_data:
                a.set_xscale("log")
            else:
                a.text(0.5, 0.5, "无有效数据\n(未达阈/样本不足)", ha="center", va="center",
                       transform=a.transAxes, fontsize=9, color="#888")
            if logy and fin: a.set_yscale("log")
            a.set_title(f"ν={nu}  {ttl}", fontsize=10); a.grid(alpha=.3)
            if i == len(NUS) - 1: a.set_xlabel("β")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E1_ucurve.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def plot_beta_star(rows):
    fig, a = plt.subplots(figsize=(6.4, 4.6))
    for src, c, lab in [("L2", "#2E6E8E", "β*(L2)"), ("l2_band", "#94D8C3", "β*(激波带)")]:
        xs, mid, lo, hi = [], [], [], []
        for nu in NUS:
            per = []
            for seed in sorted({r["seed"] for r in rows}):
                z = [(r["beta"], r[ src ]) for r in rows if r["nu"] == nu and r["seed"] == seed]
                z = [(b, v) for b, v in z if np.isfinite(v)]
                if z: per.append(min(z, key=lambda t: t[1])[0])
            if per:
                xs.append(nu); mid.append(np.median(per)); lo.append(min(per)); hi.append(max(per))
        a.plot(xs, mid, "o-", color=c, label=lab); a.fill_between(xs, lo, hi, alpha=.15, color=c)
    a.invert_xaxis(); a.set_xlabel("ν（越小激波越薄越难）"); a.set_ylabel("最优 β*（跨种子区间）")
    a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=.3); a.legend()
    a.set_title("β*(ν) 迁移：显式单调关系（不报指数）")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E1_beta_star_nu.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def plot_mechanism(rows):
    nu = 0.005
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4))
    panels = [("focus_t", "focus=带内/带外权重均值（机制A 再分配）", False),
              ("grad_energy_t", "激波/光滑 输出梯度能量比（机制A）", True),
              ("gap_final_best", "末段-最优 L2 回摆（右支过冲）", False),
              ("wshock_t", "带内平均权重 w（饱和检查）", False)]
    bs = sorted({r["beta"] for r in rows if r["nu"] == nu})
    for a_, (key, ttl, logy) in zip(ax.ravel(), panels):
        med, lo, hi = [], [], []
        for b in bs:
            v = np.asarray([r[ key ] for r in rows if r["nu"] == nu and r["beta"] == b], float)
            v = v[np.isfinite(v)]
            med.append(np.median(v) if v.size else np.nan)
            lo.append(np.quantile(v, .25) if v.size else np.nan)
            hi.append(np.quantile(v, .75) if v.size else np.nan)
        a_.plot(bs, med, "s-", color="#8E5EA1"); a_.fill_between(bs, lo, hi, alpha=.18, color="#8E5EA1")
        a_.set_xscale("log")
        if logy and any(np.isfinite(v) and v > 0 for v in med): a_.set_yscale("log")
        a_.set_title(ttl, fontsize=10); a_.set_xlabel("β"); a_.grid(alpha=.3)
    fig.suptitle("U 型左右支机制（ν=.005）：聚焦单调↑，过大 β 出现回摆/饱和")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E1_mechanism.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def plot_bridge(bridges):
    fig, b = plt.subplots(figsize=(6.6, 5.2))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(bridges)))
    for c, (beta, (rw, ee)) in zip(cmap, sorted(bridges.items())):
        if rw.size >= 2:
            b.plot(rw, ee, ".", ms=3, color=c, alpha=.6, label=f"β={beta:g}")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("||r||_W（训练轨迹）"); b.set_ylabel("||e|| 相对 L2")
    b.set_title("分 β 桥接：同 β 内下降，跨 β 不成共带（loss 不可跨 β 比）")
    b.legend(fontsize=7, ncol=2); b.grid(alpha=.3, which="both")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E1_bridge.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def _q3(rows_sub, key):
    v = np.asarray([r[ key ] for r in rows_sub], float); v = v[np.isfinite(v)]
    if v.size == 0: return np.nan, np.nan, np.nan
    return np.median(v), np.quantile(v, .25), np.quantile(v, .75)


def plot_grad_partition(probe_rows):
    """机制 A 直接证据：同一 θ 跨 β，参数梯度能量比 η=‖∇θL_band‖²/‖∇θL_smooth‖² 随 β↑。"""
    nu = PROBE_NU
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    styles = [("gate", "#2E6E8E", "-", "gate 起点（同 θ 跨 β，最干净）"),
              ("mid", "#E8A24B", "o--", f"中段 @{MID_EP}（首种子）"),
              ("end", "#3a8a18", "-", "best 近解点")]
    for point, c, ls, lab in styles:
        sub = [r for r in probe_rows if r["nu"] == nu and r["point"] == point]
        if not sub: continue
        bs = sorted({r["beta"] for r in sub})
        med = [_q3([r for r in sub if r["beta"] == b], "eta")[ 0 ] for b in bs]
        ax[ 0 ].plot(bs, med, ls, color=c, label=lab)
        if point != "mid":
            lo = [_q3([r for r in sub if r["beta"] == b], "eta")[ 1 ] for b in bs]
            hi = [_q3([r for r in sub if r["beta"] == b], "eta")[ 2 ] for b in bs]
            ax[ 0 ].fill_between(bs, lo, hi, alpha=.15, color=c)
        cosmed = [_q3([r for r in sub if r["beta"] == b], "cos")[ 0 ] for b in bs]
        ax[ 1 ].plot(bs, cosmed, ls, color=c, label=lab)
    ax[ 0 ].set_xscale("log")
    if any(np.isfinite(v) and v > 0 for r in probe_rows for v in [r["eta"]]): ax[ 0 ].set_yscale("log")
    ax[ 0 ].set_xlabel("β"); ax[ 0 ].set_ylabel("η = ‖∇θL_激波‖² / ‖∇θL_光滑‖²")
    ax[ 0 ].set_title("机制A 直接证据：β 把参数梯度能量再分配到激波带"); ax[ 0 ].grid(alpha=.3); ax[ 0 ].legend(fontsize=8)
    ax[ 1 ].axhline(0, color="k", lw=1); ax[ 1 ].set_xscale("log")
    ax[ 1 ].set_xlabel("β"); ax[ 1 ].set_ylabel("cos(∇θL_激波, ∇θL_光滑)")
    ax[ 1 ].set_title("两区域参数梯度方向关系"); ax[ 1 ].grid(alpha=.3); ax[ 1 ].legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "fig_E1_grad_partition.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def plot_hessian(hess_rows):
    """U 型右支：近解点参数 Hessian 极端谱，β 过大时 λmin 走低/负曲率维数上升。"""
    if not hess_rows:
        print("[warn] 无 Hessian 数据，跳过 fig_E1_hessian"); return
    hs = sorted(hess_rows, key=lambda r: r["beta"]); bs = [r["beta"] for r in hs]
    fig, a1 = plt.subplots(figsize=(7.6, 4.8))
    a1.plot(bs, [r["lam_max"] for r in hs], "o-", color="#2E6E8E", label="λ_max")
    a1.plot(bs, [r["lam_min"] for r in hs], "s-", color="#EA6668", label="λ_min（<0=负曲率）")
    a1.axhline(0, color="k", lw=1); a1.set_xscale("log"); a1.set_xlabel("β")
    a1.set_ylabel("极端特征值"); a1.set_title("近解点参数 Hessian（右支负曲率检查，首种子）")
    a2 = a1.twinx()
    a2.plot(bs, [r["n_neg"] for r in hs], "^--", color="#8E5EA1", label="负曲率维数 n_neg（右轴）")
    a2.set_ylabel("n_neg", color="#8E5EA1")
    l1, t1 = a1.get_legend_handles_labels(); l2, t2 = a2.get_legend_handles_labels()
    a1.legend(l1 + l2, t1 + t2, fontsize=8, loc="best"); a1.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(OUT, "fig_E1_hessian.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


def main():
    rows, _agg, bridges, probe_rows, hess_rows = collect()
    plot_ucurve(rows); plot_beta_star(rows); plot_mechanism(rows); plot_bridge(bridges)
    plot_grad_partition(probe_rows); plot_hessian(hess_rows)
    print("\n[E1] 完成。产物在", OUT)


if __name__ == "__main__":
    main()
