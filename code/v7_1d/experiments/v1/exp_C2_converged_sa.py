# -*- coding: utf-8 -*-
"""
================================================================================
实验 C2（V8 第 9 章，C1 的关键判别实验）：谱对齐 [SA] 是不是“临近真解才成立”的渐近性质？
--------------------------------------------------------------------------------
动机（C1 的负结果）：
  C1 在【Gate 刚进盆地的冻结点 θ_gate】测残差空间谱，得到 E_off≈0.08~0.14 平台（斜率
  +0.12，不是 n^-1/2）、谱形 a≈0（R²≈.15）、μ 谱不被压平、κ_new 不沿 1/β、命题 6.2 通过率 0。
  但 [SA] 的原始前提是“已在正确盆地（临近真解）”，Gate 点只是粗解。本实验区分两种可能：
    (A) [SA] 是渐近性质：网络沿同一盆地从 θ_gate 逼近真解时，E_off 单调下降、a 转正、
        μ 被压平、improve→1/β —— 则第 6 章改写为“临近真解的局部结论”；
    (B) 结构性否定：即便充分收敛，E_off 仍是 ~0.1 平台 —— 则 [SA]/定理 6.1 在本问题彻底降格。

设计（单变量 = 离真解多近；其余全部与 C1 对齐，便于直接对照）：
  1) Phase0 复用 run_two_stage_v6（σ课程+multi-start 进同一好盆地，选 truth 最好）。
  2) 从 θ_gate 出发，用【β=1 无加权】（w≡1，避免“加权帮你收敛”的循环论证）、固定 σ=σ_end、
     关闭 Gate/best 回拉/空间 EMA，沿一条真实 Adam 轨迹续训，在若干累计步存快照
     θ_0(=gate),θ_1,...,θ_M(=充分收敛)，记录每点 L2。
  3) 对每个快照：用【该点当前残差现场冻结对比度场 t^(s)(x)】（关 EMA 时 freeze 走精确当前
     残差），再与 C1 完全相同地扫 n∈NS、β∈BETAS，调用 residual_spectral.spectral_alignment。
     —— 冻结场随快照更新、测量口径不变；K_r 只由该快照网络决定，W 只随 β 变。
  4) stage0(=gate) 应复现 C1（内部一致性校验）。

产物 results/v6/exp_C2_converged_sa/：
  c2_runs.csv                  每 (seed,stage,L2_at,n,β) 的全部谱标量（列同 C1 + stage/L2_at）
  c2_stage_info.csv            每 (seed,stage) 累计 epoch 与 L2（收敛轨迹）
  c2_eoff_slope_by_stage.csv   β=15 下每 stage 的 E_off-n log-log 斜率+bootstrap CI（看是否转向 -.5）
  fig_C2_eoff_trajectory.png   E_off 随训练/L2/n 的三张主判据图
  fig_C2_spectral.png          谱形 a、μ 全谱、improve_ratio 随 stage 变化
运行：python exp_C2_converged_sa.py（设 V8_SMOKE=1 为极小配置自检）
================================================================================
"""
import sys as _sys, os as _os, copy
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve(); _PKG = _HERE.parent.parent; _EXP = _HERE.parent
for _p in (str(_PKG), str(_EXP)):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import csv
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_two_stage_v6 as r2s
import v8_basin as vb
from trainers.pinn_trainer import PINNTrainer
from utils import residual_spectral as rs

OUT = str(_PKG.parent / "results" / "v6" / "exp_C2_converged_sa"); _os.makedirs(OUT, exist_ok=True)
vb.set_out_dir(OUT)
DEVICE = vb.DEVICE

# ----------------------------- 配置（跟实验走） -----------------------------
SMOKE = 0
NUS = [0.005]
SEEDS = [42, 123, 2024, 7] if not SMOKE else [42]
NS = [64, 128, 256, 512] if not SMOKE else [32, 48]
BETAS = [1.0, 5.0, 15.0, 40.0] if not SMOKE else [1.0, 15.0]
# 续训快照（累计 Adam 步）；stage0=0 即 Gate 点（应复现 C1），末点为充分收敛
SNAP_EPOCHS = [0, 1500, 3000, 6000, 9000, 12000] if not SMOKE else [0, 120, 240]
CONVERGE_BETA = 1.0      # 续训路径用无加权，避免“加权帮你收敛”的循环；若 β=1 无法收敛可改 15 并重述口径
CONVERGE_LR = 5e-4
FOCUS_BETA = 15.0        # 主判据图固定的 β（与 C1 斜率表一致，避免多 β 拥挤）
CONVERGED_L2 = 5e-2      # 末快照 L2 低于此才算“充分收敛点”，否则在 stage_info 标注未收敛
r2s.MULTISTART_K = 1 if SMOKE else 4
r2s.SELECT_BY = "truth"
if SMOKE:
    r2s.PHASE0_EPOCHS = 300
# ---------------------------------------------------------------------------


# ========== 1. 沿 β=1 真实轨迹续训，取快照（复用 trainer，不复制训练内循环） ==========
def drive_snapshots(basin, nu, xt, ut):
    cfg = r2s.base_cfg(nu, f"c2conv_s{basin['seed']}")
    cfg.update(fourier_scale=float(basin["sigma_end"]), spatial_ema_decay=0.0,
               lr=CONVERGE_LR, use_gate=False, use_best_ckpt=False,
               beta_schedule="const", loss_beta=CONVERGE_BETA, beta_init=CONVERGE_BETA)
    model, pde, crit, opt, samp, vis, cfg = r2s.build_all(cfg, CONVERGE_LR, CONVERGE_BETA)
    model.load_state_dict(basin["state"]); model.set_sigma(basin["sigma_end"])
    trainer = PINNTrainer(model, pde, crit, opt, samp, vis, DEVICE, cfg,
                          {"x": xt, "u_true": ut})
    snaps, cur = [], 0
    for s, ep in enumerate(SNAP_EPOCHS):
        if ep > cur:   # 分段续训：σ/β 均与 epoch 无关，optimizer/参数连续，best 回拉已关
            trainer.train(cfg, {"x": xt, "u_true": ut}, ep - cur, 0, adaptive_freq=500)
            cur = ep
        l2 = r2s.quick_l2(model, xt, ut)
        snaps.append(dict(stage=s, epoch=ep, L2=l2,
                          state=copy.deepcopy(model.state_dict())))
        print(f"  [C2-snap] stage{s} epoch{ep} L2={l2:.3e}")
    trainer.log_file.close()
    return snaps


# ========== 2. 测量器：一次 build，逐快照装载 + 现场冻结当前残差场 ==========
def build_measurer(nu):
    # 测量器无需在构建时知道 σ：每个快照装载后都会 set_sigma(sigma_end) 覆盖运行时带宽
    cfg = r2s.base_cfg(nu, "c2meas")
    cfg.update(spatial_ema_decay=0.0)
    model, pde, crit, opt, samp, vis, cfg = r2s.build_all(cfg, 1e-3, 1.0, "adaptive")
    return model, pde, crit


def measure_seed(basin, nu, seed, xt, ut, mmodel, mpde, mcrit, rep_spectra):
    snaps = drive_snapshots(basin, nu, xt, ut)
    rows, stage_info = [], []
    for snap in snaps:
        s, l2s = snap["stage"], snap["L2"]
        stage_info.append(dict(nu=nu, seed=seed, stage=s, epoch=snap["epoch"],
                               L2=l2s, converged=bool(l2s < CONVERGED_L2)))
        mmodel.load_state_dict(snap["state"]); mmodel.set_sigma(basin["sigma_end"]); mmodel.eval()
        # 关键：用【该快照当前残差】现场冻结对比度场（EMA 已关 → freeze 走精确当前残差）
        lo, hi = mcrit.freeze_residual_field(mmodel, mpde)
        for n in NS:
            x = torch.linspace(-1, 1, n, device=DEVICE).view(-1, 1)
            for beta in BETAS:
                sc, arr = rs.spectral_alignment(mmodel, mpde, mcrit, x, beta=float(beta))
                row = dict(nu=nu, seed=seed, stage=s, snap_epoch=snap["epoch"], L2_at=l2s)
                row.update(sc); rows.append(row)
                if seed == SEEDS[ 0 ] and n == max(NS):
                    rep_spectra[(seed, s, float(beta))] = arr
                if n == max(NS):
                    print(f"[C2] seed{seed} stage{s}(L2={l2s:.1e}) n={n} β={beta:g}: "
                          f"E_off={sc['E_off']:.3e} κ {sc['kappa_orig']:.2e}->"
                          f"{sc['kappa_new']:.2e} improve={sc['improve_ratio']:.3g} SA={sc['SA_holds']}")
        del snap["state"]
    return rows, stage_info


# ========== 聚合小工具 ==========
def _pass(fixed, r):
    return all(r[ k ] == v for k, v in fixed.items())


def group_med(rows, xkey, ykey, fixed):
    xs = sorted({r[ xkey ] for r in rows if _pass(fixed, r)})
    ys = [float(np.median([r[ ykey ] for r in rows if r[ xkey ] == x and _pass(fixed, r)]))
          for x in xs]
    return xs, ys


def collect(rows, xkey, ykey, fixed):
    gx, gy = [], []
    for r in rows:
        if _pass(fixed, r):
            gx.append(r[ xkey ]); gy.append(r[ ykey ])
    return np.asarray(gx, float), np.asarray(gy, float)


def stage_l2_med(stage_info):
    out = {}
    for s in sorted({z["stage"] for z in stage_info}):
        out[ s ] = float(np.median([z["L2"] for z in stage_info if z["stage"] == s]))
    return out


# ========== 3. 出图 ==========
def analyze(rows, stage_info, rep_spectra):
    sl2 = stage_l2_med(stage_info)
    stages = sorted({r["stage"] for r in rows})
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(stages)))

    # ---------- 图 1：E_off 的三张主判据 ----------
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    # (a) E_off 随训练 stage（固定 β=FOCUS_BETA，分 n）
    for n in NS:
        xs, ys = group_med(rows, "stage", "E_off", {"beta": FOCUS_BETA, "n": n})
        ax[ 0 ].plot(xs, ys, "o-", label=f"n={n}")
    ax[ 0 ].axhline(0.1, ls="--", c="gray", lw=1, label="C1 gate 平台 ~0.1")
    ax[ 0 ].set_yscale("log"); ax[ 0 ].set_xlabel("训练阶段 stage（0=Gate → 充分收敛）")
    ax[ 0 ].set_ylabel("E_off（中位数）"); ax[ 0 ].set_title(f"① E_off 随逼近真解变化（β={FOCUS_BETA:g}）")
    ax[ 0 ].legend(fontsize=8); ax[ 0 ].grid(alpha=.3, which="both")

    # (b) E_off vs L2（log-log，沿 stage 连线，分 n）—— 越左越接近真解
    for n in NS:
        lx, ly = group_med(rows, "L2_at", "E_off", {"beta": FOCUS_BETA, "n": n})
        order = np.argsort(lx)[ ::-1 ]   # 从粗解(L2大)到精解(L2小)
        lx, ly = np.asarray(lx)[ order ], np.asarray(ly)[ order ]
        ax[ 1 ].plot(lx, ly, "o-", label=f"n={n}")
    ax[ 1 ].set_xscale("log"); ax[ 1 ].set_yscale("log"); ax[ 1 ].invert_xaxis()
    ax[ 1 ].tick_params(axis="x", labelrotation=30, labelsize=8)
    ax[ 1 ].set_xlabel("该快照 L2 误差（越左越接近真解）"); ax[ 1 ].set_ylabel("E_off")
    ax[ 1 ].set_title("② E_off–L2：渐近应对齐应向左下方走"); ax[ 1 ].legend(fontsize=8)
    ax[ 1 ].grid(alpha=.3, which="both")

    # (c) E_off vs n，分 stage（看斜率是否从 +.12 转向 -.5）
    slope_rows = []
    for si, s in enumerate(stages):
        xs, ys = group_med(rows, "n", "E_off", {"beta": FOCUS_BETA, "stage": s})
        gx, gy = collect(rows, "n", "E_off", {"beta": FOCUS_BETA, "stage": s})
        fit = rs.fit_loglog_slope(xs, ys)
        lo, hi = rs.bootstrap_slope_ci(gx, gy, n_boot=300)
        slope_rows.append(dict(stage=s, L2_med=sl2.get(s, float("nan")),
                               slope=fit["slope"], ci_lo=lo, ci_hi=hi, r2=fit["r2"]))
        ax[ 2 ].plot(xs, [max(y, 1e-12) for y in ys], "o-", color=cmap[ si ],
                     label=f"stage{s} L2={sl2.get(s, float('nan')):.1e} s={fit['slope']:.2f}")
    ref = np.asarray(NS, float)
    ax[ 2 ].plot(ref, 0.8 * ref.astype(float) ** -0.5, "k--", alpha=.6, label="n^-1/2 参考")
    ax[ 2 ].set_xscale("log"); ax[ 2 ].set_yscale("log")
    ax[ 2 ].set_xlabel("配点数 n"); ax[ 2 ].set_ylabel("E_off")
    ax[ 2 ].set_title("③ E_off–n：斜率是否由正转 −.5"); ax[ 2 ].legend(fontsize=7)
    ax[ 2 ].grid(alpha=.3, which="both")
    fig.tight_layout(); p1 = _os.path.join(OUT, "fig_C2_eoff_trajectory.png")
    fig.savefig(p1, dpi=140); plt.close(fig)

    # ---------- 图 2：谱形 a / μ 全谱 / improve ----------
    fig, bx = plt.subplots(1, 3, figsize=(15.5, 4.3))
    # (a) 谱形 a 随 stage（代表 seed，分 β）
    for beta in BETAS:
        if abs(beta - 1.0) < 1e-9:
            continue
        av, xs = [], []
        for s in stages:
            arr = rep_spectra.get((SEEDS[ 0 ], s, float(beta)))
            if arr is None:
                continue
            lam, what = arr["lam"], arr["what"]; msk = lam > 1e-8 * lam.max()
            est = rs.estimate_spectral_a(lam[ msk ], np.clip(what[ msk ], 1e-6, None), n_boot=200)
            av.append(est["a"]); xs.append(s)
        bx[ 0 ].plot(xs, av, "s-", label=f"β={beta:g}")
    bx[ 0 ].axhline(0.0, ls="--", c="gray", lw=1)
    bx[ 0 ].set_xlabel("stage"); bx[ 0 ].set_ylabel("谱形指数 a（>0 才支持幂律）")
    bx[ 0 ].set_title("④ a 是否随收敛转正"); bx[ 0 ].legend(fontsize=8); bx[ 0 ].grid(alpha=.3)

    # (b) μ 全谱分 stage（代表 seed，β=FOCUS）
    for si, s in enumerate(stages):
        arr = rep_spectra.get((SEEDS[ 0 ], s, FOCUS_BETA))
        if arr is None:
            continue
        mu = np.sort(arr["mu"])[ ::-1 ]
        bx[ 1 ].plot(np.arange(1, mu.size + 1), mu / mu.max(), color=cmap[ si ], label=f"stage{s}")
    bx[ 1 ].set_yscale("log"); bx[ 1 ].set_xlabel("模态序号 k（降序）")
    bx[ 1 ].set_ylabel("归一化 μ_k"); bx[ 1 ].set_title(f"⑤ μ 谱是否随收敛压平（β={FOCUS_BETA:g}）")
    bx[ 1 ].legend(fontsize=8); bx[ 1 ].grid(alpha=.3, which="both")

    # (c) improve_ratio 随 stage（分 n，β=FOCUS）
    for n in NS:
        xs, ys = group_med(rows, "stage", "improve_ratio", {"beta": FOCUS_BETA, "n": n})
        bx[ 2 ].plot(xs, ys, "o-", label=f"n={n}")
    bx[ 2 ].axhline(1.0 / FOCUS_BETA, ls=":", c="k", lw=1.2, label=f"1/β={1.0 / FOCUS_BETA:.3f}")
    bx[ 2 ].axhline(1.0, ls="--", c="gray", lw=1)
    bx[ 2 ].set_yscale("log"); bx[ 2 ].set_xlabel("stage")
    bx[ 2 ].set_ylabel("κ_new/κ_orig"); bx[ 2 ].set_title("⑥ 条件数改善是否随收敛趋近 1/β")
    bx[ 2 ].legend(fontsize=8); bx[ 2 ].grid(alpha=.3, which="both")
    fig.tight_layout(); p2 = _os.path.join(OUT, "fig_C2_spectral.png")
    fig.savefig(p2, dpi=140); plt.close(fig)

    _write("c2_eoff_slope_by_stage.csv", slope_rows)
    print("saved", p1, "\nsaved", p2)
    print("\n=== E_off–n 斜率随 stage（期望从 +.12 转向 −.5）===")
    for z in slope_rows:
        print(f"  stage{z['stage']} L2={z['L2_med']:.2e}: slope={z['slope']:.3f} "
              f"CI[{z['ci_lo']:.3f},{z['ci_hi']:.3f}] r2={z['r2']:.2f}")


def _write(name, data):
    if not data:
        return
    with open(_os.path.join(OUT, name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(data[ 0 ].keys())); w.writeheader(); w.writerows(data)


def main():
    rows_all, stage_all, rep_spectra = [], [], {}
    for nu in NUS:
        xt, ut = r2s.build_test(nu)
        mmodel, mpde, mcrit = build_measurer(nu)
        for seed in SEEDS:
            basin = r2s.run_phase0(nu, seed, xt, ut)
            r, si = measure_seed(basin, nu, seed, xt, ut, mmodel, mpde, mcrit, rep_spectra)
            rows_all.extend(r); stage_all.extend(si)
    _write("c2_runs.csv", rows_all)
    _write("c2_stage_info.csv", stage_all)
    analyze(rows_all, stage_all, rep_spectra)
    print("\n完成。产物在", OUT)


if __name__ == "__main__":
    main()
