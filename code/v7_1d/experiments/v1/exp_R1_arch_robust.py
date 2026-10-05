# -*- coding: utf-8 -*-
"""
实验 R1（V10.1）：架构/采样鲁棒性 —— 核心定性结论不依赖单一网络规模与配点数
--------------------------------------------------------------------------------
单因素(one-at-a-time)扫描，中心点 hidden=64 / layers=4 / n_pde=10000（主线配置）：
  宽度轴 width : {32, 64, 128}
  深度轴 depth : layers {3, 4, 6}（中心 4 已在宽度轴）
  配点轴 npde  : n_pde {2500, 10000, 40000}（中心 10000 已在宽度轴）
在两个代表 ν 上：.005（激波端，β*>1、机制 A 最强）、.1（近退化端）。
要验证的“结论不变性”（只看定性，不抠效应量，故种子降到 5、β 精简 {1,3,8}）：
  (i)  β* 始终落在同一 regime（不随宽/深/配点跳变）；
  (ii) Spearman(β, η_gate) 符号不变（机制 A 方向稳定）；
  (iii)好盆地内族排序 rational ≥ uniform ≥ down_inv 不反转（仅在最窄/最宽两个端点
        额外跑 uniform/down_inv 验证，避免全因子爆炸）；
  (iv) 最终 L2 同量级、且都进入平台（conv_epoch）。
不改公共层：进程内 patch r2s.HIDDEN/NLAYERS/N_PDE；Phase0 盆地自建【带架构后缀】的
隔离缓存（mc.enter_basin_cached 的 key 不含架构，直接复用会撞 state 形状，故这里自管）。
产出 results/v6/exp_R1_arch/：r1_runs.csv、r1_invariance.csv、3 张图。SMOKE：V8_SMOKE=1
"""
import os, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager

_HERE = Path(__file__).resolve(); _PKG = _HERE.parent.parent
for _p in (str(_PKG), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import run_two_stage_v6 as r2s
import v8_matrix_common as mc

for _fp in (r"C:\Windows\Fonts\msyh.ttc",):
    if os.path.exists(_fp):
        try: font_manager.fontManager.addfont(_fp)
        except Exception: pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = str(mc.RES_ROOT / "exp_R1_arch"); os.makedirs(OUT, exist_ok=True)
BASIN_CACHE = mc.RES_ROOT / "exp_R1_arch" / "_basin_cache"; os.makedirs(BASIN_CACHE, exist_ok=True)
r2s.OUT = OUT
AXIS_COLOR = {"width": "#4B86B4", "depth": "#E8A24B", "npde": "#7FB3D5"}

# (axis, level, hidden, layers, n_pde) —— 中心 (64,4,10000)
CONFIGS = [("width", "32", 32, 4, 10000), ("width", "64", 64, 4, 10000), ("width", "128", 128, 4, 10000),
           ("depth", "3", 64, 3, 10000), ("depth", "6", 64, 6, 10000),
           ("npde", "2500", 64, 4, 2500), ("npde", "40000", 64, 4, 40000)]
ORDER_CFG = {(32, 4, 10000), (128, 4, 10000)}     # 端点宽度上验证族排序
FAM_ORDER = ["rational", "uniform", "down_inv"]
NUS = [0.005, 0.1]; BETAS = [1.0, 3.0, 8.0]; SEEDS = mc.SEEDS_NEW
EP = 8000; PROBE_N = 2048
if mc.SMOKE:
    CONFIGS = [("width", "32", 32, 4, 10000), ("width", "64", 64, 4, 10000)]
    NUS = [0.1]; BETAS = [1.0, 5.0]; SEEDS = [mc.SEEDS_NEW[0]]; EP = 200; PROBE_N = 256
CONV_WIN, CONV_TOL = 5, 1e-3

def conv_epoch(dyn, win=CONV_WIN, tol=CONV_TOL):
    es = np.asarray(dyn.get("eval_step", []), float); l2 = np.asarray(dyn.get("L2_traj", []), float)
    if l2.size < win + 1: return (int(es[-1]) if es.size else -1, False)
    for i in range(0, l2.size - win + 1):
        w = l2[i:i + win]
        if (np.max(w) - np.min(w)) / (abs(np.median(w)) + 1e-12) < tol: return int(es[i]), True
    return int(es[-1]), False

def _spearman(x, y):
    try:
        from scipy.stats import spearmanr
        x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 3 and np.unique(x[m]).size >= 2 and np.unique(y[m]).size >= 2:
            r = spearmanr(x[m], y[m]); return float(r.statistic if hasattr(r, "statistic") else r[0])
    except Exception:
        pass
    return np.nan

def get_basin(h, nl, nnp, nu, seed, xt, ut):
    # 进程内对齐全局规模（base_cfg/run_phase0 实时读这些模块全局）
    r2s.HIDDEN, r2s.NLAYERS, r2s.N_PDE = h, nl, nnp
    r2s.MULTISTART_K = 1 if mc.SMOKE else 4
    r2s.SELECT_BY = "truth"
    r2s.SIG_HI = r2s.SIGMA_HI_CFG = 15.0; r2s.SIGMA_LO, r2s.SIGMA_T = 5.0, 1200
    r2s.PHASE0_EPOCHS = 60 if mc.SMOKE else 2000
    tag = f"nu{nu}_s{seed}_h{h}_l{nl}_n{nnp}.pt"; path = BASIN_CACHE / tag
    if path.exists():
        d = torch.load(path, map_location=mc.DEVICE)
        print(f"[R1 basin hit] {tag} L2={d['l2']:.3e}"); return d
    b = r2s.run_phase0(nu, seed, xt, ut); torch.save(b, path); return b

def collect():
    rows = []
    for axis, lvl, h, nl, nnp in CONFIGS:
        for nu in NUS:
            xt, ut = r2s.build_test(nu)
            xp = torch.linspace(-1, 1, PROBE_N, device=mc.DEVICE).view(-1, 1)
            fams = FAM_ORDER if (not mc.SMOKE and (h, nl, nnp) in ORDER_CFG) else ["rational"]
            for seed in SEEDS:
                basin = get_basin(h, nl, nnp, nu, seed, xt, ut)
                for family in fams:
                    for beta in BETAS:
                        m, dyn, model, crit, tr = mc.train_phase1_family(
                            basin, nu, beta, seed, xt, ut, family=family, epochs=EP,
                            tag=f"R1_{axis}{lvl}", grad_probe=(xp, nu))
                        ce, isc = conv_epoch(dyn); gp = tr.grad_probe
                        rows.append(dict(axis=axis, level=lvl, hidden=h, layers=nl, n_pde=nnp,
                                         nu=nu, seed=seed, family=family, beta=float(beta),
                                         phase0_L2=basin["l2"], L2=m.get("L2_Error", np.nan),
                                         Linf=m.get("L_inf_Error", np.nan), l2_band=m.get("l2_band", np.nan),
                                         eta_gate=(gp["gate"] or {}).get("eta", np.nan),
                                         best_epoch=int(getattr(tr, "best_epoch", -1)),
                                         conv_epoch=ce, converged=int(isc)))
                        print(f"[R1 {axis}={lvl}] ν{nu} s{seed} {family} β{beta:g}: L2={rows[-1]['L2']:.3e}")
    mc.write_csv(os.path.join(OUT, "r1_runs.csv"), rows)
    return rows

def cfg_key(r): return (r["axis"], r["level"], r["hidden"], r["layers"], r["n_pde"])
def summarize(rows):
    inv = []
    seen = []
    for r in rows:
        k = cfg_key(r)
        if k not in seen: seen.append(k)
    for (axis, lvl, h, nl, nnp) in seen:
        for nu in NUS:
            for family in sorted({r["family"] for r in rows if cfg_key(r) == (axis, lvl, h, nl, nnp)}):
                sub = [r for r in rows if cfg_key(r) == (axis, lvl, h, nl, nnp)
                       and r["nu"] == nu and r["family"] == family]
                if not sub: continue
                bstars, spear = [], []
                l2_at = {b: np.nan for b in BETAS}
                for seed in sorted({r["seed"] for r in sub}):
                    z = sorted([(r["beta"], r["L2"]) for r in sub if r["seed"] == seed])
                    zf = [(b, v) for b, v in z if np.isfinite(v)]
                    if zf: bstars.append(min(zf, key=lambda t: t[1])[0])
                    sp = _spearman([b for b, _ in z],
                                   [r["eta_gate"] for r in sub if r["seed"] == seed])
                    if np.isfinite(sp): spear.append(sp)
                for b in BETAS:
                    v = np.asarray([r["L2"] for r in sub if r["beta"] == b], float)
                    l2_at[b] = float(np.nanmedian(v))
                inv.append(dict(axis=axis, level=lvl, hidden=h, layers=nl, n_pde=nnp, nu=nu, family=family,
                                n=len(bstars), beta_star_med=np.median(bstars) if bstars else np.nan,
                                eta_beta_spearman=np.median(spear) if spear else np.nan,
                                L2_b1=l2_at.get(1.0, np.nan), L2_b3=l2_at.get(3.0, np.nan),
                                L2_b8=l2_at.get(8.0, np.nan),
                                conv_epoch_med=np.median([r["conv_epoch"] for r in sub
                                                          if r.get("conv_epoch", -1) >= 0]),
                                frac_converged=np.mean([r.get("converged", 0) for r in sub])))
    mc.write_csv(os.path.join(OUT, "r1_invariance.csv"), inv)
    return inv

def _ordered_levels(rows, axis):
    out = []
    for c in CONFIGS:
        if c[0] == axis and (c[2], c[3], c[4]) in {(cfg_key(r)[2], cfg_key(r)[3], cfg_key(r)[4]) for r in rows}:
            if c[1] not in out: out.append(c[1])
    return out

def plot_beta_star(rows):
    fig, ax = plt.subplots(1, max(1, len(NUS)), figsize=(5.5 * max(1, len(NUS)), 4.4), squeeze=False)
    for j, nu in enumerate(NUS):
        a = ax[0][j]
        for axis in ("width", "depth", "npde"):
            lv = _ordered_levels([r for r in rows if r["nu"] == nu and r["family"] == "rational"], axis)
            xs, ym = [], []
            for lvl in lv:
                sub = [r for r in rows if r["nu"] == nu and r["family"] == "rational"
                       and r["axis"] == axis and r["level"] == lvl]
                if not sub: continue
                bs = [min([(r["beta"], r["L2"]) for r in sub if r["seed"] == s and np.isfinite(r["L2"])]
                       , key=lambda t: t[1])[0] for s in {r["seed"] for r in sub}
                      if any(r["seed"] == s and np.isfinite(r["L2"]) for r in sub)]
                xs.append(f"{axis[:1]}{lvl}"); ym.append(np.median(bs))
            a.plot(xs, ym, "o-", color=AXIS_COLOR[axis], label=axis)
        a.axhline(1, color="k", ls=":", lw=1); a.set_xlabel("配置（单因素）"); a.set_ylabel("β* 中位")
        a.set_title(f"ν={nu}：β* 跨架构稳定"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "fig_R1_beta_star.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

def plot_L2(rows):
    fig, ax = plt.subplots(1, max(1, len(NUS)), figsize=(5.5 * max(1, len(NUS)), 4.4), squeeze=False)
    for j, nu in enumerate(NUS):
        a = ax[0][j]
        for c in CONFIGS:
            axis, lvl, h, nl, nnp = c
            sub = [r for r in rows if r["nu"] == nu and r["family"] == "rational"
                   and r["hidden"] == h and r["layers"] == nl and r["n_pde"] == nnp]
            if not sub: continue
            xs, ym = [], []
            for b in BETAS:
                v = np.asarray([r["L2"] for r in sub if r["beta"] == b], float); v = v[np.isfinite(v)]
                xs.append(b); ym.append(np.median(v) if v.size else np.nan)
            is_center = (h, nl, nnp) == (64, 4, 10000)
            a.plot(xs, ym, "-o", color="k" if is_center else AXIS_COLOR[axis],
                   lw=2 if is_center else 1, alpha=1 if is_center else .65,
                   label="中心64/4/1e4" if is_center else None)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("β"); a.set_ylabel("L2 中位")
        a.set_title(f"ν={nu}：L2-β 跨架构（黑线=中心）"); a.grid(alpha=.3); a.legend(fontsize=8)
    # 非中心配置用统一图例补充
    handles = [plt.Line2D([0], [0], color=AXIS_COLOR[a], lw=2, label=a) for a in ("width", "depth", "npde")]
    ax[0][-1].legend(handles=handles, fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "fig_R1_L2_beta.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

def plot_family_order(rows):
    sub = [r for r in rows if r["family"] in FAM_ORDER and (r["hidden"], r["layers"], r["n_pde"]) in ORDER_CFG]
    if not sub: print("[R1] 无族排序数据（SMOKE 跳过）"); return
    fig, ax = plt.subplots(1, max(1, len(NUS)), figsize=(5.5 * max(1, len(NUS)), 4.4), squeeze=False)
    fcolor = {"rational": "#E8A24B", "uniform": "#9A9A9A", "down_inv": "#4B86B4"}
    for j, nu in enumerate(NUS):
        a = ax[0][j]; groups, labels = [], []
        for h in (32, 128):
            for b in BETAS:
                labels.append(f"h{h}\nβ{b:g}"); groups.append((h, b))
        x = np.arange(len(groups)); w = .25
        for k, fam in enumerate(FAM_ORDER):
            vals = []
            for h, b in groups:
                v = np.asarray([r["L2"] for r in sub if r["hidden"] == h and r["beta"] == b
                                and r["family"] == fam and r["nu"] == nu], float); v = v[np.isfinite(v)]
                vals.append(np.median(v) if v.size else np.nan)
            a.bar(x + (k - 1) * w, vals, w, color=fcolor[fam], label=fam)
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=8); a.set_yscale("log")
        a.set_ylabel("L2 中位"); a.set_title(f"ν={nu}：端点宽度下族排序不反转"); a.grid(alpha=.3, axis="y")
        a.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "fig_R1_family_order.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

def main():
    rows = collect(); inv = summarize(rows)
    plot_beta_star(rows); plot_L2(rows); plot_family_order(rows)
    print("\n[R1] axis level  ν   fam      β*med  Spearman  conv_med  fracConv")
    for s in inv:
        print(f"  {s['axis']:<5} {s['level']:<5} {s['nu']:<6} {s['family']:<8} "
              f"{s['beta_star_med']:>4g}  {s['eta_beta_spearman']:+.2f}     "
              f"{s['conv_epoch_med']:>6.0f}   {s['frac_converged']:.2f}")
    print("\n[R1] 完成。产物在", OUT)

if __name__ == "__main__":
    main()
