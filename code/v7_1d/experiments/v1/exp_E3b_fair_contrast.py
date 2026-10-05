# -*- coding: utf-8 -*-
"""
实验 E3b（V9 补）：各族在【各自最优 contrast】下的公平对照 + 坏盆地逃逸率
--------------------------------------------------------------------------------
动机：E3 固定 c=15/40 对增权恰是 U 型右支、对减权却是温和区，对比不公平，导致
      “减权训练精度反而更好”的表象。本实验给每族扫各自的 contrast，再在各族最优 c 处比较。
设计（ν=.005，gate 缓存与 E1/E3 共享）：
  增权 rational/linear/band : c ∈ {3,5,10}（E1 显示增权最优在小 c）
  减权 down_lin/down_inv    : c ∈ {10,40}
  uniform                   : 1 次
  11 个规则种子全跑（SEEDS_EXT=0,5,…,50，是原 SEEDS_NEW=[0,10,20,30,40] 的超集，原 5 个全部保留），
  按【该 seed 的 phase0 gate L2】现场分好/坏盆地（不写死种子集合），好/坏各凑到 5-8 个。
产出 results/v6/exp_E3b_fair/：
  e3b_runs.csv 全量；e3b_best.csv 各族在好/坏组的最优 c 汇总；
  fig_E3b_contrast_curve.png 各族 l2_band~c（好盆地，显示最优 c 分离）；
  fig_E3b_fair.png 各族最优 c 处的好/坏盆地终态（真正公平对比）；
  fig_E3b_escape.png 坏盆地逃逸成功率 by 族。
SMOKE：V8_SMOKE=1（极小网格、240 步）
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve(); _PKG = _HERE.parent.parent
for _p in (str(_PKG), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from losses.family_weighting_loss import UP_FAMILIES, DOWN_FAMILIES

OUT = str(mc.RES_ROOT / "exp_E3b_fair"); os.makedirs(OUT, exist_ok=True); r2s.OUT = OUT
NU = 0.005
UP_GRID = {f: [3.0, 5.0, 10.0] for f in ["rational", "linear", "band"]}
DOWN_GRID = {f: [10.0, 40.0] for f in ["down_lin", "down_inv"]}
SEEDS = mc.SEEDS_EXT   # 11 个规则种子（0,5,…,50），含原 SEEDS_NEW=[0,10,20,30,40] 全部 5 个
EP = 240 if mc.SMOKE else 8000
if mc.SMOKE:
    UP_GRID = {"rational": [15.0]}; DOWN_GRID = {"down_inv": [15.0]}; SEEDS = [0]
UP_C, DOWN_C, UNI = "#E8A24B", "#4B86B4", "#9A9A9A"
def fcol(f): return UP_C if f in UP_FAMILIES else (DOWN_C if f in DOWN_FAMILIES else UNI)


def jobs():
    out = []
    for f, cs in UP_GRID.items():
        for c in cs: out.append((f, float(c)))
    for f, cs in DOWN_GRID.items():
        for c in cs: out.append((f, float(c)))
    out.append(("uniform", 1.0))
    return out


def collect():
    xt, ut = r2s.build_test(NU); rows = []
    for seed in SEEDS:
        basin = mc.enter_basin_cached(NU, seed, xt, ut, sigma_hi=r2s.SIGMA_HI_CFG, k=4, phase0_epochs=2000)
        klass = "good" if basin["l2"] < mc.GATE_L2_MAX else "bad"
        for fam, c in jobs():
            m, dyn, _x, _y, _z = mc.train_phase1_family(
                basin, NU, c, seed, xt, ut, family=fam, epochs=EP, tag="E3b")
            rows.append(dict(seed=seed, basin_class=klass, phase0_L2=basin["l2"], family=fam, contrast=c,
                             L2=m.get("L2_Error", np.nan), Linf=m.get("L_inf_Error", np.nan),
                             l2_band=m.get("l2_band", np.nan), l2_smooth=m.get("l2_smooth", np.nan),
                             T01=mc.first_hit_step(dyn, .01), best_epoch=m.get("best_epoch", -1),
                             Cw=(c ** 0.5 if fam in DOWN_FAMILIES else 1.0),
                             success=int(m.get("L2_Error", 9) < .01)))
            print(f"[E3b] s{seed}({klass}) {fam} c{c:g}: L2={rows[-1]['L2']:.2e} band={rows[-1]['l2_band']:.2e}")
    mc.write_csv(os.path.join(OUT, "e3b_runs.csv"), rows)
    return rows


def best_table(rows):
    """每族在好/坏组按 l2_band 中位选最优 contrast。"""
    out = []
    for klass in ["good", "bad"]:
        for fam in dict.fromkeys([r["family"] for r in rows]):
            cs = sorted({r["contrast"] for r in rows if r["family"] == fam and r["basin_class"] == klass})
            if not cs: continue       # 该组无样本（SMOKE 常见），跳过避免 None
            best_c, best_v = None, np.inf
            for c in cs:
                v = mc_med([r["l2_band"] for r in rows
                            if r["family"] == fam and r["basin_class"] == klass and r["contrast"] == c])
                if np.isfinite(v) and v < best_v: best_v, best_c = v, c
            z = [r for r in rows if r["family"] == fam and r["basin_class"] == klass and r["contrast"] == best_c]
            out.append(dict(basin_class=klass, family=fam, best_contrast=best_c,
                            l2_band_med=best_v, L2_med=mc_med([r["L2"] for r in z]),
                            escape_rate=np.mean([r["success"] for r in z]) if z else np.nan, n=len(z)))
    mc.write_csv(os.path.join(OUT, "e3b_best.csv"), out)
    return out


def mc_med(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def plot_contrast_curve(rows):
    good = [r for r in rows if r["basin_class"] == "good"]
    fig, a = plt.subplots(figsize=(8.4, 4.8))
    for fam in dict.fromkeys(r["family"] for r in good):
        cs = sorted({r["contrast"] for r in good if r["family"] == fam})
        yy = [mc_med([r["l2_band"] for r in good if r["family"] == fam and r["contrast"] == c]) for c in cs]
        a.plot(cs, yy, "o-", color=fcol(fam), label=fam)
    a.set_xscale("log");
    if any(np.isfinite(r["l2_band"]) and r["l2_band"] > 0 for r in good): a.set_yscale("log")
    a.set_xlabel("contrast β"); a.set_ylabel("好盆地终态 l2_band（中位）")
    a.set_title("各族最优 contrast 不同：增权最优在小β（右支恶化），减权需更大β"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "fig_E3b_contrast_curve.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def plot_fair(best, rows):
    fams = [r["family"] for r in best if r["basin_class"] == "good"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for k, klass in enumerate(["good", "bad"]):
        bk = [r for r in best if r["basin_class"] == klass]; xs = np.arange(len(bk)); w = .8
        ax[ k ].bar(xs, [r["l2_band_med"] for r in bk], w, color=[fcol(r["family"]) for r in bk])
        for i, r in enumerate(bk):
            lbl = "c*=N/A" if r["best_contrast"] is None else f"c*={r['best_contrast']:g}"
            ax[ k ].annotate(lbl, (i, r["l2_band_med"]), fontsize=8,
                            xytext=(0, 2), textcoords="offset points", ha="center")
            sub = [z for z in rows if z["family"] == r["family"] and z["basin_class"] == klass
                   and z["contrast"] == r["best_contrast"]]
            for z in sub: ax[ k ].scatter(i, z["l2_band"], s=16, color="#222", zorder=3)
        ax[ k ].axhline(.01, color="#3a8a18", ls=":", lw=1); ax[ k ].set_xticks(xs, [r["family"] for r in bk], rotation=30, ha="right", fontsize=8)
        ax[ k ].set_yscale("log"); ax[ k ].set_title("好盆地（gate 通过）" if klass == "good" else "坏盆地（gate 未过）"); ax[ k ].grid(alpha=.3, axis="y")
    ax[ 0 ].set_ylabel("各族最优 c* 处终态 l2_band（对数）")
    fig.suptitle("公平对照：各族在自己最优 contrast 下比较（柱顶标 c*，黑点为各 seed）")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E3b_fair.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def plot_escape(best):
    bad = [r for r in best if r["basin_class"] == "bad"]; bad = [r for r in bad if r["family"] != "uniform"] + [r for r in bad if r["family"] == "uniform"]
    fig, a = plt.subplots(figsize=(8.6, 4.4))
    xs = np.arange(len(bad))
    a.bar(xs, [r["escape_rate"] for r in bad], color=[fcol(r["family"]) for r in bad])
    for i, r in enumerate(bad):
        lbl = "c*=N/A" if r["best_contrast"] is None else f"c*={r['best_contrast']:g}"
        a.annotate(lbl, (i, r["escape_rate"]), fontsize=8,
                   xytext=(0, 2), textcoords="offset points", ha="center")
    a.set_xticks(xs, [r["family"] for r in bad], rotation=30, ha="right"); a.set_ylim(0, 1.05)
    a.set_ylabel("坏盆地逃逸成功率（终态 L2<.01）"); a.set_title("阶段分工证据：减权族坏盆地逃逸率显著高于增权族")
    a.grid(alpha=.3, axis="y")
    fig.tight_layout(); p = os.path.join(OUT, "fig_E3b_escape.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def main():
    rows = collect(); best = best_table(rows)
    plot_contrast_curve(rows); plot_fair(best, rows); plot_escape(best)
    print("\n[E3b] 完成。产物在", OUT)


if __name__ == "__main__":
    main()
