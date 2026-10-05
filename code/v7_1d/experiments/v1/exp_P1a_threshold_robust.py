# -*- coding: utf-8 -*-
"""
实验 P1-a（只读、不重训）：定义 5.2 两层概念的阈值稳健性
================================================================================
对应 framework_v10_1 定义 5.2（def:basin）末尾“列为待补实验（由现有配点现算、不重训）”：
  (1) 按 ν 分别报告 Gate 点（Phase0 结束）激波带误差 ℓ_band 的分布，检验“好盆地中位<.17、
      未逃逸坏盆地>.95”的自然间隙是否在每个 ν 下都成立；
  (2) τ_band∈[.3,.7]（加密扫描 .1… .9）好/坏计数稳健性表：以 τ=.5 为基准，统计标签翻转数/率，
      正面回应“操作阈值 0.5 是固定绝对值、不近似 ρ(ν)=O(ν)，但分类标签在区间内稳定”。

口径（务必与全文一致）：
  * ℓ_band = ||uθ-u*||_{L2(|x|<3ν)} / ||u*||_{L2(|x|<3ν)}，即 PINNMetrics.compute_full 的 l2_band；
  * 数据来源：results/v6/_gate_cache/ 里 Phase0 已训练好的 state（本脚本只前向、绝不重训）；
  * 参照标签用 exp_G1_basin_expand/g1_runs.csv 的 basin_class——它按 Gate 点【全域】相对 L2<0.06
    （mc.GATE_L2_MAX，truth 选起点）打标，与缓存同源（同 (nu,seed) 的 phase0_L2 等于缓存 l2）。
    注意它是“全域误差参照”，不是 Phase1 结果论的“最终是否精修成功”；本实验正是比较
    “仅用激波带内误差 ℓ_band 的阈值分类”与“全域 L2 参照分类”，二者在边界点的分歧本身就是结果。

产出 results/v6/exp_P1a_threshold/：
  p1a_gate_lband.csv      每 (nu,seed)：现算 Gate 点全域 L2（自检=缓存 l2）、ℓ_band、oracle 好/坏
  p1a_lband_by_nu.csv     按 ν 的 ℓ_band 分布（全体/好组/坏组 中位[Q1,Q3]、好max/坏min、自然间隙）
  p1a_threshold_scan.csv  τ×ν：好计数、相对 τ=.5 翻转数、与 oracle 的 sens/spec/acc
  p1a_robust_summary.csv  汇总：[.3,.7] 内最大翻转数/率、自然间隙、τ=.5 与最优 τ(Youden)
  fig_p1a_lband_dist_{zh,en}.png   ℓ_band 分布（好/坏 + τ=.5 线 + [.3,.7] 灰带）
  fig_p1a_count_{zh,en}.png        好计数随 τ（每 ν 一条，带内平）
  fig_p1a_flip_{zh,en}.png         相对 .5 的翻转率随 τ
环境：V8_SMOKE=1 用单 ν、3 seed 打通全链路（秒级）。
运行：python exp_P1a_threshold_robust.py
"""
import os, sys, csv, glob
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve(); _PKG = _HERE.parent.parent           # experiments/
_REL = _HERE.parents[4]            # paper1_release/
for _p in (str(_PKG.parent), str(_PKG), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from utils.metrics import PINNMetrics

# 只读后处理：输入来自归档原始数据 data/raw/v6（含 _gate_cache 模型权重与 g1_runs 参照标签）
_DRAW = _REL / "data" / "raw" / "v6"
OUT = os.environ.get("REPRO_OUT", str(_REL / "data" / "aggregated" / "v6_exp_P1a_threshold"))
os.makedirs(OUT, exist_ok=True); r2s.OUT = OUT
CACHE = str(_DRAW / "_gate_cache")
G1_CSV = str(_DRAW / "exp_G1_basin_expand" / "g1_runs.csv")
NUS = [0.001, 0.003, 0.005, 0.01]
SEEDS = list(range(0, 100, 5))          # 与 G1/G2 同批：0,5,…,95 共 20 颗，实验前固定
BAND_C = 3.0
TAUS = [round(0.1 * i, 1) for i in range(1, 10)]   # .1 … .9
TAU0 = 0.5
ROBUST_LO, ROBUST_HI = 0.3, 0.7
MET = PINNMetrics()
if mc.SMOKE: NUS, SEEDS = [0.01], [0, 5, 10]

# 双语标签
LAB = {
    "zh": dict(fix=dict(supt="P1-a  Gate 点 ℓ_band 分布（|x|<3ν 带内相对 L2）",
                        x="粘度 ν", y="ℓ_band（激波带内相对误差）",
                        good="参照好(全域L2<.06)", bad="参照坏", base="τ=0.5", band="稳健区间[.3,.7]"),
              cnt=dict(supt="好盆地计数随阈值 τ（现算，不重训）", x="阈值 τ_band", y="判为好盆地的数量"),
              flip=dict(supt="相对 τ=0.5 的标签翻转率", x="阈值 τ_band", y="翻转种子占比"),
              gap="自然间隙(坏min−好max)"),
    "en": dict(fix=dict(supt="P1-a  Gate-point ℓ_band distribution (in-band rel. L2, |x|<3ν)",
                        x="viscosity ν", y="ℓ_band (in-band relative error)",
                        good="ref. good (global L2<.06)", bad="ref. bad", base="τ=0.5", band="robust range [.3,.7]"),
              cnt=dict(supt="Good-basin count vs threshold τ (computed, no retraining)",
                       x="threshold τ_band", y="# classified good"),
              flip=dict(supt="Label-flip fraction relative to τ=0.5", x="threshold τ_band",
                        y="fraction of flipped seeds"),
              gap="gap (bad min − good max)"),
}
C_GOOD, C_BAD = "#3a8a18", "#c0392b"


def set_font(lang):
    if lang == "zh":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_oracle():
    """g1_runs.csv: (nu,seed)->basin_class（Gate 点全域 L2<0.06 的参照标签，与缓存同源）。"""
    m = {}
    with open(G1_CSV, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            m[(float(r["nu"]), int(float(r["seed"])))] = r["basin_class"]
    return m


def cache_path(nu, seed):
    # 只取 σ 课程终点 sig15.0 的 Fourier 主线 Gate 缓存；sig1.0 为 MLP 对照档
    # （net.0 维度不同），不能加载进 Fourier 模型，故不得用 sig* 通配取第一个。
    hits = glob.glob(os.path.join(CACHE, f"nu{nu:g}_s{seed}_sig15.0_k4_ep2000.pt"))
    hits = [h for h in hits if "smoke" not in h]
    return hits[0] if hits else None


def gate_lband(nu, seed):
    """加载已训练 Phase0 state，前向现算 Gate 点全域 L2 与 ℓ_band（不重训）。"""
    fp = cache_path(nu, seed)
    if fp is None: return None
    basin = torch.load(fp, map_location=mc.DEVICE, weights_only=False)
    cfg = r2s.base_cfg(nu, f"p1a_{nu:g}_{seed}")
    model, pde, crit, opt, samp, vis, cfg = r2s.build_all(cfg, r2s.LR_PHASE0, 1.0)
    model.load_state_dict(basin["state"]); model.set_sigma(float(basin["sigma_end"])); model.eval()
    xt, ut = r2s.build_test(nu)
    with torch.no_grad():
        up = model(xt).detach().cpu().numpy().reshape(-1)
    xn = xt.detach().cpu().numpy().reshape(-1)
    full = float(np.linalg.norm(up - ut) / np.linalg.norm(ut))
    mm = MET.compute_full(up, ut, x=xn, nu=nu, shock_band_c=BAND_C)
    return dict(cache_l2=float(basin["l2"]), gate_L2=full, lband=float(mm["l2_band"]))


def quant(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if a.size == 0: return (np.nan, np.nan, np.nan, np.nan, np.nan)
    q1, q2, q3 = np.quantile(a, [.25, .5, .75])
    return float(np.min(a)), float(q1), float(q2), float(q3), float(np.max(a))


def collect():
    oracle = load_oracle(); rows = []; miss = []
    for nu in NUS:
        xt, ut = r2s.build_test(nu)  # noqa（确保该 ν 真解可建）
        for seed in SEEDS:
            res = gate_lband(nu, seed)
            if res is None: miss.append((nu, seed)); continue
            oc = oracle.get((nu, seed), "?")
            dres = abs(res["gate_L2"] - res["cache_l2"])
            rows.append(dict(nu=nu, seed=seed, **res,
                             oracle=oc, oracle_good=int(oc == "good"), reload_err=dres))
            if dres > 1e-3:
                print(f"[warn] nu={nu:g} s{seed} 现算L2={res['gate_L2']:.5f} 与缓存l2={res['cache_l2']:.5f} 差{dres:.2e}")
    print(f"[P1a] 载入 {len(rows)} 个 Gate 点；缺缓存 {len(miss)}: {miss[:6]}")
    cols = ["nu", "seed", "cache_l2", "gate_L2", "reload_err", "lband", "oracle", "oracle_good"]
    mc.write_csv(os.path.join(OUT, "p1a_gate_lband.csv"),
                 [{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows], cols)
    return rows


def analyze(rows):
    # ---------- (1) 按 ν 的 ℓ_band 分布 ----------
    by_nu, dist_rows = {}, []
    for r in rows: by_nu.setdefault(r["nu"], []).append(r)
    for nu in sorted(by_nu):
        rs = by_nu[nu]; lb = np.array([r["lband"] for r in rs], float)
        g = np.array([r["lband"] for r in rs if r["oracle_good"] == 1], float)
        b = np.array([r["lband"] for r in rs if r["oracle_good"] == 0], float)
        ag = quant(g); ab = quant(b)
        gap = (ab[0] - ag[4]) if (g.size and b.size and np.isfinite(ab[0]) and np.isfinite(ag[4])) else np.nan
        dist_rows.append(dict(nu=nu, n=len(rs), n_good=g.size, n_bad=b.size,
                              all_med=quant(lb)[2], good_min=ag[0], good_Q1=ag[1], good_med=ag[2], good_Q3=ag[3], good_max=ag[4],
                              bad_min=ab[0], bad_Q1=ab[1], bad_med=ab[2], bad_Q3=ab[3], bad_max=ab[4], natural_gap=gap))
    mc.write_csv(os.path.join(OUT, "p1a_lband_by_nu.csv"),
                 [{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in d.items()} for d in dist_rows])

    # ---------- (2) 阈值扫描：计数稳定性 + 与 oracle 一致性 ----------
    scan, robust = [], []
    for nu in sorted(by_nu):
        rs = by_nu[nu]; lb = np.array([r["lband"] for r in rs], float)
        y = np.array([r["oracle_good"] for r in rs], int)
        base_pred = lb < TAU0
        for tau in TAUS:
            pred = lb < tau
            n_good = int(pred.sum()); n_flip = int((pred != base_pred).sum())
            tp = int(((pred == 1) & (y == 1)).sum()); fn = int(y.sum()) - tp
            fp = int(((pred == 1) & (y == 0)).sum()); tn = int((1 - y).sum()) - fp
            sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); acc = (tp + tn) / len(rs)
            scan.append(dict(nu=nu, tau=tau, n_good=n_good, n_flip_vs_05=n_flip,
                             flip_rate=n_flip / len(rs), sens=sens, spec=spec, acc=acc, youden=sens + spec - 1))
    mc.write_csv(os.path.join(OUT, "p1a_threshold_scan.csv"),
                 [{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in d.items()} for d in scan])

    # ---------- (3) 汇总：[.3,.7] 最大翻转、自然间隙、τ=.5 与最优 τ ----------
    for nu in sorted(by_nu):
        sub = [s for s in scan if s["nu"] == nu]
        inband = [s for s in sub if ROBUST_LO <= s["tau"] <= ROBUST_HI]
        max_flip = max(s["n_flip_vs_05"] for s in inband); n = len(by_nu[nu])
        gc_in = [s["n_good"] for s in inband]; count_range = max(gc_in) - min(gc_in)
        best = max(sub, key=lambda s: s["youden"])
        at05 = next(s for s in sub if abs(s["tau"] - TAU0) < 1e-9)
        gap = next(d["natural_gap"] for d in dist_rows if d["nu"] == nu)
        robust.append(dict(nu=nu, n=n, max_flip_in_03_07=max_flip, max_flip_rate=max_flip / n,
                           goodcount_range_in_03_07=count_range,
                           natural_gap=gap, acc_at_05=at05["acc"], youden_at_05=at05["youden"],
                           best_tau_youden=best["tau"], best_acc=best["acc"]))
    mc.write_csv(os.path.join(OUT, "p1a_robust_summary.csv"),
                 [{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in d.items()} for d in robust])
    return by_nu, dist_rows, scan, robust


# ---------------- 图（zh/en 各一套）----------------
def _series(scan, nu, key):
    s = sorted((x for x in scan if x["nu"] == nu), key=lambda z: z["tau"])
    return [x["tau"] for x in s], [x[key] for x in s]


def plots(by_nu, scan):
    nus = sorted(by_nu)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(nus)))
    for lang in ("zh", "en"):
        set_font(lang); T = LAB[lang]
        # 图1：ℓ_band 分布
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for i, nu in enumerate(nus):
            rs = by_nu[nu]; x = i
            for cls, cc, mk in [(1, C_GOOD, "o"), (0, C_BAD, "x")]:
                v = [r["lband"] for r in rs if r["oracle_good"] == cls]
                ax.scatter(np.random.normal(x, .06, len(v)), v, c=cc, marker=mk, s=26, alpha=.8,
                           label=(T["fix"]["good"] if cls == 1 else T["fix"]["bad"]) if i == 0 else None)
        ax.axhline(TAU0, color="#333", lw=1.4, label=T["fix"]["base"])
        ax.axhspan(ROBUST_LO, ROBUST_HI, color="#999", alpha=.15, label=T["fix"]["band"])
        ax.set_yscale("log"); ax.set_xticks(range(len(nus))); ax.set_xticklabels([f"{n:g}" for n in nus])
        ax.set_xlabel(T["fix"]["x"]); ax.set_ylabel(T["fix"]["y"]); ax.set_title(T["fix"]["supt"], fontsize=11)
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); p = os.path.join(OUT, f"fig_p1a_lband_dist_{lang}.png")
        fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)
        # 图2：好计数随 τ
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for i, nu in enumerate(nus):
            xs, ys = _series(scan, nu, "n_good"); ax.plot(xs, ys, "-o", ms=4, color=cmap[i], label=f"ν={nu:g}")
        ax.axvspan(ROBUST_LO, ROBUST_HI, color="#999", alpha=.15); ax.axvline(TAU0, color="#333", ls="--", lw=1)
        ax.set_xlabel(T["cnt"]["x"]); ax.set_ylabel(T["cnt"]["y"]); ax.set_title(T["cnt"]["supt"], fontsize=11)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); p = os.path.join(OUT, f"fig_p1a_count_{lang}.png")
        fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)
        # 图3：翻转率随 τ
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for i, nu in enumerate(nus):
            xs, ys = _series(scan, nu, "flip_rate"); ax.plot(xs, ys, "-o", ms=4, color=cmap[i], label=f"ν={nu:g}")
        ax.axvspan(ROBUST_LO, ROBUST_HI, color="#999", alpha=.15); ax.axvline(TAU0, color="#333", ls="--", lw=1)
        ax.set_ylim(-.02, 1.02); ax.set_xlabel(T["flip"]["x"]); ax.set_ylabel(T["flip"]["y"])
        ax.set_title(T["flip"]["supt"], fontsize=11); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); p = os.path.join(OUT, f"fig_p1a_flip_{lang}.png")
        fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def main():
    rows = collect()
    if not rows: print("[P1a] 无可用 Gate 缓存，退出。"); return
    by_nu, dist_rows, scan, robust = analyze(rows)
    plots(by_nu, scan)
    print("\n===== 按 ν 的 ℓ_band 分布（参照好/坏 中位[Q1,Q3]、自然间隙）=====")
    for d in dist_rows:
        print(f"  ν={d['nu']:g} n={d['n']}(参照好{d['n_good']}/参照坏{d['n_bad']}) "
              f"好中位={d['good_med']:.3f} 好max={d['good_max']:.3f} | 坏min={d['bad_min']:.3f} 坏中位={d['bad_med']:.3f} "
              f"| 间隙={d['natural_gap']:.3f}")
    print("\n===== [.3,.7] 阈值稳健性（相对 τ=.5）=====")
    for d in robust:
        flag = "标签零翻转" if d["max_flip_in_03_07"] == 0 else f"最多翻转{d['max_flip_in_03_07']}/{d['n']}"
        print(f"  ν={d['nu']:g}: {flag}，带内好计数极差={d['goodcount_range_in_03_07']}；"
              f"τ=.5 对参照一致率={d['acc_at_05']:.3f}，Youden最优τ={d['best_tau_youden']:g}"
              f"(acc={d['best_acc']:.3f})，自然间隙={d['natural_gap']:.3f}")
    print("[P1a] 完成，产物在", OUT)


if __name__ == "__main__":
    main()
