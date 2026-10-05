# -*- coding: utf-8 -*-
"""
V2 实验 S1：σ(Fourier 带宽) × ν(粘性) 二维扫描 —— 系统刻画“进入可精修盆地所需带宽随 ν 的迁移”
================================================================================
科学目的（替代 V1 过强的“刚性端必须高 Fourier”断言）：
  在【统一网格、统一 seed、统一训练预算】下测量 Phase0 盆地进入质量 L2(σ;ν) 与进入率 p_entry(σ,ν)，
  检验 σ*(ν) 的反向迁移：薄激波(小ν)需较高带宽，光滑端(大ν)高带宽反而高频错配、低带宽/普通 MLP 即可。

两种口径（严格区分，勿混）：
  * fixed[主结果，默认]：全程【固定带宽 σ】、关闭 σ 表示课程（sigma_lo=None）、β=1(uniform)、
      K 起点 multistart 取 best-of-K。这是“带宽本身”的干净单因子（不被课程退火污染）。
  * course[零算力对照]：直接聚合 results/v7/_gate_cache 里 V1 遗留的“σ 课程 5→σ_hi 退火”盆地缓存
      的 best-of-K L2，只读数、不训练，用于先看趋势 & 与 fixed 口径对照（变量是课程终点 σ_hi）。

工程：
  * 增量断点续跑：fixed 每完成一个 (ν,σ,base_seed) 立即 append 到 csv，重跑自动跳过已有单元；
  * 规则等距种子 SEEDS_EXT(0,5,…,50)，实验前固定、不挑种子；K=4 内部 start=seed..seed+3（间隔10不重叠）；
  * 结果到 results/v7/V2/exp_V2_S1_sigma_nu_scan/，不写 V1/results-v6。
运行：
  python exp_V2_sigma_nu_scan.py --mode both                # 先 course 秒出，再 fixed 缺啥补啥
  python exp_V2_sigma_nu_scan.py --mode fixed --nus 0.005,0.5 --sigmas 1,15
  V8_SMOKE=1 python exp_V2_sigma_nu_scan.py --mode fixed     # 冒烟
================================================================================
"""
import os, sys, csv, argparse, copy
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager

_HERE = Path(__file__).resolve()
_V7 = _HERE.parents[2]              # V2 -> experiments -> v7/
_V1EXP = _V7 / "experiments" / "v1"
for _p in (str(_V7), str(_V1EXP), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from trainers.pinn_trainer import PINNTrainer

# ---- 结果目录重定向到 V2（不碰 V1/results-v6）----
OUT = str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S1_sigma_nu_scan"); os.makedirs(OUT, exist_ok=True)
r2s.OUT = OUT
FIXED_CSV = os.path.join(OUT, "sigmanu_fixed_runs.csv")
COURSE_CSV = os.path.join(OUT, "sigmanu_course_cache.csv")

for _c in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(_c):
        font_manager.fontManager.addfont(_c)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_c).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 正式网格（SMOKE 自动缩小）----------------
NUS_ALL = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
SIGMA_ALL = [0.15, 0.3, 0.6, 1.5, 3.0, 7.5, 15.0, 30.0]
SEEDS_ALL = mc.SEEDS_EXT            # 0,5,…,50（11 个规则种子）
K_ALL, EP_ALL, THR_ALL = 4, 2000, mc.GATE_L2_MAX
if mc.SMOKE:
    NUS_ALL = [0.005, 0.5]; SIGMA_ALL = [1.0, 15.0]; SEEDS_ALL = [0]; K_ALL, EP_ALL = 2, 120

# feature: "fourier"=固定随机傅里叶带宽 σ（σ>0）；"mlp"=关闭傅里叶、直接以 x 输入（零带宽端点，σ 记 0）
FIXED_FIELDS = ["nu", "sigma", "base_seed", "K", "epochs", "feature", "best_seed",
                "best_L2", "success", "l2_starts"]
FEATURES_ALL = ["fourier", "mlp"]


# ======================== fixed：固定带宽 multistart ========================
def run_fixed_unit(nu, sigma, base_seed, xt, ut, K, epochs, feature="fourier"):
    """固定表示全程、β=1 uniform 的 K 起点 multistart，返回 best-of-K 的 L2 与各起点 L2。
    feature='fourier'：固定随机傅里叶带宽 σ；feature='mlp'：关闭傅里叶（零带宽，sigma 入参忽略）。"""
    starts = []
    for k in range(K):
        seed = int(base_seed) + k
        r2s.set_seed(seed)
        cfg = r2s.base_cfg(nu, f"s1_{feature}_nu{nu}_sig{sigma}_s{base_seed}_k{k}")
        # 【bc_amp 必传，与 E1b/R2 一致】硬边界提升 A=-bc_amp*x，大 ν 真解端点为 ±tanh(1/2ν)，
        # 缺省会用 1.0（仅 ν→0 对），否则大 ν 存在由错误端点幅值决定的误差地板，与 σ/种子无关。
        cfg.update(dict(lr=r2s.LR_PHASE0, use_gate=False,
                        feature=feature,
                        bc_amp=float(np.tanh(1.0 / (2.0 * float(nu)))),
                        fourier_scale=float(sigma), sigma_lo=None, sigma_hi=None, sigma_anneal_T=0,
                        beta_schedule="const", loss_beta=1.0, beta_init=1.0,
                        adam_epochs=int(epochs), lbfgs_epochs=0,
                        use_best_ckpt=True, best_metric="L2"))
        model, pde, crit, opt, samp, vis, cfg = r2s.build_all(cfg, r2s.LR_PHASE0, 1.0)
        if feature == "fourier": model.set_sigma(float(sigma))  # mlp 无傅里叶层，set_sigma 为 no-op
        trainer = PINNTrainer(model, pde, crit, opt, samp, vis, mc.DEVICE, cfg, {"x": xt, "u_true": ut})
        trainer.train(cfg, {"x": xt, "u_true": ut}, int(epochs), 0, adaptive_freq=500)
        l2 = r2s.quick_l2(model, xt, ut)
        trainer.log_file.close()
        starts.append((seed, float(l2)))
    best_seed, best_l2 = min(starts, key=lambda z: z[1])
    return dict(nu=nu, sigma=float(sigma), base_seed=int(base_seed), K=int(K), epochs=int(epochs),
                feature=feature, best_seed=int(best_seed), best_L2=float(best_l2),
                success=int(best_l2 < THR_ALL),
                l2_starts=";".join(f"{s}:{v:.6e}" for s, v in starts))


def _done_keys(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                ft = r.get("feature") or "fourier"  # 旧表无 feature 列，按 fourier 兼容
                done.add((float(r["nu"]), float(r["sigma"]), int(r["base_seed"]), ft))
    return done


def collect_fixed(nus, sigmas, seeds, K, epochs, features, limit=None):
    done = _done_keys(FIXED_CSV); new_n = 0
    newfile = not os.path.exists(FIXED_CSV)
    fout = open(FIXED_CSV, "a", newline="", encoding="utf-8-sig")
    w = csv.DictWriter(fout, fieldnames=FIXED_FIELDS)
    if newfile: w.writeheader(); fout.flush()
    # 单元序列：fourier 遍历全部 σ；mlp 为零带宽端点（σ 记 0，每个 ν×seed 一次）
    try:
        for nu in nus:
            xt, ut = r2s.build_test(nu)
            units = [("fourier", float(sg)) for sg in sigmas]
            if "mlp" in features: units.append(("mlp", 0.0))
            for feature, sigma in units:
                for seed in seeds:
                    key = (float(nu), float(sigma), int(seed), feature)
                    if key in done: continue
                    row = run_fixed_unit(nu, sigma, seed, xt, ut, K, epochs, feature=feature)
                    w.writerow(row); fout.flush(); done.add(key); new_n += 1
                    print(f"[fixed][{feature}] ν{nu} σ{sigma:g} s{seed}: bestL2={row['best_L2']:.3e} "
                          f"ok={row['success']} (best_start={row['best_seed']})")
                    if limit is not None and new_n >= limit:
                        print(f"[limit] 已补 {new_n} 个缺项，停止本批（断点续跑，下次继续）"); raise StopIteration
    except StopIteration:
        pass
    finally:
        fout.close()
    rows = _read_csv(FIXED_CSV)
    agg = aggregate_grid(rows, "best_L2")
    mc.write_csv(os.path.join(OUT, "sigmanu_fixed_agg.csv"), agg)
    return rows, agg


# ======================== course：零算力复用 V1 课程缓存 ========================
def collect_course():
    import re
    gc = _V7.parent / "results" / "v7" / "_gate_cache"
    pat = re.compile(r"nu([0-9.]+)_s(\d+)_sig([0-9.eE+-]+)_k(\d+)_ep(\d+)\.pt$")
    rows = []
    for fn in sorted(os.listdir(gc)):
        m = pat.match(fn)
        if not m or fn.startswith("smoke"): continue
        nu, s, sig, k, ep = m.groups()
        try:
            d = torch.load(gc / fn, map_location="cpu")
        except Exception as e:
            print("[course] unreadable", fn, e); continue
        l2 = float(d["l2"])
        rows.append(dict(nu=float(nu), sigma=float(sig), base_seed=int(s), K=int(k), epochs=int(ep),
                         best_seed=int(d.get("seed", -1)), best_L2=l2, success=int(l2 < THR_ALL)))
    mc.write_csv(COURSE_CSV, rows)
    agg = aggregate_grid(rows, "best_L2")
    mc.write_csv(os.path.join(OUT, "sigmanu_course_agg.csv"), agg)
    print(f"[course] 复用 V1 课程缓存 {len(rows)} 个盆地（零训练）")
    return rows, agg


def _read_csv(path):
    if not os.path.exists(path): return []
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("nu", "sigma", "best_L2"): r[k] = float(r[k])
        for k in ("base_seed", "success", "K", "epochs", "best_seed"): r[k] = int(float(r[k]))
    return rows


def aggregate_grid(rows, val="best_L2"):
    out = []
    feats = sorted({(r.get("feature") or "fourier") for r in rows})
    for ft in feats:
        rf = [r for r in rows if (r.get("feature") or "fourier") == ft]
        for nu in sorted({r["nu"] for r in rf}):
            for sg in sorted({r["sigma"] for r in rf}):
                sub = [r for r in rf if r["nu"] == nu and r["sigma"] == sg]
                v = np.asarray([r[val] for r in sub], float); ok = np.isfinite(v)
                if not ok.any(): continue
                v = v[ok]
                out.append(dict(feature=ft, nu=nu, sigma=sg, n=len(v),
                                L2_med=float(np.median(v)), L2_q1=float(np.quantile(v, .25)),
                                L2_q3=float(np.quantile(v, .75)), L2_min=float(v.min()), L2_max=float(v.max()),
                                p_entry=float(np.mean(np.asarray([r["success"] for r in sub]) == 1))))
    return out


# ============================== 图 ==============================
def _pivot(agg, key):
    nus = sorted({r["nu"] for r in agg}); sgs = sorted({r["sigma"] for r in agg})
    Z = np.full((len(nus), len(sgs)), np.nan)
    for r in agg:
        i, j = nus.index(r["nu"]), sgs.index(r["sigma"]); Z[i, j] = r[key]
    return nus, sgs, Z


def plot_heatmap(agg, tag, title):
    if not agg: return
    agg = [r for r in agg if (r.get("feature") or "fourier") == "fourier"]  # 热图只画傅里叶二维网格
    nus, sgs, Z = _pivot(agg, "p_entry")
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(sgs))); ax.set_xticklabels([f"{g:g}" for g in sgs])
    ax.set_yticks(range(len(nus))); ax.set_yticklabels([f"{n:g}" for n in nus])
    ax.set_xlabel("Fourier 带宽 σ（固定全程）" if tag == "fixed" else "σ 课程终点 σ_hi")
    ax.set_ylabel("粘性 ν（↓ 越薄激波越难）")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isfinite(Z[i, j]): ax.text(j, i, f"{Z[i,j]:.0%}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="进入可精修盆地比例 p_entry (L2<%.2g)" % THR_ALL)
    ax.set_title(title); fig.tight_layout()
    p = os.path.join(OUT, f"fig_{tag}_heatmap_pentry.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def plot_curves(agg, tag, title):
    if not agg: return
    fig, ax = plt.subplots(figsize=(7.6, 5.0)); nus = sorted({r["nu"] for r in agg})
    cmap = plt.cm.viridis(np.linspace(0, 1, len(nus)))
    for c, nu in zip(cmap, nus):
        z = sorted((r for r in agg if r["nu"] == nu and (r.get("feature") or "fourier") == "fourier"),
                   key=lambda r: r["sigma"])
        xs = [r["sigma"] for r in z]; m = [r["L2_med"] for r in z]
        lo = [r["L2_q1"] for r in z]; hi = [r["L2_q3"] for r in z]
        ax.plot(xs, m, "o-", ms=3, color=c, label=f"ν={nu:g}"); ax.fill_between(xs, lo, hi, color=c, alpha=.12)
        mz = [r for r in agg if r["nu"] == nu and r.get("feature") == "mlp"]  # 零带宽端点
        if mz: ax.scatter([0.075], [mz[0]["L2_med"]], marker="*", s=95, color=c, edgecolor="k", lw=.4, zorder=5)
    ax.plot([], [], "k*", ms=11, label="★ 普通 MLP（零傅里叶带宽）")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.06, 38)
    ax.set_xlabel("Fourier 带宽 σ（最左 ★ 为普通 MLP 零带宽）"); ax.set_ylabel("Phase0 best-of-K L2（中位/IQR）")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7, ncol=2); ax.set_title(title)
    fig.tight_layout(); p = os.path.join(OUT, f"fig_{tag}_L2_vs_sigma.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def plot_sigma_star(agg, tag, title):
    if not agg: return
    agg = [r for r in agg if (r.get("feature") or "fourier") == "fourier"]  # σ* 只在傅里叶网格上取
    xs, ys = [], []
    for nu in sorted({r["nu"] for r in agg}):
        z = [r for r in agg if r["nu"] == nu and np.isfinite(r["L2_med"])]
        if z: xs.append(nu); ys.append(min(z, key=lambda r: r["L2_med"])["sigma"])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(xs, ys, "o-", color="#2E6E8E")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("ν（↓ 越刚性）"); ax.set_ylabel("经验最优带宽 σ*(ν)")
    ax.grid(alpha=.3); ax.set_title(title)
    fig.tight_layout(); p = os.path.join(OUT, f"fig_{tag}_sigma_star.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["fixed", "course", "both"])
    ap.add_argument("--nus", default=""); ap.add_argument("--sigmas", default=""); ap.add_argument("--seeds", default="")
    ap.add_argument("--K", type=int, default=K_ALL); ap.add_argument("--epochs", type=int, default=EP_ALL)
    ap.add_argument("--limit", type=int, default=None, help="本批最多补多少个缺项（分批跑）")
    ap.add_argument("--features", default="fourier,mlp", help="fourier / mlp / fourier,mlp")
    a = ap.parse_args()
    nus = [float(x) for x in a.nus.split(",") if x] or NUS_ALL
    sigmas = [float(x) for x in a.sigmas.split(",") if x] or SIGMA_ALL
    seeds = [int(x) for x in a.seeds.split(",") if x] or SEEDS_ALL
    features = [x.strip() for x in a.features.split(",") if x.strip()] or FEATURES_ALL
    if a.mode in ("course", "both"):
        r_c, a_c = collect_course()
        plot_heatmap(a_c, "course", "课程口径（V1 缓存，σ 5→σ_hi 退火）p_entry")
        plot_curves(a_c, "course", "课程口径：L2 vs σ_hi")
        plot_sigma_star(a_c, "course", "课程口径 σ*(ν) 迁移")
    if a.mode in ("fixed", "both"):
        r_f, a_f = collect_fixed(nus, sigmas, seeds, a.K, a.epochs, features, limit=a.limit)
        plot_heatmap(a_f, "fixed", "固定带宽口径 p_entry(σ,ν)：σ*(ν) 反向迁移")
        plot_curves(a_f, "fixed", "固定带宽：Phase0 L2 vs σ（各 ν）")
        plot_sigma_star(a_f, "fixed", "固定带宽 σ*(ν)：刚性端高、光滑端低")
    print("\n[S1] 完成，产物在", OUT)


if __name__ == "__main__":
    main()
