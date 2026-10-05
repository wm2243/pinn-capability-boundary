# -*- coding: utf-8 -*-
"""
实验 R2（V10.1）：训练预算收敛性 —— 证明 β* 与 L2-β 结论不依赖训练步数、且大 ν 非“欠收敛假退化”
--------------------------------------------------------------------------------
配对设计：同一 Phase0 盆地（与预算无关、共享），只把 Phase1 预算沿翻倍链 {2000,4000,8000} 拉长。
ν=.005（激波）/.1（近退化）/.5（真退化端，专门证 β*=1 不是步数少）；β∈{1,3,8}、rational。

种子规范（与全文一致，防“挑种子”）：
  * 主结果用 SEEDS_EXT 共 11 颗（步长 5：0,5,…,50），其中原批次 SEEDS_NEW=[0,10,20,30,40]
    是其等距子集（5 颗）。本脚本【断点续跑】：启动先读已有 r2_runs.csv，已完成的
    (ν,seed,budget,β) 四元组直接跳过、只补新 6 颗 [5,15,25,35,45,50]，旧 5 颗绝不重训；
    Phase0 盆地走 enter_basin_cached，旧种子命中缓存、同样不重算。
  * 补完后做 11 抽 5 bootstrap（无放回抽 SEEDS_NEW 规模，B=1000、固定随机种子），
    量化“小样本 n=5 的波动 vs 全量 n=11 的结论稳定性”，产出 r2_bootstrap.csv。

判据（预注册）：
  (i)   4000→8000 翻倍后 β* 不变、各 β 的 L2 相对变化 <5%（进入平台）；
  (ii)  若 2000 与 8000 结论不同而 4000/8000 相同 → 2000 欠收敛、正式结论取平台段；
  (iii) ν=.5 在三档预算下都 β*=1、L2-β 平坦，排除“光滑端步数少→假退化”。
产出 results/v6/exp_R2_budget/：r2_runs.csv、r2_traj.csv（最大预算的 L2 轨迹）、
  r2_invariance.csv、r2_bootstrap.csv、3 图。
不改公共层。SMOKE：V8_SMOKE=1（单种子、两预算，用于冒烟，不写入正式结论）。
"""
import os, sys, csv
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
# 正确硬边界端点幅值 u(±1)=∓tanh(1/2ν)（进程内包装 r2s.base_cfg，不改公共文件）：
# 小ν≈1 与旧 A=-x 等价；ν=.5 修复"端点钉死 ±1"的物理一致性问题。
_orig_base_cfg = r2s.base_cfg
def _bc_base_cfg(nu, seed_tag):
    c = _orig_base_cfg(nu, seed_tag)
    c["bc_amp"] = float(np.tanh(1.0 / (2.0 * float(nu))))
    return c
r2s.base_cfg = _bc_base_cfg

for _fp in (r"C:\Windows\Fonts\msyh.ttc",):
    if os.path.exists(_fp):
        try: font_manager.fontManager.addfont(_fp)
        except Exception: pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = str(mc.RES_ROOT / "exp_R2_budget"); os.makedirs(OUT, exist_ok=True)
r2s.OUT = OUT
BUDGETS = [2000, 4000, 8000]
NUS = [0.005, 0.1, 0.5]; BETAS = [1.0, 3.0, 8.0]
SEEDS = list(mc.SEEDS_EXT)                       # 全量 11 颗（NEW⊂EXT，断点只补新 6 颗）
NU_COLOR = {0.005: "#4B86B4", 0.1: "#E8A24B", 0.5: "#7FB3D5"}
BCOL = ["#9A9A9A", "#4B86B4", "#E8A24B"]     # 按 BETAS 顺序取色，兼容任意 β 网格
PROBE_N = 2048
BOOT_B, BOOT_SEED = 1000, 20260910
if mc.SMOKE:
    BUDGETS = [100, 200]; NUS = [0.1]; BETAS = [1.0, 5.0]; SEEDS = [mc.SEEDS_NEW[0]]; PROBE_N = 256
CONV_WIN, CONV_TOL = 5, 1e-3
RUN_FCOLS = ["phase0_L2", "L2", "Linf", "l2_band", "eta_gate"]
RUN_ICOLS = ["best_epoch", "conv_epoch", "converged"]

def _f(x):
    try: return float(x)
    except Exception: return np.nan

def _i(x):
    try: return int(float(x))
    except Exception: return -1

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

# ---------- 断点续跑：读已有结果，规范化类型，建立已完成集合 ----------
def _load_runs():
    p = os.path.join(OUT, "r2_runs.csv")
    if not os.path.exists(p): return []
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        d = dict(nu=float(r["nu"]), seed=_i(r["seed"]), budget=_i(r["budget"]), beta=float(r["beta"]))
        for k in RUN_FCOLS: d[k] = _f(r.get(k, ""))
        for k in RUN_ICOLS: d[k] = _i(r.get(k, ""))
        out.append(d)
    return out

def _load_traj():
    p = os.path.join(OUT, "r2_traj.csv")
    if not os.path.exists(p): return []
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        out.append(dict(nu=float(r["nu"]), seed=_i(r["seed"]), beta=float(r["beta"]),
                        step=_i(r["step"]), L2=_f(r["L2"])))
    return out

def collect():
    old = _load_runs(); old_traj = _load_traj()
    done = {(r["nu"], r["seed"], r["budget"], r["beta"]) for r in old}
    rows = list(old); traj = list(old_traj); n_new = 0
    for nu in NUS:
        xt, ut = r2s.build_test(nu)
        xp = torch.linspace(-1, 1, PROBE_N, device=mc.DEVICE).view(-1, 1)
        for seed in SEEDS:
            todo = [(ep, float(b)) for ep in BUDGETS for b in BETAS
                    if (nu, seed, ep, float(b)) not in done]
            if not todo:
                continue  # 该种子全部预算×β 已完成：不重训、连 Phase0 缓存都不碰
            basin = mc.enter_basin_cached(nu, seed, xt, ut, sigma_hi=15.0, k=4, phase0_epochs=2000)
            for ep, beta in todo:
                m, dyn, model, crit, tr = mc.train_phase1_family(
                    basin, nu, beta, seed, xt, ut, family="rational", epochs=ep,
                    tag=f"R2_ep{ep}", grad_probe=(xp, nu))
                ce, isc = conv_epoch(dyn); gp = tr.grad_probe
                rows.append(dict(nu=nu, seed=seed, budget=ep, beta=beta,
                                 phase0_L2=basin["l2"], L2=m.get("L2_Error", np.nan),
                                 Linf=m.get("L_inf_Error", np.nan), l2_band=m.get("l2_band", np.nan),
                                 eta_gate=(gp["gate"] or {}).get("eta", np.nan),
                                 best_epoch=int(getattr(tr, "best_epoch", -1)),
                                 conv_epoch=ce, converged=int(isc)))
                n_new += 1
                if ep == max(BUDGETS):  # 只在最大预算存 L2 轨迹
                    for st, v in zip(dyn.get("eval_step", []), dyn.get("L2_traj", [])):
                        traj.append(dict(nu=nu, seed=seed, beta=beta, step=int(st), L2=float(v)))
                print(f"[R2] ν{nu} s{seed} ep{ep} β{beta:g}: L2={rows[-1]['L2']:.3e}")
    # 稳定排序后整体写回（旧+新合并，幂等）
    rows.sort(key=lambda r: (r["nu"], r["seed"], r["budget"], r["beta"]))
    traj.sort(key=lambda t: (t["nu"], t["seed"], t["beta"], t["step"]))
    mc.write_csv(os.path.join(OUT, "r2_runs.csv"), rows)
    mc.write_csv(os.path.join(OUT, "r2_traj.csv"), traj)
    print(f"[R2] 本次新训 {n_new} 组，合并后合计 {len(rows)} 行（{len(SEEDS)} 种子设计）")
    return rows, traj

def summarize(rows):
    inv = []
    for nu in NUS:
        for ep in BUDGETS:
            sub = [r for r in rows if r["nu"] == nu and r["budget"] == ep]
            if not sub: continue
            nseed = len({r["seed"] for r in sub})
            bstars, spear = [], []
            l2_at = {}
            for seed in sorted({r["seed"] for r in sub}):
                z = sorted([(r["beta"], r["L2"]) for r in sub if r["seed"] == seed])
                zf = [(b, v) for b, v in z if np.isfinite(v)]
                if zf: bstars.append(min(zf, key=lambda t: t[1])[0])
                sp = _spearman([b for b, _ in z], [r["eta_gate"] for r in sub if r["seed"] == seed])
                if np.isfinite(sp): spear.append(sp)
            for b in BETAS:
                v = np.asarray([r["L2"] for r in sub if r["beta"] == b], float); v = v[np.isfinite(v)]
                l2_at[b] = float(np.median(v)) if v.size else np.nan
            inv.append(dict(nu=nu, budget=ep, n=nseed,
                            beta_star_med=np.median(bstars) if bstars else np.nan,
                            eta_beta_spearman=np.median(spear) if spear else np.nan,
                            eta_applicable=int(3.0*nu < 1.0-1e-12),  # 3ν>=1: 激波带覆盖全域、无带外光滑区, η 不适用
                            L2_b1=l2_at.get(1.0, np.nan), L2_b3=l2_at.get(3.0, np.nan),
                            L2_b8=l2_at.get(8.0, np.nan),
                            conv_epoch_med=np.median([r["conv_epoch"] for r in sub if r.get("conv_epoch", -1) >= 0]),
                            frac_converged=np.mean([r.get("converged", 0) for r in sub])))
    for nu in NUS:
        for lo, hi in zip(BUDGETS[:-1], BUDGETS[1:]):
            a = next((s for s in inv if s["nu"] == nu and s["budget"] == lo), None)
            b = next((s for s in inv if s["nu"] == nu and s["budget"] == hi), None)
            if a and b:
                parts = []
                for be, k in [(1, "L2_b1"), (3, "L2_b3"), (8, "L2_b8")]:
                    va, vb = a.get(k), b.get(k); rv = np.nan
                    if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb) and va > 0:
                        rv = abs(vb - va) / (abs(va) + 1e-12)
                    parts.append(f"b{be}={rv:.3f}")
                print(f"[R2 翻倍 {lo}->{hi}] ν={nu}: " + " ".join(parts))
    mc.write_csv(os.path.join(OUT, "r2_invariance.csv"), inv)
    return inv

def _seed_beta_star(sub, seed):
    z = [(r["beta"], r["L2"]) for r in sub if r["seed"] == seed and np.isfinite(r["L2"])]
    return min(z, key=lambda t: t[1])[0] if z else np.nan

def bootstrap(rows, B=BOOT_B, k=len(mc.SEEDS_NEW), rng_seed=BOOT_SEED):
    """11 抽 k=5（SEEDS_NEW 规模）无放回子采样，量化小样本波动；与全量 11 颗对照。"""
    rng = np.random.default_rng(rng_seed)
    recs = []
    for nu in NUS:
        seeds = sorted({r["seed"] for r in rows if r["nu"] == nu})
        if len(seeds) <= k:
            print(f"[R2 bootstrap] ν={nu} 现有种子 {len(seeds)} ≤ {k}，暂不做子采样（补齐 11 后再算）"); continue
        for ep in BUDGETS:
            sub = [r for r in rows if r["nu"] == nu and r["budget"] == ep]
            # 全量结论
            full_bs = [_seed_beta_star(sub, s) for s in seeds]; full_bs = [x for x in full_bs if np.isfinite(x)]
            full_bstar = float(np.median(full_bs)) if full_bs else np.nan
            def medL2(b, ss):
                v = np.asarray([r["L2"] for r in sub if r["beta"] == b and r["seed"] in ss and np.isfinite(r["L2"])])
                return float(np.median(v)) if v.size else np.nan
            full_l2 = {b: medL2(b, set(seeds)) for b in BETAS}
            # 翻倍 4k->8k 最大相对变化（全量）
            def maxrel(ss):
                hi = {b: medL2(b, ss) for b in BETAS}
                lo_sub = [r for r in rows if r["nu"] == nu and r["budget"] == 4000]
                lo = {b: medL2_from(lo_sub, b, ss) for b in BETAS}
                vals = [abs(hi[b]-lo[b])/(abs(lo[b])+1e-12) for b in BETAS
                        if np.isfinite(hi[b]) and np.isfinite(lo[b]) and lo[b] > 0]
                return max(vals) if vals else np.nan
            def medL2_from(sub, b, ss):
                v = np.asarray([r["L2"] for r in sub if r["beta"] == b and r["seed"] in ss and np.isfinite(r["L2"])])
                return float(np.median(v)) if v.size else np.nan
            full_maxrel = maxrel(set(seeds))
            bag = {"bstar": [], "L2_b1": [], "L2_b3": [], "L2_b8": [], "maxrel_4k_8k": []}
            bstar1_frac = []
            for _ in range(B):
                ss = set(rng.choice(seeds, size=k, replace=False).tolist())
                bs = [_seed_beta_star(sub, s) for s in ss]; bs = [x for x in bs if np.isfinite(x)]
                if bs:
                    bm = float(np.median(bs)); bag["bstar"].append(bm); bstar1_frac.append(float(np.mean([x == 1.0 for x in bs])))
                for b, key in zip(BETAS, ["L2_b1", "L2_b3", "L2_b8"]):
                    v = medL2(b, ss)
                    if np.isfinite(v): bag[key].append(v)
                mr = maxrel(ss)
                if np.isfinite(mr): bag["maxrel_4k_8k"].append(mr)
            def q(a, qq):
                a = np.asarray(a, float); a = a[np.isfinite(a)]
                return float(np.percentile(a, qq)) if a.size else np.nan
            for metric in ["bstar", "L2_b1", "L2_b3", "L2_b8", "maxrel_4k_8k"]:
                fullv = {"bstar": full_bstar, "L2_b1": full_l2[1.0], "L2_b3": full_l2[3.0],
                         "L2_b8": full_l2[8.0], "maxrel_4k_8k": full_maxrel}[metric]
                recs.append(dict(nu=nu, budget=ep, metric=metric, n_full=len(seeds), n_sub=k, B=B,
                                 full11=fullv, p05=q(bag[metric], 5), p50=q(bag[metric], 50),
                                 p95=q(bag[metric], 95)))
            if bstar1_frac:
                recs.append(dict(nu=nu, budget=ep, metric="frac_seed_bstar_eq1", n_full=len(seeds),
                                 n_sub=k, B=B, full11=float(np.mean([x == 1.0 for x in full_bs])),
                                 p05=q(bstar1_frac, 5), p50=q(bstar1_frac, 50), p95=q(bstar1_frac, 95)))
            print(f"[R2 bootstrap] ν={nu} budget={ep}: 全量β*={full_bstar:g}; "
                  f"抽{k} β* 5/50/95 分位={q(bag['bstar'],5):g}/{q(bag['bstar'],50):g}/{q(bag['bstar'],95):g}; "
                  f"4k→8k 最大相对变化全量={full_maxrel:.3f}、子样本中位={q(bag['maxrel_4k_8k'],50):.3f}")
    if recs:
        mc.write_csv(os.path.join(OUT, "r2_bootstrap.csv"), recs)
    else:
        print("[R2 bootstrap] 种子未补齐，未写 r2_bootstrap.csv")
    return recs

def plot_beta_star(rows):
    fig, a = plt.subplots(figsize=(6.6, 4.4))
    for nu in NUS:
        xs, ym = [], []
        for ep in BUDGETS:
            sub = [r for r in rows if r["nu"] == nu and r["budget"] == ep]
            bs = []
            for s in {r["seed"] for r in sub}:
                z = [(r["beta"], r["L2"]) for r in sub if r["seed"] == s and np.isfinite(r["L2"])]
                if z: bs.append(min(z, key=lambda t: t[1])[0])
            xs.append(ep); ym.append(np.median(bs) if bs else np.nan)
        a.plot(xs, ym, "o-", color=NU_COLOR[nu], label=f"ν={nu} (n={len({r['seed'] for r in rows if r['nu']==nu})})")
    a.axhline(1, color="k", ls=":", lw=1); a.set_xscale("log")
    a.set_xlabel("Phase1 训练预算（步）"); a.set_ylabel("β* 中位")
    a.set_title("β* 不随预算翻倍而改变"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); p = os.path.join(OUT, "fig_R2_beta_star.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

def plot_L2_budget(rows):
    fig, ax = plt.subplots(1, len(NUS), figsize=(5.2 * len(NUS), 4.2), squeeze=False)
    for j, nu in enumerate(NUS):
        a = ax[0][j]
        for bi, b in enumerate(BETAS):
            xs, ym = [], []
            for ep in BUDGETS:
                v = np.asarray([r["L2"] for r in rows if r["nu"] == nu and r["budget"] == ep and r["beta"] == b], float)
                v = v[np.isfinite(v)]; xs.append(ep); ym.append(np.median(v) if v.size else np.nan)
            a.plot(xs, ym, "o-", color=BCOL[bi], label=f"β={b:g}")
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("训练预算（步）"); a.set_ylabel("L2 中位")
        a.set_title(f"ν={nu}：预算翻倍后 L2 进入平台"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "fig_R2_L2_budget.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

def plot_convergence(traj):
    if not traj: print("[R2] 无轨迹数据"); return
    fig, ax = plt.subplots(1, len(NUS), figsize=(5.2 * len(NUS), 4.2), squeeze=False)
    for j, nu in enumerate(NUS):
        a = ax[0][j]
        for bi, b in enumerate(BETAS):
            d = [t for t in traj if t["nu"] == nu and t["beta"] == b]
            if not d: continue
            steps = sorted({t["step"] for t in d})
            ym = [np.median([t["L2"] for t in d if t["step"] == s]) for s in steps]
            a.plot(steps, ym, "-", color=BCOL[bi], lw=1.6, label=f"β={b:g}")
        for ep in BUDGETS[:-1]:
            a.axvline(ep, color="r", ls="--", lw=.8, alpha=.6)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("训练步"); a.set_ylabel("验证 L2")
        a.set_title(f"ν={nu}：L2 收敛轨迹（红虚线=预算档）"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "fig_R2_convergence.png"); fig.savefig(p, dpi=150); plt.close(fig); print("saved", p)

def main():
    rows, traj = collect(); inv = summarize(rows)
    boot = bootstrap(rows)
    plot_beta_star(rows); plot_L2_budget(rows); plot_convergence(traj)
    print("\n[R2] ν    budget  n   β*med  Spearman  conv_med  fracConv")
    for s in inv:
        print(f"  {s['nu']:<6} {s['budget']:<6} {s['n']:<3} {s['beta_star_med']:>4g}  "
              f"{s['eta_beta_spearman']:+.2f}     {s['conv_epoch_med']:>6.0f}   {s['frac_converged']:.2f}")
    print("\n[R2] 完成。产物在", OUT)

if __name__ == "__main__":
    main()
