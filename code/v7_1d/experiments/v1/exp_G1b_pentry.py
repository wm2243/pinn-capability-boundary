# -*- coding: utf-8 -*-
"""
实验 G1b（补 G1 的口径缺口）：权重【前置】的好盆地进入率 p_entry + ν=.01 傅里叶尺度对照
=====================================================================================
为什么需要 G1b（G1 测不到的那一环）：
  G1 的 Phase0 是【统一、β=1】的（enter_basin_cached 缓存），只在 Gate 点之后才换族续训，
  所以 G1 回答的是“事后换权重能否【救援】已形成的盆地”（结论：坏盆地梯度僵死、不可逆）。
  但 F3 课程“先减权提高进入好盆地概率”的前提，是【从一开始就用不同权重】会改变到达 Gate
  时的好/坏盆地占比——这必须让 Phase0 本身就带权重，正是本实验 Part A。

Part A  权重前置 p_entry（核心）：同一 (nu,seed)，Phase0(σ课程 + K=4 multistart + truth 选择，
        与标准 Gate 完全同流程) 分别用
          uniform  c=1    标准 PINN（直接复用 _gate_cache，零额外算力，与 G1 同一缓存）
          down_inv c=10   逆式减权（与 G1/F2b 减权口径一致；激波带端点权重 1/10）
          down_lin c=10   = 统一有理族 β=0.1（w=1-(1-1/10)q，激波带端点也是 0.1，等端点）
                           —— 同时检验“分子/分母两种减权在早期 p_entry 上是否等价”
          rational c=10   一开始就增权（反向对照，预期把激波放大、更易焊进坏盆地、p_entry 最低）
        比较四族 Phase0 末 L2、good=(L2<.06) 占比 p_entry（Wilson95），并做【同种子配对翻转】
        （减权相对 uniform：救回 bad→good 几个、丢 good→bad 几个，McNemar 式，比边际比例更有力）。

Part B  ν=.01 傅里叶尺度对照（排查 G1 中 .01 卡在 .03-.1 的高频错配）：
        仅 ν=.01，σ课程终点 SIG ∈ {15(基线,复用缓存),10,5}（σ_lo=σ_hi/3 保持 1:3 比例），
        Phase0(uniform,K=4) → Phase1 rational c=10 训 8000（Phase1 带宽自动跟随 basin.sigma_end）。
        若降尺度后终态 L2 显著进入 .01 以下 → 论文注明“频率尺度须匹配解光滑度”，.01 用低尺度；
        若仍卡 .03-.1 → .01 移出主扫描。判据在跑前写死，不事后调阈值。

设计规范（与 G1/E 系列一致）：不改公共代码；规则等距种子 SEEDS_WIDE=0,5,..,95（SEEDS_EXT 超集）；
        逐条落盘、可中断续跑、重跑自动跳过已完成 key；真解只用于离线 truth 选择/评估。
产出 results/v6/exp_G1b_pentry/：
        g1b_pentry_runs.csv   Part A 逐 (nu,seed,family) 的 Phase0 结果
        g1b_pentry_agg.csv    Part A 分(nu,family)/pooled 的 n/p_entry/Wilson/L2 中位·IQR
        g1b_flip.csv          Part A 减权 vs uniform 同种子配对翻转（2x2）
        g1b_nu01_sigma.csv    Part B 逐(seed,sig) phase0+phase1
        g1b_nu01_agg.csv      Part B 分 sig 汇总
        fig_g1b_pentry.png / fig_g1b_flip.png / fig_g1b_nu01.png
SMOKE：V8_SMOKE=1（单 ν、单种子、K=1、240 步，自检流程）；运行：python exp_G1b_pentry.py
=====================================================================================
"""
import os, sys, csv, copy, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager

_HERE = Path(__file__).resolve(); _PKG = _HERE.parent.parent
for _p in (str(_PKG), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import run_two_stage_v6 as r2s
import v8_matrix_common as mc
from trainers.pinn_trainer import PINNTrainer
from losses.family_weighting_loss import FamilyWeightingLoss, UP_FAMILIES, DOWN_FAMILIES

for _c in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(_c):
        font_manager.fontManager.addfont(_c); plt.rcParams["font.family"] = font_manager.FontProperties(fname=_c).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

OUT = str(mc.RES_ROOT / "exp_G1b_pentry"); os.makedirs(OUT, exist_ok=True)
PA_CSV = os.path.join(OUT, "g1b_pentry_runs.csv")
PB_CSV = os.path.join(OUT, "g1b_nu01_sigma.csv")
DEVICE = mc.DEVICE

# ----------------------------- 固定配置（实验前锁死，不按结果挑）-----------------------------
NUS = [0.01, 0.005, 0.003, 0.001]
SEEDS_WIDE = list(range(0, 100, 5))          # 20 个，G1 同口径、SEEDS_EXT(11) 超集
PA_JOBS = [("down_inv", 10.0), ("down_lin", 10.0), ("rational", 10.0)]   # uniform 走缓存，不在此列
GATE = mc.GATE_L2_MAX                         # 0.06，与 G1 同一好/坏阈值
K_MULTI = 4
P0_EP = 2000
P1_EP = 8000
P1_CONTRAST = 10.0
# Part B：ν=.01 尺度网格（15 为基线，复用标准缓存；低尺度保持 σ_lo:σ_hi=1:3）
SIG_GRID = [15.0, 10.0, 5.0]
NU01 = 0.01

if mc.SMOKE:
    NUS = [0.005]; SEEDS_WIDE = [0]; K_MULTI = 1; P0_EP, P1_EP = 240, 240
    PA_JOBS = [("down_inv", 10.0)]; SIG_GRID = [15.0, 5.0]

UP_C, DOWN_C, UNI_C = "#E8A24B", "#4B86B4", "#9A9A9A"
def fcol(f): return UP_C if f in UP_FAMILIES else (DOWN_C if f in DOWN_FAMILIES else UNI_C)


def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k / n; den = 1 + z*z/n
    cen = (p + z*z/(2*n))/den
    half = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)))/den
    return max(0., cen-half), min(1., cen+half)


# ============ Phase0（复刻 r2s.run_phase0，唯一区别：criterion 换族 + 可覆盖 σ 带宽）============
def run_phase0_family(nu, base_seed, xt, ut, family, contrast,
                      sigma_lo=r2s.SIGMA_LO, sigma_hi=r2s.SIGMA_HI_CFG, sigma_T=r2s.SIGMA_T):
    """同 seed 的 B0 随机方向由 set_seed 保证一致，故跨族/跨 σ 只改权重或带宽，不改随机方向（控制变量）。"""
    cands = []
    for k in range(K_MULTI):
        seed = base_seed + k
        r2s.set_seed(seed)
        cfg = r2s.base_cfg(nu, f"g1b_{family}_s{base_seed}_{k}")
        cfg.update(dict(lr=r2s.LR_PHASE0, use_gate=True, gate_patience=3, gate_lr_gamma=0.3,
                        sigma_lo=float(sigma_lo), sigma_hi=float(sigma_hi), sigma_anneal_T=int(sigma_T),
                        fourier_scale=float(sigma_hi),
                        beta_schedule="const", loss_beta=float(contrast), beta_init=float(contrast),
                        adam_epochs=P0_EP, lbfgs_epochs=0, use_best_ckpt=True,
                        best_metric="max_L2_band"))
        model, pde, _crit0, opt, samp, vis, cfg = r2s.build_all(cfg, r2s.LR_PHASE0, float(contrast))
        crit = FamilyWeightingLoss(cfg, DEVICE, family=family)     # 唯一替换：权函数族
        trainer = PINNTrainer(model, pde, crit, opt, samp, vis, DEVICE, cfg, {"x": xt, "u_true": ut})
        trainer.train(cfg, {"x": xt, "u_true": ut}, P0_EP, 0, adaptive_freq=500)
        if not crit.field_frozen:
            crit.freeze_residual_field(model, pde)
        l2 = r2s.quick_l2(model, xt, ut)
        proxy = float(np.mean(trainer.loss_history[-200:]))
        cands.append(dict(seed=seed, l2=l2, proxy=proxy, gate=trainer.gate_epoch,
                          state=copy.deepcopy(model.state_dict()),
                          t_field=crit._frozen_t_field.detach().clone(),
                          sigma_end=model.get_sigma()))
        trainer.log_file.close()
    best = min(cands, key=lambda z: z["l2"])     # truth 选择，与标准 Gate 口径一致
    return best


def _f(x, d=float("nan")):
    try: return float(x)
    except Exception: return d


def _load(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def _akey(nu, seed, fam, c, tag=""):
    return (f"{float(nu):g}", str(seed), fam, f"{float(c):g}", tag)


# ================================= Part A 采集 =================================
def collect_A():
    rows = _load(PA_CSV); done = {_akey(r["nu"], r["seed"], r["family"], r["contrast"]) for r in rows}
    for nu in NUS:
        xt, ut = r2s.build_test(nu)
        for seed in SEEDS_WIDE:
            # uniform 走标准缓存（与 G1 同一 Phase0，保证可直接比较，且零额外算力）
            ku = _akey(nu, seed, "uniform", 1.0)
            if ku not in done:
                b = mc.enter_basin_cached(nu, seed, xt, ut, sigma_hi=r2s.SIGMA_HI_CFG,
                                          k=K_MULTI, phase0_epochs=P0_EP)
                rows.append(dict(part="A", nu=nu, seed=seed, family="uniform", contrast=1.0,
                                 phase0_L2=b["l2"], proxy=b.get("proxy", float("nan")),
                                 gate_epoch=b.get("gate", -1), sigma_end=b.get("sigma_end", r2s.SIGMA_HI_CFG),
                                 good=int(b["l2"] < GATE)))
                done.add(ku); mc.write_csv(PA_CSV, rows)
            for fam, c in PA_JOBS:
                kk = _akey(nu, seed, fam, c)
                if kk in done: continue
                b = run_phase0_family(nu, seed, xt, ut, fam, c)
                rows.append(dict(part="A", nu=nu, seed=seed, family=fam, contrast=c,
                                 phase0_L2=b["l2"], proxy=b["proxy"], gate_epoch=b["gate"],
                                 sigma_end=b["sigma_end"], good=int(b["l2"] < GATE)))
                done.add(kk); mc.write_csv(PA_CSV, rows)
                print(f"[G1b-A] ν={nu:g} s{seed} {fam} c{c:g}: phase0_L2={b['l2']:.3e} good={int(b['l2']<GATE)}")
    for r in rows:
        r["nu"]=_f(r["nu"]); r["seed"]=int(float(r["seed"])); r["contrast"]=_f(r["contrast"])
        r["phase0_L2"]=_f(r["phase0_L2"]); r["proxy"]=_f(r["proxy"]); r["good"]=int(float(r["good"]))
    return rows


def agg_A(rows):
    out = []
    fams = ["uniform"] + [f for f, _ in PA_JOBS]
    for scope, nus in [("pooled", sorted({r["nu"] for r in rows}))] + \
                      [("by_nu", [nu]) for nu in sorted({r["nu"] for r in rows})]:
        for fam in fams:
            z = [r for r in rows if r["family"] == fam and (scope == "pooled" or r["nu"] in nus)]
            if not z: continue
            k = int(np.sum([r["good"] for r in z])); n = len(z); lo, hi = wilson(k, n)
            L = np.asarray([r["phase0_L2"] for r in z], float)
            out.append(dict(scope=scope, nu=("all" if scope=="pooled" else nus[0]), family=fam, n=n,
                            good_k=k, p_entry=k/n, wilson_lo=lo, wilson_hi=hi,
                            L2_med=float(np.median(L)), L2_q1=float(np.quantile(L,.25)),
                            L2_q3=float(np.quantile(L,.75))))
    mc.write_csv(os.path.join(OUT, "g1b_pentry_agg.csv"), out)
    return out, fams


def paired_flip_A(rows):
    """同 (nu,seed)：每个减权族 vs uniform 的 good 翻转 2x2。"""
    uni = {(r["nu"], r["seed"]): r["good"] for r in rows if r["family"] == "uniform"}
    out = []
    for fam in [f for f, _ in PA_JOBS]:
        cell = {"GG":0,"GB":0,"BG":0,"BB":0}    # 第一个字母=uniform(G/B)，第二个=该族
        for r in rows:
            if r["family"] != fam or (r["nu"], r["seed"]) not in uni: continue
            gu, gx = uni[(r["nu"], r["seed"])], r["good"]
            cell["GG" if gu and gx else "GB" if gu else "BG" if gx else "BB"] += 1
        n = sum(cell.values())
        out.append(dict(family=fam, n=n, uniG_xG=cell["GG"], uniG_xB=cell["GB"],
                        uniB_xG=cell["BG"], uniB_xB=cell["BB"],
                        rescued_BtoG=cell["BG"], lost_GtoB=cell["GB"],
                        net=cell["BG"]-cell["GB"]))
    mc.write_csv(os.path.join(OUT, "g1b_flip.csv"), out)
    return out


# ================================= Part B 采集（ν=.01 尺度）=================================
def collect_B():
    xt, ut = r2s.build_test(NU01)
    rows = _load(PB_CSV); done = {_akey(r["nu"], r["seed"], "uniform", r["sig"], "B") for r in rows}
    for seed in SEEDS_WIDE:
        for sig in SIG_GRID:
            kk = _akey(NU01, seed, "uniform", sig, "B")
            if kk in done: continue
            slo = sig/3.0
            if abs(sig-15.) < 1e-9:
                basin = mc.enter_basin_cached(NU01, seed, xt, ut, sigma_hi=15., k=K_MULTI, phase0_epochs=P0_EP)
            else:
                basin = run_phase0_family(NU01, seed, xt, ut, "uniform", 1.0,
                                          sigma_lo=slo, sigma_hi=sig, sigma_T=r2s.SIGMA_T)
            m, _d, _x, _c, _t = mc.train_phase1_family(
                basin, NU01, P1_CONTRAST, seed, xt, ut,
                family="rational", epochs=P1_EP, tag="G1b")
            rows.append(dict(part="B", nu=NU01, seed=seed, sig=sig,
                             phase0_L2=basin["l2"], p0_good=int(basin["l2"] < GATE),
                             L2=m.get("L2_Error", np.nan), l2_band=m.get("l2_band", np.nan),
                             success=int(m.get("L2_Error", 9.) < 1e-2)))
            done.add(kk); mc.write_csv(PB_CSV, rows)
            print(f"[G1b-B] .01 s{seed} sig{sig:g}: p0={basin['l2']:.3e} -> L2={rows[-1]['L2']:.3e} "
                  f"band={rows[-1]['l2_band']:.3e} succ={rows[-1]['success']}")
    for r in rows:
        for c in ("nu","sig","phase0_L2","L2","l2_band"): r[c]=_f(r[c])
        r["seed"]=int(float(r["seed"])); r["p0_good"]=int(float(r["p0_good"])); r["success"]=int(float(r["success"]))
    return rows


def agg_B(rows):
    out=[]
    for sig in SIG_GRID:
        z=[r for r in rows if abs(r["sig"]-sig)<1e-9]
        if not z: continue
        L=np.asarray([r["L2"] for r in z],float); B=np.asarray([r["l2_band"] for r in z],float); P=np.asarray([r["phase0_L2"] for r in z],float)
        k=int(np.sum([r["success"] for r in z])); lo,hi=wilson(k,len(z))
        out.append(dict(sig=sig,n=len(z),p0_good_rate=float(np.mean([r["p0_good"] for r in z])),
                        succ_k=k,success=k/len(z),wilson_lo=lo,wilson_hi=hi,
                        L2_med=float(np.median(L)),band_med=float(np.median(B)),p0_med=float(np.median(P))))
    mc.write_csv(os.path.join(OUT,"g1b_nu01_agg.csv"),out); return out


# ================================= 出图 =================================
def plot_A(agg, fams):
    a=[r for r in agg if r["scope"]=="pooled"]; order=[f for f in fams if any(r["family"]==f for r in a)]
    a=sorted(a,key=lambda r:fams.index(r["family"]))
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    xs=np.arange(len(a)); rates=[r["p_entry"] for r in a]
    lo=[r["p_entry"]-r["wilson_lo"] for r in a]; hi=[r["wilson_hi"]-r["p_entry"] for r in a]
    ax.bar(xs,rates,.66,color=[fcol(r["family"]) for r in a],yerr=[lo,hi],capsize=4,
           error_kw=dict(ecolor="#222",lw=1.2))
    for i,r in enumerate(a): ax.text(i,r["p_entry"]+.03,f"n={r['n']}\n{r['p_entry']:.0%}",ha="center",fontsize=9)
    ax.set_xticks(xs,[r["family"] for r in a]); ax.set_ylim(0,1.12)
    ax.set_ylabel("Phase0 末好盆地占比 p_entry（L2<.06）"); ax.grid(alpha=.3,axis="y")
    ax.set_title("Part A 权重前置 p_entry（合并4ν×20种子，误差棒=Wilson95%）")
    fig.tight_layout(); p=os.path.join(OUT,"fig_g1b_pentry.png"); fig.savefig(p,dpi=150); plt.close(fig); print("saved",p)


def plot_flip(flips):
    if not flips: return
    fig,ax=plt.subplots(figsize=(8.2,4.4))
    labels=["uniG→xG\n保持好","uniG→xB\n好变坏","uniB→xG\n坏救回","uniB→xB\n保持坏"]
    keys=["uniG_xG","uniG_xB","rescued_BtoG","uniB_xB"]
    xs=np.arange(len(labels)); w=.36
    for i,row in enumerate(flips):
        off=(i-(len(flips)-1)/2)*w
        ax.bar(xs+off,[row[k] for k in keys],w,
               label=f"{row['family']}（净救回 {row['net']:+d}）",color=fcol(row["family"]))
    ax.set_xticks(xs,labels); ax.set_ylabel("同 (ν,seed) 配对数"); ax.legend(fontsize=9); ax.grid(alpha=.3,axis="y")
    ax.set_title("Part A 配对翻转：相对标准 uniform（净救回=坏→好 − 好→坏）")
    fig.tight_layout(); p=os.path.join(OUT,"fig_g1b_flip.png"); fig.savefig(p,dpi=150); plt.close(fig); print("saved",p)


def plot_B(aggB):
    if not aggB: return
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    sigs=[r["sig"] for r in aggB]
    ax[0].plot(sigs,[r["L2_med"] for r in aggB],"o-",label="终态 L2 中位",color="#c0392b")
    ax[0].plot(sigs,[r["p0_med"] for r in aggB],"s--",label="Phase0 L2 中位",color="#2c3e50")
    ax[0].axhline(1e-2,color="#3a8a18",ls=":",label="成功线 .01"); ax[0].axhline(.06,color="#888",ls=":")
    ax[0].invert_xaxis(); ax[0].set_yscale("log"); ax[0].set_xlabel("σ 课程终点带宽（降尺度→）"); ax[0].set_ylabel("L2")
    ax[0].set_title("ν=.01：降傅里叶尺度能否解除高频错配"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    xs=np.arange(len(aggB)); rates=[r["success"] for r in aggB]
    lo=[r["success"]-r["wilson_lo"] for r in aggB]; hi=[r["wilson_hi"]-r["success"] for r in aggB]
    ax[1].bar(xs,rates,.6,color="#4B86B4",yerr=[lo,hi],capsize=4,error_kw=dict(ecolor="#222"))
    for i,r in enumerate(aggB): ax[1].text(i,r["success"]+.03,f"σ={r['sig']:g}\nn={r['n']}\n{r['success']:.0%}",ha="center",fontsize=8.5)
    ax[1].set_xticks(xs,[f"σ={r['sig']:g}" for r in aggB]); ax[1].set_ylim(0,1.12)
    ax[1].set_ylabel("Phase1 终态 L2<.01 成功率"); ax[1].set_title("ν=.01 成功率随尺度"); ax[1].grid(alpha=.3,axis="y")
    fig.tight_layout(); p=os.path.join(OUT,"fig_g1b_nu01.png"); fig.savefig(p,dpi=150); plt.close(fig); print("saved",p)


def main():
    rowsA = collect_A(); agg, fams = agg_A(rowsA); flips = paired_flip_A(rowsA)
    rowsB = collect_B(); aggB = agg_B(rowsB)
    plot_A(agg, fams); plot_flip(flips); plot_B(aggB)
    print("\n===== Part A p_entry（pooled）=====")
    for r in sorted([z for z in agg if z["scope"]=="pooled"], key=lambda z:-z["p_entry"]):
        print(f"  {r['family']:<9} n={r['n']:<3} p_entry={r['p_entry']:.2f} "
              f"[{r['wilson_lo']:.2f},{r['wilson_hi']:.2f}] L2med={r['L2_med']:.2e}")
    print("\n===== Part A 配对翻转（vs uniform）=====")
    for r in flips:
        print(f"  {r['family']:<9} n={r['n']} 救回B→G={r['rescued_BtoG']} 丢G→B={r['lost_GtoB']} 净={r['net']:+d}")
    print("\n===== Part B ν=.01 尺度 =====")
    for r in aggB:
        print(f"  σ={r['sig']:g}: p0good={r['p0_good_rate']:.2f} success={r['success']:.2f} "
              f"L2med={r['L2_med']:.2e} bandmed={r['band_med']:.2e}")
    print("\n[G1b] 完成，产物在", OUT)


if __name__ == "__main__":
    main()
