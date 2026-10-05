# -*- coding: utf-8 -*-
"""
实验 Gstab：梯度稳定性诊断 —— 检验“减权是否抑制梯度爆炸”（机理 D1 梯度平衡 / D2 有界影响-软裁剪 / D4 间断弱化）
=====================================================================================
为什么单独做这个实验（上一轮用 p_entry/终值 L2 判减权是对错了指标）：
  * G1b 已证：常驻/前置减权【不提高】好盆地进入率 p_entry；G1 good 已证：好盆地内减权【不利于】终值精度。
    但减权最初的设计目标是【抑制激波窄带主导的梯度尖峰/爆炸】，这是一条独立通道，目标量是
    “裁剪前原始梯度范数的峰值/尾部、Hessian λmax/条件数、NaN-发散率、可承受最大 lr、LBFGS 失稳率”，
    与 p_entry、终值 L2 无关。

【关键：两道“减权”的因子拆解（clip × weight）】
  默认训练管线 grad_clip=1.0 本身就是第一道限幅——它是【全局、各向同性】的 L2 范数裁剪；
  空间减权是第二道——按残差对比度【逐点、各向异性】地降权。二者叠加等于“两次减权”，若只在 clip
  常开下比较，第二道的边际会被第一道掩盖（post-clip 范数都被砍到 ≤1）。故本实验做 2×4 因子：
    clip=0（关裁剪）：看减权【原生】对 pre-clip 梯度峰值/λmax 的压制（D1/D2 最干净证据）；
    clip=1（默认裁剪）：看减权是否【降低全局裁剪触发率 frac_clip】——若 down 组更少触发裁剪，
                       说明第二道把尖峰在反传前就抹平、与第一道互补；若各族 post-clip 无差异则二者冗余。

三个 Stage：
  Stage-E 早期原生稳定性：同一随机初始化(set_seed 保证 B0/θ0/采样一致，只换权族)，clip{0,1}×四族，
           带 σ 表示课程、adaptive 缓变权场，从零训 N_E 步；记 pre-clip 梯度峰值/尾部、裁剪触发率、
           静态冻结 λmax/κ、发散率。
  Stage-B1 可承受 lr：统一 Phase0 到同一【好盆地】Gate，从同一 state+冻结场换族，clip{0,1}×四族×lr 网格
           各短跑，找“不 NaN/不发散”的最大稳定 lr（D1/D2 预测：减权能承受更大步长）。
  Stage-B2 LBFGS：同一好盆地换族，Adam 短预热→冻结场→LBFGS（二阶法无梯度裁剪，对 λmax 最敏感），比失稳率。

四族（唯一变量是权函数方向/形状，contrast=10，减权端点 1/10）：
  uniform β=1 基线；rational β=10 增权（预期抬高峰值、更易炸，反向对照）；down_inv/down_lin c=10 减权（被检验对象）。

【跑前写死的判据 pre-registered，跑完照此判，不事后挪阈值】
  H1 早期峰值：clip=0 下 pre-clip g_peak、静态 λmax 中位满足 down ≤ uniform ≤ rational；发散率同向。
  H1b 两道减权关系：clip=1 下【裁剪触发率 frac_clip】满足 down ≤ uniform ≤ rational（减权减少对全局裁剪的依赖）。
  H2 可承受步长：最大稳定 lr down ≥ uniform ≥ rational（分 clip=0/1 各报一次）。
  H3 二阶法：LBFGS NaN/发散率 down ≤ uniform ≤ rational。
  全部给同种子配对与效应量；数据不支持就如实写“不支持”，不粉饰。

规范：不改公共代码；规则等距种子；逐条落盘、可中断续跑、重跑跳过；真解只用于离线评估。
产出 results/v6/exp_Gstab_grad/：gstab_early_runs.csv / gstab_lr_runs.csv / gstab_lbfgs_runs.csv /
  gstab_agg.csv；fig_gstab_early.png / fig_gstab_lr.png / fig_gstab_lbfgs.png
SMOKE：V8_SMOKE=1 自检；运行：python exp_Gstab_grad_stability.py
=====================================================================================
"""
import os, sys, csv, copy, math
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
from losses.family_weighting_loss import FamilyWeightingLoss, UP_FAMILIES, DOWN_FAMILIES

for _c in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(_c):
        font_manager.fontManager.addfont(_c); plt.rcParams["font.family"] = font_manager.FontProperties(fname=_c).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

OUT = str(mc.RES_ROOT / "exp_Gstab_grad"); os.makedirs(OUT, exist_ok=True)
EARLY_CSV = os.path.join(OUT, "gstab_early_runs.csv")
LR_CSV = os.path.join(OUT, "gstab_lr_runs.csv")
LBFGS_CSV = os.path.join(OUT, "gstab_lbfgs_runs.csv")
DEVICE = mc.DEVICE

CONTRAST = 10.0
FAMS = ["uniform", "down_inv", "down_lin", "rational"]
CLIP_E = [0.0, 1.0]          # Stage-E 两道减权因子：关 / 默认1.0
CLIP_B = [0.0, 1.0]          # Stage-B1 同样分档
NUS_E = [0.001, 0.005, 0.01]
NUS_B = [0.001, 0.005]
SEEDS_E = mc.SEEDS_EXT
SEEDS_B = mc.SEEDS_NEW
N_E = 3000; N_B1 = 1200; N_B2_ADAM = 500; N_B2_LBFGS = 100
LR_GRID = [5e-4, 1e-3, 3e-3, 1e-2]
PROBE_FREQ = 200
HESS_STEPS_E = [0, 750, 1500, 2999]
N_LANCZOS = 30; N_PROBE_GRID = 2048; DIVERGE_L2 = 5.0
SIG_LO, SIG_HI, SIG_T = r2s.SIGMA_LO, r2s.SIGMA_HI_CFG, r2s.SIGMA_T

if mc.SMOKE:
    NUS_E=[0.005]; NUS_B=[0.005]; SEEDS_E=[0]; SEEDS_B=[0]
    N_E,N_B1,N_B2_ADAM,N_B2_LBFGS=120,120,60,8
    HESS_STEPS_E=[0,119]; N_LANCZOS=8; LR_GRID=[5e-4,3e-3]; CLIP_B=[0.0]

UP_C, DOWN_C, UNI_C = "#E8A24B", "#4B86B4", "#9A9A9A"
def fcol(f): return UP_C if f in UP_FAMILIES else (DOWN_C if f in DOWN_FAMILIES else UNI_C)


def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p=k/n; den=1+z*z/n; cen=(p+z*z/(2*n))/den
    half=(z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den
    return max(0.,cen-half), min(1.,cen+half)

def _sigma_at(ep, lo, hi, T):
    if lo is None or T<=0: return float(hi)
    if ep>=T: return float(hi)
    return float(lo+(hi-lo)*0.5*(1-math.cos(math.pi*ep/max(T,1))))

def _band_l2(model, xt, ut, nu, c=3.0):
    with torch.no_grad(): up=model(xt).detach().cpu().numpy().flatten()
    xn=xt.detach().cpu().numpy().flatten(); m=np.abs(xn)<c*nu
    return float(np.linalg.norm((up-ut)[m])/max(np.linalg.norm(ut[m]),1e-8)) if m.any() else 0.0

def _build(nu, seed, family, contrast, lr, sigma, tag, grad_clip, spatial_ema=0.9, freeze_load=None):
    r2s.set_seed(int(seed))
    cfg = r2s.base_cfg(nu, tag)
    cfg.update(dict(grad_clip=float(grad_clip), lr=lr, fourier_scale=float(sigma),
                    sigma_lo=None, sigma_hi=None, sigma_anneal_T=0,
                    loss_beta=float(contrast), spatial_ema_decay=spatial_ema,
                    update_field_freq=100, use_gate=False, weight_norm="raw"))
    model, pde, _c0, opt, samp, vis, cfg = r2s.build_all(cfg, lr, float(contrast))
    crit = FamilyWeightingLoss(cfg, DEVICE, family=family)
    if freeze_load is not None:
        state, tfield = freeze_load
        model.load_state_dict(state); model.set_sigma(float(sigma))
        crit.load_frozen_field(tfield, beta=float(contrast))
    return model, pde, crit, opt, samp, cfg


def raw_adam_run(nu, seed, family, contrast, n_step, lr, xt, ut, grad_clip=0.0,
                 sigma_course=True, freeze_load=None, probe_freq=PROBE_FREQ,
                 hess_steps=(), tag="Gstab", verbose=False, sigma=None):
    """轻量 Adam 循环；grad_clip=0 关裁剪。记录【裁剪前】g_pre 与（可选）裁剪触发率。"""
    sigma0 = float(SIG_HI if sigma is None else sigma)
    model, pde, crit, opt, samp, cfg = _build(nu, seed, family, contrast, lr, sigma0,
                                              f"{tag}_c{grad_clip}_{family}_s{seed}", grad_clip,
                                              freeze_load=freeze_load)
    x_probe = torch.linspace(-1,1,N_PROBE_GRID,device=DEVICE).view(-1,1)
    gpre_seq, gpost_seq, clip_seq, part_seq, hess_seq = [], [], [], [], []
    blown, blow_step, blow_reason = 0, -1, ""
    for ep in range(n_step):
        sig = _sigma_at(ep,SIG_LO,SIG_HI,SIG_T) if (sigma_course and freeze_load is None) else float(sigma0)
        model.set_sigma(sig)
        if (not crit.field_frozen) and ep % cfg["update_field_freq"]==0:
            crit.update_residual_field(model, pde)
        data=samp.sample(cfg["n_pde"],mode="random"); x=data["x_pde"]
        res=pde.compute_residual(model,x); loss,_=crit(res,x_pde=x)
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            gsq=sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None)
            gpre=float(torch.sqrt(gsq)) if gsq is not None else float("nan")
        lv=float(loss.detach())
        if not math.isfinite(gpre) or not math.isfinite(lv):
            blown,blow_step,blow_reason=1,ep,"nan_grad_or_loss"; break
        # 第一道限幅（可选）：L2 裁剪，裁剪后范数=min(gpre,thr)；记录是否触发
        is_clip=0; gpost=gpre
        if grad_clip and grad_clip>0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            is_clip=int(gpre > grad_clip*1.001); gpost=min(gpre,float(grad_clip))
        gpre_seq.append(gpre); gpost_seq.append(gpost); clip_seq.append(is_clip)
        opt.step()
        if ep%probe_freq==0 or ep==n_step-1:
            pe=mc.grad_partition_energy(model,pde,crit,x_probe,nu); pe["step"]=ep; part_seq.append(pe)
        if ep in hess_steps:
            was=crit.field_frozen
            if not was: crit.freeze_residual_field(model,pde)
            h=mc.hessian_min_curvature(model,pde,crit,x_probe,num_lanczos=N_LANCZOS); h["step"]=ep; hess_seq.append(h)
            if not was: crit.unfreeze_residual_field()
    L2=r2s.quick_l2(model,xt,ut) if not blown else float("nan")
    band=_band_l2(model,xt,ut,nu) if not blown else float("nan")
    if not blown and (not math.isfinite(L2) or L2>DIVERGE_L2): blown,blow_reason=1,"l2_diverge"
    model.zero_grad()
    g=np.asarray(gpre_seq,float); g=g[np.isfinite(g)]
    etas=[p["eta"] for p in part_seq if math.isfinite(p.get("eta",float("nan")))]
    lmax=[h["lam_max"] for h in hess_seq if math.isfinite(h.get("lam_max",float("nan")))]
    kaps=[h["kappa"] for h in hess_seq if math.isfinite(h.get("kappa",float("nan")))]
    base=np.median(g) if g.size else float("nan")
    out=dict(blown=blown,blow_step=blow_step,blow_reason=blow_reason,
             g_peak=float(g.max()) if g.size else float("nan"),
             g_p95=float(np.quantile(g,.95)) if g.size else float("nan"),
             g_med=float(np.median(g)) if g.size else float("nan"),
             g_std=float(g.std()) if g.size else float("nan"),
             g_spikes=int(np.sum(g>10*base)) if g.size and base>0 else 0,
             frac_clip=float(np.mean(clip_seq)) if clip_seq else float("nan"),
             gpost_peak=float(np.max(gpost_seq)) if gpost_seq else float("nan"),
             eta_med=float(np.median(etas)) if etas else float("nan"),
             lmax_med=float(np.median(lmax)) if lmax else float("nan"),
             lmax_max=float(np.max(lmax)) if lmax else float("nan"),
             kappa_med=float(np.median(kaps)) if kaps else float("nan"),
             L2=L2,l2_band=band,n_rec=len(gpre_seq),
             model=model,pde=pde,crit=crit,samp=samp,cfg=cfg)
    if verbose: print(f"    [{family} clip{grad_clip:g}] blown={blown} g_peak={out['g_peak']:.2e} "
                      f"frac_clip={out['frac_clip']:.2f} λmax={out['lmax_med']:.2e} L2={L2:.2e}")
    return out

def lbfgs_run(nu, seed, family, contrast, xt, ut, basin, n_adam=N_B2_ADAM, n_lbfgs=N_B2_LBFGS, tag="GstabL"):
    sigma=float(basin["sigma_end"]); fl=(basin["state"],basin["t_field"])
    pre=raw_adam_run(nu,seed,family,contrast,n_adam,r2s.LR_PHASE1,xt,ut,grad_clip=0.0,
                     sigma_course=False,freeze_load=fl,hess_steps=(),tag=tag,sigma=sigma)
    model,pde,crit,samp,cfg=pre["model"],pre["pde"],pre["crit"],pre["samp"],pre["cfg"]
    if not crit.field_frozen: crit.freeze_residual_field(model,pde)
    data=samp.sample(cfg["n_pde"],mode="fixed")
    optL=torch.optim.LBFGS(model.parameters(),lr=float(cfg.get("lbfgs_lr",1.0)),max_iter=20,
                           tolerance_grad=1e-7,tolerance_change=1e-9,history_size=50)
    blown,blow_it=0,-1
    for it in range(n_lbfgs):
        def closure():
            optL.zero_grad(); rr=pde.compute_residual(model,data["x_pde"]); ll,_=crit(rr,x_pde=data["x_pde"]); ll.backward(); return ll
        try:
            vv=float(optL.step(closure))
            if not math.isfinite(vv): blown,blow_it=1,it; break
        except Exception: blown,blow_it=1,it; break
    L2=r2s.quick_l2(model,xt,ut) if not blown else float("nan")
    if not blown and (not math.isfinite(L2) or L2>DIVERGE_L2): blown=1
    return dict(pre_blown=pre["blown"],pre_gpeak=pre["g_peak"],lbfgs_blown=blown,lbfgs_blow_it=blow_it,
                L2=L2,l2_band=_band_l2(model,xt,ut,nu) if not blown else float("nan"))

def _load(p):
    if not os.path.exists(p): return []
    with open(p,"r",encoding="utf-8-sig") as f: return list(csv.DictReader(f))
def _key(*vals):
    out=[]
    for v in vals:
        try: out.append(f"{float(v):g}")
        except Exception: out.append(str(v))
    return tuple(out)
def _f(x,d=float("nan")):
    try: return float(x)
    except Exception: return d

_E_FIELDS=["blown","blow_step","g_peak","g_p95","g_med","g_std","g_spikes","frac_clip","gpost_peak",
           "eta_med","lmax_med","lmax_max","kappa_med","L2","l2_band","n_rec"]

# ============================== Stage-E ==============================
def collect_early():
    rows=_load(EARLY_CSV); done={_key(r["nu"],r["seed"],r["family"],r["clip"]) for r in rows}
    for nu in NUS_E:
        xt,ut=r2s.build_test(nu)
        for seed in SEEDS_E:
            for clip in CLIP_E:
                for fam in FAMS:
                    kk=_key(nu,seed,fam,clip)
                    if kk in done: continue
                    r=raw_adam_run(nu,seed,fam,CONTRAST if fam!="uniform" else 1.0,N_E,r2s.LR_PHASE0,
                                   xt,ut,grad_clip=clip,sigma_course=True,hess_steps=HESS_STEPS_E,
                                   tag="GstabE",verbose=True)
                    row=dict(nu=nu,seed=seed,family=fam,clip=clip,
                             contrast=(1.0 if fam=="uniform" else CONTRAST),
                             **{k:r[k] for k in _E_FIELDS})
                    rows.append(row); done.add(kk); mc.write_csv(EARLY_CSV,rows)
                    print(f"[Gstab-E] ν={nu:g} s{seed} clip{clip:g} {fam}: g_peak={r['g_peak']:.2e} "
                          f"frac_clip={r['frac_clip']:.2f} blown={r['blown']}")
    for r in rows:
        r["nu"]=_f(r["nu"]);r["seed"]=int(float(r["seed"]));r["clip"]=_f(r["clip"]);r["blown"]=int(float(r["blown"]))
        for c in ("g_peak","g_p95","g_med","g_std","frac_clip","gpost_peak","lmax_med","lmax_max","kappa_med","L2","l2_band","eta_med"): r[c]=_f(r[c])
    return rows

# ============================== Stage-B1 lr ==============================
def collect_lr():
    rows=_load(LR_CSV); done={_key(r["nu"],r["seed"],r["family"],r["lr"],r["clip"]) for r in rows}
    for nu in NUS_B:
        xt,ut=r2s.build_test(nu)
        for seed in SEEDS_B:
            basin=mc.enter_basin_cached(nu,seed,xt,ut,sigma_hi=SIG_HI,k=4,phase0_epochs=r2s.PHASE0_EPOCHS)
            if (not mc.SMOKE) and basin["l2"]>=mc.GATE_L2_MAX: continue
            fl=(basin["state"],basin["t_field"])
            for clip in CLIP_B:
                for fam in FAMS:
                    for lr in LR_GRID:
                        kk=_key(nu,seed,fam,lr,clip)
                        if kk in done: continue
                        r=raw_adam_run(nu,seed,fam,CONTRAST if fam!="uniform" else 1.0,N_B1,lr,xt,ut,
                                       grad_clip=clip,sigma_course=False,freeze_load=fl,hess_steps=(),
                                       tag="GstabLR",sigma=basin["sigma_end"])
                        rows.append(dict(nu=nu,seed=seed,family=fam,lr=lr,clip=clip,blown=r["blown"],
                                         blow_step=r["blow_step"],g_peak=r["g_peak"],frac_clip=r["frac_clip"],
                                         L2=r["L2"],l2_band=r["l2_band"])); done.add(kk); mc.write_csv(LR_CSV,rows)
                        print(f"[Gstab-B1] ν={nu:g} s{seed} clip{clip:g} {fam} lr={lr:g}: blown={r['blown']} g_peak={r['g_peak']:.2e}")
    for r in rows:
        r["nu"]=_f(r["nu"]);r["seed"]=int(float(r["seed"]));r["lr"]=_f(r["lr"]);r["clip"]=_f(r["clip"]);r["blown"]=int(float(r["blown"]))
        for c in ("g_peak","frac_clip","L2","l2_band"): r[c]=_f(r[c])
    return rows

# ============================== Stage-B2 LBFGS ==============================
def collect_lbfgs():
    rows=_load(LBFGS_CSV); done={_key(r["nu"],r["seed"],r["family"]) for r in rows}
    for nu in NUS_B:
        xt,ut=r2s.build_test(nu)
        for seed in SEEDS_B:
            basin=mc.enter_basin_cached(nu,seed,xt,ut,sigma_hi=SIG_HI,k=4,phase0_epochs=r2s.PHASE0_EPOCHS)
            if (not mc.SMOKE) and basin["l2"]>=mc.GATE_L2_MAX: continue
            for fam in FAMS:
                kk=_key(nu,seed,fam)
                if kk in done: continue
                r=lbfgs_run(nu,seed,fam,CONTRAST if fam!="uniform" else 1.0,xt,ut,basin)
                rows.append(dict(nu=nu,seed=seed,family=fam,**r)); done.add(kk); mc.write_csv(LBFGS_CSV,rows)
                print(f"[Gstab-B2] ν={nu:g} s{seed} {fam}: LBFGS_blown={r['lbfgs_blown']} L2={r['L2']:.2e}")
    for r in rows:
        r["nu"]=_f(r["nu"]);r["seed"]=int(float(r["seed"]))
        r["pre_blown"]=int(float(r["pre_blown"]));r["lbfgs_blown"]=int(float(r["lbfgs_blown"]))
        for c in ("pre_gpeak","L2","l2_band"): r[c]=_f(r[c])
    return rows

# ============================== 聚合 ==============================
def aggregate(er,lr,lb):
    out=[]
    def medof(z,c):
        a=np.asarray([r[c] for r in z],float); a=a[np.isfinite(a)]; return float(np.median(a)) if a.size else float("nan")
    for nu in sorted({r["nu"] for r in er}):
        for clip in sorted({r["clip"] for r in er}):
            for fam in FAMS:
                z=[r for r in er if r["nu"]==nu and r["family"]==fam and abs(r["clip"]-clip)<1e-9]
                if not z: continue
                k=sum(r["blown"] for r in z); lo,hi=wilson(k,len(z))
                out.append(dict(stage="E",nu=nu,clip=clip,family=fam,n=len(z),blow_rate=k/len(z),wlo=lo,whi=hi,
                                g_peak=medof(z,"g_peak"),g_p95=medof(z,"g_p95"),frac_clip=medof(z,"frac_clip"),
                                lmax_med=medof(z,"lmax_med"),kappa_med=medof(z,"kappa_med"),L2_med=medof(z,"L2"),
                                extra=float("nan")))
    for nu in sorted({r["nu"] for r in lr}):
        for clip in sorted({r["clip"] for r in lr}):
            for fam in FAMS:
                per={}
                for r in lr:
                    if r["nu"]==nu and r["family"]==fam and abs(r["clip"]-clip)<1e-9: per.setdefault(r["seed"],[]).append(r)
                msl=[]
                for s,zz in per.items():
                    stable=[z["lr"] for z in zz if z["blown"]==0 and np.isfinite(z["L2"]) and z["L2"]<DIVERGE_L2]
                    msl.append(max(stable) if stable else 0.0)
                if not msl: continue
                out.append(dict(stage="B1",nu=nu,clip=clip,family=fam,n=len(msl),blow_rate=float("nan"),wlo=float("nan"),whi=float("nan"),
                                g_peak=float("nan"),g_p95=float("nan"),frac_clip=float("nan"),lmax_med=float("nan"),
                                kappa_med=float("nan"),L2_med=float("nan"),extra=float(np.median(msl))))
    for nu in sorted({r["nu"] for r in lb}):
        for fam in FAMS:
            z=[r for r in lb if r["nu"]==nu and r["family"]==fam]
            if not z: continue
            k=sum(r["lbfgs_blown"] for r in z); lo,hi=wilson(k,len(z))
            out.append(dict(stage="B2",nu=nu,clip=float("nan"),family=fam,n=len(z),blow_rate=k/len(z),wlo=lo,whi=hi,
                            g_peak=float("nan"),g_p95=float("nan"),frac_clip=float("nan"),lmax_med=float("nan"),
                            kappa_med=float("nan"),L2_med=medof(z,"L2"),extra=float("nan")))
    mc.write_csv(os.path.join(OUT,"gstab_agg.csv"),out); return out

# ============================== 出图 ==============================
def plot_early(er):
    nus=sorted({r["nu"] for r in er}); clips=sorted({r["clip"] for r in er})
    metrics=[("g_peak","pre-clip 梯度峰值 g_peak（log）"),("frac_clip","全局裁剪触发率（仅 clip=1 有效）"),("lmax_med","静态冻结 λmax 中位（log）")]
    fig,ax=plt.subplots(len(metrics),len(nus),figsize=(4.8*len(nus),3.4*len(metrics)),squeeze=False)
    for i,(mc_,tt) in enumerate(metrics):
        for j,nu in enumerate(nus):
            a=ax[i,j]; offs={0.0:-0.16,1.0:0.16}
            for clip in clips:
                for fam in FAMS:
                    z=[r for r in er if r["nu"]==nu and r["family"]==fam and abs(r["clip"]-clip)<1e-9]
                    if not z: continue
                    xx=FAMS.index(fam)+offs.get(clip,0.)
                    yy=[r[mc_] for r in z]
                    a.scatter([xx]*len(yy),yy,s=20,color=fcol(fam),
                              facecolors=("none" if clip>0 else fcol(fam)),edgecolors=fcol(fam),
                              linewidths=1.4 if clip>0 else 0,
                              label=f"{fam}, clip{clip:g}" if i==0 and j==0 else None,alpha=.8)
            a.set_xticks(range(len(FAMS)),FAMS,rotation=20,fontsize=8); a.grid(alpha=.3,axis="y")
            if mc_ in ("g_peak","lmax_med"): a.set_yscale("log")
            if mc_=="frac_clip": a.set_ylim(-.05,1.05)
            a.set_title(f"ν={nu:g}  {tt}",fontsize=9)
    handles,labels=ax[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=7,ncol=4,loc="upper center",bbox_to_anchor=(.5,1.0))
    fig.tight_layout(rect=(0,0,1,.96)); p=os.path.join(OUT,"fig_gstab_early.png"); fig.savefig(p,dpi=150); plt.close(fig); print("saved",p)

def plot_lr(lr):
    if not lr: return
    nus=sorted({r["nu"] for r in lr}); clips=sorted({r["clip"] for r in lr})
    fig,ax=plt.subplots(len(clips),len(nus),figsize=(5.0*len(nus),4.0*len(clips)),squeeze=False)
    ax=np.atleast_2d(np.asarray(ax))
    for i,clip in enumerate(clips):
        for j,nu in enumerate(nus):
            a=ax[i,j]
            for fam in FAMS:
                xs=sorted({r["lr"] for r in lr}); ys=[]
                for L in xs:
                    z=[r for r in lr if r["nu"]==nu and r["family"]==fam and abs(r["clip"]-clip)<1e-9 and abs(r["lr"]-L)<1e-15]
                    ys.append(np.mean([r["blown"] for r in z]) if z else float("nan"))
                a.plot(xs,ys,"o-",color=fcol(fam),label=fam)
            a.set_xscale("log");a.set_ylim(-.05,1.05);a.grid(alpha=.3)
            a.set_title(f"ν={nu:g} clip={clip:g}：发散率随 lr（越靠右越能扛大步长）",fontsize=9)
            a.set_xlabel("lr");a.set_ylabel("发散/NaN 率");a.legend(fontsize=8)
    fig.tight_layout(); p=os.path.join(OUT,"fig_gstab_lr.png"); fig.savefig(p,dpi=150); plt.close(fig); print("saved",p)

def plot_lbfgs(lb):
    if not lb: return
    nus=sorted({r["nu"] for r in lb}); fig,ax=plt.subplots(1,len(nus),figsize=(5.2*len(nus),4.2),squeeze=False)
    for j,nu in enumerate(nus):
        rate=[]
        for fam in FAMS:
            z=[r for r in lb if r["nu"]==nu and r["family"]==fam]
            rate.append(np.mean([r["lbfgs_blown"] for r in z]) if z else float("nan"))
        xs=np.arange(len(FAMS)); ax[0,j].bar(xs,rate,.6,color=[fcol(f) for f in FAMS])
        ax[0,j].set_xticks(xs,FAMS,rotation=20);ax[0,j].set_ylim(0,1.05)
        ax[0,j].set_ylabel("LBFGS NaN/发散率");ax[0,j].set_title(f"ν={nu:g} 二阶法失稳率",fontsize=10);ax[0,j].grid(alpha=.3,axis="y")
    fig.tight_layout(); p=os.path.join(OUT,"fig_gstab_lbfgs.png"); fig.savefig(p,dpi=150); plt.close(fig); print("saved",p)

def verdict(agg):
    print("\n===== Gstab 跑前判据对照（pooled over ν，中位数）=====")
    order=["down_lin","down_inv","uniform","rational"]
    def med(stage,fam,c,clip=None):
        vals=[r[c] for r in agg if r["stage"]==stage and r["family"]==fam and (clip is None or abs(r["clip"]-clip)<1e-9)]
        a=np.asarray(vals,float); a=a[np.isfinite(a)]; return float(np.median(a)) if a.size else float("nan")
    print("[H1 早期·clip=0 关裁剪] 期望 down ≤ uniform ≤ rational")
    for f in order: print(f"   {f:<9} g_peak={med('E',f,'g_peak',0):.3e}  λmax={med('E',f,'lmax_med',0):.3e}  κ={med('E',f,'kappa_med',0):.3e}")
    print("[H1b 两道减权·clip=1] 裁剪触发率 frac_clip 期望 down ≤ uniform ≤ rational（减权减少对全局裁剪的依赖）")
    for f in order: print(f"   {f:<9} frac_clip={med('E',f,'frac_clip',1):.3f}  g_peak(裁剪前)={med('E',f,'g_peak',1):.3e}")
    print("[H2 最大稳定 lr] 期望 down ≥ uniform ≥ rational（extra 列=最大稳定 lr 中位）")
    for clip in (0.0,1.0):
        s="  clip=0" if clip==0 else "  clip=1"
        for f in order: print(f"  {s} {f:<9} maxstable_lr={med('B1',f,'extra',clip):.2e}")
    print("[H3 LBFGS 失稳率] 期望 down ≤ uniform ≤ rational")
    for f in order: print(f"   {f:<9} blow_rate={med('B2',f,'blow_rate'):.2f}")

def main():
    er=collect_early(); lr=collect_lr(); lb=collect_lbfgs()
    agg=aggregate(er,lr,lb)
    plot_early(er); plot_lr(lr); plot_lbfgs(lb); verdict(agg)
    print("\n[Gstab] 完成，产物在", OUT)

if __name__ == "__main__":
    main()
