# -*- coding: utf-8 -*-
"""
make_paper_figures.py — 论文用图重制（中/英双版、去定性口语、统一配色、只读数不重算）
=====================================================================================
不改实验脚本、不重训；从 results/v6 各实验【既有聚合 csv】读数重绘。
输出 results/v6/paper_figures/{zh,en}/fig_*.png（中/英各一份，文件名语言中立）。
配色：增权 rational/linear/band 橙系；减权 down_lin/down_inv 蓝系；等权 uniform 灰。
覆盖 V10 正文图：G1b(flip/nu01)、Gstab(early/lr)、E3b、F1(SA no-go)、F2(课程)、E1(机制A/U型)。
运行：python make_paper_figures.py
"""
import os, csv, math, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_HERE=Path(__file__).resolve(); _ROOT=_HERE.parent.parent
for _p in (str(_ROOT),str(_HERE)):
    if _p not in sys.path: sys.path.insert(0,_p)
import v8_matrix_common as mc

RES=str(mc.RES_ROOT)
OUTROOT=os.path.join(RES,"paper_figures")
os.makedirs(os.path.join(OUTROOT,"zh"),exist_ok=True); os.makedirs(os.path.join(OUTROOT,"en"),exist_ok=True)

TXT={
 "zh":dict(
  fam=dict(rational="有理增权",linear="线性增权",band="带通增权",down_lin="线性减权",down_inv="逆式减权",uniform="等权"),
  f1w=dict(const="无权",smooth="光滑缓变",up="局域增权 β=15",down="局域减权 1/β",white="非局部谱白化 W*"),
  f2s={"down->up":"先减后增","up->down":"先增后减","down_inv":"恒定逆式减权","up_rational":"恒定理化增权","uniform":"恒定等权"},
  flip_t="配对盆地流向：相对等权的好/坏盆地翻转（n=80）",
  rescued="救回 B→G", lost="丢失 G→B", netlbl="净救回", family="权函数族", cnt="配对计数",
  nu01_t="好盆地命中率随 Fourier 带宽 σ（ν=0.01，Wilson 95%）", sig="Fourier 带宽 σ", succ="好盆地命中率",
  early_t="训练早期原生梯度统计（n=11/族，未裁剪口径）",
  gpeak="原生梯度峰值 g_peak（中位，对数轴）", lmax="冻结最大特征值 λmax（中位，对数轴）",
  fclip="硬裁剪触发步占比", eL2="早期全域 L2 误差（中位）", nu="粘度 ν",
  lr_t="好盆地内终态 L2 随学习率（红线 L2=0.05）", lr="学习率（对数轴）", L2="终态 L2 误差（对数轴）",
  thresh="判据 L2=0.05", clip0="关闭全局裁剪", clip1="默认裁剪(阈1)",
  e3b_t="六族公平对照：好/坏盆地激波带误差（好 n=8 起，坏 n=3 描述性）",
  goodband="好盆地 l2_band 中位", badband="坏盆地终态 l2_band 中位（n=3）", contrast="对比度 c", band="激波带误差 l2_band（对数轴）",
  f1_t="冻结线性化算符的谱数值实验（激波线性化，离散规模 n）",
  impr="条件数比 κ_new/κ_orig（对数轴）", eoff="特征基偏离度 E_off", ngrid="离散规模 n",
  f2_t="五种权重时序策略的成功率（n=5/组）", strat="权重时序策略", rate="成功率",
  e1m_t="冻结 Gate 点能量比 η 随 β（同 θ 同 r 仅改 β）", eta="能量比 η=g_band²/g_smooth²（对数轴）", beta="权重强度 β（对数轴）",
  e1u_t="好盆地全域 L2 随 β（经验 U 型，中位）"),
 "en":dict(
  fam=dict(rational="up-rational",linear="up-linear",band="up-band",down_lin="down-linear",down_inv="down-inverse",uniform="uniform"),
  f1w=dict(const="const",smooth="smooth",up="local up β=15",down="local down 1/β",white="nonlocal whitening W*"),
  f2s={"down->up":"down->up","up->down":"up->down","down_inv":"const down-inv","up_rational":"const up-rat","uniform":"uniform"},
  flip_t="Paired basin transition vs uniform (n=80)",
  rescued="rescued B->G", lost="lost G->B", netlbl="net", family="weight family", cnt="paired count",
  nu01_t="Good-basin hit rate vs Fourier bandwidth sigma (nu=0.01, Wilson 95%)", sig="Fourier bandwidth sigma", succ="good-basin hit rate",
  early_t="Raw early-training gradient statistics (n=11/family, pre-clipping)",
  gpeak="raw gradient peak g_peak (median, log)", lmax="frozen lambda_max (median, log)",
  fclip="fraction of clipped steps", eL2="early global L2 error (median)", nu="viscosity nu",
  lr_t="In-basin terminal L2 vs learning rate (red line L2=0.05)", lr="learning rate (log)", L2="terminal L2 error (log)",
  thresh="threshold L2=0.05", clip0="clipping off", clip1="clipping on (thr=1)",
  e3b_t="Six-family controlled comparison: shock-band error (good n>=8, bad n=3 descriptive)",
  goodband="good-basin median l2_band", badband="bad-basin terminal l2_band (n=3)", contrast="contrast c", band="shock-band error l2_band (log)",
  f1_t="Spectral numerics on the frozen linearized operator (shock linearization, grid n)",
  impr="condition ratio kappa_new/kappa_orig (log)", eoff="eigenbasis mismatch E_off", ngrid="grid size n",
  f2_t="Success rate of five weight schedules (n=5 per group)", strat="weight schedule", rate="success rate",
  e1m_t="Frozen Gate energy ratio eta vs beta (same theta,r; vary beta)", eta="energy ratio eta (log)", beta="weight strength beta (log)",
  e1u_t="Good-basin global L2 vs beta (empirical U-shape, median)"),
}
COL=dict(rational="#E8A24B",linear="#F0C078",band="#D98C34",down_lin="#7FB3D5",down_inv="#4B86B4",uniform="#9A9A9A")
FAM_ORDER=["rational","linear","band","down_lin","down_inv","uniform"]
F1_KEYS=["const","smooth","up","down","white"]; F1_COL=dict(const="#9A9A9A",smooth="#8BC8AA",up="#E8A24B",down="#4B86B4",white="#C0504D")
F2_ORDER=["uniform","up_rational","down_inv","down->up","up->down"]
NUS=[0.001,0.005,0.01]

def _read(p):
    with open(p,encoding="utf-8-sig") as f: return list(csv.DictReader(f))
def _f(x):
    try:
        v=float(x); return v if math.isfinite(v) else np.nan
    except Exception: return np.nan
def _med(xs):
    xs=[x for x in xs if np.isfinite(x)]; return float(np.median(xs)) if xs else np.nan
def _setfont(lang):
    plt.rcParams["axes.unicode_minus"]=False
    if lang=="zh":
        fp=r"C:\Windows\Fonts\msyh.ttc"
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp); plt.rcParams["font.family"]=font_manager.FontProperties(fname=fp).get_name()
    else: plt.rcParams["font.family"]="DejaVu Sans"

# --------------------------------------------------------------- G1b flip
def fig_g1b_flip(lang,L):
    rows=_read(os.path.join(RES,"exp_G1b_pentry","g1b_flip.csv"))
    fams=[r["family"] for r in rows]; rescued=[_f(r["rescued_BtoG"]) for r in rows]
    lost=[_f(r["lost_GtoB"]) for r in rows]; net=[_f(r["net"]) for r in rows]
    x=np.arange(len(fams)); w=.36
    fig,ax=plt.subplots(figsize=(6.4,4.0))
    ax.bar(x-w/2,rescued,w,label=L["rescued"],color="#4B86B4")
    ax.bar(x+w/2,lost,w,label=L["lost"],color="#C0504D")
    for xi,vv in zip(x-w/2,rescued): ax.text(xi,vv+.3,f"{vv:g}",ha="center",fontsize=9)
    for xi,vv in zip(x+w/2,lost): ax.text(xi,vv+.3,f"{vv:g}",ha="center",fontsize=9)
    for i,(xi,vv) in enumerate(zip(x,net)):
        ax.text(xi,max(rescued[i],lost[i])+2.0,f"{L['netlbl']}: {vv:g}",ha="center",fontsize=9,color=("#C0504D" if vv<0 else "#3a7a3a"))
    ax.set_xticks(x); ax.set_xticklabels([L["fam"][f] for f in fams])
    ax.set_ylabel(L["cnt"]); ax.set_xlabel(L["family"]); ax.set_title(L["flip_t"],fontsize=11,pad=26)
    ax.set_ylim(0,max(max(rescued),max(lost))+6); ax.grid(axis="y",alpha=.3)
    ax.legend(fontsize=9,ncol=2,loc="lower center",bbox_to_anchor=(0.5,1.0),frameon=False)
    fig.tight_layout(); out=os.path.join(OUTROOT,lang,"fig_g1b_flip.png"); fig.savefig(out,dpi=200); plt.close(fig); return out

# --------------------------------------------------------------- G1b nu01
def fig_g1b_nu01(lang,L):
    rows=sorted(_read(os.path.join(RES,"exp_G1b_pentry","g1b_nu01_agg.csv")),key=lambda r:_f(r["sig"]))
    sig=[_f(r["sig"]) for r in rows]; succ=[_f(r["success"]) for r in rows]
    lo=[succ[i]-_f(rows[i]["wilson_lo"]) for i in range(len(rows))]; hi=[_f(rows[i]["wilson_hi"])-succ[i] for i in range(len(rows))]
    x=np.arange(len(sig))
    fig,ax=plt.subplots(figsize=(5.8,4.0))
    ax.bar(x,succ,.55,color="#4B86B4",yerr=[lo,hi],capsize=5,ecolor="#333",error_kw=dict(lw=1.2))
    for xi,v in zip(x,succ): ax.text(xi,v+hi[int(xi)]+.02,f"{v:.2f}",ha="center",fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in sig]); ax.set_ylim(0,1.0)
    ax.set_xlabel(L["sig"]); ax.set_ylabel(L["succ"]); ax.set_title(L["nu01_t"],fontsize=11); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); out=os.path.join(OUTROOT,lang,"fig_g1b_nu01.png"); fig.savefig(out,dpi=200); plt.close(fig); return out

# --------------------------------------------------------------- Gstab early
def fig_gstab_early(lang,L):
    rows=[r for r in _read(os.path.join(RES,"exp_Gstab_grad","gstab_agg.csv")) if r["stage"]=="E"]
    def cell(nu,clip,fam,col):
        z=[r for r in rows if abs(_f(r["nu"])-nu)<1e-12 and abs(_f(r["clip"])-clip)<1e-9 and r["family"]==fam]
        return _f(z[0][col]) if z else np.nan
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.4)); fig.suptitle(L["early_t"],fontsize=12)
    x=np.arange(len(NUS)); w=.135
    def grouped(ax,clip,col,title,log=True):
        for k,fam in enumerate(["rational","down_lin","down_inv","uniform"]):
            ax.bar(x+(k-1.5)*w,[cell(nu,clip,fam,col) for nu in NUS],w,color=COL[fam],label=L["fam"][fam],edgecolor="white",linewidth=.4)
        ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in NUS]); ax.set_xlabel(L["nu"]); ax.set_title(title,fontsize=10)
        ax.legend(fontsize=8,ncol=2); ax.grid(axis="y",alpha=.3,which=("both" if log else "major"));
        if log: ax.set_yscale("log")
    grouped(axs[0,0],0,"g_peak",L["gpeak"],True); grouped(axs[0,1],0,"lmax_med",L["lmax"],True)
    grouped(axs[1,0],1,"frac_clip",L["fclip"],False); axs[1,0].set_ylim(0,1.08)
    grouped(axs[1,1],1,"L2_med",L["eL2"],False)
    fig.tight_layout(rect=[0,0,1,.96]); out=os.path.join(OUTROOT,lang,"fig_gstab_early.png"); fig.savefig(out,dpi=200); plt.close(fig); return out

# --------------------------------------------------------------- Gstab lr
def fig_gstab_lr(lang,L):
    rows=_read(os.path.join(RES,"exp_Gstab_grad","gstab_b1_fixed.csv"))
    nus=sorted({_f(r["nu"]) for r in rows}); clips=sorted({_f(r["clip"]) for r in rows})
    fig,axs=plt.subplots(len(nus),len(clips),figsize=(11,7.6),squeeze=False)
    for i,nu in enumerate(nus):
        for j,clip in enumerate(clips):
            ax=axs[i][j]
            for fam in ["rational","down_lin","down_inv","uniform"]:
                z=sorted([r for r in rows if abs(_f(r["nu"])-nu)<1e-12 and abs(_f(r["clip"])-clip)<1e-9 and r["family"]==fam],key=lambda r:_f(r["lr"]))
                ax.plot([_f(r["lr"]) for r in z],[_f(r["L2_med"]) for r in z],"o-",ms=4,lw=1.4,color=COL[fam],label=L["fam"][fam])
            ax.axhline(.05,color="#C0504D",ls="--",lw=1.2); ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(alpha=.3,which="both"); ax.set_title(f"ν={nu:g} | {L['clip1'] if clip>.5 else L['clip0']}",fontsize=10)
            ax.set_xlabel(L["lr"]); ax.set_ylabel(L["L2"])
    axs[0][0].legend(fontsize=8,ncol=2); fig.suptitle(L["lr_t"],fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96]); out=os.path.join(OUTROOT,lang,"fig_gstab_lr_fixed.png"); fig.savefig(out,dpi=200); plt.close(fig); return out

# --------------------------------------------------------------- E3b
def fig_e3b(lang,L):
    R=_read(os.path.join(RES,"exp_E3b_fair","e3b_runs.csv"))
    fams=["rational","linear","band","uniform","down_inv","down_lin"]
    def agg(bc,fam,col="l2_band"):
        return _med([_f(r[col]) for r in R if r["basin_class"]==bc and r["family"]==fam])
    fig,axs=plt.subplots(1,3,figsize=(15,4.3)); fig.suptitle(L["e3b_t"],fontsize=12)
    x=np.arange(len(fams)); w=.4
    g=[agg("good",f) for f in fams]; b=[agg("bad",f) for f in fams]
    axs[0].bar(x,g,color=[COL[f] for f in fams]); axs[0].set_xticks(x); axs[0].set_xticklabels([L["fam"][f] for f in fams],rotation=20,fontsize=8)
    axs[0].set_ylabel(L["band"]); axs[0].set_yscale("log"); axs[0].set_title(L["goodband"],fontsize=10); axs[0].grid(axis="y",alpha=.3,which="both")
    axs[1].bar(x,b,color=[COL[f] for f in fams]); axs[1].set_xticks(x); axs[1].set_xticklabels([L["fam"][f] for f in fams],rotation=20,fontsize=8)
    axs[1].set_ylabel(L["band"]); axs[1].set_yscale("log"); axs[1].set_title(L["badband"],fontsize=10); axs[1].grid(axis="y",alpha=.3,which="both")
    cs=sorted({_f(r["contrast"]) for r in R if r["basin_class"]=="good"})
    for fam in ["rational","linear","band","uniform","down_inv"]:
        ys=[_med([_f(r["l2_band"]) for r in R if r["basin_class"]=="good" and r["family"]==fam and abs(_f(r["contrast"])-c)<1e-9]) for c in cs]
        axs[2].plot(cs,ys,"o-",ms=4,color=COL[fam],label=L["fam"][fam])
    axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xlabel(L["contrast"]); axs[2].set_ylabel(L["band"])
    axs[2].set_title(L["goodband"],fontsize=10); axs[2].legend(fontsize=7,ncol=2); axs[2].grid(alpha=.3,which="both")
    fig.tight_layout(rect=[0,0,1,.95]); out=os.path.join(OUTROOT,lang,"fig_E3b.png"); fig.savefig(out,dpi=200); plt.close(fig); return out

# --------------------------------------------------------------- F1 SA no-go
def fig_f1(lang,L):
    R=[r for r in _read(os.path.join(RES,"exp_F1_SA_linear","f1_rows.csv")) if r["operator"].startswith("激波")]
    ns=sorted({int(_f(r["n"])) for r in R})
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.3)); fig.suptitle(L["f1_t"],fontsize=12)
    x=np.arange(len(ns)); w=.16
    def wkey(r):
        s=r["weight"]
        if s.startswith("const"):return "const"
        if s.startswith("光滑") or s.startswith("smooth"):return "smooth"
        if s.startswith("局域增"):return "up"
        if s.startswith("局域减"):return "down"
        return "white"
    for k,k_ in enumerate(F1_KEYS):
        imp=[];eoff=[]
        for n in ns:
            z=[r for r in R if int(_f(r["n"]))==n and wkey(r)==k_]
            imp.append(_f(z[0]["improve"]) if z else np.nan); eoff.append(_f(z[0]["E_off"]) if z else np.nan)
        axs[0].bar(x+(k-2)*w,imp,w,color=F1_COL[k_],label=L["f1w"][k_])
        axs[1].bar(x+(k-2)*w,eoff,w,color=F1_COL[k_],label=L["f1w"][k_])
    axs[0].axhline(1.0,color="#333",ls=":",lw=1); axs[0].set_yscale("log")
    for ax,t in [(axs[0],L["impr"]),(axs[1],L["eoff"])]:
        ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns]); ax.set_xlabel(L["ngrid"]); ax.set_title(t,fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y",alpha=.3,which=("both" if ax is axs[0] else "major"))
    fig.tight_layout(rect=[0,0,1,.95]); out=os.path.join(OUTROOT,lang,"fig_F1_sa.png"); fig.savefig(out,dpi=200); plt.close(fig); return out

# --------------------------------------------------------------- F2 curriculum
def fig_f2(lang,L):
    R=_read(os.path.join(RES,"exp_F2_curriculum","f2_runs.csv"))
    nus=sorted({_f(r["nu"]) for r in R}); x=np.arange(len(F2_ORDER)); w=.38
    fig,ax=plt.subplots(figsize=(8.4,4.4))
    for j,nu in enumerate(nus):
        rate=[np.mean([_f(r["success"]) for r in R if abs(_f(r["nu"])-nu)<1e-9 and r["strategy"]==s]) for s in F2_ORDER]
        ax.bar(x+(j-.5)*w,rate,w,label=f"ν={nu:g}")
        for xi,v in zip(x+(j-.5)*w,rate): ax.text(xi,v+.02,f"{v:.2f}",ha="center",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([L["f2s"][s] for s in F2_ORDER],fontsize=9); ax.set_ylim(0,1.0)
    ax.set_ylabel(L["rate"]); ax.set_xlabel(L["strat"]); ax.set_title(L["f2_t"],fontsize=11); ax.legend(fontsize=9); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); out=os.path.join(OUTROOT,lang,"fig_F2_curriculum.png"); fig.savefig(out,dpi=200); plt.close(fig); return out

# --------------------------------------------------------------- E1 mechanism / U
def fig_e1(lang,L):
    R=_read(os.path.join(RES,"exp_E1_beta_nu","e1_agg.csv"))
    nus=sorted({_f(r["nu"]) for r in R})
    cmap=plt.cm.viridis(np.linspace(.05,.85,len(nus)))
    # (1) 冻结 eta vs beta
    fig,ax=plt.subplots(figsize=(6.6,4.4))
    for ci,nu in enumerate(nus):
        z=sorted([r for r in R if abs(_f(r["nu"])-nu)<1e-12],key=lambda r:_f(r["beta"]))
        ax.plot([_f(r["beta"]) for r in z],[_f(r["eta_gate_med"]) for r in z],"o-",ms=4,color=cmap[ci],label=f"ν={nu:g}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(L["beta"]); ax.set_ylabel(L["eta"])
    ax.set_title(L["e1m_t"],fontsize=11); ax.legend(fontsize=8); ax.grid(alpha=.3,which="both")
    fig.tight_layout(); o1=os.path.join(OUTROOT,lang,"fig_E1_mechanism.png"); fig.savefig(o1,dpi=200); plt.close(fig)
    # (2) L2 vs beta U 型
    fig,ax=plt.subplots(figsize=(6.6,4.4))
    for ci,nu in enumerate(nus):
        z=sorted([r for r in R if abs(_f(r["nu"])-nu)<1e-12],key=lambda r:_f(r["beta"]))
        ax.plot([_f(r["beta"]) for r in z],[_f(r["L2_med"]) for r in z],"o-",ms=4,color=cmap[ci],label=f"ν={nu:g}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(L["beta"]); ax.set_ylabel(L["L2"])
    ax.set_title(L["e1u_t"],fontsize=11); ax.legend(fontsize=8); ax.grid(alpha=.3,which="both")
    fig.tight_layout(); o2=os.path.join(OUTROOT,lang,"fig_E1_ucurve.png"); fig.savefig(o2,dpi=200); plt.close(fig)
    return o1,o2

FUNCS=[fig_g1b_flip,fig_g1b_nu01,fig_gstab_early,fig_gstab_lr,fig_e3b,fig_f1,fig_f2,fig_e1]
def main():
    for lang in ("zh","en"):
        _setfont(lang); L=TXT[lang]
        for fn in FUNCS:
            try:
                r=fn(lang,L); rs=r if isinstance(r,tuple) else (r,)
                for o in rs: print("saved",o)
            except Exception as e:
                import traceback; print(f"[{lang}] {fn.__name__} FAIL: {e}"); traceback.print_exc()
if __name__=="__main__":
    main()
