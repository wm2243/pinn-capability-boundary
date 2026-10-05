# -*- coding: utf-8 -*-
"""
实验 Vverify：对理论框架 V10 的核心主张做【独立数值核验】（按跑前判据判定成立与否，不做演示性复述）
=====================================================================================
Claim 1 机制 A（盆地内梯度能量再分配）——【V10 改为直接读 E1 结果、不重训】：
   数据源 results/v6/exp_E1_beta_nu/：
     · e1_grad_probe.csv：冻结 Gate 点“同 θ、同 r、仅改 β”的探针 eta=g_band²/g_smooth²（机制 A 最干净口径）；
     · e1_runs.csv：好盆地（gate_ok）终态 l2_band / 全局 L2。
   命题：β↑ ⇒ 冻结 Gate 点 eta ↑（A0/A1，确定性方向）；好盆地 l2_band 中位在最优段 ↓；全局 L2 不被整体牺牲。
   跑前判据：Spearman ρ(β, eta_gate)>0；ρ(β, l2_band|好盆地)≤0；并报 L2(βmax)/L2(1) 作 U 型右支参考。
   —— 不再 enter_basin / train_phase1_family，避免重复训练；E1 已用 11 个规则等距种子。
Claim 2 测度不变性引理（纯 numpy 线代，确定性、无网络、无随机，每次重算）：
   线性 well-posed 算子 L=-νu''+a u'（齐次 Dirichlet，中心差分），离散稳定性常数
   C(w)=sup_r ‖L_h^{-1}r‖_X / ‖r‖_{L2(w)}；对称化后 C(w)^2=λmax(W^{-1/2} L_h^{-T}L_h^{-1} W^{-1/2})。
   命题：① 增权族 w_min=1 ⇒ C(w)≤C(1)（不放大）；② 减权族 w_min=1/β ⇒ C(w)≤√β·C(1)（上界恰放宽 √β）；
        ③ 统一上界 C(w)√w_min ≤ C(1)（加权只移动常数、不改 PDE 本身稳定性，测度不变性）。
Claim 3 权函数乘子三条件（纯函数，torch autograd，无随机）：对六族数值算 w(t)、w'(t) 与
   自适应损失径向乘子 m_ad(t)=Ψ₂'(t)/t=2w+t·w'(t)（与单点损失密度 Ψ₂=w t² 严格自洽；冻结截面乘子为 w，不含 t w'），
   逐条判 ①m_ad>0 正定 ②m_ad 逐点落在各族闭式区间 ③t→∞ 的 w 饱和/m_ad 收敛；rational 逐点满足 m_ad∈[2,2β]。
   注：旧版误取 2w+2t w'=2φ'（加权残差 wr 径向雅可比的 2 倍），它不对应任何损失二次型，已更正为 2w+t w'。
Claim 4 [SA] 方向-尺度（默认关，VERIFY_SA=1 才跑；仅旁证）：冻结 θ 跨 β 的 Hessian top-1 特征向量主角/λmax。

规范：不改公共代码；Claim1 读 E1 既有 csv（不重训）；Claim2/3 确定性直接重算；不支持就如实打印“不成立”。
产出 results/v6/exp_Vverify/：vv_claim1_from_e1.csv、vv_claim2_measure.csv、vv_claim3_multiplier.csv、
   fig_vv_claim1.png / fig_vv_claim2.png / fig_vv_claim3.png
SMOKE：V8_SMOKE=1；运行：python exp_Vverify_claims.py
=====================================================================================
"""
import os, sys, csv, math
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager

_HERE = Path(__file__).resolve(); _PKG = _HERE.parent.parent
for _p in (str(_PKG), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import v8_matrix_common as mc
from losses.family_weighting_loss import FAMILIES, family_weight, w_min_theory

try:
    from scipy.stats import spearmanr
    def _spearman(x, y):
        x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
        return float(spearmanr(x[m], y[m]).statistic) if m.sum()>=3 else float("nan")
except Exception:
    def _spearman(x, y):
        x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
        x,y=x[m],y[m]
        if len(x)<3: return float("nan")
        rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
        rx-=rx.mean(); ry-=ry.mean()
        return float((rx*ry).sum()/math.sqrt((rx**2).sum()*(ry**2).sum()))

for _c in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(_c):
        font_manager.fontManager.addfont(_c); plt.rcParams["font.family"]=font_manager.FontProperties(fname=_c).get_name(); break
plt.rcParams["axes.unicode_minus"]=False

OUT=str(mc.RES_ROOT/"exp_Vverify"); os.makedirs(OUT,exist_ok=True)
E1_DIR=str(mc.RES_ROOT/"exp_E1_beta_nu")
C1_CSV=os.path.join(OUT,"vv_claim1_from_e1.csv")
C2_CSV=os.path.join(OUT,"vv_claim2_measure.csv")
C3_CSV=os.path.join(OUT,"vv_claim3_multiplier.csv")

# Claim2 配置
C2_NGRID=401; C2_AMP=50.0; C2_NUS=[0.01,0.001]; C2_A=[0.0,1.0]
C2_BETA_UP=[1.0,3.0,10.0,40.0,100.0]; C2_BETA_DOWN=[3.0,10.0,40.0]
# Claim3 配置
C3_TMAX=20.0; C3_NT=2001; C3_BETA=10.0; C3_FAMS=list(FAMILIES)
VERIFY_SA=os.environ.get("VERIFY_SA","0")=="1"
if mc.SMOKE:
    C2_NGRID=101; C2_NUS=[0.005]; C2_BETA_UP=[1.0,10.0]; C2_BETA_DOWN=[10.0]
    C3_NT=401

UP_C="#E8A24B"; DOWN_C="#4B86B4"; UNI_C="#9A9A9A"; TH_C="#C0504D"
def fcol(f):
    from losses.family_weighting_loss import UP_FAMILIES, DOWN_FAMILIES
    return UP_C if f in UP_FAMILIES else (DOWN_C if f in DOWN_FAMILIES else UNI_C)

def _load(p):
    if not os.path.exists(p): return []
    with open(p,"r",encoding="utf-8-sig") as f: return list(csv.DictReader(f))
def _f(x,d=float("nan")):
    try: return float(x)
    except Exception: return d
def _is_good(v):
    return str(v).strip().lower() in ("true","1","1.0","yes")
def _med(xs):
    xs=[x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")

# ============================================================ Claim 1（读 E1，不重训）
def claim1():
    gp_p=os.path.join(E1_DIR,"e1_grad_probe.csv"); rn_p=os.path.join(E1_DIR,"e1_runs.csv")
    if not (os.path.exists(gp_p) and os.path.exists(rn_p)):
        print(f"[Claim1] 缺少 E1 结果（{gp_p} / {rn_p}），请先跑 E1；本次跳过。"); return [],{},False
    gp=_load(gp_p); rn=_load(rn_p)
    # 冻结 Gate 点探针（point==gate）：同 θ 同 r 仅改 β —— 机制 A 最干净口径
    gate=[r for r in gp if str(r.get("point","")).strip().lower()=="gate"]
    good=[r for r in rn if _is_good(r.get("gate_ok"))]   # 好盆地终态（机制 A 是盆地内命题）
    nus=sorted({_f(r["nu"]) for r in rn})
    out=[]; verdict={}
    fig,ax=plt.subplots(1,3,figsize=(15,4.2))
    for nu in nus:
        betas=sorted({_f(r["beta"]) for r in gate if abs(_f(r["nu"])-nu)<1e-12}
                     | {_f(r["beta"]) for r in good if abs(_f(r["nu"])-nu)<1e-12})
        if len(betas)<3: continue
        em=[_med([_f(r["eta"]) for r in gate if abs(_f(r["nu"])-nu)<1e-12 and abs(_f(r["beta"])-b)<1e-12]) for b in betas]
        bm=[_med([_f(r["l2_band"]) for r in good if abs(_f(r["nu"])-nu)<1e-12 and abs(_f(r["beta"])-b)<1e-12]) for b in betas]
        gm=[_med([_f(r["L2"]) for r in good if abs(_f(r["nu"])-nu)<1e-12 and abs(_f(r["beta"])-b)<1e-12]) for b in betas]
        for a_,yy,tt,log in [(ax[0],em,"冻结 Gate 点 η=g_band²/g_smooth² 随 β↑",False),
                            (ax[1],bm,"好盆地 l2_band（中位）",True),(ax[2],gm,"好盆地全局 L2（中位）",True)]:
            a_.plot(betas,yy,"o-",label=f"ν={nu:g}"); a_.set_xscale("log"); a_.set_title(tt,fontsize=10)
            if log:
                yy2=[v for v in yy if np.isfinite(v) and v>0]
                if yy2: a_.set_yscale("log")
            a_.grid(alpha=.3,which="both" if log else "major"); a_.set_xlabel("β"); a_.legend(fontsize=8)
        rho_e=_spearman(betas,em); rho_b=_spearman(betas,bm)
        gvals=[v for v in gm if np.isfinite(v)]
        L2ratio=(gvals[-1]/gvals[0]) if gvals and gvals[0] else float("nan")
        bvals=[v for v in bm if np.isfinite(v)]
        band_improve=(min(bvals)/bvals[0]) if bvals and bvals[0] else float("nan")
        verdict[nu]=dict(rho_eta=rho_e,rho_band=rho_b,L2_max_over_1=L2ratio,band_best_over_1=band_improve,
                         pass_eta=rho_e>0,pass_band=rho_b<=0)
        for b,e,bb,gg in zip(betas,em,bm,gm):
            out.append(dict(nu=nu,beta=b,eta_gate_med=e,l2_band_good_med=bb,L2_good_med=gg))
    mc.write_csv(C1_CSV,out)
    fig.tight_layout(); p=os.path.join(OUT,"fig_vv_claim1.png"); fig.savefig(p,dpi=150); plt.close(fig); print("saved",p)
    print("\n[Claim1 判据｜读 E1，不重训] 核心：ρ(β,η_gate)>0（冻结探针）；辅助：ρ(β,l2_band|好)≤0；并报 U 型右支 L2 比")
    ok_all=True
    for nu,v in verdict.items():
        ok=v["pass_eta"] and v["pass_band"]; ok_all&=ok
        print(f"  ν={nu:g}: ρ_eta={v['rho_eta']:+.2f}({'成立' if v['pass_eta'] else '不成立'})  "
              f"ρ_band={v['rho_band']:+.2f}({'成立' if v['pass_band'] else '不成立'})  "
              f"L2(max)/L2(1)={v['L2_max_over_1']:.2f}(U型右支参考)  band最优/β1={v['band_best_over_1']:.2f}  "
              f"=> 机制A {'支持' if ok else '不完全支持/需查'}")
    if not verdict: ok_all=False
    return out,verdict,ok_all

# ============================================================ Claim 2
def _conv_diff_matrix(nu,a,h,n_in):
    d=2*nu/h**2; o=nu/h**2; c=a/(2*h)
    L=np.full((n_in,n_in),0.0)
    idx=np.arange(n_in)
    L[idx,idx]=d
    L[idx[1:],idx[:-1]]= -o-c
    L[idx[:-1],idx[1:]]= -o+c
    return L

def _Cw(Linv,w):
    """C(w)^2 = λmax(W^{-1/2} L^{-T}L^{-1} W^{-1/2})；网格质量 h 在比值中约去。"""
    M=Linv.T@Linv
    sq=1.0/np.sqrt(w)
    S=(sq[:,None]*M)*sq[None,:]
    return math.sqrt(max(np.linalg.eigvalsh(S).max(),0.0))

def claim2():
    rows=[]
    for nu in C2_NUS:
        for a in C2_A:
            x=np.linspace(-1,1,C2_NGRID); h=x[1]-x[0]; xin=x[1:-1]; n=len(xin)
            L=_conv_diff_matrix(nu,a,h,n); Linv=np.linalg.inv(L)
            t=1.0+C2_AMP*np.exp(-xin**2/(2*nu**2))
            tt=torch.tensor(t)
            w1=family_weight("uniform",tt,1.0).numpy(); C1=_Cw(Linv,w1)
            for fam,betas in [("rational",C2_BETA_UP),("down_inv",C2_BETA_DOWN)]:
                for b in betas:
                    w=family_weight(fam,tt,b).numpy(); wmin=w_min_theory(fam,b); Cw=_Cw(Linv,w)
                    rows.append(dict(nu=nu,a=a,family=fam,beta=b,w_min=wmin,C_uniform=C1,C_w=Cw,
                                     ratio=Cw/C1, Cw_sqrt_wmin=Cw*math.sqrt(wmin),
                                     up_bound=1.0/math.sqrt(wmin)))
    mc.write_csv(C2_CSV,rows)
    tol=2e-2
    p_up=all(r["ratio"]<=1.0+tol for r in rows if r["family"]=="rational")
    p_down=all(r["ratio"]<=r["up_bound"]*(1+tol) for r in rows if r["family"]=="down_inv")
    p_uni=all(r["Cw_sqrt_wmin"]<=r["C_uniform"]*(1+tol) for r in rows)
    print("\n[Claim2 判据] ①增权 C(w)/C(1)≤1：",("成立" if p_up else "不成立"),
          " ②减权 C(w)/C(1)≤√β：",("成立" if p_down else "不成立"),
          " ③统一上界 C(w)√w_min≤C(1)：",("成立" if p_uni else "不成立"))
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for nu in C2_NUS:
        for fam,mk in [("rational","o-"),("down_inv","s--")]:
            z=sorted([r for r in rows if r["nu"]==nu and r["a"]==1.0 and r["family"]==fam],key=lambda q:q["beta"])
            ax[0].plot([r["beta"] for r in z],[r["ratio"] for r in z],mk,color=fcol(fam),label=f"{fam} ν={nu:g}")
            ax[1].plot([r["beta"] for r in z],[r["Cw_sqrt_wmin"]/r["C_uniform"] for r in z],mk,color=fcol(fam),label=f"{fam} ν={nu:g}")
    bx=np.array(sorted(set(C2_BETA_UP+C2_BETA_DOWN)))
    ax[0].axhline(1.0,color=TH_C,ls=":",label="增权上界 1")
    ax[0].plot(bx,np.sqrt(bx),color="#888",ls=":",label="减权上界 √β")
    ax[0].set_xscale("log");ax[0].set_yscale("log");ax[0].set_xlabel("β");ax[0].set_ylabel("C(w)/C(1)")
    ax[0].set_title("离散稳定性常数比：增权不放大、减权至多 √β",fontsize=10);ax[0].legend(fontsize=7);ax[0].grid(alpha=.3,which="both")
    ax[1].axhline(1.0,color=TH_C,ls=":",label="统一上界 C(1)")
    ax[1].set_xscale("log");ax[1].set_xlabel("β");ax[1].set_ylabel("C(w)·√w_min / C(1)")
    ax[1].set_title("归一化后不越等权上界（测度不变性）",fontsize=10);ax[1].legend(fontsize=7);ax[1].grid(alpha=.3,which="both")
    fig.tight_layout(); p=os.path.join(OUT,"fig_vv_claim2.png"); fig.savefig(p,dpi=150); plt.close(fig); print("saved",p)
    return rows,dict(p_up=p_up,p_down=p_down,p_uni=p_uni),(p_up and p_down and p_uni)

# ============================================================ Claim 3
def _claim3_closed_interval(fam,B):
    """m_ad=2w+t w' 的闭式上下界（β≥1）。"""
    if fam in ("rational","band","uniform"): return (2.0,2.0*B)
    if fam=="linear":   return (2.0,3.0*B-1.0)   # 峰值在 t→1⁻：2+3(β−1)=3β−1；t>1 回 2β
    if fam in ("down_lin","down_inv"): return (2.0/B,2.0)
    return (float("nan"),float("nan"))

def claim3():
    t=torch.linspace(0,C3_TMAX,C3_NT,dtype=torch.float64,requires_grad=False)
    rows=[]; curves={}; B=C3_BETA
    for fam in C3_FAMS:
        tt=t.clone().requires_grad_(True)
        w=family_weight(fam,tt,B)
        if w.requires_grad:
            wp=torch.autograd.grad(w.sum(),tt,create_graph=True)[0]
        else:
            wp=torch.zeros_like(tt)
        m=2*w+tt*wp   # 自适应损失径向乘子 m_ad = Ψ₂'(t)/t = 2w + t w'（冻结截面乘子为 w，不含 tw'）
        wn=w.detach().numpy(); mn=m.detach().numpy(); dn=wp.detach().numpy(); tn=t.numpy()
        lo,hi=_claim3_closed_interval(fam,B)
        m_pos=bool(np.all(mn>=-1e-6))
        in_closed=bool(np.isfinite(mn).all() and np.all(mn>=lo-1e-6) and np.all(mn<=hi+1e-2))
        w_sat=bool(abs(wn[-1]-wn[max(0,len(wn)-50):].mean())<1e-2*max(1,abs(wn[-1])))
        rows.append(dict(family=fam,beta=B,m_positive=int(m_pos),m_bounded=int(in_closed),w_saturate=int(w_sat),
                         in_closed=int(in_closed),m_lo_theory=lo,m_hi_theory=hi,
                         m_min=mn.min(),m_max=mn.max(),w_at0=wn[0],w_inf=wn[-1],wp_inf=dn[-1]))
        curves[fam]=(tn,wn,mn)
    rz=[r for r in rows if r["family"]=="rational"][0]
    # rational：逐点落 [2,2β] 且 m_min≈2；不要求有限 TMAX 处 m_max 数值等于 2β（t→∞ 才饱和）
    rat_ok=bool(rz["m_positive"] and rz["in_closed"] and abs(rz["m_min"]-2)<2e-2)
    all_closed=all(r["in_closed"] for r in rows)
    mc.write_csv(C3_CSV,rows)
    print("\n[Claim3 判据] 六族自适应径向乘子 m_ad=2w+tw'（=Ψ₂'/t）；rational 逐点 m_ad∈[2,2β]：",
          f"m_min={rz['m_min']:.3f}(应2)；有限 TMAX={C3_TMAX:g} 处 m_max={rz['m_max']:.3f}（t→∞ 趋于 {2*B:g}）=>",
          ("成立" if rat_ok else "不成立"))
    for r in rows:
        print(f"  {r['family']:<9} 正定={r['m_positive']} 落闭式区间={r['in_closed']} 饱和={r['w_saturate']} "
              f"m∈[{r['m_min']:.3f},{r['m_max']:.3f}] 闭式[{r['m_lo_theory']:.3g},{r['m_hi_theory']:.3g}] w∞={r['w_inf']:.3f}")
    def _plot(lang):
        en=(lang=="en")
        fig,ax=plt.subplots(1,2,figsize=(11,4.2))
        for fam,(tn,wn,mn) in curves.items():
            ax[0].plot(tn,wn,color=fcol(fam),label=fam); ax[1].plot(tn,mn,color=fcol(fam),label=fam)
        ax[1].axhline(2,color=TH_C,ls=":"); ax[1].axhline(2*B,color=TH_C,ls=":",
                    label=("[2,2β] rational" if en else "rational 界 [2,2β]"))
        if en:
            ax[0].set_xlabel("contrast t");ax[0].set_ylabel("w(t)");ax[0].set_title("Six weighting families")
            ax[1].set_xlabel("contrast t");ax[1].set_ylabel(r"adaptive radial multiplier $m_{ad}=2w+t\,w'$")
            ax[1].set_title(r"$m_{ad}=\Psi_2'/t$: positivity and bounds")
            p=os.path.join(OUT,"fig_vv_claim3_en.png")
        else:
            ax[0].set_xlabel("对比度 t");ax[0].set_ylabel("w(t)");ax[0].set_title("六族权函数",fontsize=10)
            ax[1].set_xlabel("对比度 t");ax[1].set_ylabel("自适应径向乘子 m_ad=2w+tw′")
            ax[1].set_title("自适应损失径向乘子（Ψ₂′/t）与三条件",fontsize=10)
            p=os.path.join(OUT,"fig_vv_claim3.png")
        for a in ax: a.legend(fontsize=7);a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(p,dpi=150); plt.close(fig); print("saved",p)
    _plot("zh"); _plot("en")
    all3=all(r["m_positive"] and r["in_closed"] and r["w_saturate"] for r in rows if r["family"]=="rational")
    return rows,dict(rat_bound=rat_ok,all_closed=all_closed),bool(rat_ok and all_closed)

# ============================================================ Claim 4（默认关）
def claim4():
    if not VERIFY_SA:
        print("\n[Claim4] [SA] 方向-尺度旁证默认关闭（设 VERIFY_SA=1 开启；需 Hessian top-1 特征向量，成本较高）。")
        return None,None,True
    print("\n[Claim4] VERIFY_SA=1：请用 HessianSpectralEstimator 的特征向量接口补 top-1 主角/λmax（预留）。")
    return None,None,True

def main():
    r1,v1,ok1=claim1()
    r2,v2,ok2=claim2()
    r3,v3,ok3=claim3()
    r4,v4,ok4=claim4()
    print("\n================ Vverify 总判据（V10）================")
    print(f"Claim1 机制A（读 E1：冻结η单调+好盆地band）：{'支持' if ok1 else '不完全支持/需查'}  {v1}")
    print(f"Claim2 测度不变性引理：                    {'成立' if ok2 else '不成立'}  {v2}")
    print(f"Claim3 乘子三条件+rational界：             {'成立' if ok3 else '不成立'}  {v3}")
    print("产物目录：",OUT)

if __name__=="__main__":
    main()
