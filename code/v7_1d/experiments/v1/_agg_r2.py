# -*- coding: utf-8 -*-
import csv, collections
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
_REL = Path(__file__).resolve().parents[4]   # paper1_release/
d = str(_REL/'data'/'raw'/'v6'/'exp_R2_budget') + '/'
R = list(csv.DictReader(open(d+'r2_runs.csv', encoding='utf-8-sig')))
def fn(x):
    try: return float(x)
    except: return np.nan
for r in R:
    for k in ('nu','budget','beta','phase0_L2','L2','Linf','l2_band','eta_gate','best_epoch','conv_epoch','converged'):
        r[k]=fn(r[k])

print('######## r2_invariance.csv ########')
print(open(d+'r2_invariance.csv', encoding='utf-8-sig').read())

print('######## r2_traj.csv 结构 ########')
T = list(csv.DictReader(open(d+'r2_traj.csv', encoding='utf-8-sig')))
print('traj 行数', len(T), ' 列', list(T[0].keys()))
print('样例3行', T[:3])

# 主表：nu x budget x beta 的 L2 中位[IQR]
print('\n######## 主表 L2 中位 (IQR), n=5 ########')
g = collections.defaultdict(list)
for r in R: g[(r['nu'],r['budget'],r['beta'])].append(r['L2'])
order_nu=[0.005,0.1,0.5]; order_b=[2000,4000,8000]; order_beta=[1.,3.,8.]
for nu in order_nu:
    print(f'\n--- ν={nu} ---   β1 / β3 / β8 (L2中位),  括号=增权相对β1的对数增益 log10(L2_β1/L2_β)')
    for b in order_b:
        cells=[]; gains=[]
        base=np.nanmedian(g[(nu,b,1.)])
        for bt in order_beta:
            v=np.array(g[(nu,b,bt)],float); m=np.median(v)
            q1,q3=np.percentile(v,[25,75])
            cells.append(f"{m:.2e}[{q1:.1e},{q3:.1e}]")
            gains.append(0. if bt==1 else np.log10(base/m))
        print(f" budget{b:>5}: "+" | ".join(f"β{bt:g}:{c}" for bt,c in zip(order_beta,cells)))
        print(f"          对数增益: β3={gains[1]:+.2f}  β8={gains[2]:+.2f} (正=增权降误差)")

# eta-beta spearman by nu x budget
print('\n######## η_gate–β Spearman by ν×budget ########')
for nu in order_nu:
    row=[]
    for b in order_b:
        xs=[r['beta'] for r in R if r['nu']==nu and r['budget']==b]
        ys=[r['eta_gate'] for r in R if r['nu']==nu and r['budget']==b]
        rho,p=spearmanr(xs,ys); row.append(f"b{b}:ρ={rho:+.2f}")
    print(f"ν={nu}: ", " ".join(row))

# 加速叙事：β8@(2k/4k) vs β1@8k；β3@4k vs β1@8k
print('\n######## 加速：增权短预算 vs 无增权满预算(8k) ########')
for nu in order_nu:
    full_uni=np.median(g[(nu,8000,1.)])
    print(f"ν={nu}: β1@8k={full_uni:.2e} || ", end='')
    for b in (2000,4000):
        for bt in (3.,8.):
            m=np.median(g[(nu,b,bt)])
            tag='更快达精度' if m<=full_uni else '未达'
            print(f"β{bt:g}@{b}={m:.2e}({tag},{np.log10(full_uni/m):+.2f}dec) ", end='')
    print()

# 收敛统计
print('\n######## converged / conv_epoch ########')
for nu in order_nu:
    for b in order_b:
        for bt in order_beta:
            sub=[r for r in R if r['nu']==nu and r['budget']==b and r['beta']==bt]
            fc=np.mean([r['converged'] for r in sub])
            ce=[r['conv_epoch'] for r in sub if r['conv_epoch']>0]
            cem=np.median(ce) if ce else -1
            print(f"ν={nu} b{b} β{bt:g}: frac_conv={fc:.2f} conv_epoch中位={cem:.0f}")
