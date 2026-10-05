# -*- coding: utf-8 -*-
import csv, collections, statistics as st
from pathlib import Path
import numpy as np
_REL = Path(__file__).resolve().parents[4]   # paper1_release/
d = str(_REL/'data'/'raw'/'v6'/'exp_R1_arch') + '/'
R = list(csv.DictReader(open(d + 'r1_runs.csv', encoding='utf-8-sig')))
I = list(csv.DictReader(open(d + 'r1_invariance.csv', encoding='utf-8-sig')))
print('r1_runs 列:', list(R[0].keys()))
print('\nr1_invariance 列:', list(I[0].keys()))

def fnum(x):
    try: return float(x)
    except: return np.nan

# 1) 不变量全表：每配置×ν×family 的 η-β Spearman 与 β1/3/8 中位 L2
print('\n================ invariance（按配置） ================')
hdr = ['axis', 'level', 'nu', 'family']
cand_sp = [k for k in I[0] if 'spear' in k.lower()]
cand_l2 = [k for k in I[0] if k.lower().startswith('l2_') or 'b1' in k or 'b3' in k or 'b8' in k]
print('spearman候选列', cand_sp, ' L2候选列', cand_l2)
for r in sorted(I, key=lambda r: (r['axis'], str(r['level']), float(r['nu']), r['family'])):
    line = f"{r['axis']:>6} {str(r['level']):>4} ν={float(r['nu']):<5} {r['family']:<9}| "
    for k in cand_sp + cand_l2:
        line += f"{k}={fnum(r[k]):+.3f} "
    print(line)

# 2) runs：rational 各配置×ν×β 的 L2 中位（跨5seed）
print('\n================ rational: L2 中位 by 配置×ν×β ================')
g = collections.defaultdict(list)
for r in R:
    if r['family'] != 'rational': continue
    key = (r['axis'], str(r['level']), int(float(r['hidden'])), int(float(r['layers'])),
           int(float(r['n_pde'])), float(r['nu']), float(r['beta']))
    g[key].append(fnum(r['L2']))
for key in sorted(g, key=lambda k: (k[0], str(k[1]), k[5], k[6])):
    v = np.array(g[key]); v = v[np.isfinite(v)]
    print(f"{key[0]:>6} {key[1]:>4} h{key[2]} L{key[3]} n{key[4]:<6} ν={key[5]:<5} β={key[6]:<4} n={v.size} L2med={np.median(v):.3e}")

# 3) width 轴三族对比（32/128 有三族）
print('\n================ width 轴三族对比（L2中位） ================')
for w in (32, 64, 128):
    for nu in (0.005, 0.1):
        row = f"width{w} ν={nu:<5}: "
        for fam in ('rational', 'uniform', 'down_inv'):
            cells = []
            for b in (1.0, 3.0, 8.0):
                v = [fnum(r['L2']) for r in R if r['family'] == fam and int(float(r['hidden'])) == w
                     and abs(float(r['nu']) - nu) < 1e-9 and abs(float(r['beta']) - b) < 1e-9]
                v = np.array([x for x in v if np.isfinite(v)])
                cells.append(f"β{b:g}={np.median(v):.2e}" if v.size else f"β{b:g}=-")
            row += f"{fam:<9}[{(' '.join(cells))}]  "
        print(row)

# 4) 收敛统计
conv = [fnum(r.get('converged')) for r in R]
cep = [fnum(r.get('conv_epoch')) for r in R if np.isfinite(fnum(r.get('conv_epoch')))]
print('\nfrac_converged =', np.nanmean(conv), ' conv_epoch中位 =', np.nanmedian(cep) if cep else None,
      ' 总run=', len(R))
