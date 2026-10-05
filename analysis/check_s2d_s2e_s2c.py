# -*- coding: utf-8 -*-
"""核对第七章 7.2：S2d 机制A探针 / S2e 尖刺 / S2c 对齐两阶段"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parents[1] / "data" / "aggregated"

print("=" * 70)
print("S2d 机制A探针: s2d_beta_probe.csv")
print("=" * 70)
probe = pd.read_csv(BASE / "v7_v2_exp_V2_S2d_beta_effect" / "s2d_beta_probe.csv")
print("列:", list(probe.columns))
print("eps 取值:", probe['eps'].unique())
print("beta 取值:", sorted(probe['beta'].unique()))
print("base_seed 个数:", probe['base_seed'].nunique())
probe['eta'] = probe['g_band'] / probe['g_smooth']
# 每个 base_seed 内 eta 对 beta 的 Spearman
rows = []
for s, g in probe.groupby('base_seed'):
    g = g.sort_values('beta')
    if len(g) >= 3:
        rho, p = stats.spearmanr(g['beta'], g['eta'])
        rows.append((s, rho, list(g['beta']), list(g['eta'])))
rho = np.array([r[1] for r in rows])
print(f"\nGate 点数(有3档beta的seed): {len(rows)}")
print(f"eta 随 beta 下降(rho<0): {(rho<0).sum()}, 上升(rho>0): {(rho>0).sum()}")
print(f"Spearman 均值: {rho.mean():.4f}, 中位: {np.median(rho):.4f}")
for s, r, b, e in rows:
    print(f"  seed={s}: rho={r:+.3f}  eta={['%.4f'%x for x in e]}")

print("\n" + "=" * 70)
print("S2d 配对精修: 从 s2d_beta_refine.csv 按 seed 配对重算中位差")
print("=" * 70)
try:
    ref = pd.read_csv(BASE / "v7_v2_exp_V2_S2d_beta_effect" / "s2d_beta_refine.csv")
    print("列:", list(ref.columns))
    print(ref.head(20).to_string())
except FileNotFoundError:
    print("s2d_beta_refine.csv 不存在，尝试 cmp 表")
    cmp = pd.read_csv(BASE / "v7_v2_exp_V2_S2d_beta_effect" / "s2d_beta_refine_cmp.csv")
    print(cmp.to_string())

print("\n" + "=" * 70)
print("S2e 尖刺")
print("=" * 70)
d = pd.read_csv(BASE / "v7_v2_exp_V2_S2e_spike" / "diag_seed.csv")
print("diag_seed.csv 列:", list(d.columns))
print(d.to_string())
r = pd.read_csv(BASE / "v7_v2_exp_V2_S2e_spike" / "s2e_runs.csv")
print("\ns2e_runs.csv 列:", list(r.columns))
print("行数:", len(r))
print(r.head(30).to_string())

sup = pd.read_csv(BASE / "v7_v2_exp_V2_S2e_supervised" / "s2e_supervised.csv")
print("\ns2e_supervised.csv 列:", list(sup.columns))
print(sup.to_string())

print("\n" + "=" * 70)
print("S2c 对齐两阶段")
print("=" * 70)
for f in ["diag_basin_hit.csv", "diag_lift.csv", "diag_stretch.csv"]:
    df = pd.read_csv(BASE / "v7_v2_exp_V2_S2c_aligned_twostage" / f)
    print(f"\n--- {f} 列: {list(df.columns)}")
    print(df.to_string())
runs = pd.read_csv(BASE / "v7_v2_exp_V2_S2c_aligned_twostage" / "s2c_runs.csv")
print("\ns2c_runs.csv 列:", list(runs.columns))
print("行数:", len(runs))
print(runs.head(40).to_string())
