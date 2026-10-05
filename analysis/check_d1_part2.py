# -*- coding: utf-8 -*-
"""核对 D1 第二部分：Wilcoxon / 阈值窗 / 机制A / caseB 裁剪逐位相同"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

AGG = Path(__file__).resolve().parents[1] / "data" / "aggregated" / "v8_time_varying"

def load(f):
    d = pd.read_csv(AGG / f)
    return d.set_index("seed").sort_index()

b1 = load("summary.csv")["L2_relative"]
b3 = load("summary_2.csv")["L2_relative"]
b8 = load("summary_3.csv")["L2_relative"]
b15 = load("summary_1.csv")["L2_relative"]

print("=== tab:d1beta ===")
for name, v in [("1", b1), ("3", b3), ("8", b8), ("15", b15)]:
    print(f"beta={name}: med={v.median():.4e} range=[{v.min():.4e},{v.max():.4e}] n(.06)={(v<.06).sum()}/11")

print("\n=== Wilcoxon β=1 vs 3/8/15（按 seed 配对）===")
for name, v in [("3", b3), ("8", b8), ("15", b15)]:
    d = b1 - v
    print(f"vs beta={name}: 全部11对 b1 更小? {(d<0).all()}  零差数={(d==0).sum()}")
    w, p = stats.wilcoxon(d)
    print(f"   W={w}, p={p:.5f}")

print("\n=== β=3 阈值窗 ===")
v = np.sort(b3.values)
print("β=3 排序后 L2:", np.array2string(v, precision=5))
for tau in [0.045, 0.06, 0.061, 0.0611, 0.0649, 0.065]:
    print(f"  tau={tau}: n={int((b3<tau).sum())}/11")

print("\n=== 机制 A（aggregated summary.json）===")
js = json.load(open(AGG / "summary.json"))
print("记录数:", len(js), "键:", list(js[0].keys()))
rec8 = [r for r in js if r["beta"] == 8.0]
print("beta=8 记录数:", len(rec8))
ma = pd.DataFrame([r["mechanism_A"] for r in rec8])
print("mechanism_A 字段:", list(ma.columns))
for c in ma.columns:
    col = ma[c].astype(float)
    print(f"  {c}: med={col.median():.4f} range=[{col.min():.4f},{col.max():.4f}]")

print("\n=== caseB/s8 人为坏盆地 ===")
ctrl = load("summary_11.csv")["L2_relative"]
dn3 = load("summary_12.csv")["L2_relative"]
dn8 = load("summary_13.csv")["L2_relative"]
up3 = load("summary_14.csv")["L2_relative"]
up8 = load("summary_15.csv")["L2_relative"]
print(f"对照(β=1): med={ctrl.median():.5f}, n(.06)={(ctrl<.06).sum()}/11")
print(f"减权β=3(1/3): med={dn3.median():.5f}")
print(f"减权β=8(1/8): med={dn8.median():.5f}")
print(f"增权β=3: med={up3.median():.5f}; 增权β=8: med={up8.median():.5f}")

clip = pd.read_csv(AGG / "summary_18.csv")
print("\nclip 文件列:", list(clip.columns), "行数:", len(clip))
print(clip.groupby(["group", "clip_norm"])["L2_relative"].median() if "group" in clip.columns else clip.head(15).to_string())
# 逐位相同检查
for (g, cn), sub in clip.groupby(["group", "clip_norm"]) if "group" in clip.columns else []:
    sub = sub.set_index("seed").sort_index()["L2_relative"]
    same = np.allclose(sub.values, ctrl.values, rtol=0, atol=0)
    maxdiff = np.abs(sub.values - ctrl.values).max()
    print(f"  group={g} clip={cn}: 与对照逐位相同? {same}, maxdiff={maxdiff:.3e}")

print("\n=== 另一代文件（供对照）===")
for f in ["summary_19.csv", "summary_23.csv", "summary_24.csv"]:
    v = load(f)["L2_relative"]
    print(f"{f}: med={v.median():.5f}")
