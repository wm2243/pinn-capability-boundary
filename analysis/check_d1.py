# -*- coding: utf-8 -*-
"""核对 7.3 D1：tab:d1beta + Wilcoxon + 阈值窗 + 机制A + caseB/s8
注意：第 1 部分读取原始 run 记录（data/raw/，GitHub 不含，需 Zenodo 完整档案）。"""
import json, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1] / "data"

# ---------- 1. D1 主扫描 ----------
d1dir = BASE / "raw" / "v8" / "time_varying" / "D1_timevarying_burgers_pytorch"
rows = []
for bdir in sorted(glob.glob(str(d1dir / "beta_*"))):
    beta = float(os.path.basename(bdir).split("_")[1])
    for f in glob.glob(os.path.join(bdir, "result_seed*.json")):
        r = json.load(open(f))
        r["beta"] = beta
        rows.append(r)
df = pd.DataFrame(rows)
print("D1 列:", list(df.columns))
print("每 beta 种子数:", df.groupby("beta")["seed"].count().to_dict())

# 看有哪些 L2 字段
l2cols = [c for c in df.columns if "l2" in c.lower() or "L2" in c]
print("L2 字段:", l2cols)

print("\n--- 按 beta 汇总各候选 L2 指标 ---")
for c in l2cols:
    print(f"\n指标 {c}:")
    for b, g in df.groupby("beta"):
        v = g[c].astype(float)
        print(f"  beta={b:5.1f}: med={v.median():.4e}  range=[{v.min():.4e},{v.max():.4e}]  n(tau=.06)={int((v<.06).sum())}/11")
