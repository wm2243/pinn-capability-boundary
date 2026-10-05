# -*- coding: utf-8 -*-
"""
统计工具（V8 新增；对应第 9 章 E4 统计规范与附录 D）

解决旧稿三个统计硬伤：
  1) n=5 的成功率只给点估计（p̂=0.6）毫无信息量——这里统一给 Wilson 95% 区间；
  2) 多起点覆盖率 p_K = 1-(1-p)^K 不能代入 p 的点估计，必须把 p 的 Wilson 区间整体映射；
  3) 跨方法比较用配对检验（同种子配对 Wilcoxon 符号秩），而非独立样本 t 检验。
纯标准库 + numpy + scipy，不依赖 pandas。
"""

import csv
import math
import os
from collections import defaultdict

import numpy as np


def wilson_interval(k, n, z=1.96):
    """二项成功率 Wilson 区间。k=成功数, n=试验数，返回 (p_hat, lo, hi)。"""
    n = int(n)
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return float(phat), float(max(0.0, center - half)), float(min(1.0, center + half))


def pk_interval(p_lo, p_hi, K):
    """
    K 个独立起点至少一次进入好盆地的概率 p_K = 1-(1-p)^K，区间由 p 的 Wilson 区间单调映射。
    （p 越大 p_K 越大，故下界用 p_lo、上界用 p_hi。）
    """
    if any(not math.isfinite(v) for v in (p_lo, p_hi)):
        return float("nan"), float("nan")
    return float(1.0 - (1.0 - p_lo) ** K), float(1.0 - (1.0 - p_hi) ** K)


def paired_wilcoxon(a, b):
    """配对 Wilcoxon 符号秩（同种子两方法差值）。返回 (statistic, p_value, n_effective)。"""
    from scipy.stats import wilcoxon
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    d = d[np.isfinite(d)]
    d = d[d != 0]
    n_eff = int(d.size)
    if n_eff < 6:                       # 样本太少时符号秩无功效，如实返回 nan
        return float("nan"), float("nan"), n_eff
    try:
        m = min(len(a), len(b))
        stat, p = wilcoxon(a[:m], b[:m], zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p), n_eff
    except Exception as e:  # noqa
        return float("nan"), float("nan"), n_eff


def summarize_success(values, threshold, smaller_is_better=True, z=1.96):
    """给定一组指标值与达标阈值，返回 n/成功率/Wilson 区间/中位/best/worst。"""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    n = int(v.size)
    if n == 0:
        return dict(n=0)
    k = int(np.sum(v < threshold)) if smaller_is_better else int(np.sum(v > threshold))
    phat, lo, hi = wilson_interval(k, n, z)
    return dict(n=n, success_k=k, success_rate=phat, wilson_lo=lo, wilson_hi=hi,
                median=float(np.median(v)), best=float(v.min()), worst=float(v.max()),
                std=float(v.std()))


def aggregate_csv(path, value_col, threshold, group_cols, smaller_is_better=True):
    """
    通用聚合：读 runs 级 csv，按 group_cols 分组，对 value_col 做成功率 + Wilson。
    返回 (rows:list[dict], fieldnames)。供主结果表（A1 分母=配置×种子）使用。
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        recs = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in recs:
        try:
            val = float(r[value_col])
        except (KeyError, ValueError):
            continue
        key = tuple(r.get(c, "") for c in group_cols)
        groups[key].append(val)
    rows = []
    for key, vals in sorted(groups.items()):
        stat = summarize_success(vals, threshold, smaller_is_better)
        row = {c: key[i] for i, c in enumerate(group_cols)}
        row.update(stat)
        rows.append(row)
    fields = list(group_cols) + ["n", "success_k", "success_rate", "wilson_lo", "wilson_hi",
                                 "median", "best", "worst", "std"]
    return rows, fields


def _cli():
    """命令行：python -m utils.stats_toolkit <runs.csv> <value_col> <threshold> <group_col...>"""
    import sys
    if len(sys.argv) < 5:
        print("用法: python -m utils.stats_toolkit runs.csv L2_Error 0.01 nu beta")
        return
    path, val_col, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
    group_cols = sys.argv[4:]
    rows, fields = aggregate_csv(path, val_col, thr, group_cols)
    out = os.path.splitext(path)[0] + "_success_table.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"{' / '.join(group_cols):<24}{'n':>4}{'succ':>8}{'Wilson95%':>20}{'median':>12}")
    for r in rows:
        print(f"{'/'.join(str(r[c]) for c in group_cols):<24}{r['n']:>4}"
              f"{r['success_rate']:>8.2f}   [{r['wilson_lo']:.2f},{r['wilson_hi']:.2f}]"
              f"{r['median']:>12.3e}")
    print("saved", out)


if __name__ == "__main__":
    _cli()
