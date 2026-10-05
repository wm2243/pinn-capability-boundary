# -*- coding: utf-8 -*-
"""N1+N6: D1 配对 Wilcoxon + G1b τ 扰动敏感性
N1: D1 时变 Burgers β=1/3/8/15 各 11 seeds，按 seed 配对做 Wilcoxon signed-rank
N6: G1b 阈值 τ∈[.03,.08] 扰动，重算 uniform vs 各族的 McNemar 精确 p
"""
import csv, math, os
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve()
_REL = _HERE.parents[4]                    # paper1_release/
V8 = _REL / "data" / "raw" / "v8" / "time_varying"
G1B = _REL / "data" / "raw" / "v6" / "exp_G1b_pentry"
OUT = Path(os.environ.get(
    "REPRO_OUT", str(_REL / "data" / "aggregated" / "v7_v2_exp_V2_N1_N6_stats")))
OUT.mkdir(parents=True, exist_ok=True)


def wilcoxon_signed_rank(x, y):
    """配对 Wilcoxon signed-rank，n≤30 用精确正态近似（带连续性修正）。
    返回 (W_stat, p_two_sided, n_nonzero, median_diff)。"""
    d = np.array(x, dtype=float) - np.array(y, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return (0.0, 1.0, 0, 0.0)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    W_pos = float(ranks[d > 0].sum())
    W_neg = float(ranks[d < 0].sum())
    W = min(W_pos, W_neg)
    # 正态近似（n≥10 合理）
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu + 0.5) / sigma  # +0.5 连续性修正
    # 双侧 p = 2*Phi(z)，用 erf
    p = 2.0 * (0.5 * (1 + math.erf(z / math.sqrt(2))))
    p = min(p, 1.0)
    return (W, p, n, float(np.median(np.array(x) - np.array(y))))


def mcnemar_exact(b, c):
    """精确 McNemar: b=A好B坏, c=A坏B好. H0: b=c. p=2*P(X<=min(b,c)), X~Binom(b+c,0.5)"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 精确二项累积
    log_p = 0.0
    for i in range(k + 1):
        log_p += math.comb(n, i)
    p = 2.0 * log_p / (2 ** n)
    return min(p, 1.0)


# ===== N1: D1 配对 Wilcoxon =====
print("=" * 60)
print("N1: D1 时变 Burgers 配对 Wilcoxon (n=11 seeds)")
print("=" * 60)
betas = [1.0, 3.0, 8.0, 15.0]
data = {}
for beta in betas:
    fn = V8 / f"a2_weighted_beta{beta:.1f}_K11_adam50000" / "summary.csv"
    rows = {}
    with open(fn, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows[int(r["seed"])] = float(r["L2_relative"])
    data[beta] = rows
    print(f"  β={beta:g}: median={np.median(list(rows.values())):.4e} "
          f"range=[{min(rows.values()):.4e},{max(rows.values()):.4e}]")

n1_rows = []
seeds = sorted(data[1.0].keys())
for beta in [3.0, 8.0, 15.0]:
    x = [data[1.0][s] for s in seeds]
    y = [data[beta][s] for s in seeds]
    W, p, nz, med_diff = wilcoxon_signed_rank(x, y)
    direction = "β=1 更优" if med_diff > 0 else f"β={beta:g} 更优"
    print(f"  β=1 vs β={beta:g}: W={W:.1f} p={p:.4f} n_nonzero={nz} "
          f"median_diff(1-{beta:g})={med_diff:.4e} → {direction}")
    n1_rows.append(dict(comparison=f"beta1_vs_beta{beta:g}", W=W, p_value=p,
                        n_nonzero=nz, median_diff=med_diff, direction=direction))

with open(OUT / "n1_d1_wilcoxon.csv", "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=["comparison", "W", "p_value", "n_nonzero", "median_diff", "direction"])
    wr.writeheader(); wr.writerows(n1_rows)

# ===== N6: G1b τ 扰动 =====
print("\n" + "=" * 60)
print("N6: G1b τ 扰动 McNemar 敏感性")
print("=" * 60)
runs = []
with open(G1B / "g1b_pentry_runs.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        runs.append(r)

families = ["uniform", "down_inv", "down_lin", "rational"]
taus = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
n6_rows = []

for tau in taus:
    # 按 (nu, seed) 配对，重算 good = phase0_L2 < tau
    good_map = {}  # (nu, seed, family) -> bool
    for r in runs:
        key = (float(r["nu"]), int(r["seed"]), r["family"])
        good_map[key] = float(r["phase0_L2"]) < tau
    print(f"\n  τ={tau:.2f}:")
    for fam in ["down_inv", "down_lin", "rational"]:
        b = c = 0  # b=uniform好fam坏, c=uniform坏fam好
        n_total = 0
        for (nu, seed), _ in {(k[0], k[1]): 1 for k in good_map}.items():
            gu = good_map.get((nu, seed, "uniform"))
            gf = good_map.get((nu, seed, fam))
            if gu is None or gf is None:
                continue
            n_total += 1
            if gu and not gf: b += 1
            elif not gu and gf: c += 1
        p = mcnemar_exact(b, c)
        # Holm 校正（3 族）
        print(f"    uniform vs {fam:10s}: b={b:2d} c={c:2d} n={n_total} p={p:.4f}")
        n6_rows.append(dict(tau=tau, family=fam, b=b, c=c, n_total=n_total, p_value=p))

# Holm 校正
from itertools import groupby
for tau in taus:
    subset = sorted([r for r in n6_rows if r["tau"] == tau], key=lambda x: x["p_value"])
    for i, r in enumerate(subset):
        r["holm_p"] = min(r["p_value"] * (len(subset) - i), 1.0)

with open(OUT / "n6_g1b_tau_sensitivity.csv", "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=["tau", "family", "b", "c", "n_total", "p_value", "holm_p"])
    wr.writeheader(); wr.writerows(n6_rows)

# 摘要：τ 扰动下 down_lin 是否始终显著
print("\n  --- 摘要：down_lin 在各 τ 下的 Holm 校正 p ---")
for r in n6_rows:
    if r["family"] == "down_lin":
        sig = "显著" if r["holm_p"] < 0.05 else "不显著"
        print(f"    τ={r['tau']:.2f}: p={r['p_value']:.4f} holm={r['holm_p']:.4f} {sig}")

print(f"\n结果写入 {OUT}")
