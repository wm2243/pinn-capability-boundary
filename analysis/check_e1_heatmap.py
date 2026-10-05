from pathlib import Path
import pandas as pd
import numpy as np

base = Path(__file__).resolve().parents[1] / "data" / "aggregated" / "v7_v1_exp_E1_beta_nu"
runs = pd.read_csv(base / "e1_runs.csv")
agg = pd.read_csv(base / "e1_agg.csv")

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)

# 每格中位数、IQR 相对宽度、极值跨度和成功率
rows = []
for (nu, beta), g in runs.groupby(["nu", "beta"]):
    med = g["L2"].median()
    q1 = g["L2"].quantile(.25)
    q3 = g["L2"].quantile(.75)
    rows.append({
        "nu": nu,
        "beta": beta,
        "n": len(g),
        "med": med,
        "q1": q1,
        "q3": q3,
        "iqr_over_med": (q3-q1)/med if med else np.nan,
        "min": g["L2"].min(),
        "max": g["L2"].max(),
        "mean": g["L2"].mean(),
        "gate_ok_frac": g["gate_ok"].mean(),
        "success_frac_lt_.06": (g["L2"] < .06).mean(),
    })
cell = pd.DataFrame(rows).sort_values(["nu", "beta"])
print("=== E1 每格中位数与离散度 ===")
print(cell.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# 中位数最优 beta 与逐种子最优 beta 的一致性
print("\n=== 中位数热力图最优 beta 是否代表逐种子多数 ===")
for nu, g in runs.groupby("nu"):
    med_best_beta = g.groupby("beta")["L2"].median().idxmin()
    per_seed_best = g.loc[g.groupby("seed")["L2"].idxmin()].set_index("seed")["beta"]
    frac_match = (per_seed_best == med_best_beta).mean()
    print(f"nu={nu:g}: median-best beta={med_best_beta:g}, 逐种子最优中位={per_seed_best.median():g}, "
          f"逐种子最优分布={per_seed_best.value_counts().sort_index().to_dict()}, "
          f"与median-best一致比例={frac_match:.2f}")

# 逐种子曲线的 Spearman：检验“同一组种子是否共同随 beta 单调变化”
def spearman(x, y):
    x = pd.Series(x).rank(method="average").to_numpy(float)
    y = pd.Series(y).rank(method="average").to_numpy(float)
    x = x - x.mean(); y = y - y.mean()
    den = np.sqrt((x*x).sum() * (y*y).sum())
    return float((x*y).sum()/den) if den else np.nan

print("\n=== 逐种子 L2-beta 单调性（Spearman；+1=随beta增坏，-1=随beta增好）===")
for nu, g in runs.groupby("nu"):
    rhos = []
    for seed, gs in g.groupby("seed"):
        gs = gs.sort_values("beta")
        if gs["L2"].nunique() > 1 and gs["beta"].nunique() > 1:
            rhos.append(spearman(gs["beta"], gs["L2"]))
    print(f"nu={nu:g}: median rho={np.median(rhos):+.2f}, IQR=[{np.quantile(rhos,.25):+.2f},{np.quantile(rhos,.75):+.2f}], "
          f"正/负/总数={(np.array(rhos)>0).sum()}/{(np.array(rhos)<0).sum()}/{len(rhos)}")

# 中位数 vs 均值排序是否一致
print("\n=== 中位数与均值给出的 beta 排序 ===")
for nu, g in runs.groupby("nu"):
    med_rank = g.groupby("beta")["L2"].median().sort_values()
    mean_rank = g.groupby("beta")["L2"].mean().sort_values()
    print(f"nu={nu:g}: median {list(med_rank.index)} | mean {list(mean_rank.index)}")

# agg 中 success 字段的范围
print("\n=== e1_agg success 范围（按 nu）===")
print(agg.groupby("nu")["success"].agg(["count", "min", "median", "max"]).to_string(float_format=lambda x: f"{x:.3g}"))
