from pathlib import Path
import pandas as pd
import numpy as np

base = Path(__file__).resolve().parents[1] / "data" / "aggregated" / "v7_v1_exp_E1b_smooth"
summary_path = base / "e1b_summary.csv"
runs_path = base / "e1b_runs.csv"
old_summary_path = base / "e1b_summary_1.csv"

summary = pd.read_csv(summary_path)
runs = pd.read_csv(runs_path)

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 220)

print("=== e1b_summary.csv: shape", summary.shape, "===")
print(summary.to_string(index=False))

print("\n=== e1b_runs.csv: shape", runs.shape, "===")
print("columns:", list(runs.columns))
print("feat values:", sorted(runs["feat"].unique()))
print("nu values:", sorted(runs["nu"].unique()))
print("beta grid by feat/nu:")
print(runs.groupby(["feat", "nu"])["beta"].agg(lambda s: sorted(s.unique())).to_string())
print("seeds per feat/nu/beta:")
print(runs.groupby(["feat", "nu", "beta"])["seed"].nunique().to_string())

# 按附录口径复核：gain = L2(beta=1) / min_beta L2 - 1，逐 seed 先取 min，再汇总
rows = []
for (feat, nu), g in runs.groupby(["feat", "nu"]):
    b1 = g[np.isclose(g["beta"], 1.0)].set_index("seed")["L2"]
    best = g.loc[g.groupby("seed")["L2"].idxmin()].set_index("seed")[["L2", "beta"]]
    common = b1.index.intersection(best.index)
    gain = (b1.loc[common] / best.loc[common, "L2"]) - 1.0
    ratio = b1.loc[common] / best.loc[common, "L2"]
    rows.append({
        "feat": feat,
        "nu": nu,
        "n_seed": len(common),
        "beta_star_med": best.loc[common, "beta"].median(),
        "gain_med": gain.median(),
        "gain_q1": gain.quantile(0.25),
        "gain_q3": gain.quantile(0.75),
        "gain_min": gain.min(),
        "gain_max": gain.max(),
        "ratio_med": ratio.median(),
        "ratio_min": ratio.min(),
        "ratio_max": ratio.max(),
        "L2_b1_med": b1.loc[common].median(),
        "L2_best_med": best.loc[common, "L2"].median(),
    })

check = pd.DataFrame(rows).sort_values(["feat", "nu"]).reset_index(drop=True)
print("\n=== 从 e1b_runs.csv 复算（逐 seed best；gain=ratio-1）===")
print(check.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

# 与 summary 的 gain_med 对齐检查
merge = summary.merge(
    check[["feat", "nu", "gain_med", "beta_star_med", "n_seed"]],
    on=["feat", "nu"],
    how="outer",
    suffixes=("_summary", "_recalc"),
)
merge["gain_med_diff"] = merge["gain_med_summary"] - merge["gain_med_recalc"]
print("\n=== summary 与 runs 复算对齐 ===")
print(merge[["feat", "nu", "gain_med_summary", "gain_med_recalc", "gain_med_diff", "beta_star_med_summary", "beta_star_med_recalc", "n_seed"]].to_string(index=False, float_format=lambda x: f"{x:.6g}"))

if old_summary_path.exists():
    old = pd.read_csv(old_summary_path)
    print("\n=== e1b_summary_1.csv: shape", old.shape, "===")
    print(old.to_string(index=False))
    cmp = summary.merge(old, on=["feat", "nu"], how="outer", suffixes=("_new", "_old"), indicator=True)
    diff_cols = [c for c in cmp.columns if c.endswith("_new")]
    print("\n=== summary 与 summary_1 合并键分布 ===")
    print(cmp["_merge"].value_counts().to_string())
    if "gain_med_new" in cmp.columns and "gain_med_old" in cmp.columns:
        cmp["gain_med_new_minus_old"] = cmp["gain_med_new"] - cmp["gain_med_old"]
        print(cmp[["feat", "nu", "_merge", "gain_med_new", "gain_med_old", "gain_med_new_minus_old"]].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
