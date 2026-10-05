from pathlib import Path
import pandas as pd
import numpy as np

base = Path(__file__).resolve().parents[1] / "data" / "aggregated" / "v7_v1_exp_E1b_smooth"
runs = pd.read_csv(base / "e1b_runs.csv")
csep = pd.read_csv(base / "e1b_csep.csv")

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 240)

print("=== e1b_csep.csv: shape", csep.shape, "===")
print("columns:", list(csep.columns))
print(csep.to_string(index=False))

# 对 L2 与 l2_band 两种口径分别复算逐 seed 最优与增益
rows = []
for metric in ["L2", "l2_band"]:
    for (feat, nu), g in runs.groupby(["feat", "nu"]):
        b1 = g[np.isclose(g["beta"], 1.0)].set_index("seed")[metric]
        best = g.loc[g.groupby("seed")[metric].idxmin()].set_index("seed")[["beta", metric]]
        common = b1.index.intersection(best.index)
        gain = b1.loc[common] / best.loc[common, metric] - 1.0
        rows.append({
            "metric": metric,
            "feat": feat,
            "nu": nu,
            "n": len(common),
            "beta*_med": best.loc[common, "beta"].median(),
            "gain_med": gain.median(),
            "gain_q1": gain.quantile(.25),
            "gain_q3": gain.quantile(.75),
            "Pr_gain_pos": (gain > 0).mean(),
            "ratio_med": (gain + 1).median(),
            "b1_med": b1.loc[common].median(),
            "best_med": best.loc[common, metric].median(),
        })

out = pd.DataFrame(rows).sort_values(["metric", "feat", "nu"])
print("\n=== 逐 seed 最优增益：L2 与 l2_band 两种口径 ===")
print(out.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

# 固定高带宽失配：Hi 相对 MLP/Match 的终态误差倍数（β=1 与逐 seed best 两种口径）
wide = []
for nu, g in runs.groupby("nu"):
    rec = {"nu": nu}
    for feat in ["MLP", "Match", "Hi"]:
        gf = g[g["feat"] == feat]
        b1 = gf[np.isclose(gf["beta"], 1.0)].set_index("seed")["L2"]
        best = gf.loc[gf.groupby("seed")["L2"].idxmin()].set_index("seed")["L2"]
        rec[f"{feat}_b1_med"] = b1.median()
        rec[f"{feat}_best_med"] = best.median()
    for denom in ["MLP", "Match"]:
        rec[f"Hi/{denom}_b1"] = rec["Hi_b1_med"] / rec[f"{denom}_b1_med"] if rec[f"{denom}_b1_med"] else np.nan
        rec[f"Hi/{denom}_best"] = rec["Hi_best_med"] / rec[f"{denom}_best_med"] if rec[f"{denom}_best_med"] else np.nan
    wide.append(rec)
wide = pd.DataFrame(wide).sort_values("nu")
print("\n=== Hi 相对 MLP/Match 的 L2 倍数（β=1 与逐 seed best）===")
print(wide[["nu", "MLP_b1_med", "Match_b1_med", "Hi_b1_med", "Hi/MLP_b1", "Hi/Match_b1", "MLP_best_med", "Match_best_med", "Hi_best_med", "Hi/MLP_best", "Hi/Match_best"]].to_string(index=False, float_format=lambda x: f"{x:.6g}"))

# C-Sep 是否支持光滑端平凡化：按 ν 汇总（若有多 feat/seed）
for col in ["frac_S", "Delta", "inversion", "minS", "maxO", "meanS", "meanO"]:
    if col in csep.columns:
        print(f"\n=== C-Sep {col} 按 nu 汇总 ===")
        print(csep.groupby("nu")[col].agg(["count", "median", "min", "max"]).to_string(float_format=lambda x: f"{x:.6g}"))
