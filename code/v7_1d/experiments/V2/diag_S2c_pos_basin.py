# -*- coding: utf-8 -*-
"""
S2c 归因探针（只跑 Phase0，不精修）：同厚度 bur ν=.005 与 pos ε=.01 的 multistart 盆地命中率
================================================================================
baseline 现象：bur K=4 有 1 颗进好盆地(L2~8e-3)，pos K=4 四颗全坏(~1.0)。
但 K=4、p_entry≈.15 时 0/4 概率≈.52，不能据此判系统性失败。本探针加大到 K=8，并对 pos
扫 σ_hi∈{15,30}，统计 Phase0 候选盆地 L2 分布与命中率（阈值 .06/.1/.3）：
  * pos σ15 K=8 若出现 L2~.01-.1 好盆地   -> 只是命中率低/预算问题，加大 multistart 即可对齐；
  * σ15 0/8 但 σ30 命中                   -> 贴边层需更高带宽（表示调度参数差异，可补救）；
  * σ15/σ30 K=8 全败(min>.3)             -> 系统性 ansatz/几何失配（lift_linear + (1-x^2) 湮灭）。
产物：results/v7/V2/exp_V2_S2c_aligned_twostage/diag_basin_hit.csv
"""
import os, sys, csv, importlib
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve(); _V7 = _HERE.parents[2]; _V1 = _V7 / "experiments" / "v1"
for _p in (str(_V7), str(_V1), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import run_two_stage_v6 as r2s
import v8_matrix_common as mc
S = importlib.import_module("exp_V2_S2c_aligned_twostage")

OUT = os.path.join(str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S2c_aligned_twostage"))
os.makedirs(OUT, exist_ok=True)
K = int(os.environ.get("DIAG_K", "8"))
P0 = int(os.environ.get("DIAG_P0", "2000"))
THRS = [0.06, 0.1, 0.3]


def summ(tag, l2s):
    a = np.sort(np.asarray(l2s, float))
    row = dict(config=tag, K=len(a), min=float(a.min()), median=float(np.median(a)),
               **{f"hit<{t}": int(np.mean(a < t) * len(a)) for t in THRS})
    print(f"\n[{tag}] K={len(a)}  min={a.min():.3e}  med={np.median(a):.3e}  "
          + "  ".join(f"#(L2<{t})={int((a < t).sum())}" for t in THRS))
    print("  全部候选 L2:", " ".join(f"{v:.2e}" for v in a))
    return row


def main():
    rows = []
    # --- Burgers 对照 ν=.005，K=8，σ15 ---
    r2s.MULTISTART_K = K; r2s.SELECT_BY = "truth"
    r2s.SIG_HI = 15.0; r2s.SIGMA_HI_CFG = 15.0; r2s.PHASE0_EPOCHS = P0
    xtb, utb = r2s.build_test(0.005)
    _bb, cb = r2s.run_phase0(0.005, 0, xtb, utb, return_all=True)
    rows.append(summ(f"bur nu=.005 sig15 K{K}", [c["l2"] for c in cb]))

    # --- pos ε=.01，K=8，σ=15 / 30 ---
    xtp, unp = S.test_tensors("pos", 0.01)
    for sig in [15.0, 30.0]:
        _bp, cp = S.enter_basin_pos(0.01, 0, xtp, unp, k=K, p0=P0,
                                    sigma_hi=sig, return_all=True, reuse=True)
        rows.append(summ(f"pos eps=.01 sig{sig:g} K{K}", [c["l2"] for c in cp]))

    p = os.path.join(OUT, "diag_basin_hit.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
