# -*- coding: utf-8 -*-
"""
S2e 诊断：尖峰问题大 multistart 命中率探针（只跑 Phase0），排除“K=4 运气差/低命中”。
复用 exp_V2_S2e_spike.enter_basin（含盆地形态分类 peak/negpeak/zero/other）。
"""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
for _p in (str(_HERE.parents[2]), str(_HERE.parents[2] / "experiments" / "v1"), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import exp_V2_S2e_spike as S2e

for eps in [0.01, 0.0044, 0.002]:
    xt, un = S2e.test_tensors(eps)
    K = 8
    basin = S2e.enter_basin(eps, 0, xt, un, k=K, p0=2000, reuse=False)
    co = basin["counts"]
    print(f"[diag] eps={eps:g} K={K}: 正峰 {co['peak']} 负峰 {co['negpeak']} "
          f"塌零 {co['zero']} 其他 {co['other']} | best L2={basin['l2']:.3e} cls={basin['cls']}")
