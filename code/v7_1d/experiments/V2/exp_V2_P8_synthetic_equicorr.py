# -*- coding: utf-8 -*-
"""P8：定理 4.2 的合成验证（纯线性代数，不依赖网络/训练数据）。

两部分：
  (1) 等相关矩阵  K = (1-ρ)I + ρ·11^T，ρ∈(0,1)：多起点 L-BFGS-B 求解
      min_{E≻0,对角} κ(EKE)，验证 κ*_diag/κ(K) ≈ 1（定理 4.2 预言的精确等号）。
  (2) 非对称带状（两块常相关）相关阵：少数激波配点块内强相关 ρ_in、
      多数光滑配点块内弱相关 ρ_out、跨块 ρ_cross；相关剖面非均匀，
      对角权存在有限改善 r<1，复现真实 Burgers 冻结 Jacobian（F1d）的 r≈.77--.87。

口径说明：相关阵 R = D^{-1/2} K D^{-1/2} 对一切正对角缩放严格不变，对角
（幅度）信息在归一化时已被移除，故 r<1 只可能来自 R 的非对角相关剖面非均匀，
与幅度失配无关（见论文 §4.4；旧版"幅度共线场"臂已据此删除）。

环境变量：
  V8_SMOKE=1   缩减 N/ρ/多起点，秒级冒烟测试。
  REPRO_OUT    覆盖输出目录（默认回写 release 的 data/raw/v7_v2/...）。
"""
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_V1EXP = _HERE.parent.parent / "v1"
for _p in (str(_V1EXP), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from exp_F1b_diag_capability import opt_diag_kappa  # noqa: E402

SMOKE = os.environ.get("V8_SMOKE", "0") == "1"
_REL = _HERE.parents[4]  # paper1_release 根（code/v7_1d/experiments/V2/*.py）
OUT = Path(os.environ.get(
    "REPRO_OUT",
    str(_REL / "data" / "raw" / "v7_v2" / "exp_V2_P8_synthetic_equicorr")))
OUT.mkdir(parents=True, exist_ok=True)


def equicorr_matrix(N, rho):
    """K=(1-ρ)I + ρ·11^T，等相关矩阵，特征值 1+(N-1)ρ 与 1-ρ（N-1 重）。"""
    return (1 - rho) * np.eye(N) + rho * np.ones((N, N))


def banded_block_corr(N, n_min, rho_in, rho_out, rho_cross):
    """非对称带状（两块常相关）相关阵：前 n_min 个点为激波带（块内 ρ_in），
    其余为光滑带（块内 ρ_out），跨块 ρ_cross，对角为 1。"""
    R = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            if i < n_min and j < n_min:
                r = rho_in
            elif i >= n_min and j >= n_min:
                r = rho_out
            else:
                r = rho_cross
            R[i, j] = R[j, i] = r
    return R


def kappa(K):
    ev = np.linalg.eigvalsh((K + K.T) / 2.0)
    return float(ev[-1] / max(ev[0], 1e-30)), float(ev[0])


def main():
    # ---------- (1) 等相关类：精确等号 r=1 ----------
    fn_eq = OUT / "p8_synthetic_equicorr.csv"
    fields_eq = ["N", "rho", "kappa_theory", "kappa_numeric", "kappa_opt_diag",
                 "r_ratio", "nstart", "opt_s", "abs_err"]
    if SMOKE:
        Ns, rhos = [4, 16], [0.5, 0.9]
    else:
        Ns = [4, 8, 16, 32, 64, 128]
        rhos = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    rows = []
    for N in Ns:
        for rho in rhos:
            K = equicorr_matrix(N, rho)
            k_num, _ = kappa(K)
            k_theory = (1 + (N - 1) * rho) / max(1 - rho, 1e-30)
            nstart = 3 if SMOKE else (20 if N <= 32 else (12 if N <= 64 else 8))
            t0 = time.perf_counter()
            k_opt = opt_diag_kappa(K, nstart=nstart, seed=3000 + N * 100 + int(rho * 1000))
            dt = time.perf_counter() - t0
            r = k_opt / k_num
            rows.append(dict(N=N, rho=rho, kappa_theory=float(k_theory),
                             kappa_numeric=float(k_num), kappa_opt_diag=float(k_opt),
                             r_ratio=float(r), nstart=nstart, opt_s=float(dt),
                             abs_err=float(abs(r - 1.0))))
            print(f"  [equicorr] N={N:3d} rho={rho:.2f}: k_num={k_num:.4e} "
                  f"k_opt={k_opt:.4e} r={r:.6f} err={abs(r-1):.2e} ({dt:.2f}s)", flush=True)
    with open(fn_eq, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields_eq)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n[P8] 等相关 {len(rows)} 组 -> {fn_eq.name}；"
          f"max|r-1|={max(x['abs_err'] for x in rows):.2e}\n", flush=True)

    # ---------- (2) 非对称带状（两块常相关）：r<1 ----------
    fn_bd = OUT / "p8_banded_corr.csv"
    fields_bd = ["N", "n_min", "rho_in", "rho_out", "rho_cross", "kappa_R",
                 "kappa_opt_diag", "r_ratio", "lambda_min", "psd", "nstart", "opt_s"]
    band_params = [(4, 0.70, 0.60, 0.40), (5, 0.75, 0.65, 0.40)]
    Nb = 16
    nstart_b = 5 if SMOKE else 30
    brows = []
    print("--- 非对称带状（两块常相关）相关阵 ---", flush=True)
    for (n_min, rin, rout, rcross) in band_params:
        R = banded_block_corr(Nb, n_min, rin, rout, rcross)
        kR, lmin = kappa(R)
        t0 = time.perf_counter()
        k_opt = opt_diag_kappa(R, nstart=nstart_b, seed=8000 + n_min)
        dt = time.perf_counter() - t0
        r = k_opt / kR
        brows.append(dict(N=Nb, n_min=n_min, rho_in=rin, rho_out=rout, rho_cross=rcross,
                          kappa_R=float(kR), kappa_opt_diag=float(k_opt), r_ratio=float(r),
                          lambda_min=float(lmin), psd=bool(lmin > -1e-8),
                          nstart=nstart_b, opt_s=float(dt)))
        print(f"  [banded] N={Nb} n_min={n_min} rin={rin} rout={rout} cross={rcross}: "
              f"PSD={lmin > -1e-8} k(R)={kR:.3f} k_opt={k_opt:.3f} r={r:.4f} ({dt:.2f}s)",
              flush=True)
    with open(fn_bd, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields_bd)
        wr.writeheader()
        wr.writerows(brows)
    print(f"\n[P8] 带状 {len(brows)} 组 -> {fn_bd.name}", flush=True)
    print(f"[P8] 完成，产物目录 {OUT}", flush=True)


if __name__ == "__main__":
    main()
