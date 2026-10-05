# -*- coding: utf-8 -*-
"""
V2 实验 S2e-sup（M4 补充诊断）：Allen-Cahn homoclinic 尖峰的【纯监督拟合能力】对照
================================================================================
目的（回应第三轮审稿 M4 / C4）：
  S2e 主实验里无种子时 12/12 塌到另一合法稳态 u≡0，这是“可表达≠可到达”还是
  “网络类根本表达不了 √2 sech(x/ε) 窄峰”？本脚本剥离 PDE 残差，做纯数据拟合：
  固定目标 u*(x)=√2 sech(x/ε)，损失只含 (uθ-u*)²（无 PDE 残差、无空间加权、无 Gate、
  无课程），网络/带宽/硬边界与 S2e 精修端完全一致（HardBCPINN，Fourier σ=15，
  lift_linear 两端精确小尾值）。若纯监督能把【全域归一化 L2】压到远低于观测到的
  塌零态 L2≈1（论文 .999 地板），则窄峰在该网络类中 ε0-可表达、12/12 塌零是
  优化/初值（θ、S–θ 接口）问题，而非表示能力（S）问题。

不修改公共代码；结果写独立目录 exp_V2_S2e_supervised/s2e_supervised.csv。
SMOKE：set V8_SMOKE=1  -> 2 seed × 200 步自检。
全量：python exp_V2_S2e_supervised.py            (12 seed × Adam 20000 + L-BFGS 500)
================================================================================
"""
import os, sys, csv, copy
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve(); _V7 = _HERE.parents[2]; _V1EXP = _V7 / "experiments" / "v1"
for _p in (str(_V7), str(_V1EXP), str(_HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)

import v8_matrix_common as mc
import exp_V2_S2e_spike as S2e
from models.pinn_FourierFeatures_model import HardBCPINN

DEV = mc.DEVICE
OUT = str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S2e_supervised")
os.makedirs(OUT, exist_ok=True)
CSVP = os.path.join(OUT, "s2e_supervised.csv")
FIELDS = ["eps", "sigma", "n_train", "seed", "adam_steps", "lbfgs_steps",
          "L2_adam", "L2", "Linf", "L2_peak", "peak_amp", "cls"]

EPS = 0.0044
SIGMA = 15.0
N_TRAIN = 2048
N_SEED = 12
ADAM_STEPS = 20000
LBFGS_STEPS = 500


def build_supervised_data(eps, n):
    """固定密集均匀监督网格（与 N_TEST 同密度），目标 u* 与精确边界。"""
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
    pde = S2e.SpikePDE1D(dict(spike_eps=eps))
    u = pde.exact_np(x).astype(np.float64)
    xt = torch.tensor(x, dtype=torch.float32, device=DEV)
    ut = torch.tensor(u, dtype=torch.float32, device=DEV)
    return xt, ut


def run_one(eps, seed, adam_steps, lbfgs_steps, n_train, sigma):
    S2e.r2s.set_seed(seed)
    cfg = S2e.sp_base_cfg(eps, f"sup_sp{eps:g}_s{seed}")
    cfg.update(dict(fourier_scale=float(sigma), sigma_lo=None, sigma_hi=None, sigma_anneal_T=0))
    model = HardBCPINN(cfg).to(DEV)
    try:
        model.set_sigma(float(sigma))
    except Exception:
        pass
    xt, ut = build_supervised_data(eps, n_train)
    xte, un = S2e.test_tensors(eps)

    def loss_fn():
        pred = model(xt)
        return torch.mean((pred - ut) ** 2)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for it in range(adam_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    ev_a = S2e.eval_pred(model, xte, un, eps)

    # L-BFGS 收尾（确定性精修，逼近表示能力上界）
    if lbfgs_steps > 0:
        opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_steps,
                                 history_size=50, tolerance_grad=1e-12, tolerance_change=1e-14,
                                 line_search_fn="strong_wolfe")
        def closure():
            opt2.zero_grad(); l = loss_fn(); l.backward(); return l
        try:
            opt2.step(closure)
        except Exception as e:
            print(f"  [seed {seed}] L-BFGS early stop: {e}")
    ev = S2e.eval_pred(model, xte, un, eps)
    return ev_a, ev


def main():
    smoke = bool(mc.SMOKE)
    n_seed = 2 if smoke else N_SEED
    adam_steps = 200 if smoke else ADAM_STEPS
    lbfgs_steps = 0 if smoke else LBFGS_STEPS
    n_train = 512 if smoke else N_TRAIN

    xt_e, un_e = S2e.test_tensors(EPS)
    g = float(np.linalg.norm(un_e.flatten()))
    print(f"[S2e-sup] eps={EPS} sigma={SIGMA} n_train={n_train} seeds={n_seed} "
          f"adam={adam_steps} lbfgs={lbfgs_steps}  ||u*||={g:.4f} dev={DEV}")

    fout = open(CSVP, "w", newline="", encoding="utf-8-sig")
    wcsv = csv.DictWriter(fout, fieldnames=FIELDS); wcsv.writeheader()
    L2s = []
    try:
        for seed in range(n_seed):
            ev_a, ev = run_one(EPS, seed, adam_steps, lbfgs_steps, n_train, SIGMA)
            L2s.append(ev["L2"])
            wcsv.writerow(dict(eps=EPS, sigma=SIGMA, n_train=n_train, seed=seed,
                               adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
                               L2_adam=ev_a["L2"], L2=ev["L2"], Linf=ev["Linf"],
                               L2_peak=ev["L2_peak"], peak_amp=ev["peak_amp"], cls=ev["cls"]))
            fout.flush()
            print(f"  seed {seed:2d}: Adam L2={ev_a['L2']:.3e} -> 精修 L2={ev['L2']:.3e} "
                  f"Linf={ev['Linf']:.3e} 峰区={ev['L2_peak']:.3e} 峰幅={ev['peak_amp']:+.3f} ({ev['cls']})")
    finally:
        fout.close()
    L2s = np.asarray(L2s)
    print("\n===== S2e 纯监督拟合能力对照（应远低于塌零态 L2≈1 / .999 地板）=====")
    print(f"  全域归一化 L2  median={np.median(L2s):.3e}  min={L2s.min():.3e}  max={L2s.max():.3e}  n={len(L2s)}")
    print(f"  结论判据：median L2 ≪ 0.999 即窄峰 ε0-可表达，12/12 塌零归 θ/初值，非 S 表示能力。")
    print("csv:", CSVP)


if __name__ == "__main__":
    main()
