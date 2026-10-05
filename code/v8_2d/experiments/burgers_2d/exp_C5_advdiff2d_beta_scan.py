# -*- coding: utf-8 -*-
"""
C5: 2D 线性对流扩散 — β 加权/减权扫描
======================================
验证：好盆地内空间加权是否单调恶化精度（和 1D 时变 A2 结论对齐）

权重：径向 w(x,y) = 1 + (beta-1) * (1 - r/pi)，r=sqrt(x^2+y^2)
  beta > 1: 中心加权（中心权重高）
  beta < 1: 中心减权（中心权重低）

方程：u_t + a u_x + b u_y = nu (u_xx + u_yy)
解析解同 C4
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import time

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "C5_advdiff2d_beta_scan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 方程参数
NU = 0.01
A = 1.0
B = 0.5

# 训练参数
SEEDS = list(range(11))
BETAS = [1.0, 3.0, 8.0]   # 典型值，不跑密扫描
K_FEAT = 32
HIDDEN = 50
LAYERS = 4
LR = 1e-3
STEPS = 50000
NUM_DOMAIN = 4000
NUM_IC = 200
LAMBDA_IC = 100.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- smoke 开关：V8_SMOKE=1 时缩到 1 种子/单 β/20 步，输出独立 _smoke 目录（不污染正式结果）----
import os as _os
SMOKE = bool(_os.environ.get("V8_SMOKE"))
if SMOKE:
    SEEDS, BETAS, STEPS = [0], [1.0], 20
    OUT_DIR = HERE / "results" / "C5_advdiff2d_beta_scan_smoke"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def true_solution(x, y, t):
    return (torch.exp(-NU * t) * torch.sin(x - A * t)
            + 0.5 * torch.exp(-4 * NU * t) * torch.cos(2 * (y - B * t))
            + 0.3 * torch.exp(-2 * NU * t) * torch.sin(x + y - (A + B) * t))


def true_solution_np(x, y, t):
    return (np.exp(-NU * t) * np.sin(x - A * t)
            + 0.5 * np.exp(-4 * NU * t) * np.cos(2 * (y - B * t))
            + 0.3 * np.exp(-2 * NU * t) * np.sin(x + y - (A + B) * t))


def weight_radial(x, y, beta):
    """径向权重：中心(0,0)权重=beta，边界权重=1"""
    r = torch.sqrt(x ** 2 + y ** 2)
    return 1.0 + (beta - 1.0) * (1.0 - r / np.pi)


class Fourier2D_T(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.k = int(k)
        freqs = torch.arange(1, self.k + 1, dtype=torch.float32)
        self.register_buffer('freqs', freqs)

    def forward(self, x, y, t):
        xf = x * self.freqs; yf = y * self.freqs
        return torch.cat([x, y, t, torch.cos(xf), torch.sin(xf),
                          torch.cos(yf), torch.sin(yf)], dim=1)


class PINN2D(nn.Module):
    def __init__(self, k_feat=32, hidden=50, layers=4):
        super().__init__()
        self.feat = Fourier2D_T(k_feat)
        d_in = 3 + 4 * k_feat
        mods = [nn.Linear(d_in, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x, y, t):
        return self.net(self.feat(x, y, t))


def pde_residual(model, x, y, t, nu, a, b):
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    u = model(x, y, t)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0]
    return (u_t + a * u_x + b * u_y - nu * (u_xx + u_yy)).squeeze(-1)


def evaluate(model):
    model.eval()
    x_ref = np.linspace(-np.pi, np.pi, 128)
    y_ref = np.linspace(-np.pi, np.pi, 128)
    t_ref = np.linspace(0, 1.0, 6)
    l2_per_time = []
    for tt in t_ref:
        xx, yy = np.meshgrid(x_ref, y_ref, indexing='ij')
        X = torch.tensor(xx.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
        Y = torch.tensor(yy.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
        T = torch.full((xx.size, 1), float(tt), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            u_pred = model(X, Y, T).cpu().numpy().reshape(len(x_ref), len(y_ref))
        u_true = true_solution_np(xx, yy, tt)
        l2_per_time.append(float(np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)))
    return float(np.mean(l2_per_time)), l2_per_time


def train_one(seed, beta):
    set_seed(seed)
    model = PINN2D(K_FEAT, HIDDEN, LAYERS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(STEPS):
        opt.zero_grad()
        x = (torch.rand(NUM_DOMAIN, 1, device=DEVICE) * 2 * np.pi - np.pi)
        y = (torch.rand(NUM_DOMAIN, 1, device=DEVICE) * 2 * np.pi - np.pi)
        t = torch.rand(NUM_DOMAIN, 1, device=DEVICE)
        r_pde = pde_residual(model, x, y, t, NU, A, B)
        w = weight_radial(x, y, beta)
        loss_pde = torch.mean(w * r_pde ** 2)

        x_ic = (torch.rand(NUM_IC, 1, device=DEVICE) * 2 * np.pi - np.pi)
        y_ic = (torch.rand(NUM_IC, 1, device=DEVICE) * 2 * np.pi - np.pi)
        t_ic = torch.zeros(NUM_IC, 1, device=DEVICE)
        u_pred_ic = model(x_ic, y_ic, t_ic)
        u_true_ic = true_solution(x_ic, y_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_true_ic) ** 2)

        loss = loss_pde + LAMBDA_IC * loss_ic
        loss.backward()
        opt.step()

    l2_total, l2_per_time = evaluate(model)
    return {
        'seed': seed, 'beta': beta, 'L2_total': l2_total,
        'L2_per_time': l2_per_time, 'good_basin': l2_total < 0.1,
    }


def main():
    print("=" * 60)
    print("C5: 2D 线性对流扩散 β 扫描")
    print(f"  BETAS={BETAS}, seeds={SEEDS}")
    print(f"  K_FEAT={K_FEAT}, hidden={HIDDEN}, steps={STEPS}")
    print("=" * 60)

    all_results = []
    for beta in BETAS:
        beta_dir = OUT_DIR / f"beta_{beta}"
        beta_dir.mkdir(exist_ok=True)
        print(f"\n=== beta={beta} ===")
        beta_results = []
        for seed in SEEDS:
            rf = beta_dir / f"result_seed{seed}.json"
            if rf.exists():
                with open(rf) as f:
                    r = json.load(f)
                print(f"  seed={seed}: L2={r['L2_total']:.4e}（跳过）")
                beta_results.append(r)
                continue
            t0 = time.time()
            r = train_one(seed, beta)
            r['train_time_s'] = time.time() - t0
            with open(rf, 'w') as f:
                json.dump(r, f, indent=2)
            print(f"  seed={seed}: L2={r['L2_total']:.4e}, "
                  f"good={r['good_basin']}, time={r['train_time_s']:.0f}s")
            beta_results.append(r)
        all_results.append({'beta': beta, 'results': beta_results})

    # 汇总
    print("\n" + "=" * 60)
    print("汇总：")
    for item in all_results:
        beta = item['beta']
        l2s = [r['L2_total'] for r in item['results']]
        good = sum(1 for r in item['results'] if r['good_basin'])
        print(f"  beta={beta}: median={np.median(l2s):.4e}, "
              f"range=[{min(l2s):.4e}, {max(l2s):.4e}], "
              f"good={good}/{len(l2s)}")

    with open(OUT_DIR / "summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
