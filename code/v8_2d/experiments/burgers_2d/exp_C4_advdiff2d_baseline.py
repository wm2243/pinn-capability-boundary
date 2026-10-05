# -*- coding: utf-8 -*-
"""
C4: 2D 线性对流扩散方程（跨方程验证 baseline）
================================================
方程：u_t + a u_x + b u_y = nu (u_xx + u_yy)
域：x,y in [-pi, pi]（周期），t in [0, 1]
初始条件：u(x,y,0) = sin(x) + 0.5 cos(2y) + 0.3 sin(x+y)
解析解：u(x,y,t) = exp(-nu*t) sin(x-a*t) + 0.5 exp(-4nu*t) cos(2(y-b*t))
                    + 0.3 exp(-2nu*t) sin(x+y-(a+b)*t)

用途：第一篇论文的 2D 跨方程验证（替代 2D Burgers，后者是 PINN 经典失败模式）
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import time

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "C4_advdiff2d_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 方程参数
NU = 0.01
A = 1.0   # x 方向对流速度
B = 0.5   # y 方向对流速度

# 训练参数
SEEDS = list(range(11))
K_FEAT = 32
HIDDEN = 50
LAYERS = 4
LR = 1e-3
STEPS = 50000
NUM_DOMAIN = 4000
NUM_IC = 200
LAMBDA_IC = 100.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- smoke 开关：V8_SMOKE=1 时缩到 1 种子/20 步，输出独立 _smoke 目录（不污染正式结果）----
import os as _os
SMOKE = bool(_os.environ.get("V8_SMOKE"))
if SMOKE:
    SEEDS, STEPS = [0], 20
    OUT_DIR = HERE / "results" / "C4_advdiff2d_baseline_smoke"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def true_solution(x, y, t):
    """解析解"""
    return (torch.exp(-NU * t) * torch.sin(x - A * t)
            + 0.5 * torch.exp(-4 * NU * t) * torch.cos(2 * (y - B * t))
            + 0.3 * torch.exp(-2 * NU * t) * torch.sin(x + y - (A + B) * t))


def true_solution_np(x, y, t):
    """numpy 版本解析解（用于评估）"""
    return (np.exp(-NU * t) * np.sin(x - A * t)
            + 0.5 * np.exp(-4 * NU * t) * np.cos(2 * (y - B * t))
            + 0.3 * np.exp(-2 * NU * t) * np.sin(x + y - (A + B) * t))


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
    """r = u_t + a u_x + b u_y - nu(u_xx + u_yy)"""
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
        l2_t = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        l2_per_time.append(float(l2_t))
    return float(np.mean(l2_per_time)), l2_per_time, t_ref


def train_one(seed):
    set_seed(seed)
    model = PINN2D(K_FEAT, HIDDEN, LAYERS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for step in range(STEPS):
        opt.zero_grad()
        x = (torch.rand(NUM_DOMAIN, 1, device=DEVICE) * 2 * np.pi - np.pi)
        y = (torch.rand(NUM_DOMAIN, 1, device=DEVICE) * 2 * np.pi - np.pi)
        t = torch.rand(NUM_DOMAIN, 1, device=DEVICE)
        r_pde = pde_residual(model, x, y, t, NU, A, B)
        loss_pde = torch.mean(r_pde ** 2)

        x_ic = (torch.rand(NUM_IC, 1, device=DEVICE) * 2 * np.pi - np.pi)
        y_ic = (torch.rand(NUM_IC, 1, device=DEVICE) * 2 * np.pi - np.pi)
        t_ic = torch.zeros(NUM_IC, 1, device=DEVICE)
        u_pred_ic = model(x_ic, y_ic, t_ic)
        u_true_ic = true_solution(x_ic, y_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_true_ic) ** 2)

        loss = loss_pde + LAMBDA_IC * loss_ic
        loss.backward()
        opt.step()

        if step % 10000 == 0 or step == STEPS - 1:
            print(f"  seed={seed} step={step}: loss={loss.item():.4e} "
                  f"(pde={loss_pde.item():.4e}, ic={loss_ic.item():.4e})")

    train_time = time.time() - t0
    l2_total, l2_per_time, t_ref = evaluate(model)
    print(f"  seed={seed}: L2_total={l2_total:.4e}, "
          f"L2(t)={[f'{v:.3e}' for v in l2_per_time]}, time={train_time:.1f}s")

    result = {
        'seed': seed, 'L2_total': l2_total, 'L2_per_time': l2_per_time,
        'train_time_s': train_time, 'steps': STEPS,
        'good_basin': l2_total < 0.1,
    }
    with open(OUT_DIR / f"result_seed{seed}.json", 'w') as f:
        json.dump(result, f, indent=2)
    return result


def main():
    print("=" * 60)
    print("C4: 2D 线性对流扩散（baseline，均匀权重）")
    print(f"  NU={NU}, A={A}, B={B}, K_FEAT={K_FEAT}, hidden={HIDDEN}")
    print(f"  lr={LR}, steps={STEPS}, seeds={SEEDS}")
    print("=" * 60)
    results = []
    for seed in SEEDS:
        rf = OUT_DIR / f"result_seed{seed}.json"
        if rf.exists():
            with open(rf) as f:
                r = json.load(f)
            print(f"  seed={seed}: 已完成 L2={r['L2_total']:.4e}（跳过）")
            results.append(r)
            continue
        results.append(train_one(seed))

    l2s = [r['L2_total'] for r in results]
    good = sum(1 for r in results if r['good_basin'])
    print(f"\n汇总：L2 median={np.median(l2s):.4e}, range=[{min(l2s):.4e}, {max(l2s):.4e}]")
    print(f"好盆地 (L2<0.1): {good}/{len(results)}")


if __name__ == "__main__":
    main()
