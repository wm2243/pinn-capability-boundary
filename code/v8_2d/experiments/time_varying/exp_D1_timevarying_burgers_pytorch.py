# -*- coding: utf-8 -*-
"""
D1: 1D 时变 Burgers（原生 PyTorch 版，与 DeepXDE 版对齐）
==========================================================
目的：统一实现框架，消除 DeepXDE vs 原生 PyTorch 的混杂因子
配置与 caseA_standard_burgers.py (DeepXDE) 完全一致：
  - 方程：u_t + u u_x = nu u_xx, nu = 0.01/pi
  - 域：x in [-1,1], t in [0, 0.99]
  - IC: u(x,0) = -sin(pi x)
  - BC: u(±1,t) = 0（软边界，作为损失项）
  - 网络：纯 MLP [2, 50, 50, 50, 50, 1]，tanh，Glorot uniform
  - 优化器：Adam lr=1e-3，50000 步
  - 采样：num_domain=4000, num_boundary=100, num_initial=200
  - 权重：w(x) = 1 + (beta-1)(1-|x|)，中心加权

实验：A2 β 扫描，β ∈ {1, 3, 8, 15}，各 11 seed
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import json
import time

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "D1_timevarying_burgers_pytorch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REF_MAT = HERE / "true_solution" / "burgers_shock.mat"

# 方程参数
NU = 0.01 / np.pi

# 训练参数（与 DeepXDE 版对齐）
SEEDS = list(range(11))
BETAS = [1.0, 3.0, 8.0, 15.0]
HIDDEN = 50
LAYERS = 4
LR = 1e-3
STEPS = 50000
NUM_DOMAIN = 4000
NUM_BOUNDARY = 100
NUM_INITIAL = 200
LAMBDA_BC = 1.0   # DeepXDE 默认边界损失权重=1
LAMBDA_IC = 1.0   # DeepXDE 默认初始损失权重=1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- smoke 开关：V8_SMOKE=1 时缩到 1 种子/单 β/20 步，输出独立 _smoke 目录（不污染正式结果）----
import os as _os
SMOKE = bool(_os.environ.get("V8_SMOKE"))
if SMOKE:
    SEEDS, BETAS, STEPS = [0], [1.0], 20
    OUT_DIR = HERE / "results" / "D1_timevarying_burgers_pytorch_smoke"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def glorot_uniform(m):
    """Glorot uniform 初始化（与 DeepXDE 一致）"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def load_reference():
    """加载 Raissi 标准真解"""
    d = loadmat(str(REF_MAT))
    x = d['x'].flatten()
    t = d['t'].flatten()
    u = d['usol']  # shape (len(x), len(t))
    return x, t, u


class MLP(nn.Module):
    """纯 MLP，与 DeepXDE FNN 一致"""
    def __init__(self, layers=[2, 50, 50, 50, 50, 1]):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)
        self.apply(glorot_uniform)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


def pde_residual(model, x, t, nu):
    """r = u_t + u u_x - nu u_xx"""
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    return (u_t + u * u_x - nu * u_xx).squeeze(-1)


def weight_spatial(x, beta):
    """空间权重：w(x) = 1 + (beta-1)(1-|x|)，中心 x=0 权重=beta，边界权重=1"""
    return 1.0 + (beta - 1.0) * (1.0 - torch.abs(x))


def evaluate(model, x_ref, t_ref, u_ref):
    """在真解网格上计算 L2 相对误差"""
    model.eval()
    xx, tt = np.meshgrid(x_ref, t_ref, indexing='ij')
    X = torch.tensor(xx.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    T = torch.tensor(tt.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_pred = model(X, T).cpu().numpy().reshape(len(x_ref), len(t_ref))
    l2 = float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))
    return l2, u_pred


def mechanism_A_analysis(model, beta, x_ref, t_ref, u_ref):
    """
    机制 A 检测：冻结网络，分析加权对梯度分布和误差分布的影响。
    仅对 beta in MECHANISM_A_BETAS 的组执行。
    """
    model.eval()
    results = {}

    # === 1. 分区域 L2 ===
    xx, tt = np.meshgrid(x_ref, t_ref, indexing='ij')
    X = torch.tensor(xx.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    T = torch.tensor(tt.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_pred = model(X, T).cpu().numpy().reshape(len(x_ref), len(t_ref))
    error = u_pred - u_ref

    center_mask = np.abs(xx) < 0.3   # 中心区（激波位置）
    boundary_mask = ~center_mask       # 边界区
    l2_center = float(np.linalg.norm(error[center_mask]) / np.linalg.norm(u_ref[center_mask]))
    l2_boundary = float(np.linalg.norm(error[boundary_mask]) / np.linalg.norm(u_ref[boundary_mask]))
    results['l2_center'] = l2_center
    results['l2_boundary'] = l2_boundary
    results['l2_ratio_center_over_boundary'] = l2_center / l2_boundary if l2_boundary > 0 else float('inf')

    # === 2. 梯度分布检测（冻结网络，固定配点） ===
    N_probe = 2048
    torch.manual_seed(42)  # 固定配点，确保不同 beta 可比
    x_p = (torch.rand(N_probe, 1, device=DEVICE) * 2 - 1)
    t_p = torch.rand(N_probe, 1, device=DEVICE) * 0.99
    x_p.requires_grad_(True)
    t_p.requires_grad_(True)

    u = model(x_p, t_p)
    u_x = torch.autograd.grad(u, x_p, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t_p, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_p, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    r = (u_t + u * u_x - NU * u_xx).squeeze(-1)  # (N,)

    # Jacobian: dr/dθ (N × P)
    params = list(model.parameters())
    P = sum(p.numel() for p in params)
    J = torch.zeros(N_probe, P, device=DEVICE)
    for i in range(N_probe):
        grads = torch.autograd.grad(r[i], params, retain_graph=(i < N_probe - 1),
                                    create_graph=False)
        J[i, :] = torch.cat([g.flatten() for g in grads])

    # 均匀梯度 g = J^T r
    g = J.T @ r.detach()
    # 加权梯度 g_W = J^T (w * r)
    w = weight_spatial(x_p.detach(), beta).squeeze(-1)
    g_W = J.T @ (w * r.detach())

    # 范数比
    norm_g = float(torch.norm(g).item())
    norm_gW = float(torch.norm(g_W).item())
    results['grad_norm_ratio'] = norm_gW / norm_g if norm_g > 0 else 0.0

    # Cosine similarity
    cos_sim = float(torch.dot(g, g_W).item() / (norm_g * norm_gW)) if norm_g > 0 and norm_gW > 0 else 0.0
    results['grad_cosine_similarity'] = cos_sim

    # 分区域梯度范数（逐点梯度范数 ‖J_i‖）
    row_norms = torch.norm(J, dim=1).detach().cpu().numpy()  # (N,)
    x_np = x_p.detach().cpu().numpy().flatten()
    center_idx = np.abs(x_np) < 0.3
    boundary_idx = ~center_idx
    results['grad_row_norm_center_mean'] = float(row_norms[center_idx].mean())
    results['grad_row_norm_boundary_mean'] = float(row_norms[boundary_idx].mean())
    results['grad_row_norm_ratio'] = (float(row_norms[center_idx].mean()) /
                                       float(row_norms[boundary_idx].mean()))

    # 加权后逐点有效梯度范数 ‖w_i * J_i‖
    weighted_row_norms = (w.detach().cpu().numpy() * row_norms)
    results['weighted_grad_row_norm_center_mean'] = float(weighted_row_norms[center_idx].mean())
    results['weighted_grad_row_norm_boundary_mean'] = float(weighted_row_norms[boundary_idx].mean())
    results['weighted_grad_row_norm_ratio'] = (float(weighted_row_norms[center_idx].mean()) /
                                                float(weighted_row_norms[boundary_idx].mean()))

    # === 3. 残差分布 ===
    r_np = r.detach().cpu().numpy()
    results['residual_center_mean_abs'] = float(np.abs(r_np[center_idx]).mean())
    results['residual_boundary_mean_abs'] = float(np.abs(r_np[boundary_idx]).mean())
    results['residual_center_std'] = float(r_np[center_idx].std())
    results['residual_boundary_std'] = float(r_np[boundary_idx].std())

    return results


# 只对这些 beta 做机制 A 检测（对比最鲜明）
MECHANISM_A_BETAS = [1.0, 8.0]


def train_one(seed, beta, x_ref, t_ref, u_ref):
    set_seed(seed)
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(STEPS):
        opt.zero_grad()

        # PDE 配点（每步重新采样，与 C4/C5 及稳态 Burgers 一致）
        x_d = (torch.rand(NUM_DOMAIN, 1, device=DEVICE) * 2 - 1)
        t_d = torch.rand(NUM_DOMAIN, 1, device=DEVICE) * 0.99
        r_pde = pde_residual(model, x_d, t_d, NU)
        w = weight_spatial(x_d, beta)
        loss_pde = torch.mean(w * r_pde ** 2)

        # 边界条件（每步重新采样）
        t_bc = torch.rand(NUM_BOUNDARY, 1, device=DEVICE) * 0.99
        x_bc_left = torch.full((NUM_BOUNDARY // 2, 1), -1.0, device=DEVICE)
        x_bc_right = torch.full((NUM_BOUNDARY - NUM_BOUNDARY // 2, 1), 1.0, device=DEVICE)
        x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
        u_bc = model(x_bc, t_bc)
        loss_bc = torch.mean(u_bc ** 2)

        # 初始条件（每步重新采样）
        x_ic = (torch.rand(NUM_INITIAL, 1, device=DEVICE) * 2 - 1)
        t_ic = torch.zeros(NUM_INITIAL, 1, device=DEVICE)
        u_ic = model(x_ic, t_ic)
        u_ic_true = -torch.sin(np.pi * x_ic)
        loss_ic = torch.mean((u_ic - u_ic_true) ** 2)

        loss = loss_pde + LAMBDA_BC * loss_bc + LAMBDA_IC * loss_ic
        loss.backward()
        opt.step()

        if step % 10000 == 0 or step == STEPS - 1:
            print(f"  seed={seed} beta={beta} step={step}: "
                  f"loss={loss.item():.4e} (pde={loss_pde.item():.4e}, "
                  f"bc={loss_bc.item():.4e}, ic={loss_ic.item():.4e})")

    l2, u_pred = evaluate(model, x_ref, t_ref, u_ref)
    result = {
        'seed': seed, 'beta': beta, 'L2_total': l2,
        'good_basin': l2 < 0.1,
        'u_pred_min': float(u_pred.min()), 'u_pred_max': float(u_pred.max()),
    }

    # 机制 A 检测（仅对指定 beta）
    if beta in MECHANISM_A_BETAS:
        print(f"  [机制A检测] beta={beta} seed={seed}...")
        mech = mechanism_A_analysis(model, beta, x_ref, t_ref, u_ref)
        result['mechanism_A'] = mech
        print(f"    L2_center={mech['l2_center']:.4e}, L2_boundary={mech['l2_boundary']:.4e}")
        print(f"    grad_norm_ratio={mech['grad_norm_ratio']:.4f}, cos_sim={mech['grad_cosine_similarity']:.4f}")
        print(f"    weighted_grad_ratio={mech['weighted_grad_row_norm_ratio']:.4f}")

    return result


def main():
    print("=" * 60)
    print("D1: 1D 时变 Burgers（原生 PyTorch 版，与 DeepXDE 对齐）")
    print(f"  NU={NU:.6f}, BETAS={BETAS}, seeds={SEEDS}")
    print(f"  hidden={HIDDEN}, layers={LAYERS}, lr={LR}, steps={STEPS}")
    print("=" * 60)

    x_ref, t_ref, u_ref = load_reference()
    print(f"真解网格: x={len(x_ref)}, t={len(t_ref)}")

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
            r = train_one(seed, beta, x_ref, t_ref, u_ref)
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
