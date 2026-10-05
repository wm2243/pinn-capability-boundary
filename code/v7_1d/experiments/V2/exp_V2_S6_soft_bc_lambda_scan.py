# -*- coding: utf-8 -*-
"""
V2 实验 S6：软边界 λ 扫描——验证分岔现象（坍缩→安全→过拟合）
================================================================================
目的（第二篇 S 层论文的第一个实验）：
  软边界损失 L = L_PDE + λ * L_boundary，扫描 λ 观察：
    - λ 太小：边界不约束 → 幅值坍缩（零模方向无曲率）
    - λ 适中：安全区 → 好盆地存在，L2 低
    - λ 太大：边界过拟合 → 网络只学幅值不学形状，L2 回升
  这是定理 1（强度分岔）的实验验证。

方程：稳态 Burgers u*u_x - ν*u_xx = 0, u(-1)=1, u(1)=-1
真解：u* = -tanh(x/(2ν))
模型：软边界 MLP（不做硬边界变换，直接输出）
损失：MSE_PDE + λ * MSE_boundary

运行：
  python exp_V2_S6_soft_bc_lambda_scan.py --smoke      # 冒烟：3 λ × 3 seed，2000步
  python exp_V2_S6_soft_bc_lambda_scan.py              # 全量：7 λ × 11 seed，8000步
================================================================================
"""
import os, sys, csv, argparse, copy, random
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_V7 = _HERE.parents[2]
_V1EXP = _V7 / "experiments" / "v1"
for _p in (str(_V7), str(_V1EXP), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import v8_matrix_common as mc
from physics.burgers_pde import SteadyBurgersPDE
from samplers.pinn_sampler import PINNSampler
from utils.metrics import PINNMetrics

OUT = str(_V7.parent / "results" / "v7" / "V2" / "exp_V2_S6_soft_bc_lambda_scan")
os.makedirs(OUT, exist_ok=True)
DEV = mc.DEVICE

# ============================== 软边界模型 ==============================
class SoftBCPINN(torch.nn.Module):
    """软边界 PINN：直接输出网络值，不做硬边界变换"""
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        layers = [torch.nn.Linear(1, hidden_dim), torch.nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(hidden_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================== 软边界损失 ==============================
class SoftBCLoss(torch.nn.Module):
    """软边界损失：L_PDE + λ * L_boundary"""
    def __init__(self, config, device):
        super().__init__()
        self.lambda_bc = float(config.get('lambda_bc', 1.0))
        self.device = device
        self.x_left = torch.tensor([[-1.0]], device=device)
        self.x_right = torch.tensor([[1.0]], device=device)
        self.u_left = 1.0
        self.u_right = -1.0

    def set_lambda(self, lam):
        self.lambda_bc = float(lam)

    def forward(self, pde_residual, x_pde=None, model=None):
        loss_pde = torch.mean(pde_residual ** 2)
        u_l = model(self.x_left)
        u_r = model(self.x_right)
        loss_bc = 0.5 * ((u_l - self.u_left) ** 2 + (u_r - self.u_right) ** 2)
        total = loss_pde + self.lambda_bc * loss_bc
        return total, {'pde': loss_pde.item(), 'bc': loss_bc.item(),
                       'u_left': u_l.item(), 'u_right': u_r.item()}


# ============================== 配置 ==============================
SEEDS_EXT = [42, 123, 2024, 7, 99, 314, 271, 828, 1618, 2025, 555]
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
NU = 0.01
HIDDEN, NLAYERS = 64, 4
N_PDE = 10000
LR = 1e-3
EPOCHS_FULL = 8000
EPOCHS_SMOKE = 2000
EVAL_FREQ = 500


def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_test(nu, n=2048):
    x = torch.linspace(-1, 1, n, device=DEV).view(-1, 1)
    u = -np.tanh(x.cpu().numpy() / (2 * nu)).flatten()
    return x, u


def train_one(lam, seed, epochs, x_test, u_test):
    set_seed(seed)
    config = dict(
        domain_x=(-1, 1), hidden_dim=HIDDEN, num_layers=NLAYERS,
        nu=NU, n_pde=N_PDE, lambda_bc=lam,
        grad_clip=1.0, shock_band_c=3.0,
    )
    model = SoftBCPINN(config).to(DEV)
    pde = SteadyBurgersPDE(config)
    crit = SoftBCLoss(config, DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    samp = PINNSampler(config["domain_x"], DEV, use_adaptive=False, buffer_size=5000)

    best_l2 = float('inf')
    best_state = None

    for epoch in range(epochs):
        data = samp.sample(N_PDE, mode='random')
        x_pde = data['x_pde']
        pde_res = pde.compute_residual(model, x_pde)
        loss, info = crit(pde_res, x_pde=x_pde, model=model)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % EVAL_FREQ == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                u_pred = model(x_test).detach().cpu().numpy().flatten()
            m = PINNMetrics.compute_fast(u_pred=u_pred, u_true=u_test)
            l2 = m['L2_Error']
            model.train()
            if l2 < best_l2:
                best_l2 = l2
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test).detach().cpu().numpy().flatten()
    m = PINNMetrics.compute_fast(u_pred=u_pred, u_true=u_test)

    with torch.no_grad():
        u_l = model(torch.tensor([[-1.0]], device=DEV)).item()
        u_r = model(torch.tensor([[1.0]], device=DEV)).item()
    bc_err = abs(u_l - 1.0) + abs(u_r + 1.0)

    u_abs_max = float(np.max(np.abs(u_pred)))
    u_center = float(u_pred[len(u_pred) // 2])

    return {
        'lambda': lam, 'seed': seed, 'epochs': epochs,
        'L2_final': m['L2_Error'], 'L2_best': best_l2,
        'Linf_final': m['L_inf_Error'],
        'bc_error': bc_err, 'u_left': u_l, 'u_right': u_r,
        'u_abs_max': u_abs_max, 'u_center': u_center,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--lambdas', type=float, nargs='+', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    args = parser.parse_args()

    if args.smoke:
        lambdas = [0.1, 10.0, 1000.0]
        seeds = [42, 123, 2024]
        epochs = EPOCHS_SMOKE
    else:
        lambdas = args.lambdas if args.lambdas else LAMBDAS
        seeds = args.seeds if args.seeds else SEEDS_EXT
        epochs = EPOCHS_FULL

    x_test, u_test = build_test(NU)
    csv_path = os.path.join(OUT, 'results.csv')
    file_exists = os.path.exists(csv_path)

    fields = ['lambda', 'seed', 'epochs', 'L2_final', 'L2_best', 'Linf_final',
              'bc_error', 'u_left', 'u_right', 'u_abs_max', 'u_center']

    done = set()
    if file_exists:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((float(row['lambda']), int(row['seed'])))

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        for lam in lambdas:
            for seed in seeds:
                if (lam, seed) in done:
                    print(f"[skip] λ={lam}, seed={seed}")
                    continue
                print(f"\n=== λ={lam}, seed={seed} ===")
                res = train_one(lam, seed, epochs, x_test, u_test)
                row = {k: res[k] for k in fields}
                writer.writerow(row)
                f.flush()
                print(f"  L2_final={res['L2_final']:.4e}, L2_best={res['L2_best']:.4e}, "
                      f"bc_err={res['bc_error']:.4e}, |u|max={res['u_abs_max']:.4f}, "
                      f"u(0)={res['u_center']:.4f}")

    print("\n" + "=" * 70)
    print("聚合统计（按 λ）")
    print("=" * 70)
    from statistics import median, mean, stdev
    rows_by_lam = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            lam = float(r['lambda'])
            rows_by_lam.setdefault(lam, []).append(r)
    summary_fields = ['lambda', 'L2_median', 'L2_mean', 'L2_std', 'bc_median',
                      'uabsmax_median', 'n']
    summary_path = os.path.join(OUT, 'summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        print(f"{'lambda':>10} {'L2_med':>10} {'L2_mean':>10} {'L2_std':>10} "
              f"{'bc_med':>10} {'|u|max_med':>10} {'n':>4}")
        print("-" * 70)
        for lam in sorted(rows_by_lam.keys()):
            rows = rows_by_lam[lam]
            l2s = [float(r['L2_final']) for r in rows]
            bcs = [float(r['bc_error']) for r in rows]
            umaxs = [float(r['u_abs_max']) for r in rows]
            row = {
                'lambda': lam,
                'L2_median': median(l2s),
                'L2_mean': mean(l2s),
                'L2_std': stdev(l2s) if len(l2s) > 1 else 0.0,
                'bc_median': median(bcs),
                'uabsmax_median': median(umaxs),
                'n': len(rows),
            }
            w.writerow(row)
            print(f"{lam:>10.2e} {row['L2_median']:>10.4e} {row['L2_mean']:>10.4e} "
                  f"{row['L2_std']:>10.4e} {row['bc_median']:>10.4e} "
                  f"{row['uabsmax_median']:>10.4f} {row['n']:>4d}")
    print(f"\n结果保存在: {OUT}")


if __name__ == '__main__':
    main()
