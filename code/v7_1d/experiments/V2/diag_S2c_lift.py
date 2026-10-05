# -*- coding: utf-8 -*-
"""
S2c 归因探针（方案 B：边界层型提升）——自包含，α=0 恒等物理坐标
================================================================================
坐标拉伸探针结论：ε=.01 拉伸 4/4 失败（厚层 ε=.1 成功证明实现无误），手算 N* 显示拉伸只把
“N 在墙角发散”有限化为 O(1/ε)≈50 的大幅值抵消任务（线性提升 A=½(1+x) 层外非零所致）。
本探针只改提升函数 A(x)，湮灭因子仍 (1-x²)、物理坐标、周期傅里叶，定位“提升 ansatz 是否才是钥匙”：
  exp      A=e^{(x-1)/ε}            与真解差 O(e^{-2/ε})≈0 —— 作弊上界/管线验证（预期 L2≈0）
  logistic A=归一化 sigmoid 阶跃     只用层位置(1-2ε)与宽度 ε、形状非指数，网络仍须学指数修正
若 logistic 也能高精度成功 -> 提升只需“几何对”（层外≈0、承载阶跃），不泄露精确形状，方法可辩护；
若只有 exp 成功、logistic 失败       -> 必须编码精确指数，该线性无源算例天生尴尬，应走能力边界/换算例。
配置：ε=.01，K=4，Phase0=2000，Adam1e-3，64x4，map256，β=1 均匀残差；σ=15 主、σ=3 对照。
产物：results/v7/V2/exp_V2_S2c_aligned_twostage/diag_lift.csv
"""
import os, csv, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve(); _V7 = _HERE.parents[2]
OUT = _V7.parent / "results" / "v7" / "V2" / "exp_V2_S2c_aligned_twostage"
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = bool(os.environ.get("V8_SMOKE"))
K = 1 if SMOKE else 4
P0 = 60 if SMOKE else 2000
N_PDE = 2000 if SMOKE else 10000
N_TEST = 2048
HID, LAY, MAP = 64, 4, 256
C_LAYER = 5.0


class LiftPINN(nn.Module):
    """恒等坐标 x；u=A_lift(x)+(1-x²)N(γ(x))。lift ∈ linear|exp|logistic。"""
    def __init__(self, sigma, eps, lift="linear"):
        super().__init__()
        self.eps = float(eps); self.lift = lift
        self.register_buffer("B0", torch.randn(1, MAP)); self.sigma = float(sigma)
        layers = [nn.Linear(2 * MAP, HID), nn.Tanh()]
        for _ in range(LAY - 1):
            layers += [nn.Linear(HID, HID), nn.Tanh()]
        layers += [nn.Linear(HID, 1)]
        self.net = nn.Sequential(*layers)
        # logistic 阶跃归一化常数：中心 1-2ε、宽 ε
        self.x0 = 1.0 - 2.0 * self.eps; self.ell = self.eps
        self.s1 = float(torch.sigmoid(torch.tensor((1.0 - self.x0) / self.ell)).item())

    def lift_fn(self, x):
        if self.lift == "linear":
            return 0.5 * (1.0 + x)
        if self.lift == "exp":
            return torch.exp(torch.clamp((x - 1.0) / self.eps, -700, 700))
        if self.lift == "logistic":
            return torch.sigmoid((x - self.x0) / self.ell) / self.s1  # A(1)=1, 层外≈0
        raise ValueError(self.lift)

    def forward(self, x):
        proj = 2.0 * np.pi * x @ (self.B0 * self.sigma)
        gamma = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        N = self.net(gamma)
        return self.lift_fn(x) + (1.0 - x ** 2) * N


def exact_np(x, eps, b=1.0):
    z = np.exp(np.clip(b * (x - 1.0) / eps, -700, 700)) - np.exp(-2.0 * b / eps)
    return z / (1.0 - np.exp(-2.0 * b / eps))


def residual(model, x, eps, b=1.0):
    x = x.detach().clone().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return -eps * u_xx + b * u_x


_xg = torch.linspace(-1, 1, N_TEST, device=DEV).reshape(-1, 1)
_dx = 2.0 / (N_TEST - 1)
def eval_phys(model, eps):
    model.eval()
    with torch.no_grad():
        pred = model(_xg).cpu().numpy().flatten()
        x = _xg.cpu().numpy().flatten()
    u = exact_np(x, eps); err = pred - u
    L2 = float(np.sqrt(np.sum(err ** 2) / max(np.sum(u ** 2), 1e-12)))
    m = x >= 1.0 - C_LAYER * eps
    L2l = float(np.sqrt(np.sum(err[m] ** 2) / max(np.sum(u[m] ** 2), 1e-12)))
    model.train(); return L2, L2l


def train_one(eps, lift, sigma, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LiftPINN(sigma, eps, lift).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = (1e18, 1e18, -1); bs = None
    for ep in range(P0):
        x = torch.rand(N_PDE, 1, device=DEV) * 2.0 - 1.0
        R = residual(model, x, eps)
        loss = torch.mean(R ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 200 == 0 or ep == P0 - 1:
            L2, L2l = eval_phys(model, eps)
            if L2 < best[0]: best = (L2, L2l, ep + 1); bs = copy.deepcopy(model.state_dict())
    model.load_state_dict(bs)
    L2f, L2lf = eval_phys(model, eps)
    return dict(best_L2=best[0], best_layer=best[1], best_ep=best[2], end_L2=L2f, end_layer=L2lf)


def main():
    configs = [
        ("E_exp_eps01_sig15", "exp", 15.0),
        ("F_logistic_eps01_sig15", "logistic", 15.0),
        ("G_logistic_eps01_sig3", "logistic", 3.0),
    ]
    rows = []
    for name, lift, sig in configs:
        eps = 0.01
        print(f"\n==== {name}: lift={lift} sigma={sig} ====")
        cands = []
        for k in range(K):
            r = train_one(eps, lift, sig, seed=k)
            cands.append(r["end_L2"])
            print(f"  start{k}: bestL2={r['best_L2']:.3e}(层{r['best_layer']:.2e}@{r['best_ep']}) "
                  f"endL2={r['end_L2']:.3e}(层{r['end_layer']:.2e})")
            rows.append(dict(config=name, eps=eps, lift=lift, sigma=sig, seed=k, **r))
        a = np.sort(np.asarray(cands))
        print(f"  -> K={K} endL2: min={a.min():.3e} med={np.median(a):.3e}  "
              f"#(<.06)={int((a<.06).sum())} #(<.1)={int((a<.1).sum())} #(<.3)={int((a<.3).sum())}")
    p = OUT / "diag_lift.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
