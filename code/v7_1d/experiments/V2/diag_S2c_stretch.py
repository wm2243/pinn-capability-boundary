# -*- coding: utf-8 -*-
"""
S2c 归因探针（方案 A：边界层坐标拉伸）——自包含，不改任何公共代码、不碰 Burgers 路径
================================================================================
要回答的问题：同厚度对流扩散边界层 ε=.01 在【硬约束 u=A+(1-x^2)N + 周期傅里叶特征】下
Phase0 零命中（K=16、σ 翻倍无效），根因是否为“贴边单边指数层”的表示-几何失配？
做法：引入计算坐标 ξ∈[-1,1] 与单调拉伸 x=X(ξ)，把物理厚度 cε 的右端层拉开到计算坐标 τ 比例，
网络在 ξ 中作画（傅里叶特征作用于 ξ），PDE 奇性吸收进雅可比：
    s=(1-ξ)/2,  d(s)=2 (e^{αs}-1)/(e^α-1),  x=1-d ;
    X'=α e^{αs}/(e^α-1)>0,  X''=-(α/2)X' ;
    u_x=u_ξ/X',   u_xx=(u_ξξ+(α/2)u_ξ)/(X')^2 ;   残差 -ε u_xx + b u_x = 0。
α 由“物理层 d≤cε 占计算坐标比例 τ”反解（只用 ε、c、τ，不用真解形状）；α=0 时 x≡ξ 恒等。
评估在 ξ 均匀网格按物理测度 dx=X'dξ 加权，等价物理 x 均匀 L2，与 Burgers/S2b 同口径。

配置（均 K=4 multistart、Phase0=2000、Adam1e-3、64x4、map256、β=1 均匀残差、硬边界精确）：
  A  eps=.01 拉伸(α解出) σ=3      主判据：能否像 Burgers 那样冒出 L2~.01-.1 的好盆地
  B  eps=.01 恒等 α=0   σ=3      内部对照：不拉伸应失败（复现 S2c/S2b）
  C  eps=.1  拉伸       σ=3      厚层回归：应成功（S2b 单阶段已到 .002）
  D  eps=.01 拉伸       σ=1.5    低带宽是否已足够（证明靠坐标而非高带宽）
产物：results/v7/V2/exp_V2_S2c_aligned_twostage/diag_stretch.csv
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
# 复现 S2c/R2 环境；可用环境变量覆盖做 smoke
SMOKE = bool(os.environ.get("V8_SMOKE"))
K = 1 if SMOKE else 4
P0 = 60 if SMOKE else 2000
N_PDE = 2000 if SMOKE else 10000
N_TEST = 2048
HID, LAY, MAP = 64, 4, 256
C_LAYER, TAU = 5.0, 0.4


# ---------------- 坐标拉伸 ----------------
def solve_alpha(eps, c=C_LAYER, tau=TAU):
    """s_L(α)=ln[1+(c eps/2)(e^α-1)]/α = tau 二分；α=0 极限为 c eps/2。"""
    target_phys = c * eps / 2.0
    if target_phys >= tau:
        return 0.0  # 层已足够厚，无需拉伸
    def sL(a):
        if a < 1e-8:
            return target_phys
        return np.log1p(target_phys * (np.exp(a) - 1.0)) / a
    lo, hi = 1e-8, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sL(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_map(alpha):
    a = float(alpha)
    if a <= 1e-8:  # 恒等
        def X_of(xi):
            return xi
        def Xp_of(xi):
            return torch.ones_like(xi)
        return X_of, Xp_of, 0.0
    ea = float(np.exp(a)); den = ea - 1.0
    def X_of(xi):
        s = (1.0 - xi) * 0.5
        d = 2.0 * (torch.exp(a * s) - 1.0) / den
        return 1.0 - d
    def Xp_of(xi):
        s = (1.0 - xi) * 0.5
        return a * torch.exp(a * s) / den
    return X_of, Xp_of, a


# ---------------- 拉伸坐标硬边界网络 ----------------
class StretchPINN(nn.Module):
    """输入计算坐标 ξ；x=X(ξ)；u=A(x)+(1-x^2) N(γ(ξ))，γ 为作用在 ξ 的随机傅里叶特征。
    端点 ξ=±1↔x=±1，lift_linear 精确给 u(-1)=0,u(1)=1。"""
    def __init__(self, sigma, alpha):
        super().__init__()
        self.a = float(alpha)
        self.X_of, self.Xp_of, _ = make_map(self.a)
        self.register_buffer("B0", torch.randn(1, MAP))
        self.sigma = float(sigma)
        layers = [nn.Linear(2 * MAP, HID), nn.Tanh()]
        for _ in range(LAY - 1):
            layers += [nn.Linear(HID, HID), nn.Tanh()]
        layers += [nn.Linear(HID, 1)]
        self.net = nn.Sequential(*layers)

    def phys(self, xi):
        return self.X_of(xi)

    def forward(self, xi):
        x = self.X_of(xi)
        proj = 2.0 * np.pi * xi @ (self.B0 * self.sigma)
        gamma = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        N = self.net(gamma)
        A = 0.5 * (1.0 + x)  # lift_linear: left0 right1
        return A + (1.0 - x ** 2) * N


def exact_np(x, eps, b=1.0):
    z = np.exp(np.clip(b * (x - 1.0) / eps, -700, 700)) - np.exp(-2.0 * b / eps)
    return z / (1.0 - np.exp(-2.0 * b / eps))


def residual(model, xi, eps, b=1.0):
    xi = xi.detach().clone().requires_grad_(True)
    u = model(xi)
    u_xi = torch.autograd.grad(u, xi, torch.ones_like(u), create_graph=True)[0]
    u_xixi = torch.autograd.grad(u_xi, xi, torch.ones_like(u_xi), create_graph=True)[0]
    Xp = model.Xp_of(xi)
    u_x = u_xi / Xp
    u_xx = (u_xixi + 0.5 * model.a * u_xi) / Xp ** 2
    return -eps * u_xx + b * u_x


# ---------------- 物理测度加权评估 ----------------
_xi_grid = torch.linspace(-1, 1, N_TEST, device=DEV).reshape(-1, 1)
_dxi = 2.0 / (N_TEST - 1)

def eval_phys(model, eps):
    model.eval()
    with torch.no_grad():
        xi = _xi_grid
        Xp = model.Xp_of(xi).cpu().numpy().flatten()
        x = model.X_of(xi).cpu().numpy().flatten()
        pred = model(xi).cpu().numpy().flatten()
    u = exact_np(x, eps)
    err = pred - u
    w = Xp * _dxi  # dx = X' dξ
    L2 = float(np.sqrt(np.sum(err ** 2 * w) / max(np.sum(u ** 2 * w), 1e-12)))
    m = x >= 1.0 - C_LAYER * eps
    L2_layer = float(np.sqrt(np.sum(err[m] ** 2 * w[m]) / max(np.sum(u[m] ** 2 * w[m]), 1e-12)))
    model.train()
    return L2, L2_layer


def train_one(eps, alpha, sigma, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = StretchPINN(sigma, alpha).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = (1e18, 1e18, -1); best_state = None
    for ep in range(P0):
        xi = (torch.rand(N_PDE, 1, device=DEV) * 2.0 - 1.0)
        R = residual(model, xi, eps)
        loss = torch.mean(R ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 200 == 0 or ep == P0 - 1:
            L2, L2l = eval_phys(model, eps)
            if L2 < best[0]:
                best = (L2, L2l, ep + 1); best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    L2f, L2lf = eval_phys(model, eps)
    return dict(best_L2=best[0], best_layer=best[1], best_ep=best[2],
                end_L2=L2f, end_layer=L2lf)


def main():
    configs = [
        ("A_stretch_eps01_sig3", 0.01, None, 3.0),
        ("B_identity_eps01_sig3", 0.01, 0.0, 3.0),
        ("C_stretch_eps010_sig3", 0.10, None, 3.0),
        ("D_stretch_eps01_sig1p5", 0.01, None, 1.5),
    ]
    rows = []
    for name, eps, alpha_arg, sig in configs:
        alpha = solve_alpha(eps) if alpha_arg is None else float(alpha_arg)
        # 报告拉伸后层内雅可比量级，核对 X'(右端)~ε
        _, Xp_of, _ = make_map(alpha)
        xp_right = float(Xp_of(torch.tensor([[1.0]], device=DEV)).item())
        print(f"\n==== {name}: eps={eps} alpha={alpha:.4f} X'(x=1)={xp_right:.4e} sigma={sig} ====")
        cands = []
        for k in range(K):
            r = train_one(eps, alpha, sig, seed=k)
            cands.append(r["end_L2"])
            print(f"  start{k}: bestL2={r['best_L2']:.3e}(层{r['best_layer']:.2e}@{r['best_ep']}) "
                  f"endL2={r['end_L2']:.3e}(层{r['end_layer']:.2e})")
            rows.append(dict(config=name, eps=eps, alpha=alpha, sigma=sig, Xp_right=xp_right,
                             seed=k, **r))
        a = np.sort(np.asarray(cands))
        print(f"  -> K={K} endL2: min={a.min():.3e} med={np.median(a):.3e}  "
              f"#(<.06)={int((a<.06).sum())} #(<.1)={int((a<.1).sum())} #(<.3)={int((a<.3).sum())}")
    p = OUT / "diag_stretch.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
