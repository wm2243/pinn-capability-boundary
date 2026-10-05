# -*- coding: utf-8 -*-
"""
D1-rational: 1D 时变 Burgers，有理残差增权族 beta 扫描（P-45 方案 b）
====================================================================
目的：让论文 §7.3 声称的协议成为现实——
  - 方程：u_t + u u_x = nu u_xx, nu = 0.01/pi
  - 域：x in [-1,1], t in [0, 0.99]
  - IC: u(x,0) = -sin(pi x), BC: u(±1,t) = 0（软边界，损失项权重 1）
  - 网络：纯 MLP [2, 50, 50, 50, 50, 1]，tanh，Glorot uniform
  - 优化器：Adam lr=1e-3，50000 步，每步重采样 4000 PDE + 100 BC + 200 IC
  - 权重（与稳态 E1/R2 同族）：w(t) = (1 + beta*t) / (1 + t)，t = |r|/median|r|（批内中位数），
    adaptive 模式（w 随当前 r 现算，含 autograd），raw 口径（w∈[1,β]），仅作用于 PDE 残差项
  - 种子：S0 = {0,5,10,...,50}（与稳态同一基础种子集）
  - 评估：完整 (x,t) 时空网格相对 L2（L2_total），另记录末时刻切片 L2（L2_final）
  - 机制 A 探针：beta∈{1,8} 时冻结训练后网络，同配点比较 g 与 g_W
      （方向余弦、总范数比、中心/边界行范数比（未加权/加权））

断点续跑：每 run 自动 checkpoint（模型+优化器+步数+RNG），
  单次进程墙钟超过 D1_WALLCAP 秒（默认 280）时落盘退出，下次启动自动接续。
  结果齐全（result_seed*.json × 11 × 4 beta）后写 summary.json。

冒烟：D1_SMOKE=1 时 1 种子 / beta=1 / 200 步，输出独立 _smoke 目录。
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "exp_D1_rational_beta_scan"
REF_MAT = HERE / "true_solution" / "burgers_shock.mat"

NU = 0.01 / np.pi
SEEDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]   # S0，与稳态同一基础种子集
BETAS = [1.0, 3.0, 8.0, 15.0]
# 环境变量可缩小范围做试点：D1_SEEDS="0,5,10" D1_BETAS="1,8"
if os.environ.get("D1_SEEDS"):
    SEEDS = [int(s) for s in os.environ["D1_SEEDS"].split(",")]
if os.environ.get("D1_BETAS"):
    BETAS = [float(b) for b in os.environ["D1_BETAS"].split(",")]
LR = 1e-3
STEPS = 50000
NUM_DOMAIN = 4000
NUM_BOUNDARY = 100
NUM_INITIAL = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MECH_BETAS = [1.0, 8.0]
# 墙钟上限（秒）：<=0 不限时（默认，适合 Spyder/无人值守整跑）；
# 分批接续跑时由调用方显式设 D1_WALLCAP=280。
WALLCAP = float(os.environ.get("D1_WALLCAP", "0"))

SMOKE = bool(os.environ.get("D1_SMOKE"))
if SMOKE:
    SEEDS, BETAS, STEPS = [0], [1.0], 200
    OUT_DIR = HERE / "results" / "exp_D1_rational_beta_scan_smoke"

# 启动时打印环境，便于确认解释器/GPU 是否正确（Spyder 里请把解释器设为 pinns 环境）
print(f"[env] python={os.sys.executable}", flush=True)
print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}"
      + (f" gpu={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""), flush=True)

# CPU 线程留余量，避免打满整机
if DEVICE.type == "cpu":
    n = os.cpu_count() or 8
    torch.set_num_threads(max(1, n - 2))


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_rng_state():
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


def glorot_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def load_reference():
    d = loadmat(str(REF_MAT))
    return d["x"].flatten(), d["t"].flatten(), d["usol"]


class MLP(nn.Module):
    def __init__(self, layers=(2, 50, 50, 50, 50, 1)):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)
        self.apply(glorot_uniform)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def pde_residual(model, x, t, nu=NU):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return (u_t + u * u_x - nu * u_xx).squeeze(-1)


def rational_weight(r, beta, eps=1e-12):
    """w(t)=(1+βt)/(1+t)，t=|r|/median|r|（批内中位数），adaptive：median 由当前 r 现算。
    与稳态 RationalWeightingLoss(raw, adaptive) 一致：w 参与 autograd。"""
    a = r.abs()
    bg = torch.median(a.detach())
    if not torch.isfinite(bg) or bg < 1e-12:
        return torch.ones_like(r)
    t = a / bg
    return (1.0 + beta * t) / (1.0 + t + eps)


def evaluate(model, x_ref, t_ref, u_ref):
    model.eval()
    xx, tt = np.meshgrid(x_ref, t_ref, indexing="ij")
    X = torch.tensor(xx.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    T = torch.tensor(tt.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_pred = model(X, T).cpu().numpy().reshape(len(x_ref), len(t_ref))
    l2_total = float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))
    l2_final = float(np.linalg.norm(u_pred[:, -1] - u_ref[:, -1]) / np.linalg.norm(u_ref[:, -1]))
    return l2_total, l2_final, u_pred


def mechanism_A_analysis(model, beta):
    """冻结训练后网络，同配点比较 g 与 g_W（权重 detach，等价 frozen-weight 口径）。"""
    model.eval()
    res = {}
    N_probe = 2048
    torch.manual_seed(42)
    x_p = (torch.rand(N_probe, 1, device=DEVICE) * 2 - 1)
    t_p = torch.rand(N_probe, 1, device=DEVICE) * 0.99
    x_p.requires_grad_(True)
    t_p.requires_grad_(True)

    u = model(x_p, t_p)
    u_x = torch.autograd.grad(u, x_p, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t_p, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_p, torch.ones_like(u_x), create_graph=True)[0]
    r = (u_t + u * u_x - NU * u_xx).squeeze(-1)

    params = list(model.parameters())
    P = sum(p.numel() for p in params)
    J = torch.zeros(N_probe, P, device=DEVICE)
    for i in range(N_probe):
        grads = torch.autograd.grad(r[i], params, retain_graph=(i < N_probe - 1))
        J[i, :] = torch.cat([g.flatten() for g in grads])

    g = J.T @ r.detach()
    w = rational_weight(r.detach(), beta).detach()
    g_W = J.T @ (w * r.detach())

    norm_g = float(torch.norm(g))
    norm_gW = float(torch.norm(g_W))
    res["grad_norm_ratio"] = norm_gW / norm_g if norm_g > 0 else 0.0
    res["grad_cosine_similarity"] = (
        float(torch.dot(g, g_W) / (norm_g * norm_gW)) if norm_g > 0 and norm_gW > 0 else 0.0
    )

    row_norms = torch.norm(J, dim=1).detach().cpu().numpy()
    x_np = x_p.detach().cpu().numpy().flatten()
    cidx = np.abs(x_np) < 0.3
    bidx = ~cidx
    res["grad_row_norm_ratio"] = float(row_norms[cidx].mean() / row_norms[bidx].mean())
    wnorm = w.detach().cpu().numpy() * row_norms
    res["weighted_grad_row_norm_ratio"] = float(wnorm[cidx].mean() / wnorm[bidx].mean())
    return res


def train_one(seed, beta, x_ref, t_ref, u_ref, t0_global):
    """单 (seed,beta)：支持断点续跑。返回 (result_dict 或 None=墙钟到点未完)。"""
    beta_dir = OUT_DIR / f"beta_{beta}"
    beta_dir.mkdir(parents=True, exist_ok=True)
    ckpt_f = beta_dir / f"ckpt_seed{seed}.pt"
    res_f = beta_dir / f"result_seed{seed}.json"
    if res_f.exists():
        return json.load(open(res_f))

    if ckpt_f.exists():
        # map_location='cpu'：RNG 状态必须保持 CPU ByteTensor，模型/优化器 load 时会自动拷到 DEVICE
        ck = torch.load(ckpt_f, map_location="cpu", weights_only=False)
        model = MLP().to(DEVICE)
        model.load_state_dict(ck["model"])
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        opt.load_state_dict(ck["opt"])
        set_rng_state(ck["rng"])
        step0 = ck["step"]
    else:
        set_seed(seed)
        model = MLP().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        step0 = 0

    t_run = time.time()
    for step in range(step0, STEPS):
        opt.zero_grad()
        x_d = torch.rand(NUM_DOMAIN, 1, device=DEVICE) * 2 - 1
        t_d = torch.rand(NUM_DOMAIN, 1, device=DEVICE) * 0.99
        r_pde = pde_residual(model, x_d, t_d)
        w = rational_weight(r_pde, beta)
        loss_pde = torch.mean(w * r_pde ** 2)

        t_bc = torch.rand(NUM_BOUNDARY, 1, device=DEVICE) * 0.99
        x_bc = torch.cat([
            torch.full((NUM_BOUNDARY // 2, 1), -1.0, device=DEVICE),
            torch.full((NUM_BOUNDARY - NUM_BOUNDARY // 2, 1), 1.0, device=DEVICE),
        ])
        loss_bc = torch.mean(model(x_bc, t_bc) ** 2)

        x_ic = torch.rand(NUM_INITIAL, 1, device=DEVICE) * 2 - 1
        u_ic = model(x_ic, torch.zeros(NUM_INITIAL, 1, device=DEVICE))
        loss_ic = torch.mean((u_ic + torch.sin(np.pi * x_ic)) ** 2)

        loss = loss_pde + loss_bc + loss_ic
        loss.backward()
        opt.step()

        if step % 5000 == 0:
            print(f"  seed={seed} beta={beta} step={step} loss={loss.item():.4e}", flush=True)
            # 定期 checkpoint：崩溃/断电后可从最近 5000 步处接续
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "rng": get_rng_state(), "step": step + 1}, ckpt_f)

        # 墙钟到点：落 checkpoint 退出（留给下次接续）；WALLCAP<=0 表示不限时（无人值守全量模式）
        if WALLCAP > 0 and time.time() - t0_global > WALLCAP and step > step0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "rng": get_rng_state(), "step": step + 1}, ckpt_f)
            print(f"  [wallcap] seed={seed} beta={beta} 存于 step={step+1}", flush=True)
            return None

    l2_total, l2_final, u_pred = evaluate(model, x_ref, t_ref, u_ref)
    result = {
        "seed": seed, "beta": beta,
        "L2_total": l2_total, "L2_final": l2_final,
        "good_basin": l2_total < 0.06,
        "u_pred_min": float(u_pred.min()), "u_pred_max": float(u_pred.max()),
        "weight_family": "rational", "weight_norm": "raw", "weight_mode": "adaptive",
        "train_time_s": time.time() - t_run,
    }
    if beta in _MECH_BETAS:
        print(f"  [机制A] beta={beta} seed={seed}...", flush=True)
        result["mechanism_A"] = mechanism_A_analysis(model, beta)
    json.dump(result, open(res_f, "w"), indent=2)
    if ckpt_f.exists():
        ckpt_f.unlink()
    print(f"  seed={seed} beta={beta}: L2_total={l2_total:.4e} L2_final={l2_final:.4e} "
          f"({result['train_time_s']:.0f}s)", flush=True)
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 60, flush=True)
    print(f"D1-rational: SEEDS={SEEDS} BETAS={BETAS} steps={STEPS} device={DEVICE}", flush=True)
    x_ref, t_ref, u_ref = load_reference()
    print(f"真解网格: x={len(x_ref)}, t={len(t_ref)}", flush=True)

    for beta in BETAS:
        for seed in SEEDS:
            r = train_one(seed, beta, x_ref, t_ref, u_ref, t0)
            if r is None:      # 墙钟到点
                return
            if WALLCAP > 0 and time.time() - t0 > WALLCAP:
                print("  [wallcap] 本轮结束，下次继续", flush=True)
                return

    # 全部完成：汇总
    all_results = []
    for beta in BETAS:
        rs = []
        for seed in SEEDS:
            rs.append(json.load(open(OUT_DIR / f"beta_{beta}" / f"result_seed{seed}.json")))
        all_results.append({"beta": beta, "results": rs})
    json.dump(all_results, open(OUT_DIR / "summary.json", "w"), indent=2)
    print("\n=== 汇总 ===", flush=True)
    for item in all_results:
        l2 = [r["L2_total"] for r in item["results"]]
        print(f"  beta={item['beta']}: med={np.median(l2):.4e} "
              f"range=[{min(l2):.4e},{max(l2):.4e}] good(<0.06)={sum(v < 0.06 for v in l2)}/{len(l2)}",
              flush=True)


if __name__ == "__main__":
    main()
