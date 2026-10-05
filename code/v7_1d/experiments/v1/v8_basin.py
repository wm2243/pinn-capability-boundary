# -*- coding: utf-8 -*-
"""
V8 实验共享薄层：进好盆地 + 在冻结盆地上装载模型（只做封装，不复制训练逻辑）。
训练/课程/Gate/冻结全部复用 run_two_stage_v6，C1/D1/D2/R1 脚本统一从这里取，避免重复代码。
"""
from pathlib import Path as _Path
import sys as _sys
_HERE = _Path(__file__).resolve()
for _p in (str(_HERE.parent), str(_HERE.parent.parent)):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import numpy as np
import torch
import run_two_stage_v6 as r2s

DEVICE = r2s.DEVICE


def set_out_dir(d):
    """让复用的 r2s 函数把日志/产物落到指定实验目录。"""
    r2s.OUT = str(d)


def enter_basin(nu, seed, xt, ut, return_all=False):
    """Phase0：σ课程 + multi-start 进入并冻结一个好盆地，返回 basin(dict)。"""
    return r2s.run_phase0(nu, seed, xt, ut, return_all=return_all)


def frozen_at(basin, nu, beta, lr=5e-4, tag="vz"):
    """从同一好盆地装载冻结网络与冻结 t 场，只改 β（静态冻结算子，定理 6.1 口径）。"""
    cfg = r2s.base_cfg(nu, f"{tag}_b{beta}")
    cfg.update(loss_beta=float(beta), weight_mode="adaptive")
    model, pde, crit, opt, samp, vis, cfg = r2s.build_all(cfg, lr, float(beta))
    model.load_state_dict(basin["state"]); model.set_sigma(basin["sigma_end"])
    crit.load_frozen_field(basin["t_field"], beta=float(beta))
    model.eval()
    return model, pde, crit, opt, samp, vis, cfg


def train_phase1(basin, nu, beta, seed, xt, ut, epochs=None):
    """在好盆地上做 Phase1 固定 β 训练（D1 用），复用 r2s.run_phase1。"""
    if epochs is not None:
        r2s.PHASE1_EPOCHS = int(epochs)
    return r2s.run_phase1(nu, float(beta), basin, seed, xt, ut)


def grid_values(model, n=2048):
    """统一细网格上的网络取值（R1 多起点粗解两两距离用）。"""
    x = torch.linspace(-1, 1, n, device=DEVICE).view(-1, 1)
    with torch.no_grad():
        v = model(x).detach().cpu().numpy().reshape(-1)
    return v


def weighted_residual_norm(model, pde, crit, x):
    """冻结权重下的 ||r||_W = sqrt(mean(w r^2))，返回 (标量, r_np, w_np)。"""
    with torch.enable_grad():
        r = pde.compute_residual(model, x).reshape(-1)
    with torch.no_grad():
        w = crit.get_current_weights(r.detach(), x_pde=x.detach()).reshape(-1)
        nrw = float(torch.sqrt((w * r.reshape(-1) ** 2).mean()))
    return nrw, r.detach().cpu().numpy(), w.detach().cpu().numpy()
