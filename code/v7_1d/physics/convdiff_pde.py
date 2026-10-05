# -*- coding: utf-8 -*-
"""
一维定常对流-扩散（奇摄动）方程：S2b 边界层算例，用于检验“局域刚性”迁移到椭圆型/抛物型定常问题时，
空间加权机制是否依旧成立（与 S2a 光滑 Poisson 退化对照互补）。
================================================================================
控制方程（域 [-1,1]，非齐次 Dirichlet，由硬边界 bc_type='lift_linear', bc_left=0, bc_right=1 精确施加）：
    -eps * u''(x) + b * u'(x) = 0 ,   u(-1)=0, u(1)=1
eps 为扩散系数（奇摄动参数），b 为对流速度（默认 1）。eps -> 0 时在 x=1 附近形成厚度 O(eps/b) 的边界层，
是 Burgers 激波“局域刚性结构”在椭圆/对流扩散问题上的对应物。

解析真解（特征根 0 与 b/eps），数值稳定形式（所有指数 <=0，避免 eps 很小时 e^{b/eps} 上溢）：
    u*(x) = ( exp(b(x-1)/eps) - exp(-2b/eps) ) / ( 1 - exp(-2b/eps) )
易验 u*(-1)=0, u*(1)=1；层外 (x<1) u*≈0，层内 [1-O(eps),1] 由 0 陡升到 1。

接口与 PoissonPDE1D / SteadyBurgersPDE 对齐：只暴露 compute_residual(model,x_pde)，真解用 exact_np。
"""
import numpy as np
import torch


class ConvDiffPDE1D:
    def __init__(self, config):
        self.eps = float(config["cd_eps"])
        assert self.eps > 0, "cd_eps 必须为正"
        self.b = float(config.get("cd_b", 1.0))
        self.left = float(config.get("bc_left", 0.0))
        self.right = float(config.get("bc_right", 1.0))

    def exact_np(self, x):
        """解析真解，x 为 ndarray（任意形状），返回同形。"""
        e = self.eps; b = self.b; lo, hi = self.left, self.right
        z = np.exp(np.clip(b * (x - 1.0) / e, -700, 700)) - np.exp(-2.0 * b / e)
        den = 1.0 - np.exp(-2.0 * b / e)
        u01 = z / den                       # 左端0、右端1 的基准解
        return lo + (hi - lo) * u01

    def compute_residual(self, model, x_pde):
        """残差 r = -eps u_xx + b u_x（右端源项为 0）。"""
        x_pde = x_pde.detach().clone().requires_grad_(True)
        u = model(x_pde)
        u_x = torch.autograd.grad(u, x_pde, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, torch.ones_like(u_x), create_graph=True)[0]
        return -self.eps * u_xx + self.b * u_x
