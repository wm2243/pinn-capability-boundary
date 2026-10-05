# -*- coding: utf-8 -*-
"""
一维椭圆（Poisson）方程：制造解法（MMS），用于 V2“光滑退化对照”。
================================================================================
命题：空间加权是被【局域刚性对比度分层】（如 Burgers 激波带/光滑带）激活的机制；
光滑椭圆上残差对比度场近似空间均匀、无 S/O 分离，加权应退化为平凡（β 平坦），
且光滑解是普通 PINN/低带宽好球区。本模块提供【已知光滑真解、无激波、无粘性】的干净算例。

控制方程（域 [-1,1]，齐次 Dirichlet u(±1)=0，由硬边界 bc_type='dirichlet0' 精确施加）：
    -u''(x) = f(x)

两种制造解（均为全局光滑、无局域窄层；频率是全局分布，用来把“表示需求 σ”与“空间加权需求 W”解耦）：
  single : u*(x) = sin(k (x+1)),                 f = k^2 sin(k(x+1))，端点 sin0=sin(2k)=0 要求 2k=nπ，默认 k=π/2
  multi  : u*(x) = sin(k1(x+1)) + a2 sin(k2(x+1)),f = k1^2 sin(k1(x+1)) + a2 k2^2 sin(k2(x+1))，
           要求 2k1,2k2 为 π 整数倍（默认 k1=π/2, k2=3π/2），含全局高频但仍无局域刚性。

接口与 SteadyBurgersPDE 对齐：只暴露 compute_residual(model,x_pde)；真解用 exact_np 供实验脚本建测试网格。
"""
import numpy as np
import torch


class PoissonPDE1D:
    def __init__(self, config):
        self.mode = str(config.get("pois_mode", "single"))
        self.k = float(config.get("pois_k", np.pi / 2.0))
        self.k1 = float(config.get("pois_k1", np.pi / 2.0))
        self.k2 = float(config.get("pois_k2", 3.0 * np.pi / 2.0))
        self.a2 = float(config.get("pois_a2", 0.5))
        if self.mode == "single":
            self.exact_np = lambda x: np.sin(self.k * (x + 1.0))
        elif self.mode == "multi":
            self.exact_np = lambda x: np.sin(self.k1 * (x + 1.0)) + self.a2 * np.sin(self.k2 * (x + 1.0))
        else:
            raise ValueError(f"unknown pois_mode {self.mode!r} (single|multi)")

    def _source(self, z):
        """右端 f(x)，与 exact 严格对应（解析，不经过网络）。z 为张量坐标。"""
        if self.mode == "single":
            return (self.k ** 2) * torch.sin(self.k * (z + 1.0))
        return (self.k1 ** 2) * torch.sin(self.k1 * (z + 1.0)) \
            + self.a2 * (self.k2 ** 2) * torch.sin(self.k2 * (z + 1.0))

    def compute_residual(self, model, x_pde):
        """Poisson 残差 r = -u_xx - f(x)。"""
        x_pde = x_pde.detach().clone().requires_grad_(True)
        u = model(x_pde)
        u_x = torch.autograd.grad(u, x_pde, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, torch.ones_like(u_x), create_graph=True)[0]
        return -u_xx - self._source(x_pde)
