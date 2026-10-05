# -*- coding: utf-8 -*-
"""
一维定常转向点奇摄动（内部层）方程：S2d 算例
================================================================================
检验“局域刚性/陡变落在域内部”时，Burgers 激波的两阶段配置能否迁移到【线性、椭圆型】问题。
与 S2b/S2c 的贴边边界层相对照：内部层居中 x=0，硬约束湮灭因子 (1-x²) 在 x=0≈1，网络可自由表达。

控制方程（对流速度 a(x)=-x 在 x=0 变号，流场自两侧汇聚于转向点，形成内部过渡层）：
    -eps * u''(x) - x * u'(x) = 0 ,  x in (-1,1)，u(-1)=0, u(1)=1。
令 v=u'：eps v' + x v = 0 -> v = C exp(-x²/(2eps))，积分并归一化边界得 erf 型内部层：
    u*(x) = 1/2 [ 1 + erf(x/sqrt(2 eps)) / erf(1/sqrt(2 eps)) ]。
层厚 O(sqrt(eps))（10–90 全宽 ≈ 2.563 sqrt(eps)），居中、单调增；两端 u*→0/1，
线性提升 A=½(1+x) 在两端恰好匹配渐近值，故 N*=(u*-A)/(1-x²) 全程光滑有界（与贴边层 N* 发散对照）。

接口与 ConvDiffPDE1D 对齐：compute_residual(model,x_pde)、exact_np(x)。
"""
import numpy as np
import torch
from scipy.special import erf


class TurningPointPDE1D:
    def __init__(self, config):
        self.eps = float(config["tp_eps"])
        assert self.eps > 0, "tp_eps 必须为正"
        self.left = float(config.get("bc_left", 0.0))
        self.right = float(config.get("bc_right", 1.0))

    def exact_np(self, x):
        """erf 型内部层精确解（端点归一化，严格满足 u(-1)=0,u(1)=1）。"""
        z = 1.0 / np.sqrt(2.0 * self.eps)
        e1 = erf(z)
        u01 = 0.5 * (1.0 + erf(x / np.sqrt(2.0 * self.eps)) / e1)
        return self.left + (self.right - self.left) * u01

    def compute_residual(self, model, x_pde):
        """残差 r = -eps u_xx - x u_x（右端源项为 0）。"""
        x_pde = x_pde.detach().clone().requires_grad_(True)
        u = model(x_pde)
        u_x = torch.autograd.grad(u, x_pde, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, torch.ones_like(u_x), create_graph=True)[0]
        return -self.eps * u_xx - x_pde * u_x
