# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:50:02 2026

@author: wwm
"""

import torch

class SteadyBurgersPDE:
    """
    稳态 Burgers 物理方程计算器
    方程: u * u_x - nu * u_xx = 0
    """
    def __init__(self, config):
        self.nu = config['nu']

    def compute_residual(self, model, x_pde):
        """计算稳态 Burgers 残差"""
        x_pde = x_pde.detach().clone().requires_grad_(True)
        u = model(x_pde)  # 注意：这里已经包含了硬边界变换
        
        # 自动微分求导
        du_dx = torch.autograd.grad(u, x_pde, torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_pde, torch.ones_like(du_dx), create_graph=True)[0]
        
        # 稳态残差
        return u * du_dx - self.nu * d2u_dx2