# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:50:02 2026

@author: wwm
"""

import torch

class BurgersPDE:
    """
    纯粹的 Burgers 物理方程计算器
    职责单一：只负责根据输入坐标计算 PDE 残差，不包含任何 Loss 计算逻辑
    """
    def __init__(self, config):
        self.nu = config['nu']
        self.is_steady = config['is_steady']

    def compute_residual(self, model, x_pde, t_pde):
        """
        计算 Burgers 方程的残差: u_t + u*u_x - nu*u_xx = 0
        """
        # 开启梯度追踪，以便后续使用 autograd 求导
        x_pde.requires_grad_(True)
        t_pde.requires_grad_(True)
        
        # 1. 前向传播获取预测值 u
        u = model(x_pde, t_pde)
        
        # 2. 计算空间导数 (一阶和二阶)
        du_dx = torch.autograd.grad(u, x_pde, torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_pde, torch.ones_like(du_dx), create_graph=True)[0]
        
        # 3. 计算时间导数 (稳态模式下强制为 0)
        if self.is_steady:
            du_dt = torch.zeros_like(u)
        else:
            du_dt = torch.autograd.grad(u, t_pde, torch.ones_like(u), create_graph=True)[0]
        
        # 4. 返回纯粹的 PDE 残差张量
        return du_dt + u * du_dx - self.nu * d2u_dx2