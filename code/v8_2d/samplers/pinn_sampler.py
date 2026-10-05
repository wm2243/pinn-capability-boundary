# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:50:32 2026

@author: wwm
"""
import torch

class PINNSampler:
    """
    PINN 采样器
    根据配置动态生成 PDE 内部配点、边界配点和初始条件配点
    """
    def __init__(self, config, device):
        self.device = device
        self.x_min, self.x_max = config['domain_x']
        self.t_min, self.t_max = config['domain_t']
        self.is_steady = config['is_steady']

    def sample(self, n_pde, n_bc):
        """统一采样接口"""
        # PDE 内部配点
        x_pde = torch.rand(n_pde, 1, device=self.device) * (self.x_max - self.x_min) + self.x_min
        t_pde = torch.rand(n_pde, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        
        # 边界配点 (x = ±1)
        half_bc = n_bc // 2
        x_bc = torch.cat([
            torch.full((half_bc, 1), self.x_min, device=self.device),
            torch.full((n_bc - half_bc, 1), self.x_max, device=self.device)
        ])
        t_bc = torch.rand(n_bc, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        
        data = {'x_pde': x_pde, 't_pde': t_pde, 'x_bc': x_bc, 't_bc': t_bc}
        
        # 瞬态问题需要额外采样初始条件 (t=0)
        if not self.is_steady:
            x_ic = torch.rand(n_bc, 1, device=self.device) * (self.x_max - self.x_min) + self.x_min
            t_ic = torch.zeros(n_bc, 1, device=self.device)  # 初始时刻 t=0
            data.update({'x_ic': x_ic, 't_ic': t_ic})
            
        return data