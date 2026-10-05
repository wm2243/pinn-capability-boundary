# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:39:26 2026

@author: wwm
"""

import torch
import torch.nn as nn

class PINNLoss(nn.Module):
    """
    通用的 PINN 损失聚合器
    职责：接收各个模块算出的残差，负责加权、聚合、计算 MSE
    """
    def __init__(self, config):
        super(PINNLoss, self).__init__()
        # 从配置中读取各项损失的权重
        self.w_pde = config.get('w_pde', 1.0)
        self.w_bc = config.get('w_bc', 100.0)
        self.w_ic = config.get('w_ic', 100.0)  # 初始条件权重

    def forward(self, pde_residual, bc_pred, bc_true, ic_pred=None, ic_true=None):
        """
        聚合所有的损失项，并返回总 Loss 和监控字典
        """
        # 1. PDE 残差损失
        loss_pde = torch.mean(pde_residual ** 2)
        
        # 2. 边界条件损失
        loss_bc = torch.mean((bc_pred - bc_true) ** 2)
        
        # 3. 初始条件损失 (瞬态问题需要)
        loss_ic = torch.tensor(0.0, device=pde_residual.device)
        if ic_pred is not None and ic_true is not None:
            loss_ic = torch.mean((ic_pred - ic_true) ** 2)
        
        # 4. 加权求和
        total_loss = self.w_pde * loss_pde + self.w_bc * loss_bc + self.w_ic * loss_ic
        
        # 返回总 Loss 和各项明细 (方便监控)
        return total_loss, {
            'pde': loss_pde.item(), 
            'bc': loss_bc.item(), 
            'ic': loss_ic.item()
        }