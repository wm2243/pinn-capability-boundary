# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:39:26 2026

@author: wwm
"""

import torch
import torch.nn as nn

class PINNLoss(nn.Module):
    """
    纯 PDE 损失聚合器
    因为使用了硬边界，这里不再需要边界损失！
    """
    def __init__(self, config):
        super(PINNLoss, self).__init__()

    def forward(self, pde_residual):
        """
        仅计算 PDE 残差的 MSE
        """
        loss_pde = torch.mean(pde_residual ** 2)
        return loss_pde, {'pde': loss_pde.item()}