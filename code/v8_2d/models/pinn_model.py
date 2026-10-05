# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:48:46 2026

@author: wwm
"""

import torch
import torch.nn as nn

class PINNModel(nn.Module):
    """
    通用的 PINN 神经网络模型
    无论稳态还是瞬态，输入始终统一为 (x, t)，通过配置控制行为
    """
    def __init__(self, config):
        super(PINNModel, self).__init__()
        hidden_dim = config['hidden_dim']  # 神经网络的宽度
        num_layers = config['num_layers']  # num_layers 层全连接层（不含输出层）
        
        # 输入维度固定为 2 (空间 x, 时间 t)
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))  # 输出物理量 u
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        """将时空坐标拼接后输入网络"""
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)