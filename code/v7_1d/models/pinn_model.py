# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:48:46 2026

@author: wwm
"""


import torch
import torch.nn as nn

class HardBCPINN(nn.Module):
    """
    带有硬边界约束的 PINN 模型
    通过数学变换 u(x) = A(x) + B(x)*N(x) 强制满足边界条件
    """
    def __init__(self, config):
        super(HardBCPINN, self).__init__()
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        # 硬边界端点幅值，默认 1.0=旧行为 A=-x（兼容薄激波渐近）；大 ν 传 tanh(1/(2ν))。
        self.bc_amp = float(config.get('bc_amp', 1.0))
        
        # 基础神经网络 (输入1维x，输出1维N)
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播：应用硬边界变换
        稳态 Burgers 边界: u(-1)=1, u(1)=-1
        A(x) = -x  (满足 A(-1)=1, A(1)=-1)
        B(x) = 1 - x^2 (满足 B(-1)=0, B(1)=0)
        """
        N = self.net(x)
        A = -self.bc_amp * x
        B = 1.0 - x**2
        return A + B * N