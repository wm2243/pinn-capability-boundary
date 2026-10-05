# -*- coding: utf-8 -*-
"""
硬边界 + 随机傅里叶特征 PINN（V6 修订）

相对旧版的改动：
  1) 傅里叶层把“随机方向 B0”与“带宽 sigma”解耦：B0=randn 固定为 buffer，
     前向用 B0*sigma_eff；训练中可通过 set_sigma() 实时改带宽 -> 支持 σ 表示课程(Phase 0)。
     默认 sigma_eff=初始 fourier_scale，行为与旧版完全一致。
  2) 明确层数语义：num_layers=L 时，隐藏块数=L，实际 nn.Linear 个数 = L+1
     （首个输入层 + (L-1) 个隐藏层 + 1 个输出层），论文按“L 个 tanh 隐藏块”描述。
  3) HardBCPINN 提供 sigma 课程钩子 set_sigma / get_sigma，trainer 按 config 驱动。

硬边界（稳态 Burgers，u(-1)=1, u(1)=-1）：u(x) = -x + (1-x^2) N(γ(x))
"""

import torch
import torch.nn as nn
import numpy as np


class FourierFeatures(nn.Module):
    """
    随机傅里叶特征: γ(x) = [cos(2π (B0·σ) x), sin(2π (B0·σ) x)]
    B0 ∈ R^{d×m} 为固定随机方向（不含尺度），σ 为可在线调整的带宽。
    """
    def __init__(self, input_dim, mapping_size, scale=1.0):
        super().__init__()
        # 只存随机方向，尺度解耦到 self.sigma，便于 σ 课程实时调整
        self.register_buffer('B0', torch.randn(input_dim, mapping_size))
        self.sigma = float(scale)
        self.output_dim = 2 * mapping_size

    def set_sigma(self, sigma):
        self.sigma = float(sigma)

    def get_sigma(self):
        return self.sigma

    def forward(self, x):
        proj = 2.0 * np.pi * x @ (self.B0 * self.sigma)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class HardBCPINN(nn.Module):
    """硬边界约束 + 傅里叶特征网络：u(x)=A(x)+B(x)N(γ(x))，A=-bc_amp·x, B=1-x²；
    bc_amp 默认 1.0（A=-x，兼容薄激波渐近），大 ν 应传 tanh(1/(2ν)) 使端点与真解一致。"""
    def __init__(self, config):
        super(HardBCPINN, self).__init__()
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        # feature='fourier'（默认，行为与旧版逐字节一致）| 'mlp'（关闭随机傅里叶映射、直接以 x 输入，
        # 用于大 ν 光滑退化 null test，隔离“高频嵌入 vs 光滑解”的表示器失配）
        self.feature = str(config.get('feature', 'fourier'))
        # 硬边界端点幅值：u(x)=-amp*x+(1-x²)N，端点 u(-1)=+amp,u(1)=-amp。
        # 默认 1.0 = 旧行为 A=-x（ν→0 时 tanh(1/2ν)→1 的渐近值，逐字节兼容）；
        # 大 ν 光滑端真解端点为 ±tanh(1/2ν)，应由 config['bc_amp']=tanh(1/(2ν)) 显式传入。
        self.bc_amp = float(config.get('bc_amp', 1.0))
        # 硬边界类型（V2 方程迁移；默认 'burgers' 与 V1 逐字节一致）：
        #  burgers      非齐次 u(-1)=+amp,u(1)=-amp，提升 A=-amp*x
        #  dirichlet0   齐次 u(±1)=0（Poisson 等），A=0
        #  lift_linear  一般两点非齐次 u(-1)=bc_left,u(1)=bc_right，A 为线性提升
        # 三者湮灭因子均为 B=1-x²（端点精确为 0）。
        self.bc_type = str(config.get('bc_type', 'burgers'))
        self.bc_left = float(config.get('bc_left', 0.0))
        self.bc_right = float(config.get('bc_right', 0.0))
        self.num_layers = num_layers
        if self.feature == 'mlp':
            self.fourier = None
            in_dim = 1
        else:
            mapping_size = config.get('mapping_size', 64)
            sigma = config.get('fourier_scale', 1.0)
            self.fourier = FourierFeatures(input_dim=1, mapping_size=mapping_size, scale=sigma)
            in_dim = self.fourier.output_dim

        # num_layers 个 tanh 隐藏块 + 1 个线性输出 => Linear 总数 num_layers+1
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.n_linear = num_layers + 1

    # ------- σ 表示课程钩子（MLP 档为 no-op，保持 trainer 调用接口不变）-------
    def set_sigma(self, sigma):
        if self.fourier is not None:
            self.fourier.set_sigma(sigma)

    def get_sigma(self):
        return self.fourier.get_sigma() if self.fourier is not None else 1.0

    def boundary_lift(self, x):
        if self.bc_type == 'burgers':
            return -self.bc_amp * x
        if self.bc_type == 'dirichlet0':
            return torch.zeros_like(x)
        if self.bc_type == 'lift_linear':
            return 0.5 * (1.0 - x) * self.bc_left + 0.5 * (1.0 + x) * self.bc_right
        raise ValueError(f"unknown bc_type {self.bc_type!r} (burgers|dirichlet0|lift_linear)")

    def forward(self, x):
        gamma_x = x if self.fourier is None else self.fourier(x)
        N = self.net(gamma_x)
        A = self.boundary_lift(x)
        B = 1.0 - x ** 2
        return A + B * N
