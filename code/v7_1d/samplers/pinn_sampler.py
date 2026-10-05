# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:50:32 2026

@author: wwm
"""
import torch

class PINNSampler:
    """
    稳态 PINN 采样器
    支持随机采样（Adam阶段）和残差自适应采样（Adaptive）
    """
    def __init__(self, domain_x, device, use_adaptive=False, buffer_size=2000):
        """
        Args:
            domain_x: tuple, 计算域边界 (x_min, x_max)
            device: torch.device, 计算设备 ('cpu' 或 'cuda')
            use_adaptive: bool, 是否开启自适应采样
            buffer_size: int, 自适应采样的高残差点缓存池大小
        """
        self.domain_x = domain_x
        self.device = device
        self.use_adaptive = use_adaptive
        self.buffer_size = buffer_size
        
        # 内部状态：L-BFGS 阶段的固定点集
        self.fixed_points = None 
        
        # 内部状态：自适应采样的 Buffer
        # 初始化为空张量，并严格指定 device 和 dtype，防止后续 cat 报错
        self.adaptive_buffer = torch.empty(0, 2, device=self.device, dtype=torch.float32)

    def sample(self, n_pde, mode='random'):
        """
        采样内部配点
        Args:
            n_pde: 当前批次需要的点数
            mode: 'random' (Adam用) 或 'fixed' (L-BFGS用)
        """
        # --- 模式 1: L-BFGS 固定点模式 ---
        if mode == 'fixed':
            if self.fixed_points is None:
                self.fixed_points = self._generate_random(n_pde)
            return {'x_pde': self.fixed_points}

        # --- 模式 2: Adam 随机/自适应模式 ---
        
        # A. 基础随机采样 (保证全局探索)
        # 如果未开启自适应，或者 buffer 为空，则全部使用随机点
        if not self.use_adaptive or len(self.adaptive_buffer) == 0:
            return {'x_pde': self._generate_random(n_pde)}

        # B. 混合采样策略 (50% 随机 + 50% 自适应)
        n_random = int(n_pde * 0.5)
        x_random = self._generate_random(n_random)
        
        # 从 buffer 中随机抽取自适应点 (buffer 里的点已经是历史高残差点)
        n_adaptive = n_pde - n_random
        indices = torch.randperm(len(self.adaptive_buffer), device=self.device)[:n_adaptive]
        x_adaptive = self.adaptive_buffer[indices, 0].unsqueeze(1)

        # 拼接并返回
        x_total = torch.cat([x_random, x_adaptive], dim=0)
        return {'x_pde': x_total}

    def update_buffer(self, new_x, new_residuals):
        """
        更新高残差点缓存池 (仅在开启自适应时由 Trainer 调用)
        Args:
            new_x: 当前批次的输入点 [N, 1]
            new_residuals: 对应的 PDE 残差绝对值 [N, 1]
        """
        if not self.use_adaptive:
            return

        # 1. 确保传入的数据在正确的设备上，并打包
        new_x = new_x.to(self.device)
        new_residuals = new_residuals.to(self.device)
        new_data = torch.cat([new_x, new_residuals], dim=1)
        
        # 2. 合并旧 Buffer 和新数据
        if len(self.adaptive_buffer) == 0:
            combined = new_data
        else:
            combined = torch.cat([self.adaptive_buffer, new_data], dim=0)
        
        # 3. Top-K 淘汰机制：只保留残差最大的 buffer_size 个点
        if len(combined) > self.buffer_size:
            _, top_indices = torch.topk(combined[:, 1], self.buffer_size)
            combined = combined[top_indices]
            
        self.adaptive_buffer = combined

    def _generate_random(self, n):
        """内部方法：在物理域内生成均匀随机点"""
        x_min, x_max = self.domain_x
        x_pde = (x_max - x_min) * torch.rand(n, 1, device=self.device) + x_min
        return x_pde