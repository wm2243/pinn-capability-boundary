# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 06:16:27 2026
@author: wwm
"""

import torch
import torch.nn as nn
import numpy as np


class RationalWeightingLoss(nn.Module):
    """
    有理自适应加权 PDE 损失函数（统一公式）

    w = (1 + β·ρ) / (1 + ρ),   ρ = R² / (M + R²)

    支持两种模式:
      - adaptive: R = |PDE残差|, 每步动态计算
      - spatial:  R = |∇u_true|, 训练前一次性计算并固定
    """

    def __init__(self, config, device):
        super().__init__()
        self.beta = config.get('loss_beta', 1)
        self.M = config.get('loss_M', 0.01)
        self.weight_mode = config.get('weight_mode', 'adaptive')
        self.device = device

        # ★ spatial 模式的固定权重缓冲区
        # register_buffer: 不参与梯度更新，但会随 .to(device) 自动迁移
        self.register_buffer('w_fixed', None)

    # ==================== 统一权重计算核心 ====================

    @staticmethod
    def _compute_weights_from_signal(R, beta, M, detach_max=True):
        """
        统一有理权重计算。

        Args:
            R:           信号张量 (N,)，|残差| 或 |∇u_true|
            beta:        超参数 β
            M:           正则化常数（与 config['loss_M'] 对应）
            detach_max:  adaptive=True 阻断内生性梯度; spatial=False 省去多余操作

        Returns:
            weights: (N,) 权重张量，值域 [1, β]
        """
        R_sq = R ** 2
        M_val = M
        if detach_max:
            # adaptive 模式下 M 也参与计算图，需 detach 防止内生性
            if isinstance(M_val, torch.Tensor):
                M_val = M_val.detach()
        M_val = M_val + 1e-12  # 防除零

        rho = R_sq / (M_val + R_sq)                     # ρ ∈ [0, 1)
        weights = (1.0 + beta * rho) / (1.0 + rho)      # w ∈ [1, β]
        return weights

    # ==================== Spatial 初始化 ====================

    def set_spatial_weight(self, u_true, x_grid, config=None):
        """
        用真解梯度预计算固定空间权重。
        仅在 weight_mode=='spatial' 时生效。
        由 PINNTrainer 在初始化时调用一次。
        """
        if self.weight_mode != 'spatial':
            return

        if u_true is None or x_grid is None:
            raise ValueError("spatial mode requires u_true and x_grid")
        
        # ★ 修复：确保先转到 CPU 再转 numpy
        if isinstance(u_true, torch.Tensor):
            u_true = u_true.detach().cpu()
        if isinstance(x_grid, torch.Tensor):
            x_grid = x_grid.detach().cpu()
        
        u_np = np.asarray(u_true).flatten()
        x_np = np.asarray(x_grid).flatten()

        # R = |∇u_true|
        grad_u = np.abs(np.gradient(u_np, x_np))
        R_tensor = torch.tensor(grad_u, dtype=torch.float32, device=self.device)

        # ★ 复用统一公式，detach_max=False（numpy 转来的 tensor 本身无 grad）
        w_vals = self._compute_weights_from_signal(
            R_tensor, self.beta, self.M, detach_max=False
        )

        self.w_fixed = w_vals

        print(f"[RationalWeightingLoss] Spatial weight initialized | "
              f"β={self.beta} | M={self.M} | "
              f"range=[{w_vals.min().item():.4f}, {w_vals.max().item():.4f}] | "
              f"N={len(w_vals)}")

    # ==================== Adaptive 权重计算 ====================

    def _compute_adaptive_weights(self, residual):
        """
        自适应权重计算（保留原方法名，内部改为调用统一公式）。
        R = |residual|
        """
        return self._compute_weights_from_signal(
            torch.abs(residual), self.beta, self.M, detach_max=True
        )

    # ==================== 对外接口 ====================

    def get_current_weights(self, residual: torch.Tensor, x_pde: torch.Tensor = None) -> torch.Tensor:
        # ========== ★ 新增：输入校验（方法最开头） ==========
        if residual is None or residual.numel() == 0:
            raise ValueError(f"get_current_weights: invalid residual (None or empty)")
        # ===================================================
    
        # ---------- 原有三分支逻辑（保持不变） ----------
        if self.weight_mode == 'adaptive':
            result = self._compute_adaptive_weight(residual)       # ← 注意：改为赋值给 result
        elif self.weight_mode == 'spatial':
            if self._weight_interp is None:
                raise RuntimeError("Spatial weight interpolator not initialized. Call set_spatial_weight() first.")
            if x_pde is None:
                raise ValueError("Spatial mode requires x_pde argument")
            x_np = x_pde.detach().cpu().numpy().flatten()
            w_np = self._weight_interp(x_np)
            result = torch.tensor(w_np, dtype=residual.dtype, device=residual.device)  # ← 改为 result
        else:
            result = torch.ones_like(residual)                     # ← 改为 result
        # -----------------------------------------------
    
        # ========== ★ 新增：输出校验（return 之前） ==========
        if result is None:
            raise RuntimeError(f"get_current_weights returned None (mode={self.weight_mode})")
        if result.shape != residual.shape:
            raise RuntimeError(
                f"Weight shape mismatch: got {result.shape}, expected {residual.shape}"
            )
        # ===================================================
    
        return result    # ← 统一返回 result

    def forward(self, residual, x_pde=None):
        """
        计算加权 PDE 损失

        Args:
            residual: PDE 残差张量 (N,) 或 (N,1)
            x_pde:    对应坐标张量，仅 spatial 模式且点数不匹配时需要

        Returns:
            weighted_loss: 标量损失值
            loss_dict:     包含各项损失的字典（兼容原有接口）
        """
        
        # ★ 核心分支：根据模式选择权重来源
        if self.weight_mode == 'spatial':
            weights = self.get_current_weights(residual, x_pde=x_pde)
        else:
            weights = self._compute_adaptive_weights(residual)

        # 加权 MSE 损失（与原有逻辑完全一致）
        weighted_residual_sq = weights * (residual ** 2)
        loss = weighted_residual_sq.mean()

        # 保持原有返回格式，确保 trainer 中的日志/监控代码无需改动
        loss_dict = {
            'pde_loss': loss.item(),
            'weight_mean': weights.mean().item(),
            'weight_max': weights.max().item(),
        }

        return loss, loss_dict